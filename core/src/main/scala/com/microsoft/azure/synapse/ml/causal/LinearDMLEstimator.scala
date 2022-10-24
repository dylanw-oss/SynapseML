// Copyright (C) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License. See LICENSE in project root for information.

package com.microsoft.azure.synapse.ml.causal

import com.microsoft.azure.synapse.ml.codegen.Wrappable
import com.microsoft.azure.synapse.ml.train._
import com.microsoft.azure.synapse.ml.core.schema.SchemaConstants
import com.microsoft.azure.synapse.ml.core.utils.StopWatch
import com.microsoft.azure.synapse.ml.logging.BasicLogging
import com.microsoft.azure.synapse.ml.stages.DropColumns
import org.apache.spark.annotation.DeveloperApi
import org.apache.spark.ml._
import org.apache.spark.ml.classification._
import org.apache.spark.ml.feature.VectorAssembler
import org.apache.spark.ml.linalg.DenseVector
import org.apache.spark.ml.param._
import org.apache.spark.ml.param.shared.{HasFeaturesCol, HasLabelCol, HasWeightCol}
import org.apache.spark.ml.util._
import org.apache.spark.sql._
import org.apache.spark.sql.types._
import org.apache.spark.ml.regression.{GeneralizedLinearRegression, GeneralizedLinearRegressionModel, RandomForestRegressor, Regressor}

import scala.concurrent.Future
import scala.util.Success

/** Linear Double ML estimators. The estimator follows the two stage process,
 *  where a set of nuisance functions are estimated in the first stage in a crossfitting manner
 *  and a final stage estimates the average treatment effect (ATE) model.
 *  Our goal is to estimate the constant marginal ATE Theta(X)
 *
 *  In this estimator, the ATE is estimated by using the following estimating equations:
 *  .. math ::
 *      Y - \\E[Y | X, W] = \\Theta(X) \\cdot (T - \\E[T | X, W]) + \\epsilon
 *
 *  Thus if we estimate the nuisance functions :math:`q(X, W) = \\E[Y | X, W]` and
 *  :math:`f(X, W)=\\E[T | X, W]` in the first stage, we can estimate the final stage ate for each
 *  treatment t, by running a regression, minimizing the residual on residual square loss,
 *  estimating Theta(X) is a final regression problem, regressing tilde{Y} on X and tilde{T})
 *
 *  .. math ::
 *       \\hat{\\theta} = \\arg\\min_{\\Theta}\
 *       \E_n\\left[ (\\tilde{Y} - \\Theta(X) \\cdot \\tilde{T})^2 \\right]
 *
 * Where
 * `\\tilde{Y}=Y - \\E[Y | X, W]` and :math:`\\tilde{T}=T-\\E[T | X, W]` denotes the
 * residual outcome and residual treatment.
 *
 * The nuisance function :math:`q` is a simple regression problem and user
 * can use setOutcomeModel to set an arbitrary sparkml regressor that is internally used to solve this regression problem
 *
 * The problem of estimating the nuisance function :math:`f` is also a regression problem and user
 * can use setTreatmentModel to set an arbitrary sparkml regressor that is internally used to solve this regression problem.
 *
 * The input categorical treatment is one-hot encoded (excluding the lexicographically smallest treatment which is used as the baseline)
 * and the `predict_proba` method of the treatment model classifier is used to residualize the one-hot encoded treatment.

      The final stage is (potentially multi-task) linear regression problem with outcomes the labels
      :math:`\\tilde{Y}` and regressors the composite features
      :math:`\\tilde{T}\\otimes \\phi(X) = \\mathtt{vec}(\\tilde{T}\\cdot \\phi(X)^T)`.
      The :class:`.DML` takes as input parameter
      ``model_final``, which is any linear sparkml regressor that is internally used to solve this
      (multi-task) linear regresion problem.
 */
//noinspection ScalaStyle
class LinearDMLEstimator(override val uid: String)
  extends AutoTrainer[LinearDMLModel]
    with LinearDMLParams
    with BasicLogging {

  logClass()

  def this() = this(Identifiable.randomUID("LinearDMLEstimator"))

  override def modelDoc: String = "LinearDML to run"

  setDefault(featuresCol, this.uid + "_features")

  /** Fits the LinearDML model.
   *
   * @param dataset The input dataset to train.
   * @return The trained LinearDML model.
   */
  override def fit(dataset: Dataset[_]): LinearDMLModel = {
    logFit({
      if (get(weightCol).isDefined) {
        getTreatmentModel match {
          case w: HasWeightCol => w.set(w.weightCol, getWeightCol)
          case _ => throw new Exception("The defined treatment model does not support weightCol.")
        }
        getOutcomeModel match {
          case w: HasWeightCol => w.set(w.weightCol, getWeightCol)
          case _ => throw new Exception("The defined outcome model does not support weightCol.")
        }
      }

      getTreatmentModel match {
        case m: HasLabelCol with HasFeaturesCol =>
          m.set(m.labelCol, getTreatmentCol)
          if (isDefined(featurizationModel)) {
            m.set(m.featuresCol, getFeaturesCol)
          } else {
            m.set(m.featuresCol, "treatment_features")
          }
        case _ => throw new Exception("The defined treatment model does not support HasLabelCol and HasFeaturesCol.")
      }

      getOutcomeModel match {
        case m: HasLabelCol with HasFeaturesCol =>
          m.set(m.labelCol, getOutcomeCol)
          if (isDefined(featurizationModel)) {
            m.set(m.featuresCol, getFeaturesCol)
          } else {
            m.set(m.featuresCol, "outcome_features")
          }
        case _ => throw new Exception("The defined outcome model does not support HasLabelCol and HasFeaturesCol.")
      }

      val treatmentPredictionColName = getTreatmentModel match {
        case classifier: ProbabilisticClassifier[_, _, _] => classifier.getProbabilityCol
        case regressor: Regressor[_, _, _] => regressor.getPredictionCol
      }

      val outcomePredictionColName = getOutcomeModel match {
        case classifier: ProbabilisticClassifier[_, _, _] => classifier.getProbabilityCol
        case regressor: Regressor[_, _, _] => regressor.getPredictionCol
      }

      val preprocessedDF = if (get(featurizationModel).isDefined) {
        getFeaturizationModel.fit(dataset).transform(dataset)
      } else {
        val datasetWithTreatmentFeatures = getTreatmentModel match {
          case classifier: ProbabilisticClassifier[_, _, _] =>
            val trainer = new TrainClassifier().setFeaturesCol("treatment_features").setModel(getTreatmentModel).setLabelCol(getTreatmentCol).setExcludedFeatureCols(Array(getOutcomeCol))
            val (processedDF, _, _, _) = trainer.getFeaturizedDataAndModel(dataset)
            processedDF
          case regressor: Regressor[_, _, _] =>
            val trainer = new TrainRegressor().setFeaturesCol("treatment_features").setModel(getTreatmentModel).setLabelCol(getTreatmentCol).setExcludedFeatureCols(Array(getOutcomeCol))
            val (processedDF, _) = trainer.getFeaturizedDataAndModel(dataset)
            processedDF
        }
        val datasetWithTreatmentAndOutcomeFeatures = getOutcomeModel match {
          case classifier: ProbabilisticClassifier[_, _, _] =>
            val trainer = new TrainClassifier().setFeaturesCol("outcome_features").setModel(getOutcomeModel).setLabelCol(getOutcomeCol).setExcludedFeatureCols(Array(getTreatmentCol, "treatment_features"))
            val (processedDF, _, _, _) = trainer.getFeaturizedDataAndModel(datasetWithTreatmentFeatures)
            processedDF
          case regressor: Regressor[_, _, _] =>
            val trainer = new TrainRegressor().setFeaturesCol("outcome_features").setModel(getOutcomeModel).setLabelCol(getOutcomeCol).setExcludedFeatureCols(Array(getTreatmentCol, "treatment_features"))
            val (processedDF, _) = trainer.getFeaturizedDataAndModel(datasetWithTreatmentFeatures)
            processedDF
        }
        datasetWithTreatmentAndOutcomeFeatures
      }

      preprocessedDF.cache()

      val ate = trainInternal(preprocessedDF, treatmentPredictionColName, outcomePredictionColName)

      val dmlModel = new LinearDMLModel().setATE(ate)

      if (get(ciCalcIterations).isDefined) {
        // Confidence intervals:
        // sampling with replacement to redraw data and get ATE value
        // Run it for multiple times in parallel, get a number of ATE values,
        // Use 2.5% low end, 97.5% high as CI value

        // Create execution context based on $(parallelism)
        log.info(s"Parallelism: $getParallelism")
        val executionContext = getExecutionContextProxy

        val ateFutures = Range(0, getCICalcIterations).toArray.map { index =>
          Future[Double] {
            log.info(s"Executing ATE calculation on iteration: $index")
            println(s"Executing ATE calculation on iteration: $index")
            // sample data with replacement
            val redrewDF = preprocessedDF.sample(withReplacement = true, fraction = 1).cache()
            val ate: Option[Double] =
              try {
                val totalTime = new StopWatch
                val te = totalTime.measure {
                  trainInternal(redrewDF, treatmentPredictionColName, outcomePredictionColName)
                }
                println(s"Completed ATE calculation on iteration $index and got ATE value: $te, time elapsed: ${totalTime.elapsed() / 60000000000.0} minutes")
                Some(te)
              } catch {
                case ex: Throwable =>
                  println(s"ATE calculation got exception on iteration $index with the redrew sample data. Exception ignored.")
                  log.info(s"ATE calculation got exception on iteration $index with the redrew sample data. Exception details: $ex")
                  None
              }
            redrewDF.unpersist()
            ate.getOrElse(0.0)
          }(executionContext)
        }

        val ates = awaitFutures(ateFutures).filter(_ != 0.0).sorted
        println(s"Completed ATE calculation for $getCICalcIterations iterations and got ${ates.length} ATE values.")

        if (ates.length > 1) {
          val ci = Array(percentile[Double](ates, 2.5), percentile[Double](ates, 97.5))
          dmlModel.setCI(ci)
        }
      }

      preprocessedDF.unpersist()
      dmlModel
    })
  }

  private def percentile[T](ates: Seq[T], percentile: Double): T = {
    val index = Math.ceil(percentile / 100.0 * ates.size).toInt
    ates(index - 1)
  }


  private def trainInternal(dataset: Dataset[_], treatmentPredictionColName: String, outcomePredictionColName: String): Double = {
    // treatment effect:
    // 1. split sample, e.g. 50/50
    // 2. use first sample data to fit treatment model and outcome model,
    // 3. use two models on the second sample data to get residuals,
    // 4. cross fitting to get another residuals,
    // 5. apply regressor fit to get treatment effect T1 = lrm1.coefficients(0) and T2 = lrm2.coefficients(0)
    // 5. average treatment effects = (T1 + T2) / 2

    // Step 1 - split sample
    val splits = dataset.randomSplit(getSampleSplitRatio)
    val train = splits(0).cache()
    val test = splits(1).cache()

    // Step 2 - use first sample data to fit treatment model and outcome model
    val treatmentPredictor = getTreatmentModel
    val outcomePredictor = getOutcomeModel

    val treatmentModelV1 = treatmentPredictor.fit(train)
    val outcomeModelV1 = outcomePredictor.fit(train)

    // Step 3 - use second sample data to get predictions and compute residuals
    val treatmentPredictedDFV1 = treatmentModelV1.transform(test)
    val treatmentResidualDFV1 =
      new ComputeResidualTransformer()
        .setObservedCol(getTreatmentCol)
        .setPredictedCol(treatmentPredictionColName)
        .setOutputCol(SchemaConstants.TreatmentResidualColumn)
        .transform(treatmentPredictedDFV1)

    val outcomePredictedDFV1 = outcomeModelV1.transform(treatmentResidualDFV1)
    val residualsDFV1 =
      new ComputeResidualTransformer()
        .setObservedCol(getOutcomeCol)
        .setPredictedCol(outcomePredictionColName)
        .setOutputCol(SchemaConstants.OutcomeResidualColumn)
        .transform(outcomePredictedDFV1)

    // Step 4 - cross fitting to get another residuals
    val treatmentModelV2 = treatmentPredictor.fit(test)
    val outcomeModelV2 = outcomePredictor.fit(test)

    val treatmentPredictedDFV2 = treatmentModelV2.transform(train)
    val treatmentResidualDFV2 =
      new ComputeResidualTransformer()
        .setObservedCol(getTreatmentCol)
        .setPredictedCol(treatmentPredictionColName)
        .setOutputCol(SchemaConstants.TreatmentResidualColumn)
        .transform(treatmentPredictedDFV2)

    val outcomePredictedDFV2 = outcomeModelV2.transform(treatmentResidualDFV2)
    val residualsDFV2 =
      new ComputeResidualTransformer()
        .setObservedCol(getOutcomeCol)
        .setPredictedCol(outcomePredictionColName)
        .setOutputCol(SchemaConstants.OutcomeResidualColumn)
        .transform(outcomePredictedDFV2)

    // 5. apply regressor fit to get treatment effect T1 = lrm1.coefficients(0) and T2 = lrm2.coefficients(0)
    val va: Array[PipelineStage] = Array(
      new VectorAssembler()
        .setInputCols(Array(SchemaConstants.TreatmentResidualColumn))
        .setOutputCol("treatmentResidualVec")
        .setHandleInvalid("skip"),
      new DropColumns().setCols(Array(SchemaConstants.TreatmentResidualColumn))
    )

    val regressor = new GeneralizedLinearRegression()
      .setLabelCol(SchemaConstants.OutcomeResidualColumn)
      .setFeaturesCol("treatmentResidualVec")
      .setFamily("gaussian")
      .setLink("identity")
      .setFitIntercept(false)

    val lrmPipelineModelV1 = new Pipeline().setStages(va :+ regressor).fit(residualsDFV1)
    val lrmV1 = lrmPipelineModelV1.stages.last.asInstanceOf[GeneralizedLinearRegressionModel]
    val lrmPipelineModelV2 = new Pipeline().setStages(va :+ regressor).fit(residualsDFV2)
    val lrmV2 = lrmPipelineModelV2.stages.last.asInstanceOf[GeneralizedLinearRegressionModel]

    // Step 6 - final treatment effects = (T1 + T2) / 2
    val ate = (lrmV1.coefficients.asInstanceOf[DenseVector].values(0)
      + lrmV2.coefficients.asInstanceOf[DenseVector].values(0)) / 2.0

    ate
  }


  override def copy(extra: ParamMap): Estimator[LinearDMLModel] = {
    defaultCopy(extra)
  }

  @DeveloperApi
  override def transformSchema(schema: StructType): StructType = {
    LinearDMLEstimator.validateTransformSchema(schema)
  }
}

object LinearDMLEstimator extends ComplexParamsReadable[LinearDMLEstimator] {

  def validateTransformSchema(schema: StructType): StructType = {
    StructType(schema.fields)
  }
}

/** Model produced by [[LinearDMLEstimator]]. */
class LinearDMLModel(val uid: String)
  extends AutoTrainedModel[LinearDMLModel] with Wrappable with BasicLogging {
  logClass()

  def this() = this(Identifiable.randomUID("LinearDMLModel"))

  val ate = new Param[Double](this, "ate", "average treatment effect")
  def getATE: Double = $(ate)
  def setATE(v: Double): this.type = set(ate, v)

  var ci = new Param[Array[Double]](this, "ci", "treatment effect's confidence interval")
  def getCI: Array[Double] = $(ci)
  def setCI(v:Array[Double] ): this.type = set(ci, v)

  override def copy(extra: ParamMap): LinearDMLModel = defaultCopy(extra)

  //scalastyle:off
  override def transform(dataset: Dataset[_]): DataFrame = {
    logTransform[DataFrame]({
      throw new Exception("transform is invalid for LinearDMLEstimator.")
    })
  }

  @DeveloperApi
  override def transformSchema(schema: StructType): StructType =
    throw new Exception("transform is invalid for LinearDMLEstimator.")
}

object LinearDMLModel extends ComplexParamsReadable[LinearDMLModel]
