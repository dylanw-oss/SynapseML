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
import org.apache.spark.ml.param.shared.HasWeightCol
import org.apache.spark.ml.util._
import org.apache.spark.sql._
import org.apache.spark.sql.types._
import org.apache.spark.ml.regression.{GeneralizedLinearRegression, GeneralizedLinearRegressionModel, Regressor}

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

      val preprocessedDF = if (get(featurizationModel).isDefined) {
        getFeaturizationModel.fit(dataset).transform(dataset)
      } else dataset

      preprocessedDF.cache()

      val ate = trainInternal(preprocessedDF)

      val dmlModel = new LinearDMLModel().setATE(ate)

      if (get(ciCalcIterations).isDefined) {
        // Confidence intervals:
        // sampling with replacement to redraw data and get TE value
        // Run it for multiple times in parallel, get a TE list,
        // Use 2.5% low end, 97.5% high as CI value

        // Create execution context based on $(parallelism)
        log.info(s"Parallelism: $getParallelism")
        val executionContext = getExecutionContextProxy

        val ateFutures = Range(0, getCICalcIterations).toArray.map { index =>
          Future[Double] {
            log.info(s"Executing TE estimator on iteration: $index")
            println(s"Executing TE estimator on iteration: $index")
            // sample data with replacement
            val trainingDataset = preprocessedDF.sample(withReplacement = true, fraction = 1).cache()
            val ate: Option[Double] =
              try {
                val totalTime = new StopWatch
                val te = totalTime.measure {
                  trainInternal(trainingDataset)
                }
                println(s"Completed TE estimator on iteration $index and got TE value: $te, time elapsed: ${totalTime.elapsed() / 60000000000.0} minutes")
                Some(te)
              } catch {
                case ex: Throwable =>
                  println(s"TE estimator got exception on iteration $index with a redrew sample data. Exception ignored. Exception details: $ex")
                  log.info(s"TE estimator got exception on iteration $index with a redrew sample data. Exception ignored. Exception details: $ex")
                  None
              }
            trainingDataset.unpersist()
            ate.getOrElse(0.0)
          }(executionContext)
        }

        val ates = awaitFutures(ateFutures).filter(_ != 0.0).sorted
        println(s"Completed all TE estimators fitting tasks and got ${ates.length} TE results.")

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


  private def trainInternal(dataset: Dataset[_]): Double = {
    // treatment effect:
    // 1. split sample, e.g. 50/50
    // 2. use first sample data to fit treatment model and outcome model,
    // 3. use two models on the second sample data to get residuals, apply regressor fit to get treatment effect T1 = lrm1.coefficients(0)
    // 4. cross fitting to get another treatment effect, T2 = lrm2.coefficients(0)
    // 5. final treatment effects = (T1 + T2) / 2

    // Step 1 - split sample
    val splits = dataset.randomSplit(getSampleSplitRatio)
    val train = splits(0).cache()
    val test = splits(1).cache()

    // Step 2 - use first sample data to fit treatment model and outcome model
    val (treatmentEstimator, treatmentPredictionColName) = getTreatmentModel match {
      case classifier: ProbabilisticClassifier[_, _, _] => (
        new TrainClassifier().setFeaturesCol("_features_").setModel(getTreatmentModel).setLabelCol(getTreatmentCol).setExcludedFeatureCols(Array(getOutcomeCol)),
        classifier.getProbabilityCol
      )
      case regressor: Regressor[_, _, _] => (
        new TrainRegressor().setFeaturesCol("_features_").setModel(getTreatmentModel).setLabelCol(getTreatmentCol).setExcludedFeatureCols(Array(getOutcomeCol)),
        regressor.getPredictionCol
      )
    }

    val (outcomeEstimator, outcomePredictionColName) = getOutcomeModel match {
      case classifier: ProbabilisticClassifier[_, _, _] => (
        new TrainClassifier().setFeaturesCol("_features_").setModel(getOutcomeModel).setLabelCol(getOutcomeCol).setExcludedFeatureCols(Array(getTreatmentCol)),
        classifier.getProbabilityCol
      )
      case regressor: Regressor[_, _, _] => (
        new TrainRegressor().setFeaturesCol("_features_").setModel(getOutcomeModel).setLabelCol(getOutcomeCol).setExcludedFeatureCols(Array(getTreatmentCol)),
        regressor.getPredictionCol
      )
    }

    val treatmentModel_iteration1 = treatmentEstimator.fit(train)
    val outcomeModel_iteration1 = outcomeEstimator.fit(train)

    // Step 3 - use second sample data to get predictions and compute residuals
    val treatmentDF_iteration1 = treatmentModel_iteration1.transform(test)
    val treatmentResidualDF_iteration1 =
      new ComputeResidualTransformer()
        .setObservedCol(getTreatmentCol)
        .setPredictedCol(treatmentPredictionColName)
        .setOutputCol(SchemaConstants.TreatmentResidualColumn).transform(treatmentDF_iteration1)

    val outcomeDF_iteration1 = outcomeModel_iteration1.transform(treatmentResidualDF_iteration1)
    val residualDF_iteration1 = new ComputeResidualTransformer()
      .setObservedCol(getOutcomeCol)
      .setPredictedCol(outcomePredictionColName)
      .setOutputCol(SchemaConstants.OutcomeResidualColumn).transform(outcomeDF_iteration1)

    val treatmentModel_iteration2 = treatmentEstimator.fit(test)
    val outcomeModel_iteration2 = outcomeEstimator.fit(test)

    // Step 4 - cross fitting to get another treatment effect, T2 = lrm2.coefficients(0)
    val treatmentDF_iteration2 = treatmentModel_iteration2.transform(train)
    val treatmentResidualDF_iteration2 =
      new ComputeResidualTransformer()
        .setObservedCol(getTreatmentCol)
        .setPredictedCol(treatmentPredictionColName)
        .setOutputCol(SchemaConstants.TreatmentResidualColumn).transform(treatmentDF_iteration2)

    val outcomeDF_iteration2 = outcomeModel_iteration2.transform(treatmentResidualDF_iteration2)
    val residualDF_iteration2 = new ComputeResidualTransformer()
      .setObservedCol(getOutcomeCol)
      .setPredictedCol(outcomePredictionColName)
      .setOutputCol(SchemaConstants.OutcomeResidualColumn).transform(outcomeDF_iteration2)

    // Step 5 - final treatment effects = (T1 + T2) / 2
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

    val pipeline_finalModel = new Pipeline().setStages(va :+ regressor).fit(residualDF_iteration1)
    val lrmT1 = pipeline_finalModel.stages.last.asInstanceOf[GeneralizedLinearRegressionModel]
    val pipeline_finalModel2 = new Pipeline().setStages(va :+ regressor).fit(residualDF_iteration2)
    val lrmT2 = pipeline_finalModel2.stages.last.asInstanceOf[GeneralizedLinearRegressionModel]

    val ate = (lrmT1.coefficients.asInstanceOf[DenseVector].values(0)
      + lrmT2.coefficients.asInstanceOf[DenseVector].values(0)) / 2.0

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
