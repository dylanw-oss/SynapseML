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
import org.apache.spark.ml.regression.{GeneralizedLinearRegression, GeneralizedLinearRegressionModel, Regressor}

import scala.concurrent.Future

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
      (multi-task) linear regression problem.
 */
//noinspection ScalaStyle
class LinearDMLEstimator(override val uid: String)
  extends Estimator[LinearDMLModel] with ComplexParamsWritable
    with LinearDMLParams with BasicLogging with Wrappable {

  logClass()

  def this() = this(Identifiable.randomUID("LinearDMLEstimator"))

  /** Fits the LinearDML model.
   *
   * @param dataset The input dataset to train.
   * @return The trained LinearDML model, from which you can get Ate and Ci values
   */
  override def fit(dataset: Dataset[_]): LinearDMLModel = {
    logFit({
      if (get(weightCol).isDefined) {
        getTreatmentModel match {
          case w: HasWeightCol => w.set(w.weightCol, getWeightCol)
          case _ => throw new Exception("""The selected treatment model does not support sample weight,
            but the weightCol parameter was set for the LinearDMLEstimator.
            Please select a treatment model that supports sample weight.""".stripMargin)
        }
        getOutcomeModel match {
          case w: HasWeightCol => w.set(w.weightCol, getWeightCol)
          case _ => throw new Exception("""The selected outcome model does not support sample weight,
            but the weightCol parameter was set for the LinearDMLEstimator.
            Please select a outcome model that supports sample weight.""".stripMargin)
        }
      }

      getTreatmentModel match {
        case m: HasLabelCol with HasFeaturesCol =>
          m.set(m.labelCol, getTreatmentCol).set(m.featuresCol, "treatment_features")
        case _ => throw new Exception("The defined treatment model does not support HasLabelCol and HasFeaturesCol.")
      }

      getOutcomeModel match {
        case m: HasLabelCol with HasFeaturesCol =>
          m.set(m.labelCol, getOutcomeCol).set(m.featuresCol, "outcome_features")
        case _ => throw new Exception("The defined outcome model does not support HasLabelCol and HasFeaturesCol.")
      }

      val dmlModel = new LinearDMLModel()

      // sampling with replacement to redraw data and get TE value
      // Run it for multiple times in parallel, get a number of TE values,
      // Use average as Ate value, and 2.5% low end, 97.5% high end as Ci value
      // Create execution context based on $(parallelism)
      log.info(s"Parallelism: $getParallelism")
      val executionContext = getExecutionContextProxy

      val ateFutures = Range(1, getMaxIter+1).toArray.map { index =>
        Future[Double] {
          log.info(s"Executing ATE calculation on iteration: $index")
          println(s"Executing ATE calculation on iteration: $index")
          // sample data with replacement
          val redrewDF =  if (getMaxIter == 1) dataset else dataset.sample(withReplacement = true, fraction = 1)
          redrewDF.cache()
          val ate: Option[Double] =
            try {
              val totalTime = new StopWatch
              val oneAte = totalTime.measure {
                trainInternal(redrewDF)
              }
              println(s"Completed ATE calculation on iteration $index and got ATE value: $oneAte, time elapsed: ${totalTime.elapsed() / 60000000000.0} minutes")
              Some(oneAte)
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
      val finalAte = if (getMaxIter == 1) ates.head else ates.sum / ates.length
      println(s"Completed $maxIter iteration ATE calculations and got ${ates.length} values, final ATE = $finalAte")

      dmlModel.setAte(finalAte)
      if (ates.length > 1) {
        val ci = Array(percentile[Double](ates, 2.5), percentile[Double](ates, 97.5))
        dmlModel.setCi(ci)
      }

      dmlModel
    })
  }

  private def percentile[T](ates: Seq[T], percentile: Double): T = {
    val index = Math.ceil(percentile / 100.0 * ates.size).toInt
    ates(index - 1)
  }


  private def trainInternal(dataset: Dataset[_]): Double = {
    // Note, we perform these steps to get ATE
    /*
      1. split sample, e.g. 50/50
      2. use first sample data to fit treatment model and outcome model,
      3. use two models on the second sample data to get residuals,
      4. cross fitting to get another residuals,
      5. apply regressor fit to get treatment effect T1 = lrm1.coefficients(0) and T2 = lrm2.coefficients(0)
      5. average treatment effects = (T1 + T2) / 2
    */
    // Step 1 - split sample
    val splits = dataset.randomSplit(getSampleSplitRatio)
    val train = splits(0).cache()
    val test = splits(1).cache()

    // Step 2 - use first sample data to fit treatment model and outcome model
    val (treatmentEstimator, treatmentResidualPredictionColName, treatmentPredictionColsToDrop) = getTreatmentModel match {
      case classifier: ProbabilisticClassifier[_, _, _] => (
        new TrainClassifier()
          .setFeaturesCol("treatment_features")
          .setModel(getTreatmentModel)
          .setLabelCol(getTreatmentCol)
          .setExcludedFeatures(Array(getOutcomeCol)),
        classifier.getProbabilityCol,
        Seq(classifier.getPredictionCol, classifier.getProbabilityCol, classifier.getRawPredictionCol)
      )
      case regressor: Regressor[_, _, _] => (
        new TrainRegressor()
          .setFeaturesCol("treatment_features")
          .setModel(getTreatmentModel)
          .setLabelCol(getTreatmentCol)
          .setExcludedFeatures(Array(getOutcomeCol)),
        regressor.getPredictionCol,
        Seq(regressor.getPredictionCol)
      )
    }

    val (outcomeEstimator, outcomeResidualPredictionColName, outcomePredictionColsToDrop) = getOutcomeModel match {
      case classifier: ProbabilisticClassifier[_, _, _] => (
        new TrainClassifier()
          .setFeaturesCol("outcome_features")
          .setModel(getOutcomeModel)
          .setLabelCol(getOutcomeCol)
          .setExcludedFeatures(Array(getTreatmentCol)),
        classifier.getProbabilityCol,
        Seq(classifier.getPredictionCol, classifier.getProbabilityCol, classifier.getRawPredictionCol)
      )
      case regressor: Regressor[_, _, _] => (
        new TrainRegressor()
          .setFeaturesCol("outcome_features")
          .setModel(getOutcomeModel)
          .setLabelCol(getOutcomeCol)
          .setExcludedFeatures(Array(getTreatmentCol)),
        regressor.getPredictionCol,
        Seq(regressor.getPredictionCol)
      )
    }

    val treatmentModelV1 = treatmentEstimator.fit(train)
    val outcomeModelV1 = outcomeEstimator.fit(train)

    // Step 3 - use second sample data to get predictions and compute residuals
    val treatmentResidualDFV1 =
      new ComputeResidualTransformer()
        .setObservedCol(getTreatmentCol)
        .setPredictedCol(treatmentResidualPredictionColName)
        .setOutcomeCol(SchemaConstants.TreatmentResidualColumn)

    val dropTreatmentPredictedColumnsTransformer = new DropColumns().setCols(treatmentPredictionColsToDrop.toArray)

    val residualsDFV1 =
      new ComputeResidualTransformer()
        .setObservedCol(getOutcomeCol)
        .setPredictedCol(outcomeResidualPredictionColName)
        .setOutcomeCol(SchemaConstants.OutcomeResidualColumn)
    val dropOutcomePredictedColumnsTransformer = new DropColumns().setCols(outcomePredictionColsToDrop.toArray)


    val stage1Pipeline = new Pipeline()
      .setStages(
        Array(treatmentModelV1,
          treatmentResidualDFV1,
          dropTreatmentPredictedColumnsTransformer,
          outcomeModelV1,
          residualsDFV1,
          dropOutcomePredictedColumnsTransformer,
          new VectorAssembler()
            .setInputCols(Array(SchemaConstants.TreatmentResidualColumn))
            .setOutputCol("treatmentResidualVec")
            .setHandleInvalid("skip"),
          new DropColumns().setCols(Array(SchemaConstants.TreatmentResidualColumn))
      )
    )
    val dataset1 = stage1Pipeline.fit(test).transform(test)

    // Step 4 - cross fitting to get another residuals
    val treatmentModelV2 = treatmentEstimator.fit(test)
    val outcomeModelV2 = outcomeEstimator.fit(test)

    val stage2Pipeline = new Pipeline().setStages(
      Array(treatmentModelV2,
        treatmentResidualDFV1,
        dropTreatmentPredictedColumnsTransformer,
        outcomeModelV2,
        residualsDFV1,
        dropOutcomePredictedColumnsTransformer))
    val dataset2 = stage2Pipeline.fit(train).transform(train)
    
    val regressor = new GeneralizedLinearRegression()
      .setLabelCol(SchemaConstants.OutcomeResidualColumn)
      .setFeaturesCol("treatmentResidualVec")
      .setFamily("gaussian")
      .setLink("identity")
      .setFitIntercept(false)

      val lrmV1 = regressor.fit(dataset1)
      val lrmV2 = regressor.fit(dataset2)

    // Step 6 - final treatment effects = (T1 + T2) / 2
    val ate = Seq(lrmV1, lrmV2).map(_.coefficients(0)).sum / 2

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
  extends Model[LinearDMLModel] with ComplexParamsWritable with Wrappable with BasicLogging {
  logClass()



  def this() = this(Identifiable.randomUID("LinearDMLModel"))

  val ate = new Param[Double](this, "ate", "average treatment effect")
  def getAte: Double = $(ate)
  def setAte(v: Double): this.type = set(ate, v)

  var ci = new Param[Array[Double]](this, "ci", "treatment effect's confidence interval")
  def getCi: Array[Double] = $(ci)
  def setCi(v: Array[Double]): this.type = set(ci, v)

  override def copy(extra: ParamMap): LinearDMLModel = defaultCopy(extra)

  // Todo: return ate array
  //scalastyle:off
  override def transform(dataset: Dataset[_]): DataFrame = {
    // TODO: transform return dataset
    logTransform[DataFrame]({
      throw new Exception("transform is invalid for LinearDMLEstimator.")
    })
  }

  @DeveloperApi
  override def transformSchema(schema: StructType): StructType =
    throw new Exception("transform is invalid for LinearDMLEstimator.")
}

object LinearDMLModel extends ComplexParamsReadable[LinearDMLModel]
