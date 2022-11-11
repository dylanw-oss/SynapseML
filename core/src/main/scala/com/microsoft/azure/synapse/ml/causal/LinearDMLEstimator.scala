// Copyright (C) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License. See LICENSE in project root for information.

package com.microsoft.azure.synapse.ml.causal

import com.microsoft.azure.synapse.ml.codegen.Wrappable
import com.microsoft.azure.synapse.ml.train._
import com.microsoft.azure.synapse.ml.core.schema.{DatasetExtensions, SchemaConstants}
import com.microsoft.azure.synapse.ml.core.utils.StopWatch
import com.microsoft.azure.synapse.ml.logging.BasicLogging
import com.microsoft.azure.synapse.ml.stages.DropColumns
import org.apache.commons.math3.stat.descriptive.rank.Percentile
import org.apache.spark.annotation.DeveloperApi
import org.apache.spark.ml._
import org.apache.spark.ml.classification._
import org.apache.spark.ml.feature.VectorAssembler
import org.apache.spark.ml.param._
import org.apache.spark.ml.param.shared.{HasFeaturesCol, HasLabelCol, HasWeightCol}
import org.apache.spark.ml.util._
import org.apache.spark.sql._
import org.apache.spark.sql.types._
import org.apache.spark.ml.regression.{GeneralizedLinearRegression, Regressor}

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

      // sampling with replacement to redraw data and get TE value
      // Run it for multiple times in parallel, get a number of TE values,
      // Use average as Ate value, and 2.5% low end, 97.5% high end as Ci value
      // Create execution context based on $(parallelism)
      log.info(s"Parallelism: $getParallelism")
      val executionContext = getExecutionContextProxy

      val ateFutures = Range(1, getMaxIter+1).toArray.map { index =>
        Future[Option[Double]] {
          log.info(s"Executing ATE calculation on iteration: $index")
          // If the algorithm runs over 1 iteration, do not bootstrap from dataset, otherwise, draw sample with replacement
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
                log.warn(s"ATE calculation got exception on iteration $index with the redrew sample data. Exception details: $ex")
                None
            }
          redrewDF.unpersist()
          ate
        }(executionContext)
      }

      val ates = awaitFutures(ateFutures).flatten.sorted
      println(ates)
      val finalAte = if (getMaxIter == 1) ates.head else ates.sum / ates.length
      println(s"Completed $getMaxIter iteration ATE calculations and got ${ates.length} values, final ATE = $finalAte")
      val dmlModel = new LinearDMLModel().setAtes(ates.toArray).setAte(finalAte)

      if (ates.length > 1) {
        val ci = Array(percentile(ates, getPercentileLowCutOff), percentile(ates, 100-getPercentileLowCutOff))
        dmlModel.setCi(ci)
      }

      dmlModel
    })
  }

  private def percentile(values: Seq[Double], quantile: Double): Double = {
    val sortedValues = values.sorted
    val percentile = new Percentile()
    percentile.setData(sortedValues.toArray)
    percentile.evaluate(quantile)
  }


  private def trainInternal(dataset: Dataset[_]): Double = {
    // setup estimators
    val treatmentFeaturesColName = DatasetExtensions.findUnusedColumnName("treatment_features", dataset)
    val outcomeFeaturesColName = DatasetExtensions.findUnusedColumnName("outcome_features", dataset)

    val (treatmentEstimator, treatmentResidualPredictionColName, treatmentPredictionColsToDrop) = getTreatmentModel match {
      case classifier: ProbabilisticClassifier[_, _, _] => (
        new TrainClassifier()
          .setFeaturesCol(treatmentFeaturesColName)
          .setModel(getTreatmentModel)
          .setLabelCol(getTreatmentCol)
          .setExcludedFeatures(Array(getOutcomeCol)),
        classifier.getProbabilityCol,
        Seq(classifier.getPredictionCol, classifier.getProbabilityCol, classifier.getRawPredictionCol)
      )
      case regressor: Regressor[_, _, _] => (
        new TrainRegressor()
          .setFeaturesCol(treatmentFeaturesColName)
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
          .setFeaturesCol(outcomeFeaturesColName)
          .setModel(getOutcomeModel)
          .setLabelCol(getOutcomeCol)
          .setExcludedFeatures(Array(getTreatmentCol)),
        classifier.getProbabilityCol,
        Seq(classifier.getPredictionCol, classifier.getProbabilityCol, classifier.getRawPredictionCol)
      )
      case regressor: Regressor[_, _, _] => (
        new TrainRegressor()
          .setFeaturesCol(outcomeFeaturesColName)
          .setModel(getOutcomeModel)
          .setLabelCol(getOutcomeCol)
          .setExcludedFeatures(Array(getTreatmentCol)),
        regressor.getPredictionCol,
        Seq(regressor.getPredictionCol)
      )
    }

    // Note, we perform these steps to get ATE
    /*
      1. Split sample, e.g. 50/50
      2. Use the first split to fit the treatment model and the outcome model.
      3. Use the two models to fit a residual model on the second split.
      4. Cross-fit by fitting the treatment and outcome models with the second split and the residual model with the first split.
      5. Average slopes from the two residual models.
    */
    // Step 1 - split sample
    val splits = dataset.randomSplit(getSampleSplitRatio)
    val train = splits(0).cache()
    val test = splits(1).cache()

    // Step 2 - Use the first split to fit the treatment model and the outcome model.
    val treatmentModelV1 = treatmentEstimator.fit(train)
    val outcomeModelV1 = outcomeEstimator.fit(train)

    // Step 3 - Use the two models to fit a residual model on the second split.
    val treatmentResidualTransformer =
      new ComputeResidualTransformer()
        .setObservedCol(getTreatmentCol)
        .setPredictedCol(treatmentResidualPredictionColName)
        .setOutcomeCol(SchemaConstants.TreatmentResidualColumn)
    val dropTreatmentPredictedColumnsTransformer = new DropColumns().setCols(treatmentPredictionColsToDrop.toArray)

    val outcomeResidualTransformer =
      new ComputeResidualTransformer()
        .setObservedCol(getOutcomeCol)
        .setPredictedCol(outcomeResidualPredictionColName)
        .setOutcomeCol(SchemaConstants.OutcomeResidualColumn)
    val dropOutcomePredictedColumnsTransformer = new DropColumns().setCols(outcomePredictionColsToDrop.toArray)

    val treatmentResidualVA =
      new VectorAssembler()
        .setInputCols(Array(SchemaConstants.TreatmentResidualColumn))
        .setOutputCol("treatmentResidualVec")
        .setHandleInvalid("skip")

    val treatmentEffectPipelineV1 =
      new Pipeline().setStages(Array(
        treatmentModelV1, treatmentResidualTransformer, dropTreatmentPredictedColumnsTransformer,
        outcomeModelV1, outcomeResidualTransformer, dropOutcomePredictedColumnsTransformer,
        treatmentResidualVA))
    val treatmentEffectDFV1 = treatmentEffectPipelineV1.fit(test).transform(test)

    // Step 4 - Cross-fit by fitting the treatment and outcome models with the second split and the residual model with the first split.
    val treatmentModelV2 = treatmentEstimator.fit(test)
    val outcomeModelV2 = outcomeEstimator.fit(test)

    val treatmentEffectPipelineV2 =
      new Pipeline().setStages(Array(
        treatmentModelV2, treatmentResidualTransformer, dropTreatmentPredictedColumnsTransformer,
        outcomeModelV2, outcomeResidualTransformer, dropOutcomePredictedColumnsTransformer,
        treatmentResidualVA))
    val treatmentEffectDFV2 = treatmentEffectPipelineV2.fit(train).transform(train)

    // Step 5. Average slopes from the two residual models.
    val regressor = new GeneralizedLinearRegression()
      .setLabelCol(SchemaConstants.OutcomeResidualColumn)
      .setFeaturesCol("treatmentResidualVec")
      .setFamily("gaussian")
      .setLink("identity")
      .setFitIntercept(false)
    val lrmV1 = regressor.fit(treatmentEffectDFV1)
    val lrmV2 = regressor.fit(treatmentEffectDFV2)
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

  val ate = new DoubleParam(this, "ate", "average treatment effect")
  def getAte: Double = $(ate)
  def setAte(v: Double): this.type = set(ate, v)

  var ates = new DoubleArrayParam(this, "ates", "treatment effect results for each iteration")
  def getAtes: Array[Double] = $(ates)
  def setAtes(v: Array[Double]): this.type = set(ates, v)

  var ci = new DoubleArrayParam(this, "ci", "treatment effect's confidence interval")
  def getCi: Array[Double] = $(ci)
  def setCi(v: Array[Double]): this.type = set(ci, v)

  override def copy(extra: ParamMap): LinearDMLModel = defaultCopy(extra)

  //scalastyle:off
  /** LinearDMLEstimator transform does nothing and isn't supposed be called by end user. */
  override def transform(dataset: Dataset[_]): DataFrame = {
    logTransform[DataFrame]({
      dataset.toDF()
    })
  }

  @DeveloperApi
  override def transformSchema(schema: StructType): StructType =
    StructType(schema.fields)
}

object LinearDMLModel extends ComplexParamsReadable[LinearDMLModel]
