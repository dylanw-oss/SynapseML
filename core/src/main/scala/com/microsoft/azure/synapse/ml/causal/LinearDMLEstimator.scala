// Copyright (C) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License. See LICENSE in project root for information.

package com.microsoft.azure.synapse.ml.causal

import com.microsoft.azure.synapse.ml.codegen.Wrappable
import com.microsoft.azure.synapse.ml.core.contracts.{HasFeaturesCol, HasLabelCol}
import com.microsoft.azure.synapse.ml.train._
import com.microsoft.azure.synapse.ml.core.schema.SchemaConstants
import com.microsoft.azure.synapse.ml.logging.BasicLogging
import com.microsoft.azure.synapse.ml.stages.DropColumns
import org.apache.spark.annotation.DeveloperApi
import org.apache.spark.ml.{Estimator, Model, _}
import org.apache.spark.ml.classification._
import org.apache.spark.ml.feature.VectorAssembler
import org.apache.spark.ml.linalg.DenseVector
import org.apache.spark.ml.param._
import org.apache.spark.ml.param.shared.HasWeightCol
import org.apache.spark.ml.util._
import org.apache.spark.sql.{DataFrame, _}
import org.apache.spark.sql.types._
import org.apache.spark.ml.regression.{GeneralizedLinearRegression, GeneralizedLinearRegressionModel, Regressor}


/** Double ML estimators. The estimator follows the two stage process,
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
 * If the init flag `discrete_treatment` is set to `True`, then the treatment model is treated as a classifier.
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
      // treatment effect:
      // 1. split sample, e.g. 50/50
      // 2. use first sample data to fit treatment model and outcome model,
      // 3. use two models on the second sample data to get residuals, apply regressor fit to get treatment effect T1 = lrm1.coefficients(0)
      // 4. cross fitting to get another treatment effect, T2 = lrm2.coefficients(0)
      // 5. final treatment effects = (T1 + T2) / 2
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

      // Step 1 - split sample
      val splits = preprocessedDF.randomSplit(getSampleSplitRatio)
      val train = splits(0).cache()
      val test = splits(1).cache()

      // Step 2 - use first sample data to fit treatment model and outcome model
      // Train treatment model, regarding to prediction column, we use probability column for classifier and prediction column for regression.
      val (treatmentEstimator, treatmentPredictionColName) = getTreatmentModel match {
        case classifier: ProbabilisticClassifier[_, _, _] => (
          new TrainClassifier().setModel(getTreatmentModel).setLabelCol(getTreatmentCol).setExcludedFeatureCols(Array(getOutcomeCol)),
          classifier.getProbabilityCol
        )
        case regressor: Regressor[_, _, _] => (
          new TrainRegressor().setModel(getTreatmentModel).setLabelCol(getTreatmentCol).setExcludedFeatureCols(Array(getOutcomeCol)),
          regressor.getPredictionCol
        )
      }

      // Train outcome model, regarding to prediction column, we use probability column for classifier and prediction column for regression.
      val (outcomeEstimator, outcomePredictionColName) = getOutcomeModel match {
        case classifier: ProbabilisticClassifier[_, _, _] => (
          new TrainClassifier().setModel(getOutcomeModel).setLabelCol(getOutcomeCol).setExcludedFeatureCols(Array(getTreatmentCol)),
          classifier.getProbabilityCol
        )
        case regressor: Regressor[_, _, _] => (
          new TrainRegressor().setModel(getOutcomeModel).setLabelCol(getOutcomeCol).setExcludedFeatureCols(Array(getTreatmentCol)),
          regressor.getPredictionCol
        )
      }

      val testResidualDF = trainAndEnrichWithResiduals(
        train,
        test,
        treatmentEstimator,
        treatmentPredictionColName,
        outcomeEstimator,
        outcomePredictionColName
      )

      val trainResidualDF = trainAndEnrichWithResiduals(
        test,
        train,
        treatmentEstimator,
        treatmentPredictionColName,
        outcomeEstimator,
        outcomePredictionColName
      )


      // Step 3 - final regressor
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

      val pipeline_finalModel = new Pipeline().setStages(va :+ regressor).fit(testResidualDF)
      val lrmT1 = pipeline_finalModel.stages.last.asInstanceOf[GeneralizedLinearRegressionModel]
      val pipeline_finalModel2 = new Pipeline().setStages(va :+ regressor).fit(trainResidualDF)
      val lrmT2 = pipeline_finalModel2.stages.last.asInstanceOf[GeneralizedLinearRegressionModel]

      val ate = (lrmT1.coefficients.asInstanceOf[DenseVector].values(0)
        + lrmT2.coefficients.asInstanceOf[DenseVector].values(0)) / 2.0

      new LinearDMLModel()
        .setATE(ate)

      // Confidence intervals:  (spark cross validator?)
      // 6. sampling with replacement (spark maybe have function to do it), redraw, // df.sample(true, )
      // 7. with new sample, repeat 1~5, get another estimator, (can be parallel with above)
      // do 6~7 for 2K times, get 2k estimator, 2.5% low end, 97.5% high end, pyspark has quantile function,
      // (np.percentile([2k treatment effect], 0.025), np.percentile(x, 0.975)), find Spark version
      // LASSO estimator at the fist stage

      // Sarah: note 1, a, b, c,   d, e, f. => keep top 10 and put others to one
      // Sarah: note 2, do not one-hot continuous features
      // hasSampleWeight
    })
  }

  private def trainAndEnrichWithResiduals(train: Dataset[_],
                                  test: Dataset[_],
                                  treatmentEstimator: AutoTrainer[_ >: TrainedClassifierModel with TrainedRegressorModel <: Wrappable with Transformer with HasFeaturesCol with HasLabelCol with BasicLogging with ComplexParamsWritable] with BasicLogging,
                                  treatmentPredictionColName: String,
                                  outcomeEstimator: AutoTrainer[_ >: TrainedClassifierModel with TrainedRegressorModel <: Wrappable with Transformer with HasFeaturesCol with HasLabelCol with BasicLogging with ComplexParamsWritable] with BasicLogging,
                                  outcomePredictionColName: String): DataFrame = {
    val treatmentModel = treatmentEstimator.fit(train)
    val outcomeModel = outcomeEstimator.fit(train)

    // Step 3 - use second sample data to get predictions and compute residuals
    val treamentDF = treatmentModel.transform(test)
    val computeTreatmentResiduals =
      new ComputeResidualTransformer()
        .setObservedCol(getTreatmentCol)
        .setPredictedCol(treatmentPredictionColName)
        .setOutputCol(SchemaConstants.TreatmentResidualColumn)
    val treatmentResidualDF = computeTreatmentResiduals.transform(treamentDF)

    val outcomeDF = outcomeModel.transform(treatmentResidualDF)
    val computeOutcomeResiduals = new ComputeResidualTransformer()
      .setObservedCol(getOutcomeCol)
      .setPredictedCol(outcomePredictionColName)
      .setOutputCol(SchemaConstants.OutcomeResidualColumn)
    computeOutcomeResiduals.transform(outcomeDF)
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
    val treatmentSchema = StructType(schema.fields :+ StructField(SchemaConstants.TreatmentResidualColumn, DoubleType))
    StructType(treatmentSchema.fields :+ StructField(SchemaConstants.OutcomeResidualColumn, DoubleType))
  }

  def getEstimatorWithLabelFeaturesConfigured(estimator: Estimator[_ <: Model[_]], labelCol: String, featuresCol: String): Estimator[_ <: Model[_]] = {
    estimator match {
      case predictor: Predictor[_, _, _] =>
        predictor
          .setLabelCol(labelCol)
          .setFeaturesCol(featuresCol).asInstanceOf[Estimator[_ <: Model[_]]]
      case default@defaultType if defaultType.isInstanceOf[Estimator[_ <: Model[_]]] =>
        // assume label col and features col already set
        default
      case _ => throw new Exception("Unsupported learner type " + estimator.getClass.toString)
    }
  }
}

/** Model produced by [[LinearDMLEstimator]]. */
class LinearDMLModel(val uid: String)
  extends AutoTrainedModel[LinearDMLModel] with Wrappable with BasicLogging {
  logClass()

  def this() = this(Identifiable.randomUID("LinearDMLModel"))

  val ate = new Param[Double](this, "cate", "average treatment effect")
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
