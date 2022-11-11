package com.microsoft.azure.synapse.ml.causal

import com.microsoft.azure.synapse.ml.core.contracts.{HasFeaturesCol, HasWeightCol}
import org.apache.spark.ml.classification.{LogisticRegression, ProbabilisticClassifier}
import org.apache.spark.ml.{Estimator, Model}
import com.microsoft.azure.synapse.ml.param.EstimatorParam
import org.apache.spark.ml.ParamInjections.HasParallelismInjected
import org.apache.spark.ml.param.shared.HasMaxIter
import org.apache.spark.ml.param.{DoubleArrayParam, DoubleParam, Param, Params}
import org.apache.spark.ml.regression.Regressor

trait HasTreatmentCol extends Params {
  val treatmentCol = new Param[String](this, "treatmentCol", "treatment column")
  def getTreatmentCol: String = $(treatmentCol)

  /**
   * Set name of the column which will be used as treatment
   *
   * @group setParam
   */
  def setTreatmentCol(value: String): this.type = set(treatmentCol, value)
}

trait HasOutcomeCol extends Params {
  val outcomeCol: Param[String] = new Param[String](this, "outcomeCol", "outcome column")
  def getOutcomeCol: String = $(outcomeCol)

  /**
   * Set name of the column which will be used as outcome
   *
   * @group setParam
   */
  def setOutcomeCol(value: String): this.type = set(outcomeCol, value)
}

trait LinearDMLParams extends Params
  with HasTreatmentCol with HasOutcomeCol with HasFeaturesCol
  with HasMaxIter with HasWeightCol with HasParallelismInjected {

  val treatmentModel = new EstimatorParam(this, "treatmentModel", "treatment model to run")
  def getTreatmentModel: Estimator[_ <: Model[_]] = $(treatmentModel)

  /**
   * Set treatment model, it could be any model derived from 'org.apache.spark.ml.regression.Regressor' or 'org.apache.spark.ml.classification.ProbabilisticClassifier'
   *
   * @group setParam
   */
  def setTreatmentModel(value: Estimator[_ <: Model[_]]): this.type = {
    EnsureSupportedEstimator(value)
    set(treatmentModel, value)
  }

  val outcomeModel = new EstimatorParam(this, "outcomeModel", "outcome model to run")
  def getOutcomeModel: Estimator[_ <: Model[_]] = $(outcomeModel)

  /**
   * Set outcome model, it could be any model derived from 'org.apache.spark.ml.regression.Regressor' or 'org.apache.spark.ml.classification.ProbabilisticClassifier'
   *
   * @group setParam
   */
  def setOutcomeModel(value: Estimator[_ <: Model[_]]): this.type = {
    EnsureSupportedEstimator(value)
    set(outcomeModel, value)
  }

  val sampleSplitRatio = new DoubleArrayParam(
    this,
    "SampleSplitRatio",
    "Sample split ratio for cross-fitting. Default: [0.5, 0.5].",
    split => split.length == 2 && split.forall(_ >= 0)
  )
  def getSampleSplitRatio: Array[Double] = $(sampleSplitRatio).map(v => v / $(sampleSplitRatio).sum)

  /**
   * Set the sample split ratio, default is Array(0.5, 0.5)
   *
   * @group setParam
   */
  def setSampleSplitRatio(value: Array[Double]): this.type = set(sampleSplitRatio, value)


  val percentileLowCutOff = new DoubleParam(this, "percentileLowCutOff", "percentile low cutoff, e.g. 2.5 means get 2.5% percentile", value => value > 0 && value < 100)

  def getPercentileLowCutOff: Double = $(percentileLowCutOff)

  /**
   * Set the low cut-off value for percentile of distribution. Default is 2.5.
   * high cut-off will be automatically calculated as (100-percentileLowCutOff)
   * That means by default we compute 95% confidence interval, it is [2.5, 97.5] percentile of ATE distribution
   *
   * @group setParam
   */
  def setPercentileLowCutOff(value: Double): this.type = set(percentileLowCutOff, value)

  /**
   * Set the maximum number of confidence interval bootstrapping iterations.
   * Default is 1, which means it does not calculate confidence interval.
   * To get Ci values please set a meaningful value
   *
   * @group setParam
   */
  def setMaxIter(value: Int): this.type = set(maxIter, value)

  def setParallelism(value: Int): this.type = set(parallelism, value)

  setDefault(
    treatmentModel -> new LogisticRegression(),
    outcomeModel -> new LogisticRegression(),
    sampleSplitRatio -> Array(0.5, 0.5),
    percentileLowCutOff -> 2.5,
    maxIter -> 1,
    parallelism -> 10 // Best practice, a value up to 10 should be sufficient for most clusters.
  )

  private def EnsureSupportedEstimator(value: Estimator[_ <: Model[_]]): Unit = {
    val isSupportedModel = value match {
      case regressor: Regressor[_, _, _] => true // for continuous treatment
      case classifier: ProbabilisticClassifier[_, _, _] => true
      case _ => false
    }
    if (!isSupportedModel) {
      throw new Exception("LinearDML only support regressor and ProbabilisticClassifier as treatment or outcome model")
    }
  }
}
