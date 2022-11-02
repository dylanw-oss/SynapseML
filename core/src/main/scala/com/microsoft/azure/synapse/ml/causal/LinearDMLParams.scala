package com.microsoft.azure.synapse.ml.causal

import com.microsoft.azure.synapse.ml.core.contracts.{HasFeaturesCol, HasWeightCol}
import org.apache.spark.ml.classification.{LogisticRegression, ProbabilisticClassifier}
import org.apache.spark.ml.{Estimator, Model}
import com.microsoft.azure.synapse.ml.param.{EstimatorParam, UntypedArrayParam}
import org.apache.spark.ml.ParamInjections.HasParallelismInjected
import org.apache.spark.ml.param.shared.HasMaxIter
import org.apache.spark.ml.param.{IntParam, Param, Params}
import org.apache.spark.ml.regression.Regressor

trait HasTreatmentCol extends Params {
  val treatmentCol = new Param[String](this, "treatmentCol", "treatment column")
  def getTreatmentCol: String = $(treatmentCol)
  def setTreatmentCol(value: String): this.type = set(treatmentCol, value)
}

trait HasOutcomeCol extends Params {
  val outcomeCol: Param[String] = new Param[String](this, "outcomeCol", "outcome column")
  def getOutcomeCol: String = $(outcomeCol)
  def setOutcomeCol(value: String): this.type = set(outcomeCol, value)
}

trait LinearDMLParams extends Params
  with HasTreatmentCol with HasOutcomeCol with HasFeaturesCol
  with HasMaxIter with HasWeightCol with HasParallelismInjected {

  val treatmentModel = new EstimatorParam(this, "treatmentModel", "treatment model to run")
  def getTreatmentModel: Estimator[_ <: Model[_]] = $(treatmentModel)
  def setTreatmentModel(value: Estimator[_ <: Model[_]]): this.type = {

    val isSupportedModel = value match {
      case regressor: Regressor[_,_,_] =>   true // for continuous treatment
      case classifier: ProbabilisticClassifier[_, _, _] => true
      case _ => false
    }
    if (!isSupportedModel)  {
      throw new Exception("LinearDML only support regressor and ProbabilisticClassifier as treatment model")
    }
    set(treatmentModel, value)
  }

  val outcomeModel = new EstimatorParam(this, "outcomeModel", "outcome model to run")
  def getOutcomeModel: Estimator[_ <: Model[_]] = $(outcomeModel)
  def setOutcomeModel(value: Estimator[_ <: Model[_]]): this.type = {
    val isSupportedModel = value match {
      case regressor: Regressor[_,_,_] =>   true
      case classifier: ProbabilisticClassifier[_, _, _] => true
      case _ => false
    }
    if (!isSupportedModel)  {
      throw new Exception("LinearDML only support regressor and ProbabilisticClassifier as outcome model")
    }
    set(outcomeModel, value)
  }

  val featurizationModel = new EstimatorParam(this,
    "featurizationModel", "featurization model to run preprocessing data")
  def getFeaturizationModel: Estimator[_ <: Model[_]] = $(featurizationModel)
  def setFeaturizationModel(value: Estimator[_ <: Model[_]]): this.type =  set(featurizationModel, value)

  val sampleSplitRatio = new UntypedArrayParam(this,
    "SampleSplitRatio", "SampleSplitRatio")
  def getSampleSplitRatio: Array[Double] = $(sampleSplitRatio).map(_.toString.toDouble)
  def setSampleSplitRatio(value: Array[Any]): this.type = set(sampleSplitRatio, value)

  /**
   * Set the maximum number of confidence interval bootstrapping iterations.
   * Default is 1, which means it does not calculate confidence interval.
   *
   * @group setParam
   */
  def setMaxIter(value: Int): this.type = set(maxIter, value)

  def setParallelism(value: Int): this.type = set(parallelism, value)

  setDefault(
    treatmentModel -> new LogisticRegression(),
    outcomeModel -> new LogisticRegression(),
    sampleSplitRatio -> Array(0.5, 0.5),
    maxIter -> 1,
    parallelism -> 10 // Best practice, a value up to 10 should be sufficient for most clusters.
  )
}
