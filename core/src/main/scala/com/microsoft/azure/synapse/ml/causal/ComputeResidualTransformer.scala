package com.microsoft.azure.synapse.ml.causal

import com.microsoft.azure.synapse.ml.codegen.Wrappable
import com.microsoft.azure.synapse.ml.logging.BasicLogging
import org.apache.spark.ml.Transformer
import org.apache.spark.ml.linalg.{SQLDataTypes, Vector}
import org.apache.spark.ml.param._
import org.apache.spark.ml.util.{DefaultParamsReadable, DefaultParamsWritable, Identifiable}
import org.apache.spark.sql.functions.{col, udf}
import org.apache.spark.sql.{DataFrame, Dataset}
import org.apache.spark.sql.types._

/** Compute the differences between observed and predicted values of data.
 *  for classification, we compute residual as "observed - probability($(classIndex))"
 *  for regression, we compute residual as "observed - prediction"
 */
class ComputeResidualTransformer(override val uid: String) extends Transformer
  with HasOutcomeCol with DefaultParamsWritable with Wrappable with BasicLogging {

  logClass()

  def this() = this(Identifiable.randomUID("ComputeResidualsTransformer"))

  val observedCol = new Param[String](this, "observedCol", "observed data column")
  def setObservedCol(value: String): this.type = set(param = observedCol, value = value)
  final def getObservedCol: String = getOrDefault(observedCol)

  val predictedCol = new Param[String](this, "predictedCol", "predicted data column")
  def setPredictedCol(value: String): this.type = set(param = predictedCol, value = value)
  final def getPredictedCol: String = getOrDefault(predictedCol)

  val classIndex =
    new IntParam(
      this,
      "classIndex",
      "The index of the class to compute residual for classification outputs. Default value is 1.",
      ParamValidators.gtEq(0))
  def setClassIndex(value: Int): this.type = set(param = classIndex, value = value)
  final def getClassIndex: Int = getOrDefault(classIndex)
  override def copy(extra: ParamMap): ComputeResidualTransformer = defaultCopy(extra)

  override def transformSchema(schema: StructType): StructType =
    StructType(
      schema.fields :+ StructField(
        name = $(outcomeCol),
        dataType = DoubleType,
        nullable = false
      )
    )

  setDefault(observedCol -> "label", predictedCol -> "prediction", outcomeCol -> "residual", classIndex -> 1)

  override def transform(dataset: Dataset[_]): DataFrame = {
    logTransform[DataFrame]({
      transformSchema(schema = dataset.schema, logging = true)
      // Make sure the observedCol is a DoubleType
      val observedColType = dataset.schema(getObservedCol).dataType
      require(observedColType == DoubleType || observedColType == IntegerType, s"observedCol must be of type DoubleType but got $observedColType")

      val inputType = dataset.schema(getPredictedCol).dataType
      if (inputType == SQLDataTypes.VectorType) {
        // For vector input, we compute the residual as "observed - probability($index)"
        val extractionUdf = (index: Int) => udf { (observed: Double, prediction: Vector) =>
          observed - prediction(index) // TO-DO: Validate $(probabilityIndex) >= probability size
        }
        dataset.withColumn(getOutcomeCol, extractionUdf(getClassIndex)(col(getObservedCol), col(getPredictedCol)))
      } else if (inputType.isInstanceOf[NumericType]) {
        // For numeric input, we compute residual as "observed - predicted"
        dataset.withColumn(
          getOutcomeCol,
          col(getObservedCol) - col(getPredictedCol)
        )
      } else {
        throw new IllegalArgumentException(
          s"Prediction column $getPredictedCol must be of type Vector or NumericType, but is $inputType" +
            s", please use 'setPredictedCol' to set the correct predicted column"
        )
      }
    })
  }
}

object ComputeResidualTransformer extends DefaultParamsReadable[ComputeResidualTransformer]
