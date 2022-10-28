package com.microsoft.azure.synapse.ml.causal

import com.microsoft.azure.synapse.ml.codegen.Wrappable
import com.microsoft.azure.synapse.ml.core.schema.SchemaConstants
import com.microsoft.azure.synapse.ml.logging.BasicLogging
import org.apache.spark.ml.Transformer
import org.apache.spark.ml.linalg.Vector
import org.apache.spark.ml.param._
import org.apache.spark.ml.util.{DefaultParamsReadable, DefaultParamsWritable, Identifiable}
import org.apache.spark.sql.functions.{col, udf}
import org.apache.spark.sql.{DataFrame, Dataset}
import org.apache.spark.sql.types._

class ComputeResidualTransformer(override val uid: String) extends Transformer
  with DefaultParamsWritable
  with Wrappable
  with BasicLogging {

  logClass()

  def this() = this(Identifiable.randomUID("ComputeResidualsTransformer"))

  val observedCol = new Param[String](this, "observedCol", "observed data column")
  def setObservedCol(value: String): this.type = set(param = observedCol, value = value)
  final def getObservedCol: String = getOrDefault(observedCol)

  val predictedCol = new Param[String](this, "predictedCol", "predicted data column")
  def setPredictedCol(value: String): this.type = set(param = predictedCol, value = value)
  final def getPredictedCol: String = getOrDefault(predictedCol)

  val outputCol = new Param[String](this, "outputCol", "output column name")
  def setOutputCol(value: String): this.type = set(param = outputCol, value = value)
  final def getOutputCol: String = getOrDefault(outputCol)

  val probabilityIndex =
    new IntParam(
      this,
      "probabilityIndex",
      "probability index value for residual compute, by default it is 1, the second value of probability vector",
      ParamValidators.gtEq(0))
  def setProbabilityIndex(value: Int): this.type = set(param = probabilityIndex, value = value)
  final def getProbabilityIndex: Int = getOrDefault(probabilityIndex)

  override def copy(extra: ParamMap): ComputeResidualTransformer = defaultCopy(extra)

  override def transformSchema(schema: StructType): StructType =
    StructType(
      schema.fields :+ StructField(
        name = $(outputCol),
        dataType = DoubleType,
        nullable = false
      )
    )

  setDefault(outputCol -> "residual", probabilityIndex -> 1)

  override def transform(dataset: Dataset[_]): DataFrame = {
    logTransform[DataFrame]({
      val outCol  = getOutputCol
      val obsCol  = getObservedCol
      val predCol = getPredictedCol

      transformSchema(schema = dataset.schema, logging = true)

      // TO-DO: Validate $(probabilityIndex) >= probability size
      val extractProbUDF = udf { (label: Double, prob: Vector) =>
        label - prob($(probabilityIndex))
      }

      val columnsToDrop = Seq(SchemaConstants.SparkRawPredictionColumn,
        SchemaConstants.SparkProbabilityColumn,
        SchemaConstants.SparkPredictionColumn)

      if (dataset.schema(predCol).name == SchemaConstants.SparkProbabilityColumn)
      {
        // for classifier, we compute residual as "label - probability(1)"
        dataset.withColumn(
          colName = outCol,
          col = extractProbUDF(col(obsCol), col(predCol))
        ).drop(columnsToDrop: _*)
      } else {
        // for regression, we compute residual as "label - prediction"
        dataset.withColumn(
          colName = outCol,
          col = col(obsCol) - col(predCol)
        ).drop(columnsToDrop: _*)
      }
    })
  }
}

object ComputeResidualTransformer extends DefaultParamsReadable[ComputeResidualTransformer]
