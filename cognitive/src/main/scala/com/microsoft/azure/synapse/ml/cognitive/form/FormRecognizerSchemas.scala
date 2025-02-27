// Copyright (C) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License. See LICENSE in project root for information.

package com.microsoft.azure.synapse.ml.cognitive.form

import com.microsoft.azure.synapse.ml.cognitive.vision.ReadLine
import com.microsoft.azure.synapse.ml.core.schema.SparkBindings
import org.apache.spark.sql.Row
import org.apache.spark.sql.types._
import spray.json.{DefaultJsonProtocol, JsonFormat, RootJsonFormat, _}

object AnalyzeResponse extends SparkBindings[AnalyzeResponse]

case class AnalyzeResponse(status: String,
                           createdDateTime: String,
                           lastUpdatedDateTime: String,
                           analyzeResult: AnalyzeResult)

case class AnalyzeResult(version: String,
                         readResults: Seq[FormReadResult],
                         pageResults: Option[Seq[PageResult]],
                         documentResults: Option[Seq[DocumentResult]])

case class FormReadResult(page: Int,
                          language: Option[String],
                          angle: Double,
                          width: Double,
                          height: Double,
                          unit: String,
                          lines: Option[Seq[ReadLine]])

case class PageResult(page: Int,
                      keyValuePairs: Option[Seq[KeyValuePair]],
                      tables: Seq[Table])

case class Table(rows: Int,
                 columns: Int,
                 cells: Seq[Cell],
                 boundingBox: Seq[Double])

case class Cell(rowIndex: Int,
                columnIndex: Int,
                text: String,
                boundingBox: Seq[Double],
                isHeader: Option[Boolean],
                elements: Seq[String])

case class DocumentResult(docType: String,
                          pageRange: Seq[Int],
                          fields: Map[String, FieldResult])

case class FieldResult(`type`: String,
                       page: Option[Int],
                       confidence: Option[Double],
                       boundingBox: Option[Seq[Double]],
                       text: Option[String],
                       valueString: Option[String],
                       valuePhoneNumber: Option[String],
                       valueNumber: Option[Double],
                       valueDate: Option[String],
                       valueTime: Option[String],
                       valueObject: Option[String],
                       valueArray: Option[Seq[String]]) {

  def toFieldResultRecursive: FieldResultRecursive = {
    import FormsJsonProtocol._

    FieldResultRecursive(
      `type`,
      page,
      confidence,
      boundingBox,
      text,
      valueString,
      valuePhoneNumber,
      valueNumber,
      valueDate,
      valueTime,
      valueObject.map(_.parseJson.convertTo[Map[String, FieldResultRecursive]]),
      valueArray.map(seq => seq.map(str => str.parseJson.convertTo[FieldResultRecursive]))
    )
  }

}

case class FieldResultRecursive(`type`: String,
                                page: Option[Int],
                                confidence: Option[Double],
                                boundingBox: Option[Seq[Double]],
                                text: Option[String],
                                valueString: Option[String],
                                valuePhoneNumber: Option[String],
                                valueNumber: Option[Double],
                                valueDate: Option[String],
                                valueTime: Option[String],
                                valueObject: Option[Map[String, FieldResultRecursive]],
                                valueArray: Option[Seq[FieldResultRecursive]]) {

  private[ml] def toSimplifiedDataType: DataType = {
    `type`.toLowerCase match {
      case "string" => StringType
      case "number" => DoubleType
      case "date" => StringType
      case "time" => StringType
      case "array" =>
        ArrayType(valueArray.get.map(_.toSimplifiedDataType).reduce(FormOntologyLearner.combineDataTypes))
      case "object" =>
        new StructType(valueObject.get.mapValues(_.toSimplifiedDataType)
          .map({ case (fn, dt) => StructField(fn, dt) }).toArray)
      case "phonenumber" => StringType
    }
  }

  private[ml] def viewAsDataType(dt: DataType): Any = {
    (`type`.toLowerCase, dt) match {
      case ("string", StringType) => valueString.get
      case ("date", StringType) => valueDate.orElse(text).get
      case ("time", StringType) => valueTime.orElse(text).get
      case ("phonenumber", StringType) => valuePhoneNumber.orElse(text).get
      case ("number", StringType) => text.get
      case ("number", DoubleType) => valueNumber.get
      case ("array", ArrayType(et, _)) => valueArray.get.map(_.viewAsDataType(et))
      case ("object", StructType(fields)) =>
        val obj = valueObject.get
        Row.fromSeq(fields.map(sf => obj.get(sf.name).map(_.viewAsDataType(sf.dataType))))
      case _ =>
        throw new NotImplementedError()
    }
  }
}

case class ModelInfo(modelId: String,
                     status: String,
                     createDateTime: String,
                     lastUpdatedDateTime: String)

case class TrainResult(trainingDocuments: Seq[TrainingDocument],
                       fields: Seq[Field],
                       errors: Seq[String])

case class TrainingDocument(documentName: String,
                            pages: Int,
                            errors: Seq[String],
                            status: String)

case class KeyValuePair(key: Element, value: Element)

case class Element(text: String, boundingBox: Seq[Double])

object ListCustomModelsResponse extends SparkBindings[ListCustomModelsResponse]

case class ListCustomModelsResponse(summary: Summary,
                                    modelList: Seq[ModelInfo],
                                    nextLink: String)

case class Summary(count: Int, limit: Int, lastUpdatedDateTime: String)

object GetCustomModelResponse extends SparkBindings[GetCustomModelResponse]

case class GetCustomModelResponse(modelInfo: ModelInfo,
                                  keys: String,
                                  trainResult: TrainResult)

case class Key(clusters: Map[String, Seq[String]])

case class Field(fieldName: String, accuracy: Double)

object FormsJsonProtocol extends DefaultJsonProtocol {

  implicit val FieldFormat: RootJsonFormat[FieldResult] = jsonFormat12(FieldResult.apply)

  implicit val FieldResultRecursiveFormat: JsonFormat[FieldResultRecursive] = lazyFormat(jsonFormat(
    FieldResultRecursive,
    "type", "page",
    "confidence", "boundingBox", "text", "valueString",
    "valuePhoneNumber", "valueNumber", "valueDate", "valueTime",
    "valueObject", "valueArray"))

  implicit val DRFormat: RootJsonFormat[DocumentResult] = jsonFormat3(DocumentResult.apply)

}
