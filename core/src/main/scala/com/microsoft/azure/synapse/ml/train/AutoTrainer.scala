// Copyright (C) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License. See LICENSE in project root for information.

package com.microsoft.azure.synapse.ml.train

import com.microsoft.azure.synapse.ml.codegen.Wrappable
import com.microsoft.azure.synapse.ml.core.contracts.{HasExcludedFeatureCols, HasFeaturesCol, HasLabelCol}
import com.microsoft.azure.synapse.ml.param.EstimatorParam
import org.apache.spark.ml.param.{IntParam, Param}
import org.apache.spark.ml.{ComplexParamsWritable, Estimator, Model}

/** Defines common inheritance and parameters across trainers.
 */
trait AutoTrainer[TrainedModel <: Model[TrainedModel]] extends Estimator[TrainedModel]
  with HasLabelCol with HasExcludedFeatureCols with ComplexParamsWritable with HasFeaturesCol with Wrappable {

  /** Doc for model to run.
   */
  def modelDoc: String

  /** Number of features to hash to
   * @group param
   */
  val numFeatures = new IntParam(this, "numFeatures", "Number of features to hash to")

  /** @group getParam */
  def getNumFeatures: Int = $(numFeatures)
  /** @group setParam */
  def setNumFeatures(value: Int): this.type = set(numFeatures, value)

  /** Model to run.  See doc on derived classes.
   * @group param
   */
  val model = new EstimatorParam(this, "model", modelDoc)
  /** @group getParam */

  def getModel: Estimator[_ <: Model[_]] = $(model)
  /** @group setParam */
  def setModel(value: Estimator[_ <: Model[_]]): this.type = set(model, value)


  val featureColumnsStr = new Param[String](this, "featureColumnsStr", "feature columns")
  def getFeatureColumnsStr: String = $(featureColumnsStr)
  def setFeatureColumnsStr(value: String): this.type = set(featureColumnsStr, value)


  setDefault(numFeatures -> 0, excludedFeatureCols -> Array.empty[String], featureColumnsStr -> "")
}