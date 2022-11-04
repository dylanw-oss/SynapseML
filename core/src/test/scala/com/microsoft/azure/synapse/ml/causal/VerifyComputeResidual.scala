package com.microsoft.azure.synapse.ml.causal

import com.microsoft.azure.synapse.ml.core.test.fuzzing.{TestObject, TransformerFuzzing}
import org.apache.spark.ml.util.MLReadable

class VerifyComputeResidual extends TransformerFuzzing[ComputeResidualTransformer] {

  private lazy val mockDataset = spark.createDataFrame(Seq(
    (2d, 0.50d, true, 0, 0.toByte, 12F),
    (3d, 0.40d, false, 1, 100.toByte, 30F),
    (4d, 0.78d, true, 2, 50.toByte, 12F),
    (5d, 0.12d, false, 3, 0.toByte, 12F),
    (1d, 0.50d, true, 0, 0.toByte, 30F),
    (3d, 0.40d, false, 1, 10.toByte, 12F),
    (3d, 0.78d, false, 2, 0.toByte, 12F),
    (4d, 0.12d, false, 3, 0.toByte, 12F),
    (0d, 0.50d, true, 0, 0.toByte, 12F),
    (2d, 0.40d, false, 1, 127.toByte, 30F),
    (3d, 0.78d, true, 2, -128.toByte, 12F),
    (4d, 0.12d, false, 3, 0.toByte, 12F)))
    .toDF("label", "prediction", "col3", "col4", "col5", "col6")

  private lazy val expectedDF = spark.createDataFrame(Seq(
    (2d, 0.50d, true, 0, 0.toByte, 12F,                    1.5),
    (3d, 0.40d, false, 1, 100.toByte, 30F,                 2.6),
    (4d, 0.78d,  true, 2, 50.toByte, 12F,   3.2199999999999998),
    (5d, 0.12d,  false, 3, 0.toByte, 12F,                 4.88),
    (1d, 0.50d, true, 0, 0.toByte, 30F,                    0.5),
    (3d, 0.40d, false, 1, 10.toByte, 12F,                  2.6),
    (3d, 0.78d,  false, 2, 0.toByte, 12F,   2.2199999999999998),
    (4d, 0.12d,  false, 3, 0.toByte, 12F,                 3.88),
    (0d, 0.50d,  true, 0, 0.toByte, 12F,                  -0.5),
    (2d, 0.40d, false, 1, 127.toByte, 30F,                 1.6),
    (3d, 0.78d,  true, 2, -128.toByte, 12F, 2.2199999999999998),
    (4d, 0.12d,  false, 3, 0.toByte, 12F,                 3.88)))
    .toDF("label", "prediction", "col3", "col4", "col5", "col6", "diff")


  test("Compute residual") {
    val computeResiduals = new ComputeResidualTransformer()
      .setObservedCol("label")
      .setPredictedCol("prediction")
      .setOutcomeCol("diff")

    val processedDF = computeResiduals.transform(mockDataset)

    assert(verifyResult(expectedDF, processedDF))
  }

  override def testObjects(): Seq[TestObject[ComputeResidualTransformer]] = Seq(new TestObject(
    new ComputeResidualTransformer(), mockDataset))

  override def reader: MLReadable[_] = ComputeResidualTransformer
}
