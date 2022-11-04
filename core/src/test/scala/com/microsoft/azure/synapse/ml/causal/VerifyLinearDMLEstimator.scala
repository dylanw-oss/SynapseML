package com.microsoft.azure.synapse.ml.causal

import com.microsoft.azure.synapse.ml.core.test.base.TestBase
import org.apache.spark.ml.regression.RandomForestRegressor
import org.apache.spark.ml.classification.LogisticRegression

class VerifyLinearDMLEstimator extends TestBase {

  val mockLabelColumn = "Label"

  val cat = "Cat"
  val dog = "Dog"
  val bird = "Bird"
  private lazy val mockDataset = spark.createDataFrame(Seq(
    (0, 1, 0.50, 0.60, dog, cat),
    (1, 0, 0.40, 0.50, cat, dog),
    (0, 1, 0.78, 0.99, dog, bird),
    (1, 0, 0.12, 0.34, cat, dog),
    (0, 1, 0.50, 0.60, dog, bird),
    (1, 0, 0.40, 0.50, bird, dog),
    (0, 1, 0.78, 0.99, dog, cat),
    (1, 1, 0.12, 0.34, cat, dog),
    (0, 0, 0.50, 0.60, dog, cat),
    (1, 1, 0.40, 0.50, bird, dog),
    (0, 0, 0.78, 0.99, dog, bird),
    (1, 1, 0.12, 0.34, cat, dog)))
    .toDF(mockLabelColumn, "col1", "col2", "col3", "col4", "col5")


  test("Get treatment effects") {
    val ldml = new LinearDMLEstimator()
      .setTreatmentModel(new LogisticRegression())
      .setTreatmentCol(mockLabelColumn)
      .setOutcomeModel(new RandomForestRegressor())
      .setOutcomeCol("col2")

    var ldmlModel = ldml.fit(mockDataset)
    assert(ldmlModel.getAte != 0.0)
  }

  test("Get treatment effects with weight column") {
    val ldml = new LinearDMLEstimator()
      .setTreatmentModel(new LogisticRegression())
      .setTreatmentCol(mockLabelColumn)
      .setOutcomeModel(new LogisticRegression())
      .setOutcomeCol("col1")
      .setWeightCol("col3")

    var ldmlModel = ldml.fit(mockDataset)
    assert(ldmlModel.getAte != 0.0)
  }

  test("Get treatment effects and confidence intervals") {
    val ldml = new LinearDMLEstimator()
      .setTreatmentModel(new LogisticRegression())
      .setTreatmentCol(mockLabelColumn)
      .setOutcomeModel(new RandomForestRegressor())
      .setOutcomeCol("col2")
      .setMaxIter(10)

    var ldmlModel = ldml.fit(mockDataset)
    assert(ldmlModel.getCi.length == 2)
  }
}
