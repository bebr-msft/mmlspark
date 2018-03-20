// Copyright (C) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License. See LICENSE in project root for information.

package com.microsoft.ml.spark.stages

import com.microsoft.ml.spark.core.test.fuzzing.{FuzzingMethods, TestObject, TransformerFuzzing}
import org.apache.spark.ml.linalg.SQLDataTypes.VectorType
import org.apache.spark.ml.linalg.{DenseMatrix, DenseVector, Vector, Vectors}
import org.apache.spark.ml.regression.LinearRegression
import org.apache.spark.ml.util.MLReadable
import org.apache.spark.sql.DataFrame

class LIMESuite extends TransformerFuzzing[LIME] with FuzzingMethods {

  lazy val x = (0 to 10).map(_.toDouble)
  lazy val y = x.map(_ * 7)

  lazy val df: DataFrame = session
    .createDataFrame(x.map(Vectors.dense(_)).zip(y))

  lazy val lr = new LinearRegression().setFeaturesCol("_1").setLabelCol("_2")
  lazy val fitlr = lr.fit(df)

  lazy val t = new LIME()
    .setLocalModel(new LinearRegression())
    .setModel(fitlr)
    .setSampler[Vector](
    { arr: Vector =>
      (1 to 10).map { i =>
        Vectors.dense(arr(0) + scala.util.Random.nextGaussian())
      }.toArray
    }, VectorType)
    .setFeaturesCol("_1")
    .setOutputCol("weights")

  test("LIME should identify the local slope") {
    t.transform(df).select("weights").collect.foreach(row =>
      assert(7.0 === row.getAs[Vector](0)(0))
    )
  }

  test("vectorized lime") {
    val random = new java.util.Random()
    random.setSeed(0)
    val nRows = 30
    val xDim = 10
    val yDim = 1 // SparkML doesent support multi output regression: Sad
    val xs = Array.fill(nRows){Vectors.dense(Array.fill(xDim){random.nextGaussian()})}
    val randomMatrix = DenseMatrix.randn(yDim, xDim, random)
    val ys = xs.map(x => randomMatrix.multiply(x).values(0))

    lazy val df: DataFrame = session
      .createDataFrame(xs.zip(ys).toSeq)

    lazy val lr = new LinearRegression().setFeaturesCol("_1").setLabelCol("_2")
    lazy val fitlr = lr.fit(df)

    lazy val t = new LIME()
      .setLocalModel(new LinearRegression())
      .setModel(fitlr)
      .setFeaturesCol("_1")
      .setOutputCol("weights")

    t.transform(df).select("weights")
      .collect()
      .map(_.getAs[DenseVector](0))
      .foreach(s => assert(s === new DenseVector(randomMatrix.toArray)))
  }

  override def testObjects(): Seq[TestObject[LIME]] = Seq(new TestObject(t, df))

  override def reader: MLReadable[_] = LIME
}
