// Copyright (C) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License. See LICENSE in project root for information.

package com.microsoft.ml.spark.cntk

import java.awt.image.BufferedImage
import java.net.URI

import com.microsoft.ml.spark.core.env.FileUtilities.File
import com.microsoft.ml.spark.core.test.fuzzing.{FuzzingMethods, TestObject, TransformerFuzzing}
import com.microsoft.ml.spark.downloader.{ModelDownloader, ModelSchema}
import org.apache.spark.ml.linalg.SQLDataTypes.VectorType
import org.apache.spark.ml.linalg.{DenseMatrix, DenseVector, Vector, Vectors}
import org.apache.spark.ml.regression.LinearRegression
import org.apache.spark.ml.util.MLReadable
import org.apache.spark.sql.DataFrame
import com.microsoft.ml.spark.Readers.implicits._
import com.microsoft.ml.spark.core.schema.ImageSchema
import com.microsoft.ml.spark.stages.basic.UDFTransformer
import org.apache.spark.ml.NamespaceInjections
import com.microsoft.ml.spark.udfs.udfs.get_value_at
import org.apache.spark.sql.functions.udf
import org.apache.spark.sql.types.DoubleType
/*
class LIMESuite extends TransformerFuzzing[ImageLIME] with FuzzingMethods with ImageFeaturizerUtils {

lazy val x = (0 to 10).map(_.toDouble)
lazy val y = x.map(_ * 7)

lazy val df: DataFrame = session
  .createDataFrame(x.map(Vectors.dense(_)).zip(y))

lazy val lr = new LinearRegression().setFeaturesCol("_1").setLabelCol("_2")
lazy val fitlr = lr.fit(df)

lazy val t = new ImageLIME()
  .setModel(fitlr)
  .setLabelCol(fitlr.getPredictionCol)
  .setSampler[Vector](
  { arr: Vector =>
    (1 to 10).map { i =>
      Vectors.dense(arr(0) + scala.util.Random.nextGaussian())
    }.toArray
  }, VectorType)
  //.setFeaturesCol("_1")
  .setOutputCol("weights")

test("LIME should identify the local slope") {
  // TODO - set default values for cell size (16), modifier (130), input col, etc.
  t.setCellSize(16).setModifier(130).setInputCol("_1").transform(df)
    .select("weights").collect.foreach(row =>
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

  lazy val t = new ImageLIME()
    .setModel(fitlr)
    .setLabelCol(fitlr.getPredictionCol)
    //.setFeaturesCol("_1")
    .setOutputCol("weights")
    .setInputCol("_1")

  t.transform(df).select("weights")
    .collect()
    .map(_.getAs[DenseVector](0))
    .foreach(s => assert(s === new DenseVector(randomMatrix.toArray)))
}

test("image featurizer lime") {
  val resnet = resNetModel().setCutOutputLayers(0)
  val outputCol = "output"
  val model = NamespaceInjections.pipelineModel(Array(
    resnet,
    new UDFTransformer()
      .setInputCol(resnet.getOutputCol)
      .setOutputCol(outputCol)
      .setUDF(udf({vec: org.apache.spark.ml.linalg.Vector => vec(0)}, DoubleType))
  ))

  lazy val t = new ImageLIME()
    .setModel(model)
    .setLabelCol(outputCol)
    //.setFeaturesCol(inputCol)
    .setOutputCol("weights")
    .setNSamples(10)

  t.transform(images.limit(2)).show()
}

override def testObjects(): Seq[TestObject[ImageLIME]] = Seq(new TestObject(t, df))

override def reader: MLReadable[_] = ImageLIME
}
*/