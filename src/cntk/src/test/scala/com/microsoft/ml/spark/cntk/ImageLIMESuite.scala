// Copyright (C) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License. See LICENSE in project root for information.

package com.microsoft.ml.spark.cntk

import com.microsoft.ml.spark.core.test.fuzzing.{FuzzingMethods, TestObject, TransformerFuzzing}
import com.microsoft.ml.spark.opencv.{ImageTransformer, UnrollImage}
import org.apache.spark.ml.linalg.SQLDataTypes.VectorType
import org.apache.spark.ml.linalg.{DenseMatrix, DenseVector, Vector, Vectors}
import org.apache.spark.ml.regression.LinearRegression
import org.apache.spark.ml.util.MLReadable
import org.apache.spark.sql.{DataFrame, Row}
import com.microsoft.ml.spark.stages.basic.UDFTransformer
import org.apache.spark.ml.{NamespaceInjections, PipelineModel}
import org.apache.spark.sql.functions.udf
import org.apache.spark.sql.types.DoubleType
import java.awt.GridLayout
import java.nio.file.Paths

import javax.swing._
import com.microsoft.ml.spark.Readers.implicits._
import com.microsoft.ml.spark.core.test.base.{LinuxOnly, TestBase}
import com.microsoft.ml.spark.core.test.fuzzing.{FuzzingMethods, TestObject, TransformerFuzzing}
import org.apache.commons.io.FileUtils
import org.apache.spark.ml.linalg.DenseVector
import org.apache.spark.sql.{DataFrame, Row, SaveMode}
import org.opencv.core.{Mat, MatOfByte}
import org.opencv.imgcodecs.Imgcodecs
import org.opencv.imgproc.Imgproc

class ImageLIMESuite extends FuzzingMethods with ImageFeaturizerUtils {

//  lazy val x = (0 to 10).map(_.toDouble)
//  lazy val y = x.map(_ * 7)

  //lazy val df: DataFrame = session
  //  .createDataFrame(x.map(Vectors.dense(_)).zip(y))

  //lazy val lr = new LinearRegression().setFeaturesCol("_1").setLabelCol("_2")
  lazy val featurizer = new ImageFeaturizer()
//  lazy val fitlr = lr.fit(df)
//
//  lazy val t = new ImageLIME()
//    .setModel(fitlr)
//    .setLabelCol(fitlr.getPredictionCol)
//    .setSampler[Vector](
//    { arr: Vector =>
//      (1 to 10).map { i =>
//        Vectors.dense(arr(0) + scala.util.Random.nextGaussian())
//      }.toArray
//    }, VectorType)
//    //.setFeaturesCol("_1")
//    .setOutputCol("weights")

  test("Image featurizer should work with ResNet50", TestBase.Extended) {
    val resNet = resNetModel()
    val getEntryUdf = udf({
      vec: org.apache.spark.ml.linalg.Vector =>
        vec(0)
    }, DoubleType)
    val udfTransformer = new UDFTransformer()
      .setInputCol(resNet.getOutputCol)
      .setOutputCol(resNet.getOutputCol)
      .setUDF(getEntryUdf)
    val pipeline = NamespaceInjections.pipelineModel(Array(resNet, udfTransformer))

    lazy val lime = new ImageLIME()
      .setModel(pipeline)
      .setLabelCol(resNet.getOutputCol)
      .setOutputCol("weights")
      .setInputCol(inputCol)

    val count = lime.transform(groceryImages).count
    println(count)
  }

  //override def testObjects(): Seq[TestObject[ImageLIME]] = Seq(new TestObject(t, df))

  //override def reader: MLReadable[_] = ImageLIME
}
