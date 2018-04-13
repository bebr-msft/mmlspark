// Copyright (C) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License. See LICENSE in project root for information.

package com.microsoft.ml.spark.cntk

import java.awt.image.BufferedImage

import com.microsoft.ml.spark.Readers.implicits._
import com.microsoft.ml.spark.core.schema.ImageSchema
import com.microsoft.ml.spark.core.test.base.TestBase
import com.microsoft.ml.spark.core.test.fuzzing.FuzzingMethods
import com.microsoft.ml.spark.stages.basic.UDFTransformer
import org.apache.spark.ml.NamespaceInjections
import org.apache.spark.ml.linalg.DenseVector
import org.apache.spark.sql.functions.udf
import org.apache.spark.sql.types.DoubleType
import org.apache.spark.sql.{DataFrame, Row}

class ImageLIMESuite extends FuzzingMethods with ImageFeaturizerUtils {

  lazy val featurizer = new ImageFeaturizer()

  test("Image featurizer should work with ResNet50", TestBase.Extended) {
    val resNet = resNetModel().setCutOutputLayers(0)
    val getEntryUdf = udf({vec: org.apache.spark.ml.linalg.Vector => vec(0)}, DoubleType)
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
      .setCellSize(10)
      .setModifier(130)

    val testImages: DataFrame = session
      .readImages(s"$filesRoot/Images/CIFAR/00000.png", false)
      .withColumnRenamed("image", inputCol)

    val result: DataFrame = lime.transform(testImages)
    result.printSchema()

    // Gets first row from the LIME-transformed data frame
    val topRow: Row = result.take(1)(0)

    // Extracts the image, superpixels, and weights of importances from the first row of the data frame
    val imgRow: Row = topRow.getAs[Row](0)

    // Converts the row values to their appropriate types
    val superpixels1: SuperpixelData = SuperpixelData.fromRow(topRow.getAs[Row](1))
    val states1 = topRow.getAs[DenseVector](2).toArray.map(_ > 0)

    val superpixels2 = SuperpixelData.fromSuperpixel(
      new Superpixel(ImageSchema.toBufferedImage(imgRow), 10, 130))

    assert(superpixels1.clusters.map(_.sorted) === superpixels2.clusters.map(_.sorted))

    // Creates the censored image, the explanation of the model
    val censoredImage1: BufferedImage = Superpixel.censorImage(imgRow, superpixels1, states1)
    val censoredImage2: BufferedImage = Superpixel.censorImage(imgRow, superpixels2, states1)

    //Superpixel.displayImage(sp.getClusteredImage)
    //Superpixel.displayImage(censoredImage1)
    //Superpixel.displayImage(censoredImage2)

  }

  //override def testObjects(): Seq[TestObject[ImageLIME]] = Seq(new TestObject(t, df))

  //override def reader: MLReadable[_] = ImageLIME
}
