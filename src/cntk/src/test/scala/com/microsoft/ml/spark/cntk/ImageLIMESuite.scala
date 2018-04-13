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
import java.awt.image.BufferedImage
import java.io.{ByteArrayOutputStream, File}
import java.nio.file.Paths

import com.microsoft.ml.spark
import javax.swing._
import com.microsoft.ml.spark.Readers.implicits._
import com.microsoft.ml.spark.core.contracts.Wrappable
import com.microsoft.ml.spark.core.schema.ImageSchema
import com.microsoft.ml.spark.core.test.base.{LinuxOnly, TestBase}
import com.microsoft.ml.spark.core.test.fuzzing.{FuzzingMethods, TestObject, TransformerFuzzing}
import javax.imageio.ImageIO
import org.apache.commons.io.FileUtils
import org.apache.spark.ml.linalg.DenseVector
import org.apache.spark.rdd.RDD
import org.apache.spark.sql.{DataFrame, Row, SaveMode}
import org.opencv.core.{Mat, MatOfByte}
import org.opencv.imgcodecs.Imgcodecs
import org.opencv.imgproc.Imgproc

import scala.collection.mutable
import scala.util.Random

class ImageLIMESuite extends FuzzingMethods with ImageFeaturizerUtils {

  lazy val featurizer = new ImageFeaturizer()

//  test("Conversion from Row to BufferedImage should work") {
//    val bufImg: BufferedImage = ImageIO.read(new File("/home/bebr/Downloads/Turtle.jpg"))
//
//    // Original
//    val display = Superpixel.displayImage(bufImg)
//
//    val stream = new ByteArrayOutputStream()
//    ImageIO.write(bufImg, "jpg", stream)
//    val bytes = stream.toByteArray
//    val width = display.getWidth
//    val height = display.getHeight
//
//    val newBuf = ImageSchema.toBufferedImageTEST(bytes, width, height)
//
//    // Converted
//    Superpixel.displayImage(newBuf)
//
//    // Delay to view images
//    Thread.sleep(180000)
//  }

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
      .setCellSize(100)
      .setModifier(130)

    val it = new ImageTransformer()
      .setInputCol(inputCol)
      .setOutputCol(inputCol)

    lazy val testImagesPath = s"${groceriesPath}testImages/WIN_20160803_11_28_42_Pro.jpg"
    val testImages: DataFrame = it.transform(session
      .readImages(testImagesPath, true).withColumnRenamed("image", inputCol))

    val result: DataFrame = lime.transform(testImages)

    //result.write.parquet("~/mmlspark/df7")
    //val result = session.read.parquet("~/mmlspark/df7")
    //val result = session.read.parquet("~/mmlspark/df5")

    // Gets first row from the LIME-transformed data frame
    val topRow: Row = result.take(1)(0)

    // Extracts the image, superpixels, and weights of importances from the first row of the data frame
    val img: Row = topRow.getAs[Row](0)
    val spInfo = topRow.getAs[Row](1)
    val weightInfo: DenseVector = topRow.getAs[DenseVector](2)

    // Converts the row values to their appropriate types
    val superpixels: SuperpixelData = SuperpixelData.fromRow(spInfo)
    val weights = weightInfo.toArray

    // Creates the states array using the model-outputted weights
    val finalStates = new Array[Boolean](weights.length)
    for (i <- finalStates.indices) {
      finalStates(i) = weights(i) > 0
    }

    // TODO - TESTING: Creation of new Superpixel instance to create SuperpixelData used in censoring
    // TODO - This test works
    val sp = new Superpixel(ImageSchema.toBufferedImageTEST(ImageSchema.getBytes(img),
      ImageSchema.getWidth(img),
      ImageSchema.getHeight(img)), 100, 130)
    Superpixel.displayImage(sp.getClusteredImage)

    val clusterDataArr = new Array[ClusterData](sp.clusters.length)

    sp.clusters.zipWithIndex.foreach { case (cluster, cIdx) => {
      val pixelDataArr = new Array[PixelData](cluster.pixels.length)

      cluster.pixels.zipWithIndex.foreach { case (pixel, pIdx) => {
        pixelDataArr(pIdx) = new PixelData(pixel._1, pixel._2)
      }}

      clusterDataArr(cIdx) = new ClusterData(pixelDataArr)
    }}

    val superpixels2 = new SuperpixelData(clusterDataArr)


    // TODO - Assert equality
    superpixels.clusters.zipWithIndex.foreach { case (clusterData, cIdx) => {
      clusterData.pixels.zipWithIndex.foreach { case (pixelData, pIdx) => {
        val pt: PixelData = superpixels2.clusters(cIdx).pixels(pIdx)
        println(s"Does ${pixelData.x},${pixelData.y} equal ${pt.x},${pt.y} ?" )
        assert(pixelData.x === pt.x && pixelData.y === pt.y)
      }}
    }}


    // TODO - PRINT FOR SANITY
//    println("SUPERPIXELS")
//    superpixels.clusters.zipWithIndex.foreach { case (clusterData, cIdx) => if (cIdx < 3) {
//      print("Cluster " + cIdx + ": ")
//      clusterData.pixels.foreach(pixelData => {
//        print("(" + pixelData.x + ", " + pixelData.y + ") ")
//      })
//      println()
//    }}
//
//    println("SUPERPIXELS 2")
//    superpixels2.clusters.zipWithIndex.foreach { case (clusterData, cIdx) => if (cIdx < 3) {
//      print("Cluster " + cIdx + ": ")
//      clusterData.pixels.foreach(pixelData => {
//        print("(" + pixelData.x + ", " + pixelData.y + ") ")
//      })
//      println()
//    }}

    // TODO - TESTING: SuperpixelData.fromSuperpixel using the Superpixel instance works
    //val superpixels: SuperpixelData = SuperpixelData.fromSuperpixel(sp)

    // Creates the censored image, the explanation of the model
    val censoredImage: BufferedImage = Superpixel.censorImage(img, superpixels, finalStates)
    val censoredImage2: BufferedImage = Superpixel.censorImage(img, superpixels2, finalStates)


    Superpixel.displayImage(censoredImage)
    Superpixel.displayImage(censoredImage2)

    Thread.sleep(600000)
  }

  //override def testObjects(): Seq[TestObject[ImageLIME]] = Seq(new TestObject(t, df))

  //override def reader: MLReadable[_] = ImageLIME
}
