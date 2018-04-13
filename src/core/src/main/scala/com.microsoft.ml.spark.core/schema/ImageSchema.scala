// Copyright (C) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License. See LICENSE in project root for information.

package com.microsoft.ml.spark.core.schema

import java.awt.Point
import java.awt.image.{BufferedImage, DataBufferByte, Raster}
import java.io.{ByteArrayInputStream, ByteArrayOutputStream, FileInputStream}
import java.nio.{ByteBuffer, ByteOrder}

import javax.imageio.ImageIO
import javax.imageio.spi.ImageInputStreamSpi
import org.apache.spark.sql.types._
import org.apache.spark.sql.{DataFrame, Row}

case class ImageData(path: String,
                     height: Int,
                     width: Int,
                     `type`: Int,
                     bytes: Array[Byte])

object ImageData {
  def fromRow(row: Row): ImageData = {
    ImageData(
      row.getString(0),
      row.getInt(1),
      row.getInt(2),
      row.getInt(3),
      row.getAs[Array[Byte]](4)
    )
  }
}

object ImageSchema {

  /** Schema for the image column: Row(String, Int, Int, Int, Array[Byte]) */
  val columnSchema = StructType(
    StructField("path", StringType, true) ::
      StructField("height", IntegerType, true) ::
      StructField("width", IntegerType, true) ::
      // OpenCV type: CV_8U in most cases
      StructField("type", IntegerType, true) ::
      // OpenCV bytes: row-wise BGR in most cases
      StructField("bytes", BinaryType, true) :: Nil)

  // single column of images named "image"
  val schema = StructType(StructField("image", columnSchema, true) :: Nil)

  def getPath(row: Row): String      = row.getString(0)
  def getHeight(row: Row): Int        = row.getInt(1)
  def getWidth(row: Row): Int         = row.getInt(2)
  def getType(row: Row): Int          = row.getInt(3)
  def getBytes(row: Row): Array[Byte] = row.getAs[Array[Byte]](4)

  def toBufferedImage(row: Row): BufferedImage = {
    val bytes = getBytes(row)
    val w = getWidth(row)
    val h = getHeight(row)
    val img = new BufferedImage(w, h, BufferedImage.TYPE_INT_RGB)
    for (r <- 0 until h) {
      for (c <- 0 until w) {
        val index = r * w + c
        val red = bytes(index) & 0xFF
        val green = bytes(index + 1) & 0xFF
        val blue = bytes(index + 2) & 0xFF
        val rgb = (red << 16) | (green << 8) | blue
        img.setRGB(c, r, rgb)
      }
    }
    img
  }

  def toBufferedImage(bytes: Array[Byte], w: Int, h: Int): BufferedImage = {
    val img = new BufferedImage(w, h, BufferedImage.TYPE_INT_RGB)
    for (r <- 0 until h) {
      for (c <- 0 until w) {
        val index = r * w + c
        val red = bytes(index) & 0xFF
        val green = bytes(index + 1) & 0xFF
        val blue = bytes(index + 2) & 0xFF
        val rgb = (red << 16) | (green << 8) | blue
        img.setRGB(c, r, rgb)
      }
    }
    img
  }

  def toBufferedImageTEST(bytes: Array[Byte], w: Int, h: Int): BufferedImage = {
    val img = new BufferedImage(w, h, BufferedImage.TYPE_3BYTE_BGR)
    img.setData(Raster.createRaster(img.getSampleModel(), new DataBufferByte(bytes, bytes.length), new Point()))
    img
  }


  /** Check if the dataframe column contains images (i.e. has imageSchema)
    *
    * @param df
    * @param column
    * @return
    */
  def isImage(df: DataFrame, column: String): Boolean =
    df.schema(column).dataType == columnSchema

}
