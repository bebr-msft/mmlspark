package com.microsoft.ml.spark.cntk

import java.awt.Color
import java.awt.image.BufferedImage

class SuperpixelSuite extends CNTKTestUtils {

  lazy val sp = new Superpixel()
  lazy val width = 300
  lazy val height = 300
  lazy val rgbArray = new Array[Int](width * height)
  lazy val img: BufferedImage = new BufferedImage(width, height, BufferedImage.TYPE_INT_RGB)

  // Adds colors to the img
  for (y <- 0 until height) {
    val red = (y * 255) / (height - 1)
    for (x <- 0 until width) {
      val green = (x * 255) / (width - 1)
      val blue = 128
      rgbArray(x + y * height) = (red << 16) | (green << 8) | blue
    }
  }
  img.setRGB(0, 0, width, height, rgbArray, 0, width)

  lazy val allClusters = sp.cluster(img, 16, 130)
  lazy val randomClusters = Superpixel.censorImage(img, allClusters.get, 0.25)

  test("Censored clusters' pixels should be black in the censored image") {
    val outputImg = randomClusters._1
    randomClusters._3.zipWithIndex.foreach { case (state, i) => {
      if (!state) {
        randomClusters._2(i).pixels.foreach(pt => {
          val color = new Color(outputImg.getRGB(pt._1, pt._2))
          assert(color.getRed === 0 && color.getGreen === 0 && color.getBlue === 0)
        })
      }
    }
    }
  }

  test("The correct censored image gets created from clusters and their states") {
    val outputImg = randomClusters._1
    val imageFromStates = Superpixel.createImage(img, randomClusters._2, randomClusters._3)

    assert(outputImg.getWidth === imageFromStates.getWidth &&
      outputImg.getHeight === imageFromStates.getHeight)

    for (x <- 0 until outputImg.getWidth) {
      for (y <- 0 until outputImg.getHeight) {
        assert(outputImg.getRGB(x, y) === imageFromStates.getRGB(x, y))
      }
    }
  }
}
