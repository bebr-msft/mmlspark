package com.microsoft.ml.spark.cntk

import java.awt.image.BufferedImage
import java.io.File
import java.util
import javax.imageio.ImageIO

/**
  *   Based on "Superpixel algorithm implemented in Java" at
  *   popscan.blogspot.com/2014/12/superpixel-algorithm-implemented-in-java.html
  */

object Superpixel {

  def saveImage(filename: String, image: BufferedImage): Unit = {
    val file = new File(filename)
    try
      ImageIO.write(image, "png", file)
    catch {
      case e: Exception =>
        System.out.println(e.toString + " Image '" + filename + "' saving failed.")
    }
  }

  def loadImage(filename: String): Option[BufferedImage] = {
    try
      Some(ImageIO.read(new File(filename)))
    catch {
      case e: Exception =>
        System.out.println(e.toString + " Image '" + filename + "' not found.")
        None
    }
  }
}

class Superpixel() {
  // arrays to store values during process
  var distances: Option[Array[Double]] = None: Option[Array[Double]]
  var labels: Option[Array[Int]] = None: Option[Array[Int]]
  var reds: Option[Array[Int]] = None: Option[Array[Int]]
  var greens: Option[Array[Int]] = None: Option[Array[Int]]
  var blues: Option[Array[Int]] = None: Option[Array[Int]]
  var clusters: Option[Array[Cluster]] = None: Option[Array[Cluster]]
  // in case of instable clusters, max number of loops
  val maxClusteringLoops = 50

  def calculate(image: BufferedImage, S: Double, m: Double): BufferedImage = {
    val width = image.getWidth
    val height = image.getHeight
    val result = new BufferedImage(width, height, BufferedImage.TYPE_INT_RGB)
    val start = System.currentTimeMillis
    // get the image pixels
    val pixels = image.getRGB(0, 0, width, height, null, 0, width)
    // create and fill lookup tables
    distances = Some(new Array[Double](width * height))
    util.Arrays.fill(distances.get, Integer.MAX_VALUE)
    labels = Some(new Array[Int](width * height))
    util.Arrays.fill(labels.get, -1)
    // split rgb-values to own arrays
    reds = Some(new Array[Int](width * height))
    greens = Some(new Array[Int](width * height))
    blues = Some(new Array[Int](width * height))
    var y = 0
    while (y < height) {
      var x = 0
      while (x < width) {
        val pos = x + y * width
        val color = pixels(pos)
        reds(pos) = color >> 16 & 0x000000FF
        greens(pos) = color >> 8 & 0x000000FF
        blues(pos) = color >> 0 & 0x000000FF

        {
          x += 1; x - 1
        }
      }

      {
        y += 1; y - 1
      }
    }
    // create clusters
    createClusters(image, S, m)
    // loop until all clusters are stable!
    var loops = 0
    var pixelChangedCluster = true
    while (pixelChangedCluster && loops < maxClusteringLoops) {
      pixelChangedCluster = false
      loops += 1
      // for each cluster center C
      var i = 0
      while (i < clusters.get.length) {
        val c = clusters.get(i)
        // for each pixel i in 2S region around
        // cluster center
        val xs = Math.max((c.avg_x - S).toInt, 0)
        val ys = Math.max((c.avg_y - S).toInt, 0)
        val xe = Math.min((c.avg_x + S).toInt, width)
        val ye = Math.min((c.avg_y + S).toInt, height)
        y = ys
        while ( {
          y < ye
        }) {
          var x = xs
          while ( {
            x < xe
          }) {
            val pos = x + width * y
            val D = c.distance(x, y, reds.get(pos), greens.get(pos), blues.get(pos), S, m, width, height)
            if ((D < distances.get(pos)) && (labels.get(pos) != c.id)) {
              distances.get(pos) = D
              labels.get(pos) = c.id
              pixelChangedCluster = true
            }
            // end for x
            {
              x += 1; x - 1
            }
          }
          // end for y
          {
            y += 1; y - 1
          }
        }
        // end for clusters
        {
          i += 1; i - 1
        }
      }
      // reset clusters
      var index = 0
      while (index < clusters.get.length) {
        clusters.get(index).reset()

        {
          index += 1; index - 1
        }
      }
      // add every pixel to cluster based on label
      y = 0
      while (y < height) {
        var x = 0
        while (x < width) {
          val pos = x + y * width
          clusters.get(labels.get(pos)).addPixel(x, y, reds.get(pos), greens.get(pos), blues.get(pos))

          {
            x += 1; x - 1
          }
        }

        {
          y += 1; y - 1
        }
      }
      // calculate centers
      var idx = 0
      while (idx < clusters.get.length) {
        clusters.get(idx).calculateCenter()

        {
          idx += 1; idx - 1
        }
      }
    }
    // Create output image with pixel edges
    y = 1
    while (y < height - 1) {
      var x = 1
      while (x < width - 1) {
        val id1 = labels.get(x + y * width)
        val id2 = labels.get((x + 1) + y * width)
        val id3 = labels.get(x + (y + 1) * width)
        if (id1 != id2 || id1 != id3) {
          result.setRGB(x, y, 0x000000)
        }
        else result.setRGB(x, y, image.getRGB(x, y))

        {
          x += 1; x - 1
        }
      }

      {
        y += 1; y - 1
      }
    }

    val end = System.currentTimeMillis
    System.out.println("Clustered to " + clusters.get.length + " superpixels in " + loops + " loops in " + (end - start) + " ms.")
    result
  }

  /*
   * Create initial clusters.
   */
  def createClusters(image: BufferedImage, cellSize: Double, modifier: Double): Unit = {
    val temp = new util.Vector[SuperpixelJava#Cluster]
    val width = image.getWidth
    val height = image.getHeight
    var even = false
    var xstart: Double = 0
    var id = 0
    var y = cellSize / 2
    while (y < height) {
      // alternate clusters x-position to create nice hexagon grid
      if (even) {
        xstart = cellSize / 2.0
        even = false
      }
      else {
        xstart = cellSize
        even = true
      }
      var x = xstart
      while (x < width) {
        val pos = (x + y * width).toInt
        val c = new SuperpixelJava#Cluster(id, reds.get(pos), greens.get(pos), blues.get(pos), x.toInt, y.toInt, cellSize, modifier)
        temp.add(c)
        id += 1

        x += cellSize
      }

      y += cellSize
    }
    clusters = Some(new Array[Cluster](temp.size))
    var i = 0
    while (i < temp.size) {
      clusters(i) = temp.elementAt(i)

      {
        i += 1; i - 1
      }
    }
  }

  class Cluster(var id: Int, val in_red: Int, val in_green: Int, val in_blue: Int, val x: Int, val y: Int, val cellSize: Double, val modifier: Double) {
    var inv: Double = 1.0 / ((cellSize / modifier) * (cellSize / modifier)) // inv variable for optimization
    var pixelCount = .0 // pixels in this cluster
    var avg_red = .0 // average red value
    var avg_green = .0 // average green value
    var avg_blue = .0 // average blue value
    var sum_red = .0 // sum red values
    var sum_green = .0 // sum green values
    var sum_blue = .0 // sum blue values
    var sum_x = .0 // sum x
    var sum_y = .0 // sum y
    var avg_x = .0 // average x
    var avg_y = .0 // average y

    addPixel(x, y, in_red, in_green, in_blue)
    // calculate center with initial one pixel
    calculateCenter()

    def reset(): Unit = {
      avg_red = 0
      avg_green = 0
      avg_blue = 0
      sum_red = 0
      sum_green = 0
      sum_blue = 0
      pixelCount = 0
      avg_x = 0
      avg_y = 0
      sum_x = 0
      sum_y = 0
    }

    /*
     * Add pixel color values to sum of previously added
     * color values.
     */
    def addPixel(x: Int, y: Int, in_red: Int, in_green: Int, in_blue: Int): Unit = {
      sum_x += x
      sum_y += y
      sum_red += in_red
      sum_green += in_green
      sum_blue += in_blue
      pixelCount += 1
    }

    def calculateCenter(): Unit = {
      // Optimization: using "inverse"
      // to change divide to multiply
      val inv = 1 / pixelCount
      avg_red = sum_red * inv
      avg_green = sum_green * inv
      avg_blue = sum_blue * inv
      avg_x = sum_x * inv
      avg_y = sum_y * inv
    }

    def distance(x: Int, y: Int, red: Int, green: Int, blue: Int, S: Double, m: Double, w: Int, h: Int): Double = {
      // power of color difference between given pixel and cluster center
      val dx_color = (avg_red - red) * (avg_red - red) + (avg_green - green) * (avg_green - green) + (avg_blue - blue) * (avg_blue - blue)
      // power of spatial difference between
      val dx_spatial = (avg_x - x) * (avg_x - x) + (avg_y - y) * (avg_y - y)
      // Calculate approximate distance with squares to get more accurate results
      Math.sqrt(dx_color) + Math.sqrt(dx_spatial * inv)
    }
  }
}