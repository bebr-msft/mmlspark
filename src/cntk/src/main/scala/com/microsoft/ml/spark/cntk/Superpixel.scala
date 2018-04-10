package com.microsoft.ml.spark.cntk

import java.awt.FlowLayout
import java.awt.image.BufferedImage
import java.io.File
import java.util
import javax.imageio.ImageIO
import javax.swing.{ImageIcon, JFrame, JLabel}

import com.microsoft.ml.spark.IO.image.ImageReader
import com.microsoft.ml.spark.core.schema.{ImageData, ImageSchema}
import org.apache.spark.sql.Row
import org.apache.spark.sql.catalyst.ScalaReflection
import org.apache.spark.sql.expressions.UserDefinedFunction
import org.apache.spark.sql.functions.udf
import org.apache.spark.sql.types.DataType

import scala.collection.mutable
import scala.collection.mutable.{ArrayBuffer, ListBuffer}
import scala.util.Random

case class PixelData(x: Int, y: Int)

object PixelData {
  def fromRow(r: Row) = PixelData(r.getInt(0), r.getInt(1))

  val schema: DataType = ScalaReflection.schemaFor[PixelData].dataType
}

case class ClusterData(pixels: Array[PixelData])

object ClusterData {
  val schema: DataType =  ScalaReflection.schemaFor[ClusterData].dataType

  def fromCluster(c: Cluster) =
    ClusterData(c.pixels.map(p => PixelData(p._1, p._2)).toArray)

  def fromRow(r: Row): ClusterData = {
    ClusterData(r.getAs[mutable.WrappedArray[Row]](0).map(PixelData.fromRow).toArray)
  }
}

case class SuperpixelData(clusters: Array[ClusterData])

object SuperpixelData {
  val schema: DataType =  ScalaReflection.schemaFor[SuperpixelData].dataType

  def fromRow(r: Row): SuperpixelData = {
    val clusters = r.getAs[mutable.WrappedArray[Row]](0)
    SuperpixelData(clusters.map(ClusterData.fromRow).toArray)
  }

  def fromSuperpixel(sp: Superpixel): SuperpixelData = {
    SuperpixelData(sp.clusters.map(ClusterData.fromCluster))
  }

  def fromArrCluster(arrCluster: Array[Cluster]): SuperpixelData = {
    SuperpixelData(arrCluster.map(ClusterData.fromCluster))
  }
}

/**
  * Based on "Superpixel algorithm implemented in Java" at
  *   popscan.blogspot.com/2014/12/superpixel-algorithm-implemented-in-java.html
  */
object Superpixel {

  def getSuperpixelUDF(cellSize: Double, modifier: Double): UserDefinedFunction = udf(
    { row: Row => SuperpixelData.fromArrCluster(
      new Superpixel(ImageSchema.toBufferedImage(row), cellSize, modifier).clusters
    )},
    SuperpixelData.schema)

  def censorImageHelper(img: Row, sp: Row, states: mutable.WrappedArray[Boolean]): Row = {
    val bi = censorImage(img, SuperpixelData.fromRow(sp), states.toArray)
    ImageReader.decode(bi).get
  }

  val censorUDF: UserDefinedFunction = udf(censorImageHelper _, ImageSchema.columnSchema)

  def displayImage(img: BufferedImage): Unit = {
    val frame: JFrame = new JFrame()
    frame.getContentPane.setLayout(new FlowLayout())
    frame.getContentPane.add(new JLabel(new ImageIcon(img)))
    frame.pack()
    frame.setVisible(true)
  }

  def saveImage(filename: String, image: BufferedImage): Unit = {
    ImageIO.write(image, "png", new File(filename))
    ()
  }

  def loadImage(filename: String): Option[BufferedImage] = {
    Some(ImageIO.read(new File(filename)))
  }

  def copyImage(source: BufferedImage): BufferedImage = {
    val b = new BufferedImage(source.getWidth, source.getHeight, source.getType)
    val g = b.getGraphics
    g.drawImage(source, 0, 0, null)
    g.dispose()
    b
  }

  def censorImage(imgRow: Row, superpixels: SuperpixelData, clusterStates: Array[Boolean]): BufferedImage = {
    val img = ImageSchema.toBufferedImage(imgRow)
    val output = copyImage(img)

    superpixels.clusters.zipWithIndex.foreach { case (c, i) =>
      if (!clusterStates(i)) {
        c.pixels.foreach(pt => {
          output.setRGB(pt.x, pt.y, 0x000000)
        })
      }
      else {
        c.pixels.foreach(pt => {
          output.setRGB(pt.x, pt.y, img.getRGB(pt.x, pt.y))
        })
      }
    }
    output
  }

  def clusterStateSampler(decInclude: Double, numPixels: Int): Iterator[Array[Boolean]] =
    new Iterator[Array[Boolean]] {
      override def hasNext: Boolean = true

      override def next(): Array[Boolean] = {
        Array.fill(numPixels) {
          Random.nextDouble() > decInclude
        }
      }
    }
}

class Superpixel(image: BufferedImage, cellSize: Double, modifier: Double) {
  // arrays to store values during process
  private val width = image.getWidth
  private val height = image.getHeight
  private val distances: Array[Double] = new Array[Double](width * height)
  private val labels: Array[Int] = new Array[Int](width * height)
  private val reds: Array[Int] = new Array[Int](width * height)
  private val greens: Array[Int] = new Array[Int](width * height)
  private val blues: Array[Int] = new Array[Int](width * height)

  private val start: Long = System.currentTimeMillis
  // get the image pixels
  private val pixels: Array[Int] = image.getRGB(0, 0, width, height, null, 0, width)
  // create and fill lookup tables
  util.Arrays.fill(distances, Integer.MAX_VALUE)
  util.Arrays.fill(labels, -1)
  // split rgb-values to own arrays
  for(y <- 0 until height; x <- 0 until width) {
    val pos = x + y * width
    val color = pixels(pos)
    reds.update(pos, color >> 16 & 0x000000FF)
    greens.update(pos, color >> 8 & 0x000000FF)
    blues.update(pos, color >> 0 & 0x000000FF)
  }

  val clusters: Array[Cluster] = createClusters(image, cellSize, modifier)
  // in case of unstable clusters, max number of loops
  val maxClusteringLoops = 50

  // loop until all clusters are stable!
  var loops = 0
  var pixelChangedCluster = true
  while (pixelChangedCluster && loops < maxClusteringLoops) {
    pixelChangedCluster = false
    loops += 1
    // for each cluster center C
    for (c <- clusters) {
      // for each pixel i in 2S region around
      // cluster center
      val xs = Math.max((c.avg_x - cellSize).toInt, 0)
      val ys = Math.max((c.avg_y - cellSize).toInt, 0)
      val xe = Math.min((c.avg_x + cellSize).toInt, width)
      val ye = Math.min((c.avg_y + cellSize).toInt, height)
      for (y <- ys until ye; x <- xs until xe){
        val pos = x + width * y
        val D = c.distance(x, y,
          reds(pos), greens(pos), blues(pos),
          cellSize, modifier, width, height)
        if ((D < distances(pos)) && (labels(pos) != c.id)) {
          distances.update(pos, D)
          labels.update(pos, c.id)
          pixelChangedCluster = true
        }
      }
    }
    // reset clusters
    clusters.foreach(_.reset())

    // add every pixel to cluster based on label
    for (y <- 0 until height; x<- 0 until width) {
      val pos = x + y * width
      clusters(labels(pos)).addPixel(x, y, reds(pos), greens(pos), blues(pos))
    }
    // calculate centers
    clusters.foreach(_.calculateCenter())
  }

  private val end = System.currentTimeMillis
  println("Clustered to " + clusters.length +
    " superpixels in " + loops + " loops in " + (end - start) + " ms.")


  private def createClusters(image: BufferedImage, cellSize: Double, modifier: Double): Array[Cluster] = {
    val temp = new ListBuffer[Cluster]
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
      } else {
        xstart = cellSize
        even = true
      }
      var x = xstart
      while (x < width) {
        val pos = (x + y * width).toInt
        val c = new Cluster(id, reds(pos), greens(pos), blues(pos), x.toInt, y.toInt, cellSize, modifier)
        temp.append(c)
        id += 1
        x += cellSize
      }
      y += cellSize
    }
    temp.toArray
  }
}

class Cluster(var id: Int, val in_red: Int, val in_green: Int, val in_blue: Int,
              val x: Int, val y: Int, val cellSize: Double, val modifier: Double) {
  private val inv: Double = 1.0 / ((cellSize / modifier) * (cellSize / modifier)) // inv variable for optimization
  private var pixelCount = .0 // pixels in this cluster
  private var avg_red = .0 // average red value
  private var avg_green = .0 // average green value
  private var avg_blue = .0 // average blue value
  private var sum_red = .0 // sum red values
  private var sum_green = .0 // sum green values
  private var sum_blue = .0 // sum blue values
  private var sum_x = .0 // sum x
  private var sum_y = .0 // sum y
  var avg_x = .0 // average x
  var avg_y = .0 // average y
  val pixels = new ArrayBuffer[(Int, Int)]

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
    pixels.clear()
  }

  def addPixel(x: Int, y: Int, in_red: Int, in_green: Int, in_blue: Int): Unit = {
    sum_x += x
    sum_y += y
    sum_red += in_red
    sum_green += in_green
    sum_blue += in_blue
    pixelCount += 1
    pixels.append((x, y))
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
    val dx_color = (avg_red - red) * (avg_red - red) +
      (avg_green - green) * (avg_green - green) + (avg_blue - blue) * (avg_blue - blue)
    // power of spatial difference between
    val dx_spatial = (avg_x - x) * (avg_x - x) + (avg_y - y) * (avg_y - y)
    // Calculate approximate distance with squares to get more accurate results
    Math.sqrt(dx_color) + Math.sqrt(dx_spatial * inv)
  }
}
