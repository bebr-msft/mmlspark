// Copyright (C) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License. See LICENSE in project root for information.

package com.microsoft.ml.spark.cntk

import com.microsoft.ml.spark.core.contracts._
import com.microsoft.ml.spark.core.schema.{DatasetExtensions, ImageSchema}
import com.microsoft.ml.spark.core.serialize.params.TransformerParam
import com.microsoft.ml.spark.opencv.UnrollImage
import org.apache.spark.ml._
import org.apache.spark.ml.linalg.SQLDataTypes.VectorType
import org.apache.spark.ml.linalg.{DenseVector, Vector => SparkVector}
import org.apache.spark.ml.param.{DoubleParam, IntParam, ParamMap}
import org.apache.spark.ml.regression.{LinearRegression, LinearRegressionModel}
import org.apache.spark.ml.util.{ComplexParamsReadable, ComplexParamsWritable, Identifiable}
import org.apache.spark.sql.catalyst.encoders.RowEncoder
import org.apache.spark.sql.expressions.UserDefinedFunction
import org.apache.spark.sql.functions.{col, udf}
import org.apache.spark.sql.types._
import org.apache.spark.sql.{DataFrame, Dataset, Row}

import scala.collection.JavaConversions._
import scala.collection.mutable
import scala.reflect.ClassTag

object ImageLIME extends ComplexParamsReadable[ImageLIME] {

  def doubleSampler(scale: Double): Double => Iterator[Double] = { x: Double =>
    new Iterator[Double] {
      override def hasNext: Boolean = true

      override def next(): Double = {
        x + scala.util.Random.nextGaussian() * scale
      }
    }
  }

  def arraySampler[T: ClassTag](elementSampler: T => Iterator[T]): Array[T] => Iterator[Array[T]] = { xs =>
    val samplers = xs.map(x => elementSampler(x))
    new Iterator[Array[T]] {
      override def hasNext: Boolean = true

      override def next(): Array[T] = {
        samplers.map(s => s.next())
      }
    }
  }

  def vectorSampler(scale: Double): SparkVector => Iterator[SparkVector] = { v =>
    val sampler = arraySampler[Double](doubleSampler(scale))
    sampler(v.toArray).map(new DenseVector(_))
  }

  def imageSampler(scale: Double): Row => Iterator[Row] = { row =>
    vectorSampler(scale)(UnrollImage.unroll(row)).map { dv =>
      UnrollImage.roll(dv, row)
    }
  }

  private def defaultSampler(dt: DataType): Any => Iterator[Any] = dt match {
    case ArrayType(elementType, _) => arraySampler(defaultSampler(elementType)).asInstanceOf[Any => Iterator[Any]]
    case DoubleType => doubleSampler(1.0).asInstanceOf[Any => Iterator[Any]]
    case VectorType => vectorSampler(1.0).asInstanceOf[Any => Iterator[Any]]
    case dt2 if dt2 == ImageSchema.columnSchema =>
      imageSampler(30.0).asInstanceOf[Any => Iterator[Any]]
  }

  def defaultFiniteSampler(dt: DataType, n: Int): UserDefinedFunction =
    udf(finiteSampler(defaultSampler(dt), n), ArrayType(dt))

  private def finiteSampler[T: ClassTag](sampler: T => Iterator[T], n: Int): T => Array[T] = { t =>
    sampler(t).take(n).toArray
  }

  def importanceMasking(threshold: Double = 0.0, greyscaleMask: Boolean = true): UserDefinedFunction =
    udf({ x: (Row, DenseVector) =>
      x match {
        case (baseImage: Row, mask: DenseVector) =>
          val dv = if (!greyscaleMask) {
            new DenseVector(UnrollImage.unroll(baseImage).toArray
              .zip(mask.toArray)
              .map { case (e, m) => if (m > threshold) e else 0.0 })
          } else {
            throw new NotImplementedError("need to fill this in")
          }
          UnrollImage.roll(dv, baseImage)
      }
    }, ImageSchema.columnSchema)

}

/** Distributed implementation of
  * Local Interpretable Model-Agnostic Explanations (LIME)
  *
  * https://arxiv.org/pdf/1602.04938v1.pdf
  */
class ImageLIME(val uid: String) extends Transformer
  with HasInputCol with HasOutputCol with HasLabelCol
  with Wrappable with ComplexParamsWritable {
  def this() = this(Identifiable.randomUID("LIME"))

  val model = new TransformerParam(this, "model", "Model to try to locally approximate")

  def getModel: Transformer = $(model)

  def setModel(v: Transformer): this.type = set(model, v)

  val nSamples = new IntParam(this, "nSamples", "The number of samples to generate if using a default sampler")

  def getNSamples: Int = $(nSamples)

  def setNSamples(v: Int): this.type = set(nSamples, v)

  val samplingFraction = new DoubleParam(this, "samplingFraction", "TODO: doc string")

  def getSamplingFraction: Double = $(samplingFraction)

  def setSamplingFraction(d: Double): this.type = set(samplingFraction, d)

  val cellSize = new DoubleParam(this, "cellSize", "TODO: doc string")

  def getCellSize: Double = $(cellSize)

  def setCellSize(d: Double): this.type = set(cellSize, d)

  val modifier = new DoubleParam(this, "modifier", "TODO: doc string")

  def getModifier: Double = $(modifier)

  def setModifier(d: Double): this.type = set(modifier, d)

  setDefault(nSamples -> 900, cellSize -> 16, modifier -> 130, samplingFraction -> 0.3)

  override def transform(dataset: Dataset[_]): DataFrame = {
    import dataset.sparkSession.implicits._

    val df = dataset.toDF
    val model = getModel

    val superpixelCol = DatasetExtensions
      .findUnusedColumnName("superpixels", df)

    val localFeaturesCol = DatasetExtensions
      .findUnusedColumnName("localFeatures", df)

    val localModel = new LinearRegression()
      .setLabelCol(getLabelCol)
      .setFeaturesCol(localFeaturesCol)

    // Data frame with new column containing superpixels (Array[Cluster]) for each row (image)
    val spt = new SuperpixelTransformer()
      .setCellSize(getCellSize)
      .setModifier(getModifier)
      .setInputCol(getInputCol)
      .setOutputCol(superpixelCol)

    val spDF = spt.transform(df)

    // Indices of the columns containing each image and image's superpixels
    val superpixelIndex = spDF.schema.fieldIndex(spt.getOutputCol)
    val spDFSchema = spDF.schema

    val indiciesToKeep = spDF.columns.indices//.filter(_ != superpixelIndex)

    // Collects to head node and creates a data frame from each row (image)
    val sampledIterator = spDF.toLocalIterator().map { row =>

      // Gets the superpixels from the row
      val superpixels = SuperpixelData.fromRow(row.getAs[Row](superpixelIndex))

      // Generate samples for the image
      val samples = Superpixel
        .clusterStateSampler(getSamplingFraction, superpixels.clusters.length)
        .take(getNSamples).toList

      // Creates a new data frame for each image, containing samples of cluster states
      val censoredDF = samples.toDF(localFeaturesCol)
        .map(stateRow => Row.merge(row, stateRow))(
          RowEncoder(spDFSchema.add(localFeaturesCol, ArrayType(BooleanType))))
        .withColumn(getInputCol, Superpixel.censorUDF(
          col(getInputCol), col(spt.getOutputCol), col(localFeaturesCol)))
        .withColumn(localFeaturesCol,
          udf(
            { barr: mutable.WrappedArray[Boolean] => new DenseVector(barr.map(b => if (b) 1.0 else 0.0).toArray) },
            VectorType)(col(localFeaturesCol)))

      // Maps the data frame through the deep model
      val mappedLocalDF = model.transform(censoredDF)
      println(mappedLocalDF.count())

      // Fits the data frame to the local model (regression), outputting the weights of importance
      val coefficients = localModel.fit(mappedLocalDF) match {
        case lr: LinearRegressionModel => lr.coefficients
      }

      Row(indiciesToKeep.map(row.get) ++ Seq(coefficients):_*)
    }

    val outputDF = df.sparkSession.createDataFrame(sampledIterator.toSeq, spDF.schema
      .add(getOutputCol, VectorType))

    outputDF
  }

  override def copy(extra: ParamMap): Transformer = defaultCopy(extra)

  /** Add the features column to the schema
    *
    * @param schema
    * @return schema with features column
    */
  override def transformSchema(schema: StructType): StructType = {
    schema.add(getOutputCol, VectorType)
  }

}
