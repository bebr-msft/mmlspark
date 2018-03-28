// Copyright (C) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License. See LICENSE in project root for information.

package com.microsoft.ml.spark.cntk

import com.microsoft.ml.spark.core.contracts.{HasFeaturesCol, HasLabelCol, HasOutputCol, Wrappable}
import com.microsoft.ml.spark.core.schema.DatasetExtensions.findUnusedColumnName
import com.microsoft.ml.spark.core.schema.{DatasetExtensions, ImageSchema}
import com.microsoft.ml.spark.core.serialize.params.{EstimatorParam, TransformerParam, UDFParam}
import com.microsoft.ml.spark.opencv.{ImageTransformer, UnrollImage}
import com.microsoft.ml.spark.stages.basic.DropColumns
import org.apache.spark.ml.classification.{LogisticRegression, LogisticRegressionModel}
import org.apache.spark.ml.linalg.SQLDataTypes.VectorType
import org.apache.spark.ml.linalg.{DenseVector, Vector => SparkVector}
import org.apache.spark.ml.param.{IntParam, ParamMap}
import org.apache.spark.ml.regression.{LinearRegression, LinearRegressionModel}
import org.apache.spark.ml.util.{ComplexParamsReadable, ComplexParamsWritable, Identifiable}
import org.apache.spark.ml._
import org.apache.spark.sql.expressions.UserDefinedFunction
import org.apache.spark.sql.functions.{col, explode, udf}
import org.apache.spark.sql.types.{ArrayType, DataType, DoubleType, StructType}
import org.apache.spark.sql.{DataFrame, Dataset, Row}

import scala.math.round
import scala.collection.JavaConversions._
import scala.reflect.ClassTag

object LIME extends ComplexParamsReadable[LIME] {

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
    udf({ x: (Row, DenseVector) => x match {case (baseImage: Row, mask: DenseVector) =>
      val dv = if (!greyscaleMask) {
        new DenseVector(UnrollImage.unroll(baseImage).toArray
          .zip(mask.toArray)
          .map { case (e, m) => if (m > threshold) e else 0.0 })
      } else {
        throw new NotImplementedError("need to fill this in")
      }
      UnrollImage.roll(dv, baseImage)
    }}, ImageSchema.columnSchema)

}

/** Distributed implementation of
  * Local Interpretable Model-Agnostic Explanations (LIME)
  *
  * https://arxiv.org/pdf/1602.04938v1.pdf
  */
class LIME(val uid: String) extends Transformer
  with HasFeaturesCol with HasOutputCol with HasLabelCol
  with Wrappable with ComplexParamsWritable {
  def this() = this(Identifiable.randomUID("LIME"))

  val model = new TransformerParam(this, "model", "Model to try to locally approximate")

  def getModel: Transformer = $(model)

  def setModel(v: Transformer): this.type = set(model, v)

  val sampler = new UDFParam(this, "sampler", "The sampler to generate local candidates to regress")

  def getSampler: UserDefinedFunction = $(sampler)

  def setSampler(v: UserDefinedFunction): this.type = set(sampler, v)

  def setSampler[T](f: T => Array[T], elementType: DataType): this.type = {
    setSampler(udf(f, ArrayType(elementType)))
  }

  val preProcessor = new TransformerParam(this, "preProcessor",
    "Transformation to apply before use the linear regression", {x: Transformer =>
      x.hasParam("inputCol") & x.hasParam("outputCol")
    })

  def getPreProcessor: Transformer = $(preProcessor)

  def setPreProcessor(v: Transformer): this.type = set(preProcessor, v)

  val postProcessor = new TransformerParam(this, "postProcessor",
    "Transformation to apply to the trained weights", {x: Transformer =>
      x.hasParam("inputCol") & x.hasParam("outputCol")
    })

  def getPostProcessor: Transformer = $(postProcessor)

  def setPostProcessor(v: Transformer): this.type = set(postProcessor, v)


  val nSamples = new IntParam(this, "nSamples", "The number of samples to generate if using a default sampler")

  def getNSamples: Int = $(nSamples)

  def setNSamples(v: Int): this.type = set(nSamples, v)

  setDefault(nSamples -> 100)

  override def transform(dataset: Dataset[_]): DataFrame = {
    if (get(sampler).isEmpty) {
      setSampler(LIME.defaultFiniteSampler(dataset.schema(getFeaturesCol).dataType, getNSamples))
    }
    val df = dataset.toDF
    val model = getModel

    val localPredictionCol = DatasetExtensions
      .findUnusedColumnName("localPrediction", df)

    val localFeaturesCol = DatasetExtensions
      .findUnusedColumnName("localFeatures", df)

    val localModel = new LinearRegression()
          .setLabelCol(getLabelCol)
          .setFeaturesCol(localFeaturesCol)
          .setPredictionCol(localPredictionCol)

    val preProcessor = getPreProcessor
      .set("inputCol", getFeaturesCol)
      .set("outputCol", localFeaturesCol)

    val sampledIterator = df.toLocalIterator().map { row =>
      val localDF = df.sparkSession.createDataFrame(Seq(row), row.schema)
        .withColumn(getFeaturesCol, explode(getSampler(col(getFeaturesCol))))

      val featurizedLocalDF = preProcessor.transform(localDF)
      // this will have inputs, featurizedInputs

      val mappedLocalDF = model.transform(featurizedLocalDF)
      // This will have the inputs, featurizedInputs, and modelPredictions

      val coeffs = localModel.fit(mappedLocalDF) match {
        case lr: LinearRegressionModel => lr.coefficients
      }
      Row.merge(row, Row(coeffs))
    }

    val outputDF = df.sparkSession.createDataFrame(sampledIterator.toSeq, df.schema
      .add(getOutputCol, VectorType))

    getPostProcessor
      .set("inputCol", getOutputCol)
      .set("outputCol", getOutputCol)
      .transform(outputDF)
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
