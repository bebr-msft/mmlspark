// Copyright (C) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License. See LICENSE in project root for information.

package com.microsoft.ml.spark.stages

import com.microsoft.ml.spark.core.contracts.{HasFeaturesCol, HasOutputCol, Wrappable}
import com.microsoft.ml.spark.core.serialize.params.{EstimatorParam, TransformerParam, UDFParam}
import org.apache.spark.ml.classification.{LogisticRegression, LogisticRegressionModel}
import org.apache.spark.ml.linalg.SQLDataTypes.VectorType
import org.apache.spark.ml.param.{IntParam, ParamMap}
import org.apache.spark.ml.regression.{LinearRegression, LinearRegressionModel}
import org.apache.spark.ml.util.{ComplexParamsReadable, ComplexParamsWritable, Identifiable}
import org.apache.spark.ml.{Estimator, Model, Transformer}
import org.apache.spark.sql.expressions.UserDefinedFunction
import org.apache.spark.sql.functions.{col, explode, udf}
import org.apache.spark.sql.types.{ArrayType, DataType, DoubleType, StructType}
import org.apache.spark.sql.{DataFrame, Dataset, Row}
import com.microsoft.ml.spark.core.schema.DatasetExtensions.findUnusedColumnName
import org.apache.spark.ml.linalg.{DenseVector, Vector => SparkVector}

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

  def vectorSampler(scale: Double): SparkVector => Iterator[SparkVector] = {v =>
    val sampler = arraySampler[Double](doubleSampler(scale))
    sampler(v.toArray).map(new DenseVector(_))
  }

  private def defaultSampler(dt: DataType): Any => Iterator[Any] = dt match {
      case ArrayType(elementType, _) => arraySampler(defaultSampler(elementType)).asInstanceOf[Any => Iterator[Any]]
      case DoubleType => doubleSampler(1.0).asInstanceOf[Any => Iterator[Any]]
      case VectorType => vectorSampler(1.0).asInstanceOf[Any => Iterator[Any]]
  }

  def defaultFiniteSampler(dt: DataType, n: Int): UserDefinedFunction =
    udf(finiteSampler(defaultSampler(dt), n), ArrayType(dt))

  private def finiteSampler[T: ClassTag](sampler: T => Iterator[T], n: Int): T => Array[T] = {t =>
    sampler(t).take(n).toArray
  }

}

/** Distributed implementation of
  * Local Interpretable Model-Agnostic Explanations (LIME)
  *
  * https://arxiv.org/pdf/1602.04938v1.pdf
  */
class LIME(val uid: String) extends Transformer
  with HasFeaturesCol with HasOutputCol
  with Wrappable with ComplexParamsWritable {
  def this() = this(Identifiable.randomUID("LIME"))

  val model = new TransformerParam(this, "model", "Model to try to locally approximate")

  def getModel: Transformer = $(model)

  def setModel(v: Transformer): this.type = set(model, v)

  val localModel = new EstimatorParam(this, "localModel", "The model to fit locally around each point", {
    case _: LinearRegression => true
    case _: LogisticRegression => true
  })

  def getLocalModel: Estimator[_ <: Model[_]] = $(localModel)

  def setLocalModel(v: Estimator[_ <: Model[_]]): this.type = set(localModel, v)

  val sampler = new UDFParam(this, "sampler", "The sampler to generate local candidates to regress")

  def getSampler: UserDefinedFunction = $(sampler)

  def setSampler(v: UserDefinedFunction): this.type = set(sampler, v)

  def setSampler[T](f: T => Array[T], elementType: DataType): this.type = {
    setSampler(udf(f, ArrayType(elementType)))
  }

  val nSamples = new IntParam(this, "nSamples", "The number of samples to generate if using a default sampler")

  def getNSamples: Int = $(nSamples)

  def setNSamples(v: Int): this.type = set(nSamples, v)

  setDefault(nSamples -> 100)

  override def transform(dataset: Dataset[_]): DataFrame = {
    if (get(sampler).isEmpty){
      setSampler(LIME.defaultFiniteSampler(dataset.schema(getFeaturesCol).dataType, getNSamples))
    }
    val df = dataset.toDF
    val sampleFeaturesCol = findUnusedColumnName("sampleFeatures", df)
    val samplePredictionCol = findUnusedColumnName("samplePrediction", df)

    val model = getModel match {
      case m: Transformer if m.hasParam("featuresCol")
        && m.hasParam("labelCol") =>
        m.set(m.getParam("featuresCol"), sampleFeaturesCol)
          .set(m.getParam("predictionCol"), samplePredictionCol)
    }

    val localModel = getLocalModel match {
      case e: Estimator[_] if e.hasParam("featuresCol")
        && e.hasParam("labelCol") =>
        e.set(e.getParam("featuresCol"), sampleFeaturesCol)
          .set(e.getParam("labelCol"), samplePredictionCol)
          .asInstanceOf[Estimator[_ <: Model[_]]]
    }

    val sampledIterator = df.toLocalIterator().map { row =>
      val localDF = df.sparkSession.createDataFrame(Seq(row), row.schema)
        .withColumn(sampleFeaturesCol,
          explode(getSampler(col(getFeaturesCol))))
      val mappedLocalDF = model.transform(localDF)
      val coeffs = localModel.fit(mappedLocalDF) match {
        case lr: LinearRegressionModel => lr.coefficients
        case lr: LogisticRegressionModel => lr.coefficients
      }
      Row.merge(row, Row(coeffs))
    }

    df.sparkSession.createDataFrame(sampledIterator.toSeq, df.schema
      .add(getOutputCol, VectorType))
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
