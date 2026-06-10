package com.example.imageclassifier

import android.content.Context
import android.graphics.Bitmap
import android.graphics.Matrix
import android.os.SystemClock
import android.util.Log
import org.tensorflow.lite.DataType
import org.tensorflow.lite.Interpreter
import org.tensorflow.lite.gpu.CompatibilityList
import org.tensorflow.lite.gpu.GpuDelegate
import org.tensorflow.lite.support.common.FileUtil
import org.tensorflow.lite.support.common.ops.NormalizeOp
import org.tensorflow.lite.support.image.ImageProcessor
import org.tensorflow.lite.support.image.TensorImage
import org.tensorflow.lite.support.image.ops.ResizeOp
import org.tensorflow.lite.support.tensorbuffer.TensorBuffer
import java.io.BufferedReader
import java.io.InputStreamReader
import java.nio.ByteBuffer

class ImageClassifier(
    private val context: Context,
    private val modelFileName: String = MODEL_FILENAME,
    private val labelsFileName: String = LABELS_FILENAME,
    private val useGpu: Boolean = true,
    private val numThreads: Int = 4,
    private val maxResults: Int = 3
) {
    companion object {
        private const val TAG = "ImageClassifier"
        const val MODEL_FILENAME = "mobilenetv3_classifier.tflite"
        const val LABELS_FILENAME = "labels.txt"
        const val IMAGE_SIZE = 224

        // ImageNet normalization (mean/std scaled to 0-255 pixel range): (pixel/255 - mean) / std
        private val INPUT_MEAN = floatArrayOf(123.675f, 116.28f, 103.53f)
        private val INPUT_STD = floatArrayOf(58.395f, 57.12f, 57.375f)
    }
    }

    data class Recognition(val label: String, val confidence: Float, val index: Int) {
        val confidencePercent: String get() = "%.1f%%".format(confidence * 100)
    }

    data class ClassifierResult(val recognitions: List<Recognition>, val inferenceTimeMs: Long)

    private var interpreter: Interpreter? = null
    private var gpuDelegate: GpuDelegate? = null
    private val labels: List<String> by lazy { loadLabels() }

    private val imageProcessor = ImageProcessor.Builder()
        .add(ResizeOp(IMAGE_SIZE, IMAGE_SIZE, ResizeOp.ResizeMethod.BILINEAR))
        .add(NormalizeOp(INPUT_MEAN, INPUT_STD))
        .build()

    init { setupInterpreter() }

    private fun setupInterpreter() {
        val options = Interpreter.Options().apply { numThreads = this@ImageClassifier.numThreads }
        if (useGpu) {
            val compatList = CompatibilityList()
            if (compatList.isDelegateSupportedOnThisDevice) {
                gpuDelegate = GpuDelegate()
                options.addDelegate(gpuDelegate!!)
                Log.i(TAG, "GPU delegate enabled")
            } else {
                Log.w(TAG, "GPU not supported, using CPU")
            }
        }
        val modelBuffer: ByteBuffer = FileUtil.loadMappedFile(context, modelFileName)
        interpreter = Interpreter(modelBuffer, options)
        val inputShape = interpreter!!.getInputTensor(0).shape()
        Log.i(TAG, "Loaded: input=${inputShape.toList()} labels=${labels.size}")
    }

    fun classify(bitmap: Bitmap, rotationDegrees: Int = 0): ClassifierResult {
        val interp = interpreter ?: return ClassifierResult(emptyList(), 0L)
        val src = if (rotationDegrees != 0) rotateBitmap(bitmap, rotationDegrees) else bitmap
        val tensorImage = TensorImage(DataType.FLOAT32)
        tensorImage.load(src)
        val processed = imageProcessor.process(tensorImage)
        val outputShape = interp.getOutputTensor(0).shape()
        val outputBuffer = TensorBuffer.createFixedSize(outputShape, DataType.FLOAT32)
        val t0 = SystemClock.uptimeMillis()
        interp.run(processed.buffer, outputBuffer.buffer.rewind())
        val elapsed = SystemClock.uptimeMillis() - t0
        val scores = softmax(outputBuffer.floatArray)
        val recognitions = scores
            .mapIndexed { i, p -> Pair(i, p) }
            .sortedByDescending { it.second }
            .take(maxResults)
            .map { (i, p) -> Recognition(if (i < labels.size) labels[i] else "Class $i", p, i) }
        return ClassifierResult(recognitions, elapsed)
    }

    private fun softmax(logits: FloatArray): FloatArray {
        val max = logits.max()
        val exp = logits.map { Math.exp((it - max).toDouble()).toFloat() }
        val sum = exp.sum()
        return exp.map { it / sum }.toFloatArray()
    }

    private fun loadLabels(): List<String> = try {
        BufferedReader(InputStreamReader(context.assets.open(labelsFileName))).use {
            it.readLines().map(String::trim).filter(String::isNotEmpty)
        }
    } catch (e: Exception) {
        Log.e(TAG, "Label load failed: ${e.message}")
        emptyList()
    }

    private fun rotateBitmap(bitmap: Bitmap, degrees: Int): Bitmap {
        val m = Matrix().apply { postRotate(degrees.toFloat()) }
        return Bitmap.createBitmap(bitmap, 0, 0, bitmap.width, bitmap.height, m, true)
    }

    fun getNumLabels(): Int = labels.size

    fun close() {
        interpreter?.close(); interpreter = null
        gpuDelegate?.close(); gpuDelegate = null
    }
}
