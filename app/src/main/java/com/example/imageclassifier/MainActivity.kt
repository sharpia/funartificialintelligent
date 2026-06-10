package com.example.imageclassifier

import android.Manifest
import android.content.pm.PackageManager
import android.graphics.Bitmap
import android.os.Bundle
import android.util.Log
import android.view.View
import android.widget.Toast
import androidx.activity.result.contract.ActivityResultContracts
import androidx.appcompat.app.AppCompatActivity
import androidx.camera.core.*
import androidx.camera.lifecycle.ProcessCameraProvider
import androidx.core.content.ContextCompat
import androidx.lifecycle.lifecycleScope
import com.example.imageclassifier.databinding.ActivityMainBinding
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.launch
import kotlinx.coroutines.withContext
import java.util.concurrent.ExecutorService
import java.util.concurrent.Executors

class MainActivity : AppCompatActivity() {

    companion object {
        private const val TAG = "MainActivity"
        private const val INFERENCE_INTERVAL_MS = 300L
    }

    private lateinit var binding: ActivityMainBinding
    private var classifier: ImageClassifier? = null
    private lateinit var cameraExecutor: ExecutorService
    private var lastInferenceTime = 0L
    private var isClassifierReady = false

    private val requestPermissionLauncher = registerForActivityResult(
        ActivityResultContracts.RequestPermission()
    ) { granted ->
        if (granted) startCamera()
        else {
            Toast.makeText(this, getString(R.string.camera_permission_denied), Toast.LENGTH_LONG).show()
            binding.tvStatus.text = getString(R.string.camera_permission_denied)
        }
    }

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        binding = ActivityMainBinding.inflate(layoutInflater)
        setContentView(binding.root)

        cameraExecutor = Executors.newSingleThreadExecutor()
        initClassifier()

        if (ContextCompat.checkSelfPermission(this, Manifest.permission.CAMERA)
            == PackageManager.PERMISSION_GRANTED
        ) {
            startCamera()
        } else {
            requestPermissionLauncher.launch(Manifest.permission.CAMERA)
        }
    }

    private fun initClassifier() {
        binding.tvStatus.text = getString(R.string.loading_model)
        lifecycleScope.launch(Dispatchers.IO) {
            try {
                val c = ImageClassifier(applicationContext)
                withContext(Dispatchers.Main) {
                    classifier = c
                    isClassifierReady = true
                    binding.tvStatus.text = getString(R.string.ready) + " (${c.getNumLabels()} classes)"
                }
            } catch (e: Exception) {
                Log.e(TAG, "Classifier init failed: ${e.message}")
                withContext(Dispatchers.Main) {
                    binding.tvStatus.text = getString(R.string.model_load_error)
                }
            }
        }
    }

    private fun startCamera() {
        val cameraProviderFuture = ProcessCameraProvider.getInstance(this)
        cameraProviderFuture.addListener({
            val cameraProvider = cameraProviderFuture.get()

            val preview = Preview.Builder().build().also {
                it.setSurfaceProvider(binding.viewFinder.surfaceProvider)
            }

            val imageAnalyzer = ImageAnalysis.Builder()
                .setBackpressureStrategy(ImageAnalysis.STRATEGY_KEEP_ONLY_LATEST)
                .setOutputImageFormat(ImageAnalysis.OUTPUT_IMAGE_FORMAT_RGBA_8888)
                .build()
                .also {
                    it.setAnalyzer(cameraExecutor, ::analyzeImage)
                }

            try {
                cameraProvider.unbindAll()
                cameraProvider.bindToLifecycle(
                    this,
                    CameraSelector.DEFAULT_BACK_CAMERA,
                    preview,
                    imageAnalyzer
                )
            } catch (e: Exception) {
                Log.e(TAG, "Camera binding failed: ${e.message}")
            }
        }, ContextCompat.getMainExecutor(this))
    }

    private fun analyzeImage(imageProxy: ImageProxy) {
        val now = System.currentTimeMillis()
        if (!isClassifierReady || now - lastInferenceTime < INFERENCE_INTERVAL_MS) {
            imageProxy.close()
            return
        }
        lastInferenceTime = now

        val bitmap = imageProxy.toBitmap()
        val rotation = imageProxy.imageInfo.rotationDegrees
        imageProxy.close()

        val result = classifier?.classify(bitmap, rotation) ?: return

        runOnUiThread { displayResults(result) }
    }

    private fun displayResults(result: ImageClassifier.ClassifierResult) {
        binding.tvStatus.text = getString(R.string.inference_time_ms, result.inferenceTimeMs)

        val results = result.recognitions
        if (results.isEmpty()) {
            binding.tvResult1.visibility = View.GONE
            binding.tvResult2.visibility = View.GONE
            binding.tvResult3.visibility = View.GONE
            return
        }

        fun formatEntry(r: ImageClassifier.Recognition) = "${r.label}  ${r.confidencePercent}"

        binding.tvResult1.text = if (results.size > 0) formatEntry(results[0]) else ""
        binding.tvResult2.text = if (results.size > 1) formatEntry(results[1]) else ""
        binding.tvResult3.text = if (results.size > 2) formatEntry(results[2]) else ""

        binding.tvResult1.visibility = if (results.size > 0) View.VISIBLE else View.GONE
        binding.tvResult2.visibility = if (results.size > 1) View.VISIBLE else View.GONE
        binding.tvResult3.visibility = if (results.size > 2) View.VISIBLE else View.GONE
    }

    override fun onDestroy() {
        super.onDestroy()
        cameraExecutor.shutdown()
        classifier?.close()
    }
}
