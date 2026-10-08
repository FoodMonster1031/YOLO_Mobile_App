package com.evestigate.yolo.camera

import android.util.Log
import androidx.camera.core.ImageAnalysis
import androidx.camera.core.ImageProxy
import com.evestigate.yolo.detector.DetectionResult
import com.evestigate.yolo.detector.ImageUtils
import com.evestigate.yolo.detector.YoloDetector

/**
 * CameraX analyzer: converts each RGBA frame to an upright bitmap and runs detection.
 * Use with STRATEGY_KEEP_ONLY_LATEST so slow inference simply drops frames.
 */
class FrameAnalyzer(
    private val detector: YoloDetector,
    private val onResult: (DetectionResult, fps: Float) -> Unit,
) : ImageAnalysis.Analyzer {

    private var lastFrameNs = 0L
    private var smoothedFps = 0f

    override fun analyze(image: ImageProxy) {
        try {
            val bitmap = ImageUtils.rotate(image.toBitmap(), image.imageInfo.rotationDegrees)
            val result = detector.detect(bitmap)

            val now = System.nanoTime()
            if (lastFrameNs != 0L) {
                val fps = 1e9f / (now - lastFrameNs)
                smoothedFps = if (smoothedFps == 0f) fps else smoothedFps * 0.9f + fps * 0.1f
            }
            lastFrameNs = now
            onResult(result, smoothedFps)
        } catch (t: Throwable) {
            Log.e("FrameAnalyzer", "Detection failed", t)
        } finally {
            image.close()
        }
    }
}
