package com.evestigate.yolo

import androidx.compose.foundation.layout.size
import androidx.compose.ui.Modifier
import androidx.compose.ui.graphics.toPixelMap
import androidx.compose.ui.test.captureToImage
import androidx.compose.ui.test.junit4.createComposeRule
import androidx.compose.ui.test.onNodeWithTag
import androidx.compose.ui.unit.dp
import androidx.test.ext.junit.runners.AndroidJUnit4
import com.evestigate.yolo.detector.Box
import com.evestigate.yolo.detector.Detection
import com.evestigate.yolo.detector.DetectionResult
import com.evestigate.yolo.ui.DetectionOverlay
import com.evestigate.yolo.ui.colorFor
import com.google.common.truth.Truth.assertThat
import org.junit.Rule
import org.junit.Test
import org.junit.runner.RunWith
import kotlin.math.abs

@RunWith(AndroidJUnit4::class)
class DetectionOverlayTest {

    @get:Rule
    val compose = createComposeRule()

    /** Counts pixels painted in the class-0 box colour (the capture includes the window background). */
    private fun boxColoredPixels(result: DetectionResult?): Int {
        compose.setContent { DetectionOverlay(result, fill = false, modifier = Modifier.size(200.dp)) }
        val pixels = compose.onNodeWithTag("detection_overlay").captureToImage().toPixelMap()
        val target = colorFor(0)
        var count = 0
        for (x in 0 until pixels.width) for (y in 0 until pixels.height) {
            val p = pixels[x, y]
            if (abs(p.red - target.red) < 0.02f && abs(p.green - target.green) < 0.02f && abs(p.blue - target.blue) < 0.02f) count++
        }
        return count
    }

    @Test
    fun drawsNothingWithoutResult() {
        assertThat(boxColoredPixels(null)).isEqualTo(0)
    }

    @Test
    fun drawsBoxesForDetections() {
        val result = DetectionResult(
            detections = listOf(Detection(Box(20f, 20f, 80f, 80f), 0, "person", 0.9f)),
            imageWidth = 100,
            imageHeight = 100,
            inferenceMs = 5,
        )
        assertThat(boxColoredPixels(result)).isGreaterThan(0)
    }
}
