package com.evestigate.yolo.ui

import androidx.compose.foundation.Canvas
import androidx.compose.ui.Modifier
import androidx.compose.ui.geometry.Offset
import androidx.compose.ui.geometry.Size
import androidx.compose.ui.graphics.Color
import androidx.compose.ui.graphics.drawscope.Stroke
import androidx.compose.ui.graphics.drawscope.drawIntoCanvas
import androidx.compose.ui.graphics.nativeCanvas
import androidx.compose.ui.graphics.toArgb
import androidx.compose.ui.platform.testTag
import androidx.compose.runtime.Composable
import androidx.compose.runtime.remember
import com.evestigate.yolo.detector.DetectionResult
import com.evestigate.yolo.detector.OverlayTransform

private val palette = listOf(
    Color(0xFFFF3838), Color(0xFF2C99A8), Color(0xFFFF9D97), Color(0xFF00C2FF),
    Color(0xFFCFD231), Color(0xFF48F90A), Color(0xFF92CC17), Color(0xFF3DDB86),
    Color(0xFF1A9334), Color(0xFF00D4BB), Color(0xFFFF701F), Color(0xFF344593),
    Color(0xFF6473FF), Color(0xFF0018EC), Color(0xFF8438FF), Color(0xFF520085),
)

fun colorFor(classId: Int): Color = palette[classId % palette.size]

/**
 * Draws boxes and "label 0.87" tags. [fill] must match how the underlying image is shown:
 * true for a centre-cropped camera preview, false for an image scaled to fit.
 */
@Composable
fun DetectionOverlay(result: DetectionResult?, fill: Boolean, modifier: Modifier = Modifier) {
    val textPaint = remember {
        android.graphics.Paint().apply {
            isAntiAlias = true
            textSize = 36f
            color = android.graphics.Color.WHITE
        }
    }
    val bgPaint = remember { android.graphics.Paint() }

    Canvas(modifier.testTag("detection_overlay")) {
        val r = result ?: return@Canvas
        val t = OverlayTransform.compute(r.imageWidth, r.imageHeight, size.width, size.height, fill)
        for (d in r.detections) {
            val b = t.map(d.box)
            val color = colorFor(d.classId)
            drawRect(
                color = color,
                topLeft = Offset(b.left, b.top),
                size = Size(b.width, b.height),
                style = Stroke(width = 4f),
            )
            val text = "${d.label} ${"%.2f".format(d.score)}"
            val textWidth = textPaint.measureText(text)
            val textHeight = textPaint.textSize
            val tagTop = (b.top - textHeight - 8f).coerceAtLeast(0f)
            bgPaint.color = color.toArgb()
            drawIntoCanvas { canvas ->
                canvas.nativeCanvas.drawRect(b.left, tagTop, b.left + textWidth + 12f, tagTop + textHeight + 8f, bgPaint)
                canvas.nativeCanvas.drawText(text, b.left + 6f, tagTop + textHeight - 2f, textPaint)
            }
        }
    }
}
