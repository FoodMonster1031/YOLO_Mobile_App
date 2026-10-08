package com.evestigate.yolo.detector

import kotlin.math.max
import kotlin.math.min

/**
 * Maps image pixel coordinates onto a view that shows the image either scaled to fit
 * (gallery, ContentScale.Fit) or scaled to fill and centre-cropped (PreviewView FILL_CENTER).
 */
data class OverlayTransform(val scale: Float, val offsetX: Float, val offsetY: Float) {

    fun map(box: Box) = Box(
        box.left * scale + offsetX,
        box.top * scale + offsetY,
        box.right * scale + offsetX,
        box.bottom * scale + offsetY,
    )

    companion object {
        fun compute(imageWidth: Int, imageHeight: Int, viewWidth: Float, viewHeight: Float, fill: Boolean): OverlayTransform {
            if (imageWidth <= 0 || imageHeight <= 0) return OverlayTransform(1f, 0f, 0f)
            val sx = viewWidth / imageWidth
            val sy = viewHeight / imageHeight
            val scale = if (fill) max(sx, sy) else min(sx, sy)
            return OverlayTransform(
                scale = scale,
                offsetX = (viewWidth - imageWidth * scale) / 2f,
                offsetY = (viewHeight - imageHeight * scale) / 2f,
            )
        }
    }
}
