package com.evestigate.yolo.detector

import kotlin.math.max
import kotlin.math.min

/** Axis-aligned box in pixel coordinates (left, top, right, bottom). Pure Kotlin so it is JVM-testable. */
data class Box(val left: Float, val top: Float, val right: Float, val bottom: Float) {

    val width: Float get() = max(0f, right - left)
    val height: Float get() = max(0f, bottom - top)
    val area: Float get() = width * height

    fun iou(other: Box): Float {
        val iw = min(right, other.right) - max(left, other.left)
        val ih = min(bottom, other.bottom) - max(top, other.top)
        if (iw <= 0f || ih <= 0f) return 0f
        val inter = iw * ih
        val union = area + other.area - inter
        return if (union <= 0f) 0f else inter / union
    }

    fun clamp(width: Float, height: Float) = Box(
        left.coerceIn(0f, width),
        top.coerceIn(0f, height),
        right.coerceIn(0f, width),
        bottom.coerceIn(0f, height),
    )

    companion object {
        fun fromCenter(cx: Float, cy: Float, w: Float, h: Float) =
            Box(cx - w / 2f, cy - h / 2f, cx + w / 2f, cy + h / 2f)
    }
}

data class Detection(
    val box: Box,
    val classId: Int,
    val label: String,
    val score: Float,
)
