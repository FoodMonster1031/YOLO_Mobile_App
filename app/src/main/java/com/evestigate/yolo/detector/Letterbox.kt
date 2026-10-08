package com.evestigate.yolo.detector

import kotlin.math.min
import kotlin.math.roundToInt

/**
 * Geometry for resizing an image into a square model input while keeping its aspect ratio
 * (the same "letterbox" Ultralytics uses), and for mapping model boxes back to the source image.
 */
data class Letterbox(
    val srcWidth: Int,
    val srcHeight: Int,
    val dstSize: Int,
    val scale: Float,
    val scaledWidth: Int,
    val scaledHeight: Int,
    val padX: Int,
    val padY: Int,
) {

    /** Maps a box in model-input pixels back to source-image pixels, clamped to the image. */
    fun toSource(box: Box): Box = Box(
        (box.left - padX) / scale,
        (box.top - padY) / scale,
        (box.right - padX) / scale,
        (box.bottom - padY) / scale,
    ).clamp(srcWidth.toFloat(), srcHeight.toFloat())

    companion object {
        fun compute(srcWidth: Int, srcHeight: Int, dstSize: Int): Letterbox {
            require(srcWidth > 0 && srcHeight > 0) { "Source size must be positive: ${srcWidth}x$srcHeight" }
            require(dstSize > 0) { "Destination size must be positive: $dstSize" }
            val scale = min(dstSize.toFloat() / srcWidth, dstSize.toFloat() / srcHeight)
            val scaledWidth = (srcWidth * scale).roundToInt().coerceIn(1, dstSize)
            val scaledHeight = (srcHeight * scale).roundToInt().coerceIn(1, dstSize)
            return Letterbox(
                srcWidth = srcWidth,
                srcHeight = srcHeight,
                dstSize = dstSize,
                scale = scale,
                scaledWidth = scaledWidth,
                scaledHeight = scaledHeight,
                padX = (dstSize - scaledWidth) / 2,
                padY = (dstSize - scaledHeight) / 2,
            )
        }
    }
}
