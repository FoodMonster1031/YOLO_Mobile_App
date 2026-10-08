package com.evestigate.yolo.detector

/**
 * Class-aware greedy non-maximum suppression. Returns detections sorted by descending score.
 */
fun nonMaxSuppression(
    detections: List<Detection>,
    iouThreshold: Float,
    maxDetections: Int = 300,
): List<Detection> {
    val sorted = detections.sortedByDescending { it.score }
    val kept = ArrayList<Detection>(minOf(sorted.size, maxDetections))
    for (candidate in sorted) {
        if (kept.size >= maxDetections) break
        val suppressed = kept.any {
            it.classId == candidate.classId && it.box.iou(candidate.box) > iouThreshold
        }
        if (!suppressed) kept += candidate
    }
    return kept
}
