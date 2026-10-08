package com.evestigate.yolo.detector

/**
 * Decodes the raw YOLOv8/YOLO11 detection head output.
 *
 * Expects a channel-major tensor of shape [4 + numClasses][numAnchors]: rows 0..3 are
 * cx, cy, w, h (either normalized 0..1 or in model-input pixels), the remaining rows are
 * per-class scores (already sigmoid-activated; there is no objectness score).
 */
class YoloPostProcessor(
    private val labels: List<String>,
    private val scoreThreshold: Float = 0.25f,
    private val iouThreshold: Float = 0.45f,
    private val maxDetections: Int = 100,
) {

    fun process(output: Array<FloatArray>, letterbox: Letterbox): List<Detection> {
        require(output.size > 4) { "Expected at least 5 output channels, got ${output.size}" }
        val numClasses = output.size - 4
        val numAnchors = output[0].size

        val candidates = ArrayList<Int>()
        val bestClass = IntArray(numAnchors)
        val bestScore = FloatArray(numAnchors)
        for (a in 0 until numAnchors) {
            var cls = 0
            var score = output[4][a]
            for (c in 1 until numClasses) {
                val s = output[4 + c][a]
                if (s > score) {
                    score = s
                    cls = c
                }
            }
            if (score >= scoreThreshold) {
                candidates += a
                bestClass[a] = cls
                bestScore[a] = score
            }
        }
        if (candidates.isEmpty()) return emptyList()

        // TFLite exports emit normalized coordinates; other exports emit pixels.
        val normalized = candidates.all { a -> (0 until 4).all { output[it][a] <= NORMALIZED_LIMIT } }
        val coordScale = if (normalized) letterbox.dstSize.toFloat() else 1f

        val detections = candidates.map { a ->
            val modelBox = Box.fromCenter(
                cx = output[0][a] * coordScale,
                cy = output[1][a] * coordScale,
                w = output[2][a] * coordScale,
                h = output[3][a] * coordScale,
            )
            val cls = bestClass[a]
            Detection(
                box = letterbox.toSource(modelBox),
                classId = cls,
                label = labels.getOrElse(cls) { "class $it" },
                score = bestScore[a],
            )
        }
        return nonMaxSuppression(detections, iouThreshold, maxDetections)
    }

    companion object {
        private const val NORMALIZED_LIMIT = 2f

        /** Converts an anchor-major [numAnchors][channels] tensor into channel-major layout. */
        fun transpose(input: Array<FloatArray>): Array<FloatArray> {
            if (input.isEmpty()) return emptyArray()
            val rows = input.size
            val cols = input[0].size
            return Array(cols) { c -> FloatArray(rows) { r -> input[r][c] } }
        }
    }
}
