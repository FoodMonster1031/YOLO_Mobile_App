package com.evestigate.yolo.detector

import com.google.common.truth.Truth.assertThat
import org.junit.Test

class NmsTest {

    private fun det(box: Box, classId: Int, score: Float) =
        Detection(box, classId, "c$classId", score)

    @Test
    fun `empty input returns empty list`() {
        assertThat(nonMaxSuppression(emptyList(), 0.45f)).isEmpty()
    }

    @Test
    fun `overlapping boxes of same class keep the highest score`() {
        val best = det(Box(0f, 0f, 100f, 100f), 0, 0.9f)
        val dup = det(Box(5f, 5f, 105f, 105f), 0, 0.8f)
        assertThat(nonMaxSuppression(listOf(dup, best), 0.45f)).containsExactly(best)
    }

    @Test
    fun `overlapping boxes of different classes are both kept`() {
        val a = det(Box(0f, 0f, 100f, 100f), 0, 0.9f)
        val b = det(Box(5f, 5f, 105f, 105f), 1, 0.8f)
        assertThat(nonMaxSuppression(listOf(a, b), 0.45f)).containsExactly(a, b).inOrder()
    }

    @Test
    fun `boxes below iou threshold are both kept`() {
        val a = det(Box(0f, 0f, 10f, 10f), 0, 0.9f)
        val b = det(Box(5f, 0f, 15f, 10f), 0, 0.8f) // iou = 1/3
        assertThat(nonMaxSuppression(listOf(a, b), 0.45f)).hasSize(2)
    }

    @Test
    fun `result is sorted by score and capped`() {
        val dets = (0 until 10).map { i ->
            det(Box(i * 20f, 0f, i * 20f + 10f, 10f), 0, i / 10f)
        }
        val kept = nonMaxSuppression(dets, 0.45f, maxDetections = 3)
        assertThat(kept.map { it.score }).containsExactly(0.9f, 0.8f, 0.7f).inOrder()
    }
}
