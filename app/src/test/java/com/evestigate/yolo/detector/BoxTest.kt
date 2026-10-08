package com.evestigate.yolo.detector

import com.google.common.truth.Truth.assertThat
import org.junit.Test

class BoxTest {

    @Test
    fun `fromCenter converts cxcywh to xyxy`() {
        val box = Box.fromCenter(cx = 50f, cy = 40f, w = 20f, h = 10f)
        assertThat(box).isEqualTo(Box(40f, 35f, 60f, 45f))
    }

    @Test
    fun `area of degenerate box is zero`() {
        assertThat(Box(10f, 10f, 5f, 20f).area).isEqualTo(0f)
    }

    @Test
    fun `iou of identical boxes is one`() {
        val a = Box(0f, 0f, 10f, 10f)
        assertThat(a.iou(a)).isWithin(1e-6f).of(1f)
    }

    @Test
    fun `iou of disjoint boxes is zero`() {
        assertThat(Box(0f, 0f, 10f, 10f).iou(Box(20f, 20f, 30f, 30f))).isEqualTo(0f)
    }

    @Test
    fun `iou of touching boxes is zero`() {
        assertThat(Box(0f, 0f, 10f, 10f).iou(Box(10f, 0f, 20f, 10f))).isEqualTo(0f)
    }

    @Test
    fun `iou of half overlapping boxes is one third`() {
        // intersection 50, union 100 + 100 - 50 = 150
        val iou = Box(0f, 0f, 10f, 10f).iou(Box(5f, 0f, 15f, 10f))
        assertThat(iou).isWithin(1e-6f).of(1f / 3f)
    }

    @Test
    fun `clamp keeps box inside bounds`() {
        val box = Box(-5f, -1f, 105f, 60f).clamp(width = 100f, height = 50f)
        assertThat(box).isEqualTo(Box(0f, 0f, 100f, 50f))
    }
}
