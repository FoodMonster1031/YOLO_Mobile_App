package com.evestigate.yolo.detector

import com.google.common.truth.Truth.assertThat
import org.junit.Test

class OverlayTransformTest {

    @Test
    fun `fit letterboxes a wide image into a tall view`() {
        // 1000x500 image into 500x1000 view: scale 0.5, image is 500x250, centred vertically.
        val t = OverlayTransform.compute(1000, 500, 500f, 1000f, fill = false)
        assertThat(t.scale).isWithin(1e-6f).of(0.5f)
        assertThat(t.offsetX).isWithin(1e-6f).of(0f)
        assertThat(t.offsetY).isWithin(1e-6f).of(375f)
    }

    @Test
    fun `fill crops a wide image to cover a tall view`() {
        // 1000x500 image into 500x1000 view: scale 2, image is 2000x1000, centred horizontally.
        val t = OverlayTransform.compute(1000, 500, 500f, 1000f, fill = true)
        assertThat(t.scale).isWithin(1e-6f).of(2f)
        assertThat(t.offsetX).isWithin(1e-6f).of(-750f)
        assertThat(t.offsetY).isWithin(1e-6f).of(0f)
    }

    @Test
    fun `map converts image box into view coordinates`() {
        val t = OverlayTransform.compute(1000, 500, 500f, 1000f, fill = false)
        assertThat(t.map(Box(0f, 0f, 1000f, 500f))).isEqualTo(Box(0f, 375f, 500f, 625f))
    }

    @Test
    fun `zero sized image gives identity transform`() {
        val t = OverlayTransform.compute(0, 0, 500f, 1000f, fill = true)
        assertThat(t).isEqualTo(OverlayTransform(1f, 0f, 0f))
    }
}
