package com.evestigate.yolo.detector

import com.google.common.truth.Truth.assertThat
import org.junit.Test

class LetterboxTest {

    @Test
    fun `landscape image is padded top and bottom`() {
        val lb = Letterbox.compute(srcWidth = 1280, srcHeight = 720, dstSize = 640)
        assertThat(lb.scale).isWithin(1e-6f).of(0.5f)
        assertThat(lb.scaledWidth).isEqualTo(640)
        assertThat(lb.scaledHeight).isEqualTo(360)
        assertThat(lb.padX).isEqualTo(0)
        assertThat(lb.padY).isEqualTo(140)
    }

    @Test
    fun `portrait image is padded left and right`() {
        val lb = Letterbox.compute(srcWidth = 480, srcHeight = 640, dstSize = 640)
        assertThat(lb.scale).isWithin(1e-6f).of(1f)
        assertThat(lb.scaledWidth).isEqualTo(480)
        assertThat(lb.padX).isEqualTo(80)
        assertThat(lb.padY).isEqualTo(0)
    }

    @Test
    fun `square image needs no padding`() {
        val lb = Letterbox.compute(srcWidth = 320, srcHeight = 320, dstSize = 640)
        assertThat(lb.scale).isWithin(1e-6f).of(2f)
        assertThat(lb.padX).isEqualTo(0)
        assertThat(lb.padY).isEqualTo(0)
    }

    @Test
    fun `toSource maps model coordinates back to original image`() {
        val lb = Letterbox.compute(srcWidth = 1280, srcHeight = 720, dstSize = 640)
        // A box covering the whole non-padded area maps to the whole image.
        val mapped = lb.toSource(Box(0f, 140f, 640f, 500f))
        assertThat(mapped.left).isWithin(1e-3f).of(0f)
        assertThat(mapped.top).isWithin(1e-3f).of(0f)
        assertThat(mapped.right).isWithin(1e-3f).of(1280f)
        assertThat(mapped.bottom).isWithin(1e-3f).of(720f)
    }

    @Test
    fun `toSource clamps boxes that extend into padding`() {
        val lb = Letterbox.compute(srcWidth = 1280, srcHeight = 720, dstSize = 640)
        val mapped = lb.toSource(Box(-10f, 100f, 700f, 600f))
        assertThat(mapped).isEqualTo(Box(0f, 0f, 1280f, 720f))
    }

    @Test(expected = IllegalArgumentException::class)
    fun `rejects empty source`() {
        Letterbox.compute(srcWidth = 0, srcHeight = 100, dstSize = 640)
    }
}
