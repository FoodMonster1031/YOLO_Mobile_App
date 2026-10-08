package com.evestigate.yolo.detector

import com.google.common.truth.Truth.assertThat
import org.junit.Test

class YoloPostProcessorTest {

    private val labels = listOf("person", "bicycle", "car")
    private val numAnchors = 50
    private val lb = Letterbox.compute(srcWidth = 640, srcHeight = 640, dstSize = 640)

    /** Builds a channel-major [4 + numClasses][numAnchors] tensor full of zeros. */
    private fun emptyOutput() = Array(4 + labels.size) { FloatArray(numAnchors) }

    private fun Array<FloatArray>.put(
        anchor: Int, cx: Float, cy: Float, w: Float, h: Float, classId: Int, score: Float,
    ) {
        this[0][anchor] = cx; this[1][anchor] = cy; this[2][anchor] = w; this[3][anchor] = h
        this[4 + classId][anchor] = score
    }

    @Test
    fun `no scores above threshold yields no detections`() {
        val out = emptyOutput()
        out.put(0, 0.5f, 0.5f, 0.1f, 0.1f, classId = 0, score = 0.1f)
        assertThat(YoloPostProcessor(labels).process(out, lb)).isEmpty()
    }

    @Test
    fun `decodes normalized coordinates into image pixels`() {
        val out = emptyOutput()
        out.put(3, cx = 0.5f, cy = 0.25f, w = 0.25f, h = 0.125f, classId = 2, score = 0.8f)

        val dets = YoloPostProcessor(labels).process(out, lb)

        assertThat(dets).hasSize(1)
        val d = dets.single()
        assertThat(d.classId).isEqualTo(2)
        assertThat(d.label).isEqualTo("car")
        assertThat(d.score).isWithin(1e-6f).of(0.8f)
        assertThat(d.box.left).isWithin(1e-3f).of(240f)
        assertThat(d.box.top).isWithin(1e-3f).of(120f)
        assertThat(d.box.right).isWithin(1e-3f).of(400f)
        assertThat(d.box.bottom).isWithin(1e-3f).of(200f)
    }

    @Test
    fun `decodes pixel coordinates when output is not normalized`() {
        val out = emptyOutput()
        out.put(0, cx = 320f, cy = 160f, w = 160f, h = 80f, classId = 0, score = 0.9f)

        val box = YoloPostProcessor(labels).process(out, lb).single().box

        assertThat(box).isEqualTo(Box(240f, 120f, 400f, 200f))
    }

    @Test
    fun `picks the best class for each anchor`() {
        val out = emptyOutput()
        out.put(0, 0.5f, 0.5f, 0.2f, 0.2f, classId = 0, score = 0.3f)
        out[4 + 1][0] = 0.7f

        val d = YoloPostProcessor(labels).process(out, lb).single()

        assertThat(d.label).isEqualTo("bicycle")
        assertThat(d.score).isWithin(1e-6f).of(0.7f)
    }

    @Test
    fun `applies nms to duplicate anchors`() {
        val out = emptyOutput()
        out.put(0, 0.5f, 0.5f, 0.2f, 0.2f, classId = 0, score = 0.9f)
        out.put(1, 0.505f, 0.5f, 0.2f, 0.2f, classId = 0, score = 0.85f)
        out.put(2, 0.1f, 0.1f, 0.1f, 0.1f, classId = 0, score = 0.6f)

        val dets = YoloPostProcessor(labels).process(out, lb)

        assertThat(dets.map { it.score }).containsExactly(0.9f, 0.6f).inOrder()
    }

    @Test
    fun `maps through letterbox to original image size`() {
        val wide = Letterbox.compute(srcWidth = 1280, srcHeight = 720, dstSize = 640)
        val out = emptyOutput()
        // Centre of the model input, 0.5 x 0.25 of it -> 320x160 px in model space.
        out.put(0, 0.5f, 0.5f, 0.5f, 0.25f, classId = 0, score = 0.9f)

        val box = YoloPostProcessor(labels).process(out, wide).single().box

        assertThat(box.left).isWithin(1e-2f).of(320f)
        assertThat(box.right).isWithin(1e-2f).of(960f)
        assertThat(box.top).isWithin(1e-2f).of(200f)
        assertThat(box.bottom).isWithin(1e-2f).of(520f)
    }

    @Test
    fun `unknown class ids get a fallback label`() {
        val out = Array(4 + 4) { FloatArray(numAnchors) } // model has 4 classes, labels only 3
        out[0][0] = 0.5f; out[1][0] = 0.5f; out[2][0] = 0.1f; out[3][0] = 0.1f
        out[4 + 3][0] = 0.9f

        assertThat(YoloPostProcessor(labels).process(out, lb).single().label).isEqualTo("class 3")
    }

    @Test
    fun `transpose converts anchor-major output to channel-major`() {
        val anchorMajor = arrayOf(floatArrayOf(1f, 2f, 3f), floatArrayOf(4f, 5f, 6f))
        val channelMajor = YoloPostProcessor.transpose(anchorMajor)
        assertThat(channelMajor.map { it.toList() })
            .containsExactly(listOf(1f, 4f), listOf(2f, 5f), listOf(3f, 6f)).inOrder()
    }
}
