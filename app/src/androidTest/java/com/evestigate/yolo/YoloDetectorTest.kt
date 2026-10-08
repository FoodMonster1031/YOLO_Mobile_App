package com.evestigate.yolo

import android.graphics.Bitmap
import android.graphics.BitmapFactory
import androidx.test.ext.junit.runners.AndroidJUnit4
import androidx.test.platform.app.InstrumentationRegistry
import com.evestigate.yolo.detector.YoloDetector
import com.google.common.truth.Truth.assertThat
import org.junit.Test
import org.junit.runner.RunWith

/** Runs the real bundled model on the classic Ultralytics bus.jpg sample. */
@RunWith(AndroidJUnit4::class)
class YoloDetectorTest {

    private val appContext = InstrumentationRegistry.getInstrumentation().targetContext
    private val testContext = InstrumentationRegistry.getInstrumentation().context

    private fun busImage(): Bitmap =
        testContext.assets.open("bus.jpg").use { BitmapFactory.decodeStream(it) }

    private fun assertFindsBusAndPeople(useGpu: Boolean) {
        YoloDetector(appContext, useGpu = useGpu).use { detector ->
            val image = busImage()
            val result = detector.detect(image)

            assertThat(result.imageWidth).isEqualTo(image.width)
            assertThat(result.imageHeight).isEqualTo(image.height)

            val labels = result.detections.filter { it.score > 0.5f }.map { it.label }
            assertThat(labels).contains("bus")
            assertThat(labels.count { it == "person" }).isAtLeast(3)

            result.detections.forEach {
                assertThat(it.box.left).isAtLeast(0f)
                assertThat(it.box.top).isAtLeast(0f)
                assertThat(it.box.right).isAtMost(image.width.toFloat())
                assertThat(it.box.bottom).isAtMost(image.height.toFloat())
            }
        }
    }

    @Test
    fun detectsBusAndPeopleOnCpu() = assertFindsBusAndPeople(useGpu = false)

    @Test
    fun detectsBusAndPeopleWithGpuOrFallback() = assertFindsBusAndPeople(useGpu = true)

    @Test
    fun detectCanBeCalledFromManyThreads() {
        YoloDetector(appContext).use { detector ->
            val image = busImage()
            val threads = (1..3).map { Thread { detector.detect(image) } }
            threads.forEach { it.start() }
            threads.forEach { it.join() }
            assertThat(detector.detect(image).detections).isNotEmpty()
        }
    }
}
