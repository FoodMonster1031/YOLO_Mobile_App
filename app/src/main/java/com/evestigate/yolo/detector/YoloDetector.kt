package com.evestigate.yolo.detector

import android.content.Context
import android.graphics.Bitmap
import android.graphics.Canvas
import android.graphics.Color
import android.graphics.Paint
import android.graphics.Rect
import android.os.SystemClock
import android.util.Log
import org.tensorflow.lite.DataType
import org.tensorflow.lite.Interpreter
import org.tensorflow.lite.gpu.CompatibilityList
import org.tensorflow.lite.gpu.GpuDelegate
import java.io.Closeable
import java.io.FileInputStream
import java.nio.ByteBuffer
import java.nio.ByteOrder
import java.nio.MappedByteBuffer
import java.nio.channels.FileChannel
import java.util.concurrent.Callable
import java.util.concurrent.ExecutionException
import java.util.concurrent.Executors

data class DetectionResult(
    val detections: List<Detection>,
    val imageWidth: Int,
    val imageHeight: Int,
    val inferenceMs: Long,
)

/**
 * Runs a YOLOv8/YOLO11 TFLite model fully on-device. Tries the GPU delegate first and falls back
 * to multi-threaded CPU.
 *
 * The GPU delegate must be created, run and closed on one thread, so every interpreter call is
 * funnelled through a dedicated inference thread. [detect] may be called from any thread.
 */
class YoloDetector(
    context: Context,
    modelAsset: String = DEFAULT_MODEL_ASSET,
    labelsAsset: String = DEFAULT_LABELS_ASSET,
    useGpu: Boolean = true,
    scoreThreshold: Float = 0.25f,
    iouThreshold: Float = 0.45f,
) : Closeable {

    private val worker = Executors.newSingleThreadExecutor { Thread(it, "yolo-inference") }
    private val engine: InferenceEngine = try {
        onWorker { InferenceEngine(context, modelAsset, labelsAsset, useGpu, scoreThreshold, iouThreshold) }
    } catch (e: Throwable) {
        worker.shutdown()
        throw e
    }

    val usingGpu: Boolean get() = engine.usingGpu
    val inputSize: Int get() = engine.inputSize
    val modelName: String get() = engine.modelName

    fun detect(bitmap: Bitmap): DetectionResult = onWorker { engine.detect(bitmap) }

    override fun close() {
        if (worker.isShutdown) return
        try {
            onWorker { engine.close() }
        } finally {
            worker.shutdown()
        }
    }

    private fun <T> onWorker(block: () -> T): T = try {
        worker.submit(Callable { block() }).get()
    } catch (e: ExecutionException) {
        throw e.cause ?: e
    }

    companion object {
        const val DEFAULT_MODEL_ASSET = "yolo11n_float32.tflite"
        const val DEFAULT_LABELS_ASSET = "labels.txt"

        /** Uses the requested model, or the first .tflite in assets if it was renamed. */
        fun resolveModelAsset(context: Context, preferred: String): String {
            val files = context.assets.list("").orEmpty()
            if (preferred in files) return preferred
            return files.firstOrNull { it.endsWith(".tflite") }
                ?: throw ModelNotFoundException(preferred)
        }
    }
}

/** Owns the interpreter; only ever touched from [YoloDetector]'s inference thread. */
private class InferenceEngine(
    context: Context,
    modelAsset: String,
    labelsAsset: String,
    useGpu: Boolean,
    scoreThreshold: Float,
    iouThreshold: Float,
) : Closeable {

    private val interpreter: Interpreter
    private var gpuDelegate: GpuDelegate? = null

    val usingGpu: Boolean
    val inputSize: Int
    val modelName: String

    private val numChannels: Int
    private val numAnchors: Int
    private val channelMajor: Boolean

    private val postProcessor: YoloPostProcessor
    private val inputBuffer: ByteBuffer
    private val pixels: IntArray
    private val inputBitmap: Bitmap
    private val inputCanvas: Canvas
    private val paint = Paint(Paint.FILTER_BITMAP_FLAG)

    init {
        val assets = context.assets
        modelName = YoloDetector.resolveModelAsset(context, modelAsset)
        val model = loadModel(context, modelName)
        val labels = assets.open(labelsAsset).bufferedReader().useLines { lines ->
            lines.map { it.trim() }.filter { it.isNotEmpty() }.toList()
        }

        var gpuInterpreter: Interpreter? = null
        if (useGpu) {
            try {
                val compat = CompatibilityList()
                if (compat.isDelegateSupportedOnThisDevice) {
                    val delegate = GpuDelegate(compat.bestOptionsForThisDevice)
                    gpuInterpreter = Interpreter(model, Interpreter.Options().addDelegate(delegate))
                    gpuDelegate = delegate
                }
            } catch (t: Throwable) {
                Log.w(TAG, "GPU delegate unavailable, falling back to CPU", t)
                gpuDelegate?.close()
                gpuDelegate = null
            }
        }
        interpreter = gpuInterpreter ?: Interpreter(model, Interpreter.Options().setNumThreads(CPU_THREADS))
        usingGpu = gpuInterpreter != null

        val inputTensor = interpreter.getInputTensor(0)
        check(inputTensor.dataType() == DataType.FLOAT32) {
            "Only float32/float16 models are supported, got ${inputTensor.dataType()}"
        }
        val inShape = inputTensor.shape() // [1, H, W, 3]
        check(inShape.size == 4 && inShape[3] == 3 && inShape[1] == inShape[2]) {
            "Unexpected input shape ${inShape.contentToString()}"
        }
        inputSize = inShape[1]

        val outShape = interpreter.getOutputTensor(0).shape() // [1, 84, 8400] or [1, 8400, 84]
        check(outShape.size == 3) { "Unexpected output shape ${outShape.contentToString()}" }
        channelMajor = outShape[1] < outShape[2]
        numChannels = if (channelMajor) outShape[1] else outShape[2]
        numAnchors = if (channelMajor) outShape[2] else outShape[1]

        postProcessor = YoloPostProcessor(labels, scoreThreshold, iouThreshold)
        inputBuffer = ByteBuffer.allocateDirect(4 * inputSize * inputSize * 3).order(ByteOrder.nativeOrder())
        pixels = IntArray(inputSize * inputSize)
        inputBitmap = Bitmap.createBitmap(inputSize, inputSize, Bitmap.Config.ARGB_8888)
        inputCanvas = Canvas(inputBitmap)

        Log.i(TAG, "Loaded $modelName input=$inputSize output=${outShape.contentToString()} gpu=$usingGpu")
    }

    fun detect(bitmap: Bitmap): DetectionResult {
        val letterbox = Letterbox.compute(bitmap.width, bitmap.height, inputSize)
        fillInput(bitmap, letterbox)

        val start = SystemClock.uptimeMillis()
        val raw = if (channelMajor) {
            Array(1) { Array(numChannels) { FloatArray(numAnchors) } }
        } else {
            Array(1) { Array(numAnchors) { FloatArray(numChannels) } }
        }
        interpreter.run(inputBuffer, raw)
        val inferenceMs = SystemClock.uptimeMillis() - start

        val output = if (channelMajor) raw[0] else YoloPostProcessor.transpose(raw[0])
        return DetectionResult(
            detections = postProcessor.process(output, letterbox),
            imageWidth = bitmap.width,
            imageHeight = bitmap.height,
            inferenceMs = inferenceMs,
        )
    }

    private fun fillInput(bitmap: Bitmap, lb: Letterbox) {
        inputCanvas.drawColor(LETTERBOX_COLOR)
        val dst = Rect(lb.padX, lb.padY, lb.padX + lb.scaledWidth, lb.padY + lb.scaledHeight)
        inputCanvas.drawBitmap(bitmap, null, dst, paint)
        inputBitmap.getPixels(pixels, 0, inputSize, 0, 0, inputSize, inputSize)

        inputBuffer.rewind()
        for (p in pixels) {
            inputBuffer.putFloat(((p shr 16) and 0xFF) / 255f)
            inputBuffer.putFloat(((p shr 8) and 0xFF) / 255f)
            inputBuffer.putFloat((p and 0xFF) / 255f)
        }
        inputBuffer.rewind()
    }

    override fun close() {
        interpreter.close()
        gpuDelegate?.close()
        gpuDelegate = null
    }

    companion object {
        private const val TAG = "YoloDetector"
        private const val CPU_THREADS = 4
        private val LETTERBOX_COLOR = Color.rgb(114, 114, 114)

        private fun loadModel(context: Context, asset: String): MappedByteBuffer =
            context.assets.openFd(asset).use { fd ->
                FileInputStream(fd.fileDescriptor).use { stream ->
                    stream.channel.map(FileChannel.MapMode.READ_ONLY, fd.startOffset, fd.declaredLength)
                }
            }
    }
}

class ModelNotFoundException(name: String) : IllegalStateException(
    "No YOLO model found in assets. Export it with Ultralytics and copy $name into app/src/main/assets/."
)
