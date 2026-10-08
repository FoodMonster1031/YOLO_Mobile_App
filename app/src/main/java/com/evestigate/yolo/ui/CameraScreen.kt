package com.evestigate.yolo.ui

import android.Manifest
import android.content.pm.PackageManager
import androidx.activity.compose.rememberLauncherForActivityResult
import androidx.activity.result.contract.ActivityResultContracts
import androidx.camera.core.CameraSelector
import androidx.camera.core.ImageAnalysis
import androidx.camera.core.Preview
import androidx.camera.core.resolutionselector.AspectRatioStrategy
import androidx.camera.core.resolutionselector.ResolutionSelector
import androidx.camera.lifecycle.ProcessCameraProvider
import androidx.camera.view.PreviewView
import androidx.compose.foundation.background
import androidx.compose.foundation.layout.Arrangement
import androidx.compose.foundation.layout.Box
import androidx.compose.foundation.layout.Column
import androidx.compose.foundation.layout.fillMaxSize
import androidx.compose.foundation.layout.padding
import androidx.compose.material3.Button
import androidx.compose.material3.MaterialTheme
import androidx.compose.material3.Text
import androidx.compose.runtime.Composable
import androidx.compose.runtime.DisposableEffect
import androidx.compose.runtime.getValue
import androidx.compose.runtime.mutableFloatStateOf
import androidx.compose.runtime.mutableStateOf
import androidx.compose.runtime.remember
import androidx.compose.runtime.setValue
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.graphics.Color
import androidx.compose.ui.platform.LocalContext
import androidx.compose.ui.unit.dp
import androidx.compose.ui.viewinterop.AndroidView
import androidx.core.content.ContextCompat
import androidx.lifecycle.compose.LocalLifecycleOwner
import com.evestigate.yolo.camera.FrameAnalyzer
import com.evestigate.yolo.detector.DetectionResult
import com.evestigate.yolo.detector.YoloDetector
import java.util.concurrent.Executors

@Composable
fun CameraScreen(detector: YoloDetector, modifier: Modifier = Modifier) {
    val context = LocalContext.current
    var hasPermission by remember {
        mutableStateOf(
            ContextCompat.checkSelfPermission(context, Manifest.permission.CAMERA) == PackageManager.PERMISSION_GRANTED
        )
    }
    val launcher = rememberLauncherForActivityResult(ActivityResultContracts.RequestPermission()) { hasPermission = it }

    if (!hasPermission) {
        Column(
            modifier.fillMaxSize().padding(24.dp),
            verticalArrangement = Arrangement.Center,
            horizontalAlignment = Alignment.CenterHorizontally,
        ) {
            Text("Camera permission is needed for live detection.")
            Button(onClick = { launcher.launch(Manifest.permission.CAMERA) }, modifier = Modifier.padding(top = 16.dp)) {
                Text("Grant permission")
            }
        }
        return
    }

    var result by remember { mutableStateOf<DetectionResult?>(null) }
    var fps by remember { mutableFloatStateOf(0f) }

    Box(modifier.fillMaxSize()) {
        CameraPreview(detector) { r, f ->
            result = r
            fps = f
        }
        DetectionOverlay(result, fill = true, modifier = Modifier.fillMaxSize())
        StatsBadge(
            text = buildString {
                append(if (detector.usingGpu) "GPU" else "CPU")
                result?.let { append("  ${it.inferenceMs} ms  ${it.detections.size} obj") }
                if (fps > 0f) append("  %.1f fps".format(fps))
            },
            modifier = Modifier.align(Alignment.TopStart).padding(12.dp),
        )
    }
}

@Composable
private fun CameraPreview(detector: YoloDetector, onResult: (DetectionResult, Float) -> Unit) {
    val context = LocalContext.current
    val lifecycleOwner = LocalLifecycleOwner.current
    val previewView = remember {
        PreviewView(context).apply { scaleType = PreviewView.ScaleType.FILL_CENTER }
    }
    val executor = remember { Executors.newSingleThreadExecutor() }

    DisposableEffect(lifecycleOwner) {
        val providerFuture = ProcessCameraProvider.getInstance(context)
        providerFuture.addListener({
            val provider = providerFuture.get()
            val ratio = ResolutionSelector.Builder()
                .setAspectRatioStrategy(AspectRatioStrategy.RATIO_4_3_FALLBACK_AUTO_STRATEGY)
                .build()
            val preview = Preview.Builder().setResolutionSelector(ratio).build()
                .also { it.surfaceProvider = previewView.surfaceProvider }
            val analysis = ImageAnalysis.Builder()
                .setResolutionSelector(ratio)
                .setBackpressureStrategy(ImageAnalysis.STRATEGY_KEEP_ONLY_LATEST)
                .setOutputImageFormat(ImageAnalysis.OUTPUT_IMAGE_FORMAT_RGBA_8888)
                .build()
                .also { it.setAnalyzer(executor, FrameAnalyzer(detector, onResult)) }
            provider.unbindAll()
            provider.bindToLifecycle(lifecycleOwner, CameraSelector.DEFAULT_BACK_CAMERA, preview, analysis)
        }, ContextCompat.getMainExecutor(context))

        onDispose {
            if (providerFuture.isDone) providerFuture.get().unbindAll()
            executor.shutdown()
        }
    }

    AndroidView(factory = { previewView }, modifier = Modifier.fillMaxSize())
}

@Composable
fun StatsBadge(text: String, modifier: Modifier = Modifier) {
    Text(
        text = text,
        color = Color.White,
        style = MaterialTheme.typography.labelLarge,
        modifier = modifier
            .background(Color(0x99000000), MaterialTheme.shapes.small)
            .padding(horizontal = 10.dp, vertical = 6.dp),
    )
}
