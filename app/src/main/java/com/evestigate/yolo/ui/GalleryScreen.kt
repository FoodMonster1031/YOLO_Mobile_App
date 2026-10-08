package com.evestigate.yolo.ui

import android.graphics.Bitmap
import androidx.activity.compose.rememberLauncherForActivityResult
import androidx.activity.result.PickVisualMediaRequest
import androidx.activity.result.contract.ActivityResultContracts
import androidx.compose.foundation.Image
import androidx.compose.foundation.layout.Arrangement
import androidx.compose.foundation.layout.Box
import androidx.compose.foundation.layout.Column
import androidx.compose.foundation.layout.Row
import androidx.compose.foundation.layout.fillMaxSize
import androidx.compose.foundation.layout.fillMaxWidth
import androidx.compose.foundation.layout.padding
import androidx.compose.material3.Button
import androidx.compose.material3.CircularProgressIndicator
import androidx.compose.material3.Text
import androidx.compose.runtime.Composable
import androidx.compose.runtime.getValue
import androidx.compose.runtime.mutableStateOf
import androidx.compose.runtime.remember
import androidx.compose.runtime.rememberCoroutineScope
import androidx.compose.runtime.setValue
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.graphics.asImageBitmap
import androidx.compose.ui.layout.ContentScale
import androidx.compose.ui.platform.LocalContext
import androidx.compose.ui.unit.dp
import com.evestigate.yolo.detector.DetectionResult
import com.evestigate.yolo.detector.ImageUtils
import com.evestigate.yolo.detector.YoloDetector
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.launch
import kotlinx.coroutines.withContext

@Composable
fun GalleryScreen(detector: YoloDetector, modifier: Modifier = Modifier) {
    val context = LocalContext.current
    val scope = rememberCoroutineScope()
    var bitmap by remember { mutableStateOf<Bitmap?>(null) }
    var result by remember { mutableStateOf<DetectionResult?>(null) }
    var busy by remember { mutableStateOf(false) }
    var error by remember { mutableStateOf<String?>(null) }

    val picker = rememberLauncherForActivityResult(ActivityResultContracts.PickVisualMedia()) { uri ->
        if (uri == null) return@rememberLauncherForActivityResult
        busy = true
        error = null
        scope.launch {
            try {
                val (bmp, res) = withContext(Dispatchers.Default) {
                    val b = ImageUtils.decodeUpright(context, uri)
                    b to detector.detect(b)
                }
                bitmap = bmp
                result = res
            } catch (e: Exception) {
                error = e.message ?: e.toString()
            } finally {
                busy = false
            }
        }
    }

    Column(modifier.fillMaxSize()) {
        Row(
            Modifier.fillMaxWidth().padding(12.dp),
            horizontalArrangement = Arrangement.SpaceBetween,
            verticalAlignment = Alignment.CenterVertically,
        ) {
            Button(onClick = {
                picker.launch(PickVisualMediaRequest(ActivityResultContracts.PickVisualMedia.ImageOnly))
            }) { Text("Pick photo") }
            result?.let {
                StatsBadge("${if (detector.usingGpu) "GPU" else "CPU"}  ${it.inferenceMs} ms  ${it.detections.size} obj")
            }
        }
        Box(Modifier.fillMaxSize(), contentAlignment = Alignment.Center) {
            val bmp = bitmap
            when {
                busy -> CircularProgressIndicator()
                error != null -> Text("Error: $error", modifier = Modifier.padding(24.dp))
                bmp == null -> Text("Pick a photo to detect objects.")
                else -> {
                    Image(
                        bitmap = bmp.asImageBitmap(),
                        contentDescription = null,
                        contentScale = ContentScale.Fit,
                        modifier = Modifier.fillMaxSize(),
                    )
                    DetectionOverlay(result, fill = false, modifier = Modifier.fillMaxSize())
                }
            }
        }
    }
}
