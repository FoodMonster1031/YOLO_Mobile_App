package com.evestigate.yolo

import android.content.Context
import android.os.Bundle
import androidx.activity.ComponentActivity
import androidx.activity.compose.setContent
import androidx.activity.enableEdgeToEdge
import androidx.compose.foundation.isSystemInDarkTheme
import androidx.compose.foundation.layout.Box
import androidx.compose.foundation.layout.fillMaxSize
import androidx.compose.foundation.layout.padding
import androidx.compose.material.icons.Icons
import androidx.compose.material.icons.filled.Face
import androidx.compose.material.icons.filled.Search
import androidx.compose.material3.CircularProgressIndicator
import androidx.compose.material3.Icon
import androidx.compose.material3.MaterialTheme
import androidx.compose.material3.NavigationBar
import androidx.compose.material3.NavigationBarItem
import androidx.compose.material3.Scaffold
import androidx.compose.material3.Text
import androidx.compose.material3.darkColorScheme
import androidx.compose.material3.lightColorScheme
import androidx.compose.runtime.Composable
import androidx.compose.runtime.DisposableEffect
import androidx.compose.runtime.LaunchedEffect
import androidx.compose.runtime.getValue
import androidx.compose.runtime.mutableStateOf
import androidx.compose.runtime.remember
import androidx.compose.runtime.saveable.rememberSaveable
import androidx.compose.runtime.setValue
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.platform.LocalContext
import androidx.compose.ui.unit.dp
import com.evestigate.yolo.detector.YoloDetector
import com.evestigate.yolo.ui.CameraScreen
import com.evestigate.yolo.ui.GalleryScreen
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.withContext

class MainActivity : ComponentActivity() {
    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        enableEdgeToEdge()
        setContent {
            MaterialTheme(colorScheme = if (isSystemInDarkTheme()) darkColorScheme() else lightColorScheme()) {
                YoloApp()
            }
        }
    }
}

private sealed interface DetectorState {
    data object Loading : DetectorState
    data class Ready(val detector: YoloDetector) : DetectorState
    data class Failed(val message: String) : DetectorState
}

private enum class Tab(val title: String) { Camera("Camera"), Gallery("Gallery") }

@Composable
fun YoloApp() {
    val context = LocalContext.current
    var state by remember { mutableStateOf<DetectorState>(DetectorState.Loading) }

    LaunchedEffect(Unit) {
        state = withContext(Dispatchers.Default) { loadDetector(context.applicationContext) }
    }
    DisposableEffect(Unit) {
        onDispose { (state as? DetectorState.Ready)?.detector?.close() }
    }

    var tab by rememberSaveable { mutableStateOf(Tab.Camera) }

    Scaffold(
        bottomBar = {
            NavigationBar {
                NavigationBarItem(
                    selected = tab == Tab.Camera,
                    onClick = { tab = Tab.Camera },
                    icon = { Icon(Icons.Filled.Face, contentDescription = null) },
                    label = { Text(Tab.Camera.title) },
                )
                NavigationBarItem(
                    selected = tab == Tab.Gallery,
                    onClick = { tab = Tab.Gallery },
                    icon = { Icon(Icons.Filled.Search, contentDescription = null) },
                    label = { Text(Tab.Gallery.title) },
                )
            }
        },
    ) { padding ->
        val contentModifier = Modifier.fillMaxSize().padding(padding)
        when (val s = state) {
            DetectorState.Loading -> Box(contentModifier, contentAlignment = Alignment.Center) {
                CircularProgressIndicator()
            }
            is DetectorState.Failed -> Box(contentModifier.padding(24.dp), contentAlignment = Alignment.Center) {
                Text(s.message)
            }
            is DetectorState.Ready -> when (tab) {
                Tab.Camera -> CameraScreen(s.detector, contentModifier)
                Tab.Gallery -> GalleryScreen(s.detector, contentModifier)
            }
        }
    }
}

private fun loadDetector(context: Context): DetectorState = try {
    DetectorState.Ready(YoloDetector(context))
} catch (e: Exception) {
    DetectorState.Failed(e.message ?: e.toString())
}
