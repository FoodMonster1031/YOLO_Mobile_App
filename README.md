# YOLO Detect (Android)

Offline, on-device object detection with Ultralytics **YOLO11n** (80 COCO classes) running on
**LiteRT / TensorFlow Lite**. Two modes: live **Camera** and **Gallery** photo. Tries the GPU
delegate first and falls back to the CPU. The app has no internet permission.

## 1. Export the model (one time, needs internet)

Use **Python 3.12**. TensorFlow, which the TFLite exporter needs, has no Python 3.14 wheels.

The model is already exported and committed at `app/src/main/assets/yolo11n_float32.tflite`.
Redo this only to change the model.

Ultralytics' own `format=tflite` refuses to run on Windows ("only supported on Linux x86 and macOS"),
so on Windows you export to ONNX and convert it with `onnx2tf`:

```powershell
cd D:\WorkAI\yolo-export            # venv already created with Python 3.12
.venv\Scripts\activate
yolo export model=yolo11n.pt format=onnx imgsz=640 opset=17 simplify=True
onnx2tf -i yolo11n.onnx -o tflite_out
copy tflite_out\yolo11n_float32.tflite D:\WorkAI\YOLO\app\src\main\assets\
```

Use the **float32** file. The `float16` file from onnx2tf has fp16 activations, which LiteRT's CPU kernels reject.
On Linux/macOS, `yolo export model=yolo11n.pt format=tflite` works directly.

Any YOLOv8/YOLO11 *detection* `.tflite` works (outputs in normalized or pixel coordinates, in either tensor layout).
If its name differs, the app picks up the first `.tflite` file in `assets/`. For custom classes, replace `labels.txt`
(one class per line, in model order).

## 2. Build and run

Open the folder in Android Studio, let Gradle sync once while online, then enable
*Gradle → Toggle Offline Mode*. Plug in a phone with USB debugging and press **Run**.

## Sample images

`samples/` has 34 test photos covering about 50 of the 80 classes. They come from COCO val2017 (Flickr, various
Creative Commons licenses) and the Ultralytics examples. Each file is named after what the model finds in it.
To load them into an emulator or phone gallery:

```bash
adb shell mkdir -p /sdcard/Pictures/YOLO-samples
adb push samples/. /sdcard/Pictures/YOLO-samples/
adb shell content call --method scan_volume --uri content://media --arg external_primary
```

Then use **Gallery → Pick photo** in the app.

## 3. Tests (TDD gate)

```powershell
.\gradlew lintDebug testDebugUnitTest assembleDebug     # JVM tests: letterbox, NMS, decoding, overlay math
.\gradlew connectedDebugAndroidTest                     # on a device: real model on bus.jpg, overlay drawing
```

CI runs the same steps on every push and PR (`.github/workflows/android-ci.yml`). Instrumented
tests run on an emulator, and pushing a `v*` tag publishes a release APK.

## Layout

| Path | What |
|---|---|
| `detector/YoloPostProcessor.kt` | Decodes the `[1, 84, 8400]` output, applies score threshold and NMS, undoes the letterbox |
| `detector/Letterbox.kt`, `Nms.kt`, `OverlayTransform.kt` | Pure-Kotlin geometry, unit tested |
| `detector/YoloDetector.kt` | LiteRT interpreter on a dedicated thread, GPU→CPU fallback, preprocessing |
| `camera/FrameAnalyzer.kt` | CameraX frames → upright bitmap → detector |
| `ui/` | Compose screens and the box overlay |
