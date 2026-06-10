# MobileNetV3 Real-Time Image Classifier (Android)

Real-time on-device image classification using MobileNetV3-Large with TensorFlow Lite (GPU delegate, float16 quantized).

## 모델 준비 (.pth → .tflite)

PC에서 1회 실행:

```bash
pip install torch torchvision onnx onnx2tf onnx-graphsurgeon sng4onnx onnxsim tensorflow
```

```python
import torch, torch.nn as nn, tensorflow as tf
from torchvision.models import mobilenet_v3_large
import onnx2tf

NUM_CLASSES = 10  # 본인 클래스 수로 변경

# 1. PyTorch 모델 로드
model = mobilenet_v3_large(weights=None)
model.classifier[3] = nn.Linear(model.classifier[3].in_features, NUM_CLASSES)
model.load_state_dict(torch.load("your_model.pth", map_location="cpu"))
model.eval()

# 2. ONNX 변환
torch.onnx.export(model, torch.randn(1, 3, 224, 224), "model.onnx",
    input_names=["input"], output_names=["output"], opset_version=13)

# 3. ONNX → TensorFlow SavedModel (onnx2tf, onnx-tf 대체)
onnx2tf.convert(
    input_onnx_file_path="model.onnx",
    output_folder_path="saved_model",
    copy_onnx_input_output_names_to_tflite=True,
    non_verbose=True,
)

# 4. TFLite 변환 (float16 양자화)
converter = tf.lite.TFLiteConverter.from_saved_model("saved_model")
converter.optimizations = [tf.lite.Optimize.DEFAULT]
converter.target_spec.supported_types = [tf.float16]
with open("mobilenetv3_classifier.tflite", "wb") as f:
    f.write(converter.convert())
```

> `onnx2tf`는 `saved_model/` 폴더에 `.tflite` 파일도 함께 생성하므로, 4번 단계 없이 그 파일을 바로 사용해도 됩니다.

## assets 배치

```
app/src/main/assets/mobilenetv3_classifier.tflite   ← 변환된 모델
app/src/main/assets/labels.txt                      ← 클래스 이름 (1줄 1클래스)
```

**labels.txt 형식** (인덱스 0부터 순서대로):

```
고양이
강아지
새
```

학습 시 클래스 순서 확인:

```python
from torchvision.datasets import ImageFolder
ds = ImageFolder("./data")
for name, idx in sorted(ds.class_to_idx.items(), key=lambda x: x[1]):
    print(idx, name)
```

## APK 빌드

### GitHub Actions (권장)

모델 파일 추가 후 push → **Actions** 탭 → **Build APK** → `app-debug-apk` 다운로드

### 로컬 빌드

```bash
./gradlew assembleDebug
adb install app/build/outputs/apk/debug/app-debug.apk
```

## 추론 파이프라인

```
CameraX → Resize 224×224 → Normalize → TFLite (GPU delegate) → Top-3
```

- 300 ms 간격으로 추론 (배터리 절약)
- GPU delegate 자동 활성화, 미지원 기기는 4-thread CPU 폴백
- Android API 24+ (Android 7.0)

## 프로젝트 구조

```
├── .github/workflows/build-apk.yml
├── app/src/main/
│   ├── assets/                    ← .tflite + labels.txt 여기에 배치
│   ├── java/com/example/imageclassifier/
│   │   ├── ImageClassifier.kt     ← TFLite 추론 (GPU delegate)
│   │   └── MainActivity.kt        ← CameraX + UI
│   └── res/layout/activity_main.xml
└── train/train_mobilenetv3.py     ← TensorFlow 재학습용 (선택)
```
