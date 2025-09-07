
import torch
import tensorflow as tf
from ultralytics import YOLO
from deepface import DeepFace

print("========== YOLO (PyTorch) ==========")
print("torch version:", torch.__version__)
print("CUDA available:", torch.cuda.is_available())
if torch.cuda.is_available():
    print("GPU:", torch.cuda.get_device_name(0))
else:
    print("⚠️  当前 YOLO 使用 CPU")

print("\n========== DeepFace (TensorFlow) ==========")
print("tensorflow version:", tf.__version__)
gpus = tf.config.list_physical_devices('GPU')
if gpus:
    print("GPU devices:", gpus)
else:
    print("⚠️  当前 DeepFace 使用 CPU")

print("\n========== YOLOv8 quick test ==========")
try:
    model = YOLO("yolov8n-face.pt")
    results = model.predict(source="0", show=False, stream=True, verbose=False)
    print("YOLO 模型加载成功 ✅")
except Exception as e:
    print("YOLO 加载失败 ❌", e)
