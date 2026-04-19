import os
import base64
import pathlib
import platform
import numpy as np
from pathlib import Path
from PIL import Image
import io
import cv2

os.environ["TRUST_REMOTE_CODE"] = "1"
if platform.system()=="Windows":
    pathlib.PosixPath = pathlib.WindowsPath

from anomalib.deploy import TorchInferencer

MODEL_PATH = Path("saved_model/weights/torch/model.pt")

_inferencer = None

def get_inferencer() -> TorchInferencer:
    global _inferencer
    if _inferencer is None:
        _inferencer = TorchInferencer(path=MODEL_PATH)
    return _inferencer


def predict(image_bytes: bytes) -> dict:
    inferencer = get_inferencer()

    # Convert bytes to PIL image
    image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    image_np = np.array(image)

    # Run inference
    result = inferencer.predict(image=image_np)

    # Generate heatmap overlay
    anomaly_map = result.anomaly_map.squeeze().cpu().numpy()
    anomaly_map_normalized = cv2.normalize(anomaly_map, None, 0, 255, cv2.NORM_MINMAX)
    heatmap = cv2.applyColorMap(anomaly_map_normalized.astype(np.uint8), cv2.COLORMAP_JET)
    heatmap_rgb = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)

    # Overlay heatmap on original image
    image_resized = cv2.resize(image_np, (anomaly_map.shape[1], anomaly_map.shape[0]))
    overlay = cv2.addWeighted(image_resized, 0.6, heatmap_rgb, 0.4, 0)

    # Encode overlay to base64
    _, buffer = cv2.imencode(".jpg", cv2.cvtColor(overlay, cv2.COLOR_RGB2BGR))
    heatmap_b64 = base64.b64encode(buffer).decode("utf-8")

    return {
        "prediction": "ANOMALOUS" if bool(result.pred_score > 0.5) else "NORMAL",
        "anomaly_score": float(result.pred_score),
        "heatmap_base64": heatmap_b64
    }
