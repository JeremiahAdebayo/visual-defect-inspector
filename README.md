# Visual Defect Inspector

An industrial anomaly detection API built on [PatchCore](https://arxiv.org/abs/2106.08265), trained on the [MVTec AD](https://www.mvtec.com/company/research/datasets/mvtec-ad) dataset. Upload an image of a bottle and the API returns an anomaly prediction, confidence score, and a heatmap overlay highlighting suspicious regions.

**Live API:** [codehashira73-visual-defect-inspector.hf.space/docs](https://codehashira73-visual-defect-inspector.hf.space/docs)

---

## Demo

| Normal | Defective | Anomaly Heatmap |
|--------|-----------|-----------------|
| ![normal](assets/normal.png) | ![defective](assets/defective.png) | ![heatmap](assets/heatmap.png) |

> **Note:** This model is trained on MVTec AD bottle images — professionally lit, white background, standardized angles. Testing with images from this distribution will give reliable results. Random internet images may return false positives due to distribution shift (different backgrounds, lighting, angles).

---

## How It Works

PatchCore is a memory-based anomaly detection algorithm. Instead of learning what anomalies look like (which is impossible without labeled defect data), it learns what *normal* looks like and flags anything that deviates.

**Training (offline):**
1. Pass all normal training images through a pretrained CNN backbone (`resnet18`)
2. Extract intermediate feature maps from `layer2` and `layer3` — these capture both low-level textures and mid-level semantics
3. Flatten these into patch-level embeddings, one per spatial location
4. Apply coreset subsampling (ratio=0.1) to compress the embeddings into a representative memory bank — this keeps inference fast without sacrificing much accuracy

**Inference (at API call time):**
1. Pass the uploaded image through the same backbone
2. Extract patch embeddings from the same layers
3. For each patch, find its nearest neighbor in the memory bank and compute the distance
4. Large distance = that patch looks nothing like any normal patch = anomaly
5. Upsample per-patch distances back to image resolution → anomaly heatmap
6. Take the maximum patch distance as the image-level anomaly score
7. Compare score against a threshold computed during training → `NORMAL` or `ANOMALOUS`

### Why resnet18 over wide_resnet50_2?

Both backbones performed nearly identically — `wide_resnet50_2` achieved `pixel_AUROC=0.986` vs `resnet18` at `pixel_AUROC=0.978`. Since PatchCore's performance is driven primarily by the coreset memory bank and nearest-neighbor search rather than backbone capacity, the heavier backbone offers no meaningful advantage. `resnet18` was selected for its faster inference time and lower memory footprint at deployment, with negligible cost to detection performance.

---

## Results

Trained and evaluated on the **bottle** category of MVTec AD.

| Backbone | Image AUROC | Pixel AUROC | Image F1 | Model Size |
|----------|-------------|-------------|----------|------------|
| wide_resnet50_2 | 1.000 | 0.986 | 0.992 | ~1.5 GB |
| **resnet18 (deployed)** | **1.000** | **0.978** | **0.992** | **42 MB** |

Both runs logged with MLflow under the `visual-defect-inspector` experiment.

---

## API Usage

### `GET /health`
Health check endpoint.

```bash
curl https://codehashira73-visual-defect-inspector.hf.space/health
```

**Response:**
```json
{"status": "ok"}
```

---

### `POST /inspect`
Upload an image and get an anomaly prediction.

```bash
curl -X POST \
  https://codehashira73-visual-defect-inspector.hf.space/inspect \
  -F "file=@bottle.png"
```

**Response:**
```json
{
  "prediction": "ANOMALOUS",
  "anomaly_score": 0.9753,
  "heatmap_base64": "/9j/4AAQSkZJRgAB..."
}
```

| Field | Type | Description |
|-------|------|-------------|
| `prediction` | string | `"NORMAL"` or `"ANOMALOUS"` |
| `anomaly_score` | float | Score between 0 and 1. Higher = more anomalous |
| `heatmap_base64` | string | Base64-encoded JPEG of the original image overlaid with the anomaly heatmap (blue = normal, red = anomalous) |

To render the heatmap in Python:
```python
import base64
from PIL import Image
import io

heatmap_bytes = base64.b64decode(response["heatmap_base64"])
image = Image.open(io.BytesIO(heatmap_bytes))
image.show()
```

---

## Project Structure

```
visual-defect-inspector/
├── app/
│   ├── __init__.py
│   ├── main.py          # FastAPI app — routes, CORS, validation
│   └── inference.py     # Model loading (singleton) + predict logic
├── saved_model/
│   └── weights/
│       └── torch/
│           └── model.pt # PatchCore model with memory bank (via Git LFS)
├── Anomaly_detection.ipynb  # Training, evaluation, MLflow logging
├── Dockerfile
├── requirements.txt
└── .gitignore
```

---

## Tech Stack

| Component | Tool |
|-----------|------|
| Anomaly detection | [anomalib](https://github.com/openvinotoolkit/anomalib) |
| Backbone | ResNet18 (PyTorch) |
| Experiment tracking | MLflow |
| API framework | FastAPI + Uvicorn |
| Image processing | OpenCV, Pillow |
| Containerization | Docker |
| Deployment | Hugging Face Spaces |
| Model storage | Git LFS |

---

## Local Setup

**Prerequisites:** Python 3.10+, Git LFS installed

```bash
# Clone the repo
git clone https://github.com/JeremiahAdebayo/visual-defect-inspector.git
cd visual-defect-inspector

# Install dependencies
pip install -r requirements.txt

# Run the API
uvicorn app.main:app --reload
```

API will be available at `http://127.0.0.1:8000/docs`

**With Docker:**
```bash
docker build -t visual-defect-inspector .
docker run -p 7860:7860 visual-defect-inspector
```

---

## Limitations

- Trained on a single MVTec AD category (bottle). Does not generalize to other object types without retraining.
- Sensitive to distribution shift — images must closely resemble the MVTec training distribution (white background, controlled lighting, top-down angle) for reliable results.
- Anomaly threshold is fixed at training time. May need recalibration for production use cases with different defect types.

---

## Author

**Jeremiah Adebayo**  
3rd Year Information Technology Student, University of Iloilo  
[GitHub](https://github.com/JeremiahAdebayo) · [Hugging Face](https://huggingface.co/CodeHashira73)