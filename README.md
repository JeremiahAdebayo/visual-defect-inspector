# Visual Defect Inspector

An end-to-end industrial anomaly detection system built on [PatchCore](https://arxiv.org/abs/2106.08265), trained on the [MVTec AD](https://www.mvtec.com/company/research/datasets/mvtec-ad) dataset. Upload an image of a bottle and the API returns an anomaly prediction, confidence score, and a heatmap overlay highlighting suspicious regions.

**Live API:** [codehashira73-visual-defect-inspector.hf.space/docs](https://codehashira73-visual-defect-inspector.hf.space/docs)

---

## How It Works

PatchCore is a memory-based anomaly detection algorithm. Instead of learning what anomalies look like (which requires labeled defect data), it learns what *normal* looks like and flags anything that deviates.

**Training (offline):**
1. Pass all normal training images through a pretrained CNN backbone (`resnet18`)
2. Extract intermediate feature maps from `layer2` and `layer3` — capturing both low-level textures and mid-level semantics
3. Flatten into patch-level embeddings, one per spatial location
4. Apply coreset subsampling (ratio=0.1) to compress embeddings into a representative memory bank

**Inference (at API call time):**
1. Extract patch embeddings from the uploaded image using the same backbone
2. For each patch, find its nearest neighbors in the memory bank and compute distance
3. Large distance = patch looks nothing like any normal patch = anomaly
4. Upsample per-patch distances back to image resolution → anomaly heatmap
5. Maximum patch distance = image-level anomaly score
6. Compare against threshold computed at training time → `NORMAL` or `ANOMALOUS`

For a detailed breakdown of every technical decision made in this project, see [DECISIONS.md](./DECISIONS.md).

---

## Results

Trained and evaluated on the **bottle** category of MVTec AD.

| Backbone | Image AUROC | Pixel AUROC | Model Size |
|----------|-------------|-------------|------------|
| wide_resnet50_2 | 1.000 | 0.986 | ~1.5 GB |
| **resnet18 (deployed)** | **1.000** | **0.978** | **42 MB** |

Both runs tracked with MLflow under the `visual-defect-inspector` experiment.

---

## API Usage

### `GET /health`

```bash
curl https://codehashira73-visual-defect-inspector.hf.space/health
```

**Response:**
```json
{"status": "ok"}
```

---

### `POST /inspect`

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
| `anomaly_score` | float | Higher = more anomalous |
| `heatmap_base64` | string | Base64 JPEG — original image overlaid with anomaly heatmap (blue = normal, red = anomalous) |

To render the heatmap in Python:
```python
import base64
from PIL import Image
import io

heatmap_bytes = base64.b64decode(response["heatmap_base64"])
image = Image.open(io.BytesIO(heatmap_bytes))
image.show()
```

> **Note:** This model is trained on MVTec AD bottle images — white background, controlled lighting, standardized angles. Images from this distribution will give reliable results. Random internet images may return false positives due to distribution shift.

---

## Project Structure

```
visual-defect-inspector/
├── app/
│   ├── __init__.py
│   ├── main.py              # FastAPI app — routes, CORS, validation
│   └── inference.py         # Model loading (singleton) + predict logic
├── configs/
│   └── resnet18_bottle.yaml # Full experiment config — backbone, data, engine, MLflow
├── saved_model/
│   └── weights/
│       └── torch/
│           └── model.pt     # PatchCore model with memory bank (via Git LFS)
├── train.py                 # Config-driven training pipeline
├── evaluate.py              # Standalone evaluation on saved model
├── Anomaly_detection.ipynb  # Exploration, experimentation, MLflow logging
├── DECISIONS.md             # Technical decision log
├── Dockerfile
├── requirements.txt
└── .gitignore
```

---

## Local Setup

**Prerequisites:** Python 3.10+, Git LFS installed

```bash
# Clone the repo (Git LFS required to pull model weights)
git clone https://github.com/JeremiahAdebayo/visual-defect-inspector.git
cd visual-defect-inspector

# Install dependencies
pip install -r requirements.txt
```

### Train

```bash
python train.py --config configs/resnet18_bottle.yaml
```

This downloads the MVTec AD dataset automatically, trains PatchCore, logs metrics and config to MLflow, and saves the model + memory bank + metadata to `saved_model/`.

To run a different experiment, duplicate the config and modify it:

```bash
cp configs/resnet18_bottle.yaml configs/wide_resnet50_bottle.yaml
# edit the backbone field
python train.py --config configs/wide_resnet50_bottle.yaml
```

### Evaluate

```bash
python evaluate.py --config configs/resnet18_bottle.yaml
```

Loads the saved model and runs evaluation on the MVTec AD test set. Prints image AUROC, pixel AUROC, and F1.

### Run the API locally

```bash
uvicorn app.main:app --reload
```

API available at `http://127.0.0.1:8000/docs`

### Run with Docker

```bash
docker build -t visual-defect-inspector .
docker run -p 7860:7860 visual-defect-inspector
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
| Config management | PyYAML |

---

## Limitations

- Trained on a single MVTec AD category (bottle). Does not generalize to other object types without retraining.
- Sensitive to distribution shift — images must closely resemble the MVTec training distribution for reliable results.
- Anomaly threshold is fixed at training time. May need recalibration for different defect types in production.

---

## Author

**Jeremiah Adebayo**  
3rd Year Information Technology Student, University of Ilorin  
[GitHub](https://github.com/JeremiahAdebayo) · [Hugging Face](https://huggingface.co/CodeHashira73) · [LinkedIn](https://linkedin.com/in/jadebayo24)