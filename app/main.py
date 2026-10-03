import io, os, time, json
import psutil
import torch
import numpy as np
from prometheus_client import Counter, Histogram, generate_latest, CONTENT_TYPE_LATEST
from fastapi import Response
from contextlib import asynccontextmanager
from datetime import datetime, timezone
from fastapi import FastAPI, File, HTTPException, UploadFile, Form
from scipy.stats import ks_2samp
from PIL import Image
from torch.ao.quantization import quantize_dynamic
from transformers import ViTConfig, ViTForImageClassification, ViTImageProcessor

MODEL_DIR = os.getenv("MODEL_DIR", "model")
LOG_PATH = os.getenv("LOG_PATH", "logs/predictions.jsonl")
os.makedirs(os.path.dirname(LOG_PATH), exist_ok=True)
torch.set_num_threads(int(os.getenv("TORCH_THREADS", "2")))
state = {}

REQUEST_COUNT = Counter("predict_requests_total", "Total number of /predict requests")
REQUEST_LATENCY = Histogram("predict_latency_seconds", "Latency of /predict requests in seconds")

@asynccontextmanager
async def lifespan(app: FastAPI):
    cfg_dir = f"{MODEL_DIR}/merged_fp32"
    model = ViTForImageClassification(ViTConfig.from_pretrained(cfg_dir)).eval()
    model = quantize_dynamic(model, {torch.nn.Linear}, dtype=torch.qint8)
    sd = torch.load(f"{MODEL_DIR}/vit_lora_int8.pt", map_location="cpu", weights_only=False)
    model.load_state_dict(sd)
    state["model"] = model.eval()
    state["processor"] = ViTImageProcessor.from_pretrained(cfg_dir)
    with open(f"{MODEL_DIR}/reference_stats.json") as f:
        state["reference"] = json.load(f)
    yield
    state.clear()

app = FastAPI(title="ViT LoRA CIFAR-100 Classifier v1 (INT8)", lifespan=lifespan)

def log_event(event: dict):
    event["timestamp"] = datetime.now(timezone.utc).isoformat()
    with open(LOG_PATH, "a") as f:
        f.write(json.dumps(event) + "\n")

def image_stats(img: Image.Image) -> dict:
    arr = np.asarray(img.convert("RGB"), dtype=np.float32) / 255.0
    return {
        "mean": round(float(arr.mean()), 4),
        "std": round(float(arr.std()), 4),
        "width": img.width,
        "height": img.height,
    }

@app.get("/health")
def health():
    return {"status": "ok", "model_loaded": "model" in state}

@app.get("/metrics")                   
def metrics():
    process = psutil.Process(os.getpid())
    mem_mb = process.memory_info().rss / (1024 * 1024)
    cpu_pct = psutil.cpu_percent(interval=0.1)

    custom = f"""# HELP app_cpu_percent CPU usage percent of the API process
# TYPE app_cpu_percent gauge
app_cpu_percent {cpu_pct}
# HELP app_memory_mb Memory usage in MB of the API process
# TYPE app_memory_mb gauge
app_memory_mb {mem_mb:.2f}
"""
    return Response(content=generate_latest().decode() + custom, media_type=CONTENT_TYPE_LATEST)

@app.get("/drift")
def drift_check(window: int = 50):
    if not os.path.exists(LOG_PATH):
        return {"status": "no_data", "detail": "No predictions logged yet"}

    recent_means = []
    with open(LOG_PATH) as f:
        lines = f.readlines()[-window:]
    for line in lines:
        rec = json.loads(line)
        recent_means.append(rec["image"]["mean"])

    if len(recent_means) < 10:
        return {"status": "insufficient_data", "n_recent": len(recent_means)}

    ref_means = state["reference"]["mean"]
    stat, p_value = ks_2samp(ref_means, recent_means)

    drift_detected = p_value < 0.05
    return {
        "status": "drift_detected" if drift_detected else "no_drift",
        "ks_statistic": round(float(stat), 4),
        "p_value": round(float(p_value), 4),
        "n_recent": len(recent_means),
        "n_reference": len(ref_means),
        "recent_mean_brightness": round(float(np.mean(recent_means)), 4),
        "reference_mean_brightness": round(float(np.mean(ref_means)), 4),
    }

@app.get("/performance")
def performance(window: int = 100):
    if not os.path.exists(LOG_PATH):
        return {"status": "no_data", "detail": "No predictions logged yet"}

    with open(LOG_PATH) as f:
        lines = f.readlines()[-window:]

    labeled = []
    for line in lines:
        rec = json.loads(line)
        if rec.get("true_label"):
            labeled.append(rec)

    if not labeled:
        return {"status": "no_labels", "detail": "No labeled predictions in this window", "n_total": len(lines)}

    correct = sum(1 for r in labeled if r["predicted_label"] == r["true_label"])
    accuracy = correct / len(labeled)

    return {
        "status": "ok",
        "n_total_recent": len(lines),
        "n_labeled": len(labeled),
        "correct": correct,
        "accuracy": round(accuracy, 4),
        "avg_confidence": round(sum(r["confidence"] for r in labeled) / len(labeled), 4),
    }

@app.post("/predict")
async def predict(file: UploadFile = File(...), top_k: int = 5, true_label: str = Form(default=None)):
    try:
        raw = await file.read()
        img = Image.open(io.BytesIO(raw)).convert("RGB")
    except Exception:
        raise HTTPException(status_code=400, detail="Uploaded file is not a valid image")

    t0 = time.time()
    REQUEST_COUNT.inc()
    t0 = time.time()
    x = state["processor"](img, return_tensors="pt")["pixel_values"]
    with torch.inference_mode():
        probs = state["model"](pixel_values=x).logits.softmax(-1)[0]
    elapsed = time.time() - t0
    REQUEST_LATENCY.observe(elapsed)
    latency_ms = elapsed * 1000
    latency_ms = (time.time() - t0) * 1000

    top = probs.topk(min(top_k, probs.numel()))
    id2label = state["model"].config.id2label
    predictions = [
        {"label": id2label[i.item()], "probability": round(p.item(), 4)}
        for p, i in zip(top.values, top.indices)
    ]

    log_event({
        "predicted_label": predictions[0]["label"],
        "confidence": predictions[0]["probability"],
        "true_label": true_label,
        "latency_ms": round(latency_ms, 2),
        "image": image_stats(img),
    })

    return {"predictions": predictions}