import io, os
from contextlib import asynccontextmanager

import torch
from fastapi import FastAPI, File, HTTPException, UploadFile
from PIL import Image
from torch.ao.quantization import quantize_dynamic
from transformers import ViTConfig, ViTForImageClassification, ViTImageProcessor

MODEL_DIR = os.getenv("MODEL_DIR", "model")
torch.set_num_threads(int(os.getenv("TORCH_THREADS", "2")))
state = {}

@asynccontextmanager
async def lifespan(app: FastAPI):
    cfg_dir = f"{MODEL_DIR}/merged_fp32"
    model = ViTForImageClassification(ViTConfig.from_pretrained(cfg_dir)).eval()
    model = quantize_dynamic(model, {torch.nn.Linear}, dtype=torch.qint8)  # rebuild INT8 structure
    sd = torch.load(f"{MODEL_DIR}/vit_lora_int8.pt", map_location="cpu", weights_only=False)  # our own file
    model.load_state_dict(sd)
    state["model"] = model.eval()
    state["processor"] = ViTImageProcessor.from_pretrained(cfg_dir)
    yield
    state.clear()

app = FastAPI(title="ViT + LoRA (INT8) CIFAR-100 classifier", lifespan=lifespan)

@app.get("/health")
def health():
    return {"status": "ok", "model_loaded": "model" in state}

@app.post("/predict")
async def predict(file: UploadFile = File(...), top_k: int = 3):
    try:
        img = Image.open(io.BytesIO(await file.read())).convert("RGB")
    except Exception:
        raise HTTPException(status_code=400, detail="Uploaded file is not a valid image")

    x = state["processor"](img, return_tensors="pt")["pixel_values"]
    with torch.inference_mode():
        probs = state["model"](pixel_values=x).logits.softmax(-1)[0]

    top = probs.topk(min(top_k, probs.numel()))
    id2label = state["model"].config.id2label
    return {"predictions": [
        {"label": id2label[i.item()], "probability": round(p.item(), 4)}
        for p, i in zip(top.values, top.indices)
    ]}