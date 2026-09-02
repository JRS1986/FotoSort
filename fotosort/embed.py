"""CLIP image embeddings, LAION aesthetic score, zero-shot subject labels."""
from __future__ import annotations

import os
import urllib.request
from pathlib import Path

import numpy as np
import torch
from PIL import Image

AESTHETIC_URL = (
    "https://github.com/christophschuhmann/improved-aesthetic-predictor/raw/main/"
    "sac+logos+ava1-l14-linearMSE.pth"
)
CACHE_DIR = Path(os.environ.get("FOTOSORT_CACHE", Path.home() / ".cache" / "fotosort"))


def pick_device() -> str:
    if torch.backends.mps.is_available():
        return "mps"
    if torch.cuda.is_available():
        return "cuda"
    return "cpu"


class AestheticHead(torch.nn.Module):
    """LAION 'improved aesthetic predictor' MLP on top of CLIP ViT-L/14 embeddings.
    Outputs roughly 1 (ugly) .. 10 (beautiful)."""

    def __init__(self):
        super().__init__()
        self.layers = torch.nn.Sequential(
            torch.nn.Linear(768, 1024), torch.nn.Dropout(0.2),
            torch.nn.Linear(1024, 128), torch.nn.Dropout(0.2),
            torch.nn.Linear(128, 64), torch.nn.Dropout(0.1),
            torch.nn.Linear(64, 16),
            torch.nn.Linear(16, 1),
        )

    def forward(self, x):
        return self.layers(x)


def _aesthetic_weights() -> Path:
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    dst = CACHE_DIR / "aesthetic_l14.pth"
    if not dst.exists():
        print(f"Downloading aesthetic predictor weights to {dst} ...")
        urllib.request.urlretrieve(AESTHETIC_URL, dst)
    return dst


class Embedder:
    def __init__(self, device: str | None = None):
        import open_clip  # slow import, keep local

        self.device = device or pick_device()
        self.model, _, self.preprocess = open_clip.create_model_and_transforms(
            "ViT-L-14-quickgelu", pretrained="openai"
        )
        self.model = self.model.to(self.device).eval()
        self.tokenizer = open_clip.get_tokenizer("ViT-L-14-quickgelu")
        self.head = AestheticHead()
        self.head.load_state_dict(torch.load(_aesthetic_weights(), map_location="cpu"))
        self.head = self.head.to(self.device).eval()

    def prepare(self, img: Image.Image) -> torch.Tensor:
        return self.preprocess(img)

    @torch.no_grad()
    def embed_batch(self, batch: list[torch.Tensor]) -> np.ndarray:
        x = torch.stack(batch).to(self.device)
        f = self.model.encode_image(x).float()
        f = f / f.norm(dim=-1, keepdim=True)
        return f.cpu().numpy().astype(np.float32)

    @torch.no_grad()
    def aesthetic(self, emb: np.ndarray) -> np.ndarray:
        x = torch.from_numpy(emb).to(self.device)
        return self.head(x).squeeze(-1).cpu().numpy().astype(np.float32)

    @torch.no_grad()
    def text_embeddings(self, labels: list[str]) -> np.ndarray:
        tokens = self.tokenizer([f"a photo of a {l}" for l in labels]).to(self.device)
        t = self.model.encode_text(tokens).float()
        t = t / t.norm(dim=-1, keepdim=True)
        return t.cpu().numpy().astype(np.float32)


def classify(emb: np.ndarray, text_emb: np.ndarray, labels: list[str]) -> tuple[list[str], np.ndarray]:
    """Zero-shot: softmax over cosine similarities (CLIP logit scale 100)."""
    logits = 100.0 * emb @ text_emb.T
    logits -= logits.max(axis=1, keepdims=True)
    p = np.exp(logits)
    p /= p.sum(axis=1, keepdims=True)
    idx = p.argmax(axis=1)
    return [labels[i] for i in idx], p[np.arange(len(idx)), idx]
