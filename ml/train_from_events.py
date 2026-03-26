"""
Train the TwinTube model from live browse_events collected by the Chrome extension.

Uses CTR (click-through rate) as the primary training signal:
- Impressions (videos shown to the user) that were clicked = positive (label=1)
- Impressions that were NOT clicked = negative (label=0)
- Trains with binary cross-entropy on user-video dot-product similarity

This produces a model that predicts the probability a user will click a video,
which is then used for ranking recommendations.
"""

import json
import logging
import os
import time
from collections import defaultdict
from typing import Dict, List

import numpy as np
import requests
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

from model import create_model

logger = logging.getLogger(__name__)

BACKEND_URL = os.getenv("BACKEND_URL", "http://backend:4000")


class IDMapper:
    """Maps YouTube string IDs to contiguous integers for embedding layers."""

    def __init__(self):
        self.video_to_int: Dict[str, int] = {}
        self.int_to_video: Dict[int, str] = {}
        self.channel_to_int: Dict[str, int] = {}
        self.int_to_channel: Dict[int, str] = {}
        self._next_video = 1   # 0 = padding
        self._next_channel = 1

    def map_video(self, yt_id: str) -> int:
        if yt_id not in self.video_to_int:
            self.video_to_int[yt_id] = self._next_video
            self.int_to_video[self._next_video] = yt_id
            self._next_video += 1
        return self.video_to_int[yt_id]

    def map_channel(self, name: str) -> int:
        if not name:
            return 0
        if name not in self.channel_to_int:
            self.channel_to_int[name] = self._next_channel
            self.int_to_channel[self._next_channel] = name
            self._next_channel += 1
        return self.channel_to_int[name]

    @property
    def num_videos(self) -> int:
        return self._next_video

    @property
    def num_channels(self) -> int:
        return self._next_channel

    def save(self, path: str):
        with open(path, "w") as f:
            json.dump({
                "video_to_int": self.video_to_int,
                "int_to_video": {str(k): v for k, v in self.int_to_video.items()},
                "channel_to_int": self.channel_to_int,
                "int_to_channel": {str(k): v for k, v in self.int_to_channel.items()},
            }, f)

    def load(self, path: str):
        with open(path) as f:
            data = json.load(f)
        self.video_to_int = data["video_to_int"]
        self.int_to_video = {int(k): v for k, v in data["int_to_video"].items()}
        self.channel_to_int = data["channel_to_int"]
        self.int_to_channel = {int(k): v for k, v in data.get("int_to_channel", {}).items()}
        self._next_video = max(self.int_to_video.keys(), default=0) + 1
        self._next_channel = max(self.channel_to_int.values(), default=0) + 1


def _parse_views(v: str) -> float:
    if not v:
        return 0.0
    v = v.lower().replace(",", "").replace("views", "").strip()
    mul = 1.0
    if v.endswith("k"):
        mul, v = 1e3, v[:-1]
    elif v.endswith("m"):
        mul, v = 1e6, v[:-1]
    elif v.endswith("b"):
        mul, v = 1e9, v[:-1]
    try:
        return float(v) * mul
    except ValueError:
        return 0.0


def _parse_duration_sec(d: str) -> float:
    if not d:
        return 0.0
    parts = d.strip().split(":")
    try:
        p = [int(x) for x in parts]
        if len(p) == 3:
            return p[0] * 3600 + p[1] * 60 + p[2]
        if len(p) == 2:
            return p[0] * 60 + p[1]
        return float(p[0])
    except (ValueError, IndexError):
        return 0.0


class CTRDataset(Dataset):
    """
    CTR-based training dataset from Chrome extension browse events.

    Each sample = (user_watch_history, candidate_video, label)
    - label=1.0 for clicked videos (positive)
    - label=0.0 for impressed-but-not-clicked videos (negative)

    This directly models P(click | user, video) for CTR prediction.
    """

    def __init__(self, sessions: List[Dict], id_mapper: IDMapper, max_history: int = 50):
        self.samples: List[Dict] = []
        self.id_mapper = id_mapper
        self.max_history = max_history
        self.ctr_stats: Dict = {}
        self._build(sessions)

    def _build(self, sessions: List[Dict]):
        # Track global CTR stats per video
        video_impressions: Dict[str, int] = defaultdict(int)
        video_clicks: Dict[str, int] = defaultdict(int)

        for sess in sessions:
            clicks = sess.get("clicks", [])
            impressions = sess.get("impressions", [])

            # Build click history as user context
            clicked_ids = set()
            history: List[int] = []
            click_meta: Dict[str, Dict] = {}

            for click in clicks:
                vid = click.get("video_id")
                if not vid:
                    continue
                clicked_ids.add(vid)
                int_vid = self.id_mapper.map_video(vid)
                self.id_mapper.map_channel(click.get("channel_name") or click.get("channel"))
                history.append(int_vid)
                click_meta[vid] = click
                video_clicks[vid] += 1

            # Count impressions
            for imp in impressions:
                vid = imp.get("video_id")
                if vid:
                    video_impressions[vid] += 1

            # Also count clicks as impressions (clicked implies seen)
            for vid in clicked_ids:
                video_impressions[vid] += 1

            if not history:
                continue

            # ── Positive samples: clicked videos ──
            for click in clicks:
                vid = click.get("video_id")
                if not vid:
                    continue
                self.samples.append(self._make_sample(
                    history=history,
                    event=click,
                    label=1.0,
                ))

            # ── Negative samples: impressed but not clicked ──
            for imp in impressions:
                vid = imp.get("video_id")
                if not vid or vid in clicked_ids:
                    continue
                self.id_mapper.map_video(vid)
                self.id_mapper.map_channel(imp.get("channel_name") or imp.get("channel"))
                self.samples.append(self._make_sample(
                    history=history,
                    event=imp,
                    label=0.0,
                ))

        # Compute CTR stats
        all_videos = set(video_impressions.keys())
        per_video_ctr = {}
        for vid in all_videos:
            imps = video_impressions[vid]
            clks = video_clicks.get(vid, 0)
            per_video_ctr[vid] = clks / imps if imps > 0 else 0.0

        total_impressions = sum(video_impressions.values())
        total_clicks = sum(video_clicks.values())
        self.ctr_stats = {
            "total_impressions": total_impressions,
            "total_clicks": total_clicks,
            "overall_ctr": total_clicks / total_impressions if total_impressions > 0 else 0.0,
            "unique_videos": len(all_videos),
            "per_video_ctr": per_video_ctr,
        }

        n_pos = sum(1 for s in self.samples if s["label"] > 0.5)
        n_neg = len(self.samples) - n_pos
        logger.info(
            f"Built {len(self.samples)} CTR samples "
            f"({n_pos} positive, {n_neg} negative) "
            f"from {len(sessions)} sessions. "
            f"Overall CTR: {self.ctr_stats['overall_ctr']:.2%}"
        )

    def _make_sample(self, history: List[int], event: Dict, label: float) -> Dict:
        vid = event.get("video_id", "")
        return {
            "history": list(history[-self.max_history:]),
            "target_video": self.id_mapper.map_video(vid),
            "target_channel": self.id_mapper.map_channel(
                event.get("channel_name") or event.get("channel")
            ),
            "target_numerical": [
                _parse_duration_sec(event.get("duration", "")),
                _parse_views(event.get("views", "")),
                0.0, 0.0, 0.0,
            ],
            "watch_dur": float(event.get("watch_duration_sec") or 0),
            "label": label,
        }

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        s = self.samples[idx]
        h = s["history"]
        pad = self.max_history - len(h)
        if pad > 0:
            mask = [1.0] * len(h) + [0.0] * pad
            h = h + [0] * pad
        else:
            h = h[-self.max_history:]
            mask = [1.0] * self.max_history

        return {
            "watch_history": torch.tensor(h, dtype=torch.long),
            "attention_mask": torch.tensor(mask, dtype=torch.float32),
            "watch_times": torch.full((self.max_history, 1), s["watch_dur"], dtype=torch.float32),
            "engagement": torch.zeros(self.max_history, 3),
            "target_video_id": torch.tensor(s["target_video"], dtype=torch.long),
            "target_channel_id": torch.tensor(s["target_channel"], dtype=torch.long),
            "target_numerical": torch.tensor(s["target_numerical"], dtype=torch.float32),
            "label": torch.tensor(s["label"], dtype=torch.float32),
        }


# ── Fetch data from backend ────────────────────────────────────────────────

def fetch_sessions(backend_url: str = BACKEND_URL) -> List[Dict]:
    """Fetch browse events and group into sessions with click sequences."""
    try:
        resp = requests.get(f"{backend_url}/api/events", params={"limit": 5000}, timeout=10)
        resp.raise_for_status()
        events = resp.json().get("events", [])
    except Exception as e:
        logger.error(f"Failed to fetch events from {backend_url}: {e}")
        return []

    sessions: Dict[str, Dict] = defaultdict(lambda: {"clicks": [], "impressions": []})
    for ev in events:
        sid = ev["session_id"]
        if ev["event_type"] == "click":
            sessions[sid]["clicks"].append(ev)
        elif ev["event_type"] == "impression":
            sessions[sid]["impressions"].append(ev)
        elif ev["event_type"] == "watch_end":
            for c in sessions[sid]["clicks"]:
                if c["video_id"] == ev["video_id"]:
                    c["watch_duration_sec"] = ev.get("watch_duration_sec", 0)

    return list(sessions.values())


# ── Training entry point ────────────────────────────────────────────────────

def train_from_events(
    epochs: int = 10,
    batch_size: int = 32,
    lr: float = 1e-3,
    checkpoint_dir: str = "./checkpoints",
    index_dir: str = "./index",
    backend_url: str = BACKEND_URL,
) -> Dict:
    """
    End-to-end CTR-based training from browse events.

    Uses impressions as negatives and clicks as positives to train
    a two-tower model that predicts P(click | user, video).

    Returns dict with training results, CTR statistics, and status.
    """
    os.makedirs(checkpoint_dir, exist_ok=True)
    os.makedirs(index_dir, exist_ok=True)
    device = torch.device("cpu")

    # 1. Fetch sessions
    logger.info("Fetching browse sessions...")
    sessions = fetch_sessions(backend_url)
    if not sessions:
        return {"status": "error", "message": "No browse data. Browse YouTube with the extension first."}

    # 2. Build CTR dataset
    id_mapper = IDMapper()

    mapper_path = os.path.join(checkpoint_dir, "id_mapper.json")
    if os.path.exists(mapper_path):
        id_mapper.load(mapper_path)
        logger.info(f"Loaded existing ID mapper ({id_mapper.num_videos - 1} videos)")

    dataset = CTRDataset(sessions, id_mapper)
    if len(dataset) < 2:
        return {
            "status": "error",
            "message": f"Not enough training data ({len(dataset)} samples). "
                       "Need more browsing sessions with impressions and clicks.",
        }

    loader = DataLoader(
        dataset,
        batch_size=min(batch_size, len(dataset)),
        shuffle=True,
        drop_last=True,
    )

    # 3. Create model — save config so inference can recreate the same architecture
    model_config = {
        "num_videos": id_mapper.num_videos + 1000,
        "num_channels": id_mapper.num_channels + 500,
        "num_categories": 50,
        "embedding_dim": 128,
        "output_dim": 256,
    }
    model = create_model(**model_config)

    # Load existing weights if retraining
    model_path = os.path.join(checkpoint_dir, "best_model.pt")
    if os.path.exists(model_path):
        try:
            ckpt = torch.load(model_path, map_location=device)
            model.load_state_dict(ckpt["model_state_dict"], strict=False)
            logger.info("Loaded existing model weights (fine-tuning)")
        except Exception as e:
            logger.warning(f"Could not load existing weights, training from scratch: {e}")

    model.to(device)
    model.train()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    # 4. Train with CTR loss (BCE on dot-product similarity)
    logger.info(f"Training CTR model: {len(dataset)} samples, {epochs} epochs, batch_size={loader.batch_size}")
    loss_history = []
    best_loss = float("inf")

    for epoch in range(epochs):
        epoch_loss = 0.0
        epoch_correct = 0
        epoch_total = 0
        n = 0

        for batch in loader:
            # Encode user
            user_emb = model.encode_user(
                batch["watch_history"].to(device),
                batch["watch_times"].to(device),
                batch["engagement"].to(device),
                batch["attention_mask"].to(device),
            )

            # Encode candidate video
            bs = batch["target_video_id"].shape[0]
            video_emb = model.encode_video(
                batch["target_video_id"].to(device),
                batch["target_channel_id"].to(device),
                torch.zeros(bs, dtype=torch.long, device=device),
                torch.randn(bs, 384, device=device) * 0.1,
                batch["target_numerical"].to(device),
            )

            # CTR prediction: dot product similarity -> sigmoid -> P(click)
            logits = (user_emb * video_emb).sum(dim=-1)  # (batch,)
            labels = batch["label"].to(device)            # (batch,)

            # Binary cross-entropy loss for CTR prediction
            loss = F.binary_cross_entropy_with_logits(logits, labels)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()
            # Track accuracy
            preds = (logits > 0).float()
            epoch_correct += (preds == labels).sum().item()
            epoch_total += bs
            n += 1

        avg_loss = epoch_loss / max(n, 1)
        accuracy = epoch_correct / max(epoch_total, 1)
        loss_history.append(avg_loss)
        logger.info(f"Epoch {epoch + 1}/{epochs}  loss={avg_loss:.4f}  accuracy={accuracy:.2%}")

        if avg_loss < best_loss:
            best_loss = avg_loss
            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "model_config": model_config,
                },
                model_path,
            )

    # 5. Save ID mapper and CTR stats
    id_mapper.save(mapper_path)

    ctr_stats_path = os.path.join(checkpoint_dir, "ctr_stats.json")
    with open(ctr_stats_path, "w") as f:
        json.dump({
            "overall_ctr": dataset.ctr_stats["overall_ctr"],
            "total_impressions": dataset.ctr_stats["total_impressions"],
            "total_clicks": dataset.ctr_stats["total_clicks"],
            "unique_videos": dataset.ctr_stats["unique_videos"],
            "per_video_ctr": dataset.ctr_stats["per_video_ctr"],
        }, f)
    logger.info(f"CTR stats saved: overall CTR = {dataset.ctr_stats['overall_ctr']:.2%}")

    # 6. Build FAISS index from video embeddings
    logger.info("Building FAISS index...")
    model.eval()
    embs, ids = [], []

    with torch.no_grad():
        all_ints = list(range(1, id_mapper.num_videos))
        for i in range(0, len(all_ints), 128):
            batch_ints = all_ints[i:i + 128]
            vid_t = torch.tensor(batch_ints, device=device)
            ch_t = torch.zeros_like(vid_t)
            cat_t = torch.zeros_like(vid_t)
            title_t = torch.randn(len(batch_ints), 384, device=device) * 0.1
            num_t = torch.zeros(len(batch_ints), 5, device=device)
            e = model.encode_video(vid_t, ch_t, cat_t, title_t, num_t)
            embs.append(e.cpu().numpy())
            ids.extend(batch_ints)

    if embs:
        embs_np = np.vstack(embs)
        ids_np = np.array(ids)

        from inference import GPUVectorIndex, InferenceConfig
        config = InferenceConfig(use_gpu_index=False)
        idx = GPUVectorIndex(config)
        idx.build_index(embs_np, ids_np)
        idx.save(os.path.join(index_dir, "video_embeddings.faiss"))
        logger.info(f"Index built with {len(ids)} vectors")

    return {
        "status": "success",
        "epochs": epochs,
        "final_loss": loss_history[-1] if loss_history else None,
        "training_samples": len(dataset),
        "videos_indexed": id_mapper.num_videos - 1,
        "loss_history": loss_history,
        "ctr_stats": {
            "overall_ctr": dataset.ctr_stats["overall_ctr"],
            "total_impressions": dataset.ctr_stats["total_impressions"],
            "total_clicks": dataset.ctr_stats["total_clicks"],
            "unique_videos": dataset.ctr_stats["unique_videos"],
        },
    }
