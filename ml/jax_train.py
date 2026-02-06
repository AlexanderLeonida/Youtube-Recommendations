"""
TwinTube Vector: JAX/Flax Training Pipeline.

Implements a JAX-based training loop for the two-tower model with Optax.
This provides a production-ready alternative to the PyTorch trainer.
"""

import os
import time
import logging
from dataclasses import dataclass
from typing import Dict, List

import jax
import jax.numpy as jnp
import numpy as np
import optax
from flax.training import train_state

from jax_model import JaxModelConfig, create_jax_model
from train import YouTubeRecommendationDataset
from torch.utils.data import DataLoader


logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


@dataclass
class JaxTrainingConfig:
    num_videos: int = 1_000_000
    num_channels: int = 100_000
    num_categories: int = 50
    embedding_dim: int = 128
    hidden_dims: List[int] = None
    output_dim: int = 256
    temperature: float = 0.07
    dropout: float = 0.2
    num_attention_heads: int = 4
    max_history_len: int = 50
    title_embedding_dim: int = 384

    batch_size: int = 2048
    learning_rate: float = 1e-3
    weight_decay: float = 1e-5
    num_epochs: int = 10

    num_workers: int = 4

    def __post_init__(self):
        if self.hidden_dims is None:
            self.hidden_dims = [512, 256, 256]


def _to_jnp(batch: Dict[str, np.ndarray]) -> Dict[str, jnp.ndarray]:
    return {k: jnp.asarray(v) for k, v in batch.items()}


def _prepare_batch(batch: Dict[str, np.ndarray]) -> Dict[str, Dict[str, jnp.ndarray]]:
    user_features = {
        "watch_history": batch["watch_history"],
        "watch_times": batch["watch_times"],
        "engagement": batch["engagement"],
        "attention_mask": batch["attention_mask"],
    }

    video_features = {
        "video_ids": batch["target_video_id"],
        "channel_ids": batch["target_channel_id"],
        "category_ids": batch["target_category_id"],
        "title_embeddings": batch["target_title_embedding"],
        "numerical_features": batch["target_numerical"],
    }

    return {
        "user_features": _to_jnp(user_features),
        "video_features": _to_jnp(video_features),
    }


def _create_train_state(model, params, config: JaxTrainingConfig):
    tx = optax.adamw(
        learning_rate=config.learning_rate,
        weight_decay=config.weight_decay
    )
    return train_state.TrainState.create(apply_fn=model.apply, params=params, tx=tx)


def _loss_fn(params, apply_fn, batch, rng):
    outputs = apply_fn(
        params,
        batch["user_features"],
        batch["video_features"],
        deterministic=False,
        rngs={"dropout": rng}
    )
    logits = outputs["logits"]
    labels = jnp.arange(logits.shape[0])
    loss = optax.softmax_cross_entropy_with_integer_labels(logits, labels).mean()
    return loss


def _train_step(state, batch, rng):
    grad_fn = jax.value_and_grad(_loss_fn)
    loss, grads = grad_fn(state.params, state.apply_fn, batch, rng)
    state = state.apply_gradients(grads=grads)
    return state, loss


def train(config: JaxTrainingConfig):
    model_config = JaxModelConfig(
        num_videos=config.num_videos,
        num_channels=config.num_channels,
        num_categories=config.num_categories,
        embedding_dim=config.embedding_dim,
        hidden_dims=config.hidden_dims,
        output_dim=config.output_dim,
        temperature=config.temperature,
        dropout=config.dropout,
        num_attention_heads=config.num_attention_heads,
        max_history_len=config.max_history_len,
        title_embedding_dim=config.title_embedding_dim
    )

    model, params = create_jax_model(model_config)
    state = _create_train_state(model, params, config)

    train_dataset = YouTubeRecommendationDataset(
        data_path='data/train.json',
        max_history_len=config.max_history_len,
        is_training=True
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=config.batch_size,
        shuffle=True,
        num_workers=config.num_workers,
        pin_memory=False
    )

    rng = jax.random.PRNGKey(0)

    logger.info("Starting JAX training...")

    for epoch in range(1, config.num_epochs + 1):
        epoch_losses = []
        start_time = time.time()

        for batch in train_loader:
            batch_np = {k: v.numpy() for k, v in batch.items()}
            prepared = _prepare_batch(batch_np)
            rng, step_rng = jax.random.split(rng)
            state, loss = _train_step(state, prepared, step_rng)
            epoch_losses.append(float(loss))

        avg_loss = sum(epoch_losses) / max(1, len(epoch_losses))
        epoch_time = time.time() - start_time
        logger.info(f"Epoch {epoch} | Loss: {avg_loss:.4f} | Time: {epoch_time:.1f}s")

    logger.info("JAX training complete")


def main():
    config = JaxTrainingConfig(
        num_videos=1_000_000,
        num_channels=100_000,
        num_categories=50,
        batch_size=2048,
        learning_rate=1e-3,
        num_epochs=int(os.getenv('JAX_EPOCHS', '10')),
    )

    train(config)


if __name__ == '__main__':
    main()
