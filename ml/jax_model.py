"""
TwinTube Vector: JAX/Flax Two-Tower Model.

Provides a production-ready JAX implementation of the two-tower architecture
with user and video encoders projecting into a shared embedding space.
"""

from dataclasses import dataclass
from typing import Dict, List, Optional

import jax
import jax.numpy as jnp
import flax.linen as nn


@dataclass
class JaxModelConfig:
    num_videos: int
    num_channels: int
    num_categories: int
    embedding_dim: int = 128
    hidden_dims: List[int] = None
    output_dim: int = 256
    temperature: float = 0.07
    dropout: float = 0.2
    num_attention_heads: int = 4
    max_history_len: int = 50
    title_embedding_dim: int = 384

    def __post_init__(self):
        if self.hidden_dims is None:
            self.hidden_dims = [512, 256, 256]


class JaxUserTower(nn.Module):
    num_videos: int
    embedding_dim: int
    hidden_dims: List[int]
    output_dim: int
    max_history_len: int
    dropout: float
    num_attention_heads: int

    @nn.compact
    def __call__(
        self,
        watch_history: jnp.ndarray,
        watch_times: jnp.ndarray,
        engagement: jnp.ndarray,
        attention_mask: Optional[jnp.ndarray] = None,
        *,
        deterministic: bool = True
    ) -> jnp.ndarray:
        batch_size, seq_len = watch_history.shape

        video_embedding = nn.Embed(self.num_videos + 1, self.embedding_dim, name="video_embedding")
        position_embedding = nn.Embed(self.max_history_len, self.embedding_dim, name="position_embedding")

        video_embeds = video_embedding(watch_history)
        positions = jnp.arange(seq_len)[None, :]
        pos_embeds = position_embedding(positions)
        video_embeds = video_embeds + pos_embeds

        attention = nn.MultiHeadDotProductAttention(
            num_heads=self.num_attention_heads,
            dropout_rate=self.dropout,
            deterministic=deterministic
        )

        if attention_mask is not None:
            attn_mask = attention_mask.astype(bool)
            attn_mask = attn_mask[:, None, None, :]
        else:
            attn_mask = None

        attended = attention(video_embeds, video_embeds, mask=attn_mask)

        if attention_mask is not None:
            mask_expanded = attention_mask[..., None]
            sum_embeds = jnp.sum(attended * mask_expanded, axis=1)
            denom = jnp.maximum(jnp.sum(mask_expanded, axis=1), 1.0)
            mean_embeds = sum_embeds / denom
        else:
            mean_embeds = jnp.mean(attended, axis=1)

        watch_time_proj = nn.Dense(self.embedding_dim // 4)
        engagement_proj = nn.Dense(self.embedding_dim // 4)

        watch_time_feats = watch_time_proj(jnp.mean(watch_times, axis=1))
        engagement_feats = engagement_proj(jnp.mean(engagement, axis=1))

        fused = jnp.concatenate([mean_embeds, watch_time_feats, engagement_feats], axis=-1)

        x = fused
        for i, hidden_dim in enumerate(self.hidden_dims):
            x_residual = x
            x = nn.Dense(hidden_dim, name=f"user_dense_{i}")(x)
            x = nn.LayerNorm(name=f"user_ln_{i}")(x)
            x = nn.relu(x)
            x = nn.Dropout(rate=self.dropout)(x, deterministic=deterministic)
            if x_residual.shape[-1] == x.shape[-1]:
                x = x + x_residual

        x = nn.Dense(self.output_dim, name="user_output")(x)
        x = nn.LayerNorm(name="user_output_ln")(x)

        return x / (jnp.linalg.norm(x, axis=-1, keepdims=True) + 1e-8)


class JaxVideoTower(nn.Module):
    num_videos: int
    num_channels: int
    num_categories: int
    title_embedding_dim: int
    embedding_dim: int
    hidden_dims: List[int]
    output_dim: int
    dropout: float

    @nn.compact
    def __call__(
        self,
        video_ids: jnp.ndarray,
        channel_ids: jnp.ndarray,
        category_ids: jnp.ndarray,
        title_embeddings: jnp.ndarray,
        numerical_features: jnp.ndarray,
        *,
        deterministic: bool = True
    ) -> jnp.ndarray:
        video_embedding = nn.Embed(self.num_videos + 1, self.embedding_dim, name="video_embedding")
        channel_embedding = nn.Embed(self.num_channels + 1, self.embedding_dim // 2, name="channel_embedding")
        category_embedding = nn.Embed(self.num_categories + 1, self.embedding_dim // 4, name="category_embedding")

        video_embed = video_embedding(video_ids)
        channel_embed = channel_embedding(channel_ids)
        category_embed = category_embedding(category_ids)

        title_proj = nn.Dense(self.embedding_dim, name="title_proj")
        title_embed = title_proj(title_embeddings)
        title_embed = nn.LayerNorm(name="title_ln")(title_embed)
        title_embed = nn.relu(title_embed)
        title_embed = nn.Dropout(rate=self.dropout)(title_embed, deterministic=deterministic)

        numerical_features = jnp.log1p(jnp.maximum(numerical_features, 0.0))
        numerical_proj = nn.Dense(self.embedding_dim // 4, name="numerical_proj")
        numerical_embed = nn.relu(numerical_proj(numerical_features))

        fused = jnp.concatenate([
            video_embed,
            channel_embed,
            category_embed,
            title_embed,
            numerical_embed
        ], axis=-1)

        x = fused
        for i, hidden_dim in enumerate(self.hidden_dims):
            x_residual = x
            x = nn.Dense(hidden_dim, name=f"video_dense_{i}")(x)
            x = nn.LayerNorm(name=f"video_ln_{i}")(x)
            x = nn.relu(x)
            x = nn.Dropout(rate=self.dropout)(x, deterministic=deterministic)
            if x_residual.shape[-1] == x.shape[-1]:
                x = x + x_residual

        x = nn.Dense(self.output_dim, name="video_output")(x)
        x = nn.LayerNorm(name="video_output_ln")(x)

        return x / (jnp.linalg.norm(x, axis=-1, keepdims=True) + 1e-8)


class JaxTwinTubeModel(nn.Module):
    config: JaxModelConfig

    def setup(self):
        self.user_tower = JaxUserTower(
            num_videos=self.config.num_videos,
            embedding_dim=self.config.embedding_dim,
            hidden_dims=self.config.hidden_dims,
            output_dim=self.config.output_dim,
            max_history_len=self.config.max_history_len,
            dropout=self.config.dropout,
            num_attention_heads=self.config.num_attention_heads
        )
        self.video_tower = JaxVideoTower(
            num_videos=self.config.num_videos,
            num_channels=self.config.num_channels,
            num_categories=self.config.num_categories,
            title_embedding_dim=self.config.title_embedding_dim,
            embedding_dim=self.config.embedding_dim,
            hidden_dims=self.config.hidden_dims,
            output_dim=self.config.output_dim,
            dropout=self.config.dropout
        )

    def encode_user(self, user_features: Dict[str, jnp.ndarray], *, deterministic: bool = True) -> jnp.ndarray:
        return self.user_tower(
            user_features["watch_history"],
            user_features["watch_times"],
            user_features["engagement"],
            user_features.get("attention_mask"),
            deterministic=deterministic
        )

    def encode_video(self, video_features: Dict[str, jnp.ndarray], *, deterministic: bool = True) -> jnp.ndarray:
        return self.video_tower(
            video_features["video_ids"],
            video_features["channel_ids"],
            video_features["category_ids"],
            video_features["title_embeddings"],
            video_features["numerical_features"],
            deterministic=deterministic
        )

    def __call__(
        self,
        user_features: Dict[str, jnp.ndarray],
        positive_video_features: Dict[str, jnp.ndarray],
        *,
        deterministic: bool = True
    ) -> Dict[str, jnp.ndarray]:
        user_embedding = self.encode_user(user_features, deterministic=deterministic)
        positive_embedding = self.encode_video(positive_video_features, deterministic=deterministic)

        logits = jnp.matmul(user_embedding, positive_embedding.T) / self.config.temperature
        labels = jnp.arange(logits.shape[0])

        return {
            "user_embedding": user_embedding,
            "positive_embedding": positive_embedding,
            "logits": logits,
            "labels": labels
        }


def create_jax_model(config: JaxModelConfig):
    """Create JAX model and initialize parameters with dummy inputs."""
    model = JaxTwinTubeModel(config=config)

    dummy_batch = 4
    rng = jax.random.PRNGKey(0)

    user_features = {
        "watch_history": jnp.zeros((dummy_batch, config.max_history_len), dtype=jnp.int32),
        "watch_times": jnp.zeros((dummy_batch, config.max_history_len, 1), dtype=jnp.float32),
        "engagement": jnp.zeros((dummy_batch, config.max_history_len, 3), dtype=jnp.float32),
        "attention_mask": jnp.ones((dummy_batch, config.max_history_len), dtype=jnp.float32)
    }

    video_features = {
        "video_ids": jnp.zeros((dummy_batch,), dtype=jnp.int32),
        "channel_ids": jnp.zeros((dummy_batch,), dtype=jnp.int32),
        "category_ids": jnp.zeros((dummy_batch,), dtype=jnp.int32),
        "title_embeddings": jnp.zeros((dummy_batch, config.title_embedding_dim), dtype=jnp.float32),
        "numerical_features": jnp.zeros((dummy_batch, 5), dtype=jnp.float32)
    }

    params = model.init(rng, user_features, video_features, deterministic=True)

    return model, params
