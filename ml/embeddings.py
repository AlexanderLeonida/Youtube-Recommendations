"""
TwinTube Vector: Embedding Generation and Management.

This module handles:
- Text embedding generation using sentence transformers
- Video embedding batch computation
- Embedding storage and retrieval
- FAISS index management for top-k retrieval
"""

import os
import json
import logging
from typing import Dict, List, Optional, Tuple, Union
from pathlib import Path
import hashlib

import torch
import torch.nn.functional as F
import numpy as np

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Optional imports
try:
    from sentence_transformers import SentenceTransformer
    SENTENCE_TRANSFORMER_AVAILABLE = True
except ImportError:
    SENTENCE_TRANSFORMER_AVAILABLE = False
    logger.warning("sentence-transformers not installed. Using random embeddings.")

try:
    import faiss
    FAISS_AVAILABLE = True
except ImportError:
    FAISS_AVAILABLE = False
    logger.warning("FAISS not installed. Using brute-force search.")


class TextEmbedder:
    """
    Generate text embeddings for video titles and descriptions.
    
    Uses sentence-transformers for high-quality semantic embeddings.
    Supports batch processing and caching.
    """
    
    def __init__(
        self,
        model_name: str = 'all-MiniLM-L6-v2',
        embedding_dim: int = 384,
        device: str = None,
        cache_dir: str = './cache/embeddings'
    ):
        self.embedding_dim = embedding_dim
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        
        if device is None:
            self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        else:
            self.device = device
        
        # Load sentence transformer
        if SENTENCE_TRANSFORMER_AVAILABLE:
            self.model = SentenceTransformer(model_name, device=self.device)
            self.embedding_dim = self.model.get_sentence_embedding_dimension()
            logger.info(f"Loaded SentenceTransformer: {model_name} (dim={self.embedding_dim})")
        else:
            self.model = None
            logger.warning("Using random embeddings (install sentence-transformers for real embeddings)")
    
    def _get_cache_key(self, text: str) -> str:
        """Generate cache key for text."""
        return hashlib.md5(text.encode()).hexdigest()
    
    def _get_cached(self, text: str) -> Optional[np.ndarray]:
        """Get embedding from cache if available."""
        cache_key = self._get_cache_key(text)
        cache_path = self.cache_dir / f"{cache_key}.npy"
        
        if cache_path.exists():
            return np.load(cache_path)
        return None
    
    def _save_to_cache(self, text: str, embedding: np.ndarray):
        """Save embedding to cache."""
        cache_key = self._get_cache_key(text)
        cache_path = self.cache_dir / f"{cache_key}.npy"
        np.save(cache_path, embedding)
    
    def embed(self, text: str, use_cache: bool = True) -> np.ndarray:
        """
        Generate embedding for a single text.
        
        Args:
            text: Input text (title, description, etc.)
            use_cache: Whether to use disk cache
            
        Returns:
            Embedding vector (embedding_dim,)
        """
        if use_cache:
            cached = self._get_cached(text)
            if cached is not None:
                return cached
        
        if self.model is not None:
            embedding = self.model.encode(text, convert_to_numpy=True)
        else:
            # Random embedding fallback
            embedding = np.random.randn(self.embedding_dim).astype(np.float32)
            embedding = embedding / np.linalg.norm(embedding)
        
        if use_cache:
            self._save_to_cache(text, embedding)
        
        return embedding
    
    def embed_batch(
        self,
        texts: List[str],
        batch_size: int = 64,
        use_cache: bool = True,
        show_progress: bool = True
    ) -> np.ndarray:
        """
        Generate embeddings for a batch of texts.
        
        Args:
            texts: List of input texts
            batch_size: Batch size for encoding
            use_cache: Whether to use disk cache
            show_progress: Show progress bar
            
        Returns:
            Embedding matrix (num_texts, embedding_dim)
        """
        embeddings = []
        uncached_texts = []
        uncached_indices = []
        
        # Check cache first
        for i, text in enumerate(texts):
            if use_cache:
                cached = self._get_cached(text)
                if cached is not None:
                    embeddings.append((i, cached))
                    continue
            uncached_texts.append(text)
            uncached_indices.append(i)
        
        # Encode uncached texts
        if uncached_texts:
            if self.model is not None:
                new_embeddings = self.model.encode(
                    uncached_texts,
                    batch_size=batch_size,
                    show_progress_bar=show_progress,
                    convert_to_numpy=True
                )
            else:
                new_embeddings = np.random.randn(len(uncached_texts), self.embedding_dim).astype(np.float32)
                new_embeddings = new_embeddings / np.linalg.norm(new_embeddings, axis=1, keepdims=True)
            
            for idx, text, emb in zip(uncached_indices, uncached_texts, new_embeddings):
                embeddings.append((idx, emb))
                if use_cache:
                    self._save_to_cache(text, emb)
        
        # Sort by original index and stack
        embeddings.sort(key=lambda x: x[0])
        return np.stack([e[1] for e in embeddings])


class VideoEmbeddingIndex:
    """
    Manages video embeddings and FAISS index for efficient retrieval.
    
    Supports:
    - Building and updating index
    - Approximate nearest neighbor search
    - GPU acceleration
    """
    
    def __init__(
        self,
        embedding_dim: int = 256,
        index_type: str = 'IVF',
        nlist: int = 1024,
        nprobe: int = 64,
        use_gpu: bool = True
    ):
        self.embedding_dim = embedding_dim
        self.index_type = index_type
        self.nlist = nlist
        self.nprobe = nprobe
        self.use_gpu = use_gpu and torch.cuda.is_available()
        
        self.index = None
        self.video_ids = None
        self.video_metadata = {}
        self.gpu_resources = None
    
    def build(self, embeddings: np.ndarray, video_ids: np.ndarray, metadata: Optional[List[Dict]] = None):
        """
        Build FAISS index from embeddings.
        
        Args:
            embeddings: (num_videos, dim) Video embedding matrix
            video_ids: (num_videos,) Video ID array
            metadata: Optional list of video metadata dicts
        """
        num_videos, dim = embeddings.shape
        embeddings = embeddings.astype(np.float32)
        
        # Normalize for cosine similarity
        norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
        embeddings = embeddings / np.maximum(norms, 1e-8)
        
        logger.info(f"Building index for {num_videos:,} videos (dim={dim})")
        
        if not FAISS_AVAILABLE:
            logger.warning("FAISS not available, using numpy storage")
            self.embeddings = embeddings
            self.video_ids = video_ids
            if metadata:
                for i, meta in enumerate(metadata):
                    self.video_metadata[video_ids[i]] = meta
            return
        
        # Create index
        if self.index_type == 'Flat' or num_videos < 10000:
            self.index = faiss.IndexFlatIP(dim)
        else:
            # IVF index for larger datasets
            nlist = min(self.nlist, int(np.sqrt(num_videos)))
            quantizer = faiss.IndexFlatIP(dim)
            self.index = faiss.IndexIVFFlat(quantizer, dim, nlist, faiss.METRIC_INNER_PRODUCT)
            
            # Train on subset
            train_size = min(num_videos, 50000)
            train_indices = np.random.choice(num_videos, train_size, replace=False)
            self.index.train(embeddings[train_indices])
        
        # Add vectors
        self.index.add(embeddings)
        
        # Move to GPU if available
        if self.use_gpu:
            try:
                self.gpu_resources = faiss.StandardGpuResources()
                self.index = faiss.index_cpu_to_gpu(self.gpu_resources, 0, self.index)
                logger.info("Index moved to GPU")
            except Exception as e:
                logger.warning(f"GPU index failed: {e}")
        
        # Set search parameters
        if hasattr(self.index, 'nprobe'):
            self.index.nprobe = self.nprobe
        
        self.video_ids = video_ids
        
        # Store metadata
        if metadata:
            for i, meta in enumerate(metadata):
                self.video_metadata[video_ids[i]] = meta
        
        logger.info(f"Index built successfully")
    
    def search(
        self,
        query_embeddings: np.ndarray,
        top_k: int = 100,
        return_metadata: bool = False
    ) -> Union[Tuple[np.ndarray, np.ndarray], Tuple[np.ndarray, np.ndarray, List[Dict]]]:
        """
        Search for top-k nearest neighbors.
        
        Args:
            query_embeddings: (batch, dim) Query vectors
            top_k: Number of results per query
            return_metadata: Whether to return video metadata
            
        Returns:
            video_ids: (batch, top_k) Retrieved video IDs
            scores: (batch, top_k) Similarity scores
            metadata: (optional) List of metadata dicts
        """
        query_embeddings = query_embeddings.astype(np.float32)
        
        # Normalize
        norms = np.linalg.norm(query_embeddings, axis=1, keepdims=True)
        query_embeddings = query_embeddings / np.maximum(norms, 1e-8)
        
        if not FAISS_AVAILABLE:
            # Brute-force fallback
            similarities = np.dot(query_embeddings, self.embeddings.T)
            top_k_indices = np.argsort(-similarities, axis=1)[:, :top_k]
            scores = np.take_along_axis(similarities, top_k_indices, axis=1)
            video_ids = self.video_ids[top_k_indices]
        else:
            scores, indices = self.index.search(query_embeddings, top_k)
            video_ids = self.video_ids[indices]
        
        if return_metadata:
            metadata = []
            for batch_ids in video_ids:
                batch_meta = [self.video_metadata.get(vid, {}) for vid in batch_ids]
                metadata.append(batch_meta)
            return video_ids, scores, metadata
        
        return video_ids, scores
    
    def save(self, path: str):
        """Save index to disk."""
        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)
        
        if FAISS_AVAILABLE and self.index is not None:
            # Move to CPU for saving
            if self.gpu_resources:
                cpu_index = faiss.index_gpu_to_cpu(self.index)
            else:
                cpu_index = self.index
            faiss.write_index(cpu_index, str(path / 'index.faiss'))
        else:
            np.save(path / 'embeddings.npy', self.embeddings)
        
        np.save(path / 'video_ids.npy', self.video_ids)
        
        with open(path / 'metadata.json', 'w') as f:
            # Convert int keys to strings for JSON
            json.dump({str(k): v for k, v in self.video_metadata.items()}, f)
        
        logger.info(f"Index saved to {path}")
    
    def load(self, path: str):
        """Load index from disk."""
        path = Path(path)
        
        if FAISS_AVAILABLE and (path / 'index.faiss').exists():
            self.index = faiss.read_index(str(path / 'index.faiss'))
            
            if self.use_gpu and torch.cuda.is_available():
                try:
                    self.gpu_resources = faiss.StandardGpuResources()
                    self.index = faiss.index_cpu_to_gpu(self.gpu_resources, 0, self.index)
                except Exception as e:
                    logger.warning(f"GPU index failed: {e}")
            
            if hasattr(self.index, 'nprobe'):
                self.index.nprobe = self.nprobe
        else:
            self.embeddings = np.load(path / 'embeddings.npy')
        
        self.video_ids = np.load(path / 'video_ids.npy')
        
        with open(path / 'metadata.json', 'r') as f:
            self.video_metadata = {int(k): v for k, v in json.load(f).items()}
        
        logger.info(f"Index loaded from {path}")
    
    def add_videos(self, embeddings: np.ndarray, video_ids: np.ndarray, metadata: Optional[List[Dict]] = None):
        """
        Add new videos to existing index.
        
        Args:
            embeddings: (num_new, dim) New video embeddings
            video_ids: (num_new,) New video IDs
            metadata: Optional metadata for new videos
        """
        embeddings = embeddings.astype(np.float32)
        norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
        embeddings = embeddings / np.maximum(norms, 1e-8)
        
        if FAISS_AVAILABLE and self.index is not None:
            self.index.add(embeddings)
        else:
            self.embeddings = np.vstack([self.embeddings, embeddings])
        
        self.video_ids = np.concatenate([self.video_ids, video_ids])
        
        if metadata:
            for i, meta in enumerate(metadata):
                self.video_metadata[video_ids[i]] = meta
        
        logger.info(f"Added {len(video_ids)} videos to index")


class EmbeddingPipeline:
    """
    End-to-end pipeline for generating and indexing video embeddings.
    
    Combines text embedding with model encoding for full video representations.
    """
    
    def __init__(
        self,
        model_path: str = None,
        embedding_dim: int = 256,
        device: str = None
    ):
        self.device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
        self.embedding_dim = embedding_dim
        
        # Initialize components
        self.text_embedder = TextEmbedder(device=self.device)
        self.index = VideoEmbeddingIndex(embedding_dim=embedding_dim)
        
        # Load model if provided
        if model_path and os.path.exists(model_path):
            self._load_model(model_path)
        else:
            self.model = None
    
    def _load_model(self, model_path: str):
        """Load TwinTube model for video encoding."""
        from model import create_model
        
        checkpoint = torch.load(model_path, map_location=self.device)
        config = checkpoint.get('config', {})
        
        self.model = create_model(
            num_videos=config.get('num_videos', 1_000_000),
            num_channels=config.get('num_channels', 100_000),
            num_categories=config.get('num_categories', 50),
            output_dim=self.embedding_dim
        )
        
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.model.to(self.device)
        self.model.eval()
        
        logger.info(f"Model loaded from {model_path}")
    
    def process_videos(
        self,
        videos: List[Dict],
        batch_size: int = 256,
        save_path: str = None
    ) -> VideoEmbeddingIndex:
        """
        Process videos and build embedding index.
        
        Args:
            videos: List of video dicts with 'id', 'title', 'channel_id', 'category_id', etc.
            batch_size: Processing batch size
            save_path: Optional path to save index
            
        Returns:
            Built VideoEmbeddingIndex
        """
        logger.info(f"Processing {len(videos)} videos...")
        
        # Generate title embeddings
        titles = [v.get('title', '') for v in videos]
        title_embeddings = self.text_embedder.embed_batch(titles, batch_size=batch_size)
        
        # Generate video embeddings
        if self.model is not None:
            embeddings = self._encode_videos(videos, title_embeddings, batch_size)
        else:
            # Use title embeddings directly (with projection)
            embeddings = self._project_embeddings(title_embeddings)
        
        # Build index
        video_ids = np.array([v['id'] for v in videos])
        self.index.build(embeddings, video_ids, videos)
        
        if save_path:
            self.index.save(save_path)
        
        return self.index
    
    def _encode_videos(
        self,
        videos: List[Dict],
        title_embeddings: np.ndarray,
        batch_size: int
    ) -> np.ndarray:
        """Encode videos using the model."""
        all_embeddings = []
        
        with torch.no_grad():
            for i in range(0, len(videos), batch_size):
                batch = videos[i:i + batch_size]
                batch_titles = title_embeddings[i:i + batch_size]
                
                video_ids = torch.tensor([v['id'] for v in batch], device=self.device)
                channel_ids = torch.tensor([v.get('channel_id', 1) for v in batch], device=self.device)
                category_ids = torch.tensor([v.get('category_id', 1) for v in batch], device=self.device)
                title_emb = torch.tensor(batch_titles, dtype=torch.float32, device=self.device)
                
                numerical = torch.tensor([
                    [
                        v.get('duration', 0),
                        v.get('views', 0),
                        v.get('likes', 0),
                        v.get('comments', 0),
                        v.get('age_days', 0)
                    ] for v in batch
                ], dtype=torch.float32, device=self.device)
                
                embeddings = self.model.encode_video(
                    video_ids, channel_ids, category_ids,
                    title_emb, numerical
                )
                
                all_embeddings.append(embeddings.cpu().numpy())
        
        return np.vstack(all_embeddings)
    
    def _project_embeddings(self, title_embeddings: np.ndarray) -> np.ndarray:
        """Project title embeddings to target dimension."""
        if title_embeddings.shape[1] == self.embedding_dim:
            return title_embeddings
        
        # Simple linear projection
        projection = np.random.randn(title_embeddings.shape[1], self.embedding_dim).astype(np.float32)
        projection = projection / np.linalg.norm(projection, axis=0, keepdims=True)
        
        projected = np.dot(title_embeddings, projection)
        projected = projected / np.linalg.norm(projected, axis=1, keepdims=True)
        
        return projected


def main():
    """Test embedding pipeline."""
    # Create sample videos
    videos = [
        {'id': i, 'title': f'Sample Video {i}', 'channel_id': i % 100, 'category_id': i % 20}
        for i in range(1000)
    ]
    
    # Process videos
    pipeline = EmbeddingPipeline(embedding_dim=256)
    index = pipeline.process_videos(videos, save_path='./index')
    
    # Test search
    query = pipeline.text_embedder.embed("machine learning tutorial")
    query = query.reshape(1, -1)
    
    # Project query to match index dimension
    if query.shape[1] != 256:
        projection = np.random.randn(query.shape[1], 256).astype(np.float32)
        query = np.dot(query, projection)
        query = query / np.linalg.norm(query)
    
    video_ids, scores = index.search(query, top_k=10)
    
    print("\nTop 10 results:")
    for i, (vid, score) in enumerate(zip(video_ids[0], scores[0])):
        print(f"  {i+1}. Video {vid} (score: {score:.4f})")


if __name__ == '__main__':
    main()
