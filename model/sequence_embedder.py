"""Sequence embedding module for conditioning diffusion models."""
import torch
import torch.nn as nn
import numpy as np
from typing import Dict, Optional, Tuple

class SequenceEmbedder(nn.Module):
    """Module for generating or loading sequence embeddings for conditioning."""
    
    def __init__(self, model_conf):
        super(SequenceEmbedder, self).__init__()
        self._model_conf = model_conf
        self._embed_conf = model_conf.sequence_embed
        
        # Embedding dimension
        self.embed_dim = self._embed_conf.embed_dim
        
        # Embedding projection layers
        self.projection = nn.Sequential(
            nn.Linear(self.embed_dim, self._model_conf.node_embed_size),
            nn.ReLU(),
            nn.Linear(self._model_conf.node_embed_size, self._model_conf.node_embed_size),
            nn.LayerNorm(self._model_conf.node_embed_size),
        )
        
        # Load embedding from text file
        self.embedding = self.load_embedding(self._embed_conf.embedding_path)
    
    def load_embedding(self, path: str) -> torch.Tensor:
        """Load embedding from text file."""
        try:
            # Read the text file
            with open(path, 'r') as f:
                # Read the line and split by comma
                embedding_str = f.read().strip()
                embedding_values = [float(x) for x in embedding_str.split(',')]
            
            # Convert to tensor
            embedding = torch.tensor(embedding_values, dtype=torch.float32)
            
            # Verify dimension
            if embedding.shape[0] != self.embed_dim:
                raise ValueError(f"Embedding dimension mismatch. Expected {self.embed_dim}, got {embedding.shape[0]}")
            
            print(f"Loaded embedding from {path} with shape {embedding.shape}")
            return embedding
            
        except Exception as e:
            print(f"Error loading embedding from {path}: {e}")
            return torch.zeros(self.embed_dim, dtype=torch.float32)
    
    def forward(self, sequences: list, device: torch.device) -> torch.Tensor:
        """Get embedding for a batch of sequences and project them."""
        batch_size = len(sequences)
        
        # Expand single embedding to batch size
        embeddings = self.embedding.unsqueeze(0).expand(batch_size, -1).to(device)
        
        # Return raw embeddings without projection - this matches what Embedder expects
        return embeddings