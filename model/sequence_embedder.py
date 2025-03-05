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
        
        # Whether to adapt dimensions
        self.adapt_dimensions = self._embed_conf.get('adapt_dimensions', True)
        
        # Embedding projection layers
        self.projection = nn.Sequential(
            nn.Linear(self.embed_dim, self._model_conf.node_embed_size),
            nn.ReLU(),
            nn.Linear(self._model_conf.node_embed_size, self._model_conf.node_embed_size),
            nn.LayerNorm(self._model_conf.node_embed_size),
        )
        
        # Load embedding from text file
        self.embedding, actual_dim = self.load_embedding(self._embed_conf.embedding_path)
        
        # If the loaded embedding dimension doesn't match the expected dimension,
        # create a dimension adapter layer
        if actual_dim != self.embed_dim and self.adapt_dimensions:
            print(f"Creating dimension adapter: {actual_dim} -> {self.embed_dim}")
            self.dim_adapter = nn.Linear(actual_dim, self.embed_dim)
        else:
            if actual_dim != self.embed_dim and not self.adapt_dimensions:
                print(f"WARNING: Embedding dimensions don't match ({actual_dim} != {self.embed_dim}) but adapt_dimensions=False")
            self.dim_adapter = None
    
    def load_embedding(self, path: str) -> tuple:
        """Load embedding from text file.
        
        Returns:
            Tuple of (embedding tensor, actual dimension)
        """
        try:
            # Read the text file
            with open(path, 'r') as f:
                # Read the line and split by comma
                embedding_str = f.read().strip()
                
                # Remove square brackets if present
                if embedding_str.startswith('[') and embedding_str.endswith(']'):
                    embedding_str = embedding_str[1:-1]
                
                # Clean the string (remove newlines, extra spaces)
                embedding_str = embedding_str.replace('\n', '').replace(' ', '')
                
                embedding_values = [float(x) for x in embedding_str.split(',') if x.strip()]
            
            # Convert to tensor
            embedding = torch.tensor(embedding_values, dtype=torch.float32)
            actual_dim = embedding.shape[0]
            
            # Log the actual dimension
            print(f"Loaded embedding from {path} with shape {embedding.shape}")
            
            # Note: We're not checking against self.embed_dim here, as we'll handle
            # dimension mismatches with a projection layer
            
            return embedding, actual_dim
            
        except Exception as e:
            print(f"Error loading embedding from {path}: {e}")
            # Return zero tensor with the expected dimension
            return torch.zeros(self.embed_dim, dtype=torch.float32), self.embed_dim
    
    def forward(self, sequences, device: torch.device) -> torch.Tensor:
        """Get embedding for a batch of sequences and project them.
        
        Args:
            sequences: Either a list of embeddings or a tensor of shape [batch_size, embed_dim]
            device: The device to place tensors on
            
        Returns:
            torch.Tensor: Projected embeddings of shape [batch_size, node_embed_size]
        """
        # Handle both list input and tensor input
        if isinstance(sequences, list):
            batch_size = len(sequences)
            # Expand single embedding to batch size
            embeddings = self.embedding.unsqueeze(0).expand(batch_size, -1).to(device)
        else:
            # If we received a tensor, use it directly
            embeddings = sequences.to(device)
        
        # Apply dimension adapter if needed
        if self.dim_adapter is not None:
            embeddings = self.dim_adapter(embeddings)
            
        # Project embeddings to the desired dimension
        projected_embeddings = self.projection(embeddings)
        
        return projected_embeddings