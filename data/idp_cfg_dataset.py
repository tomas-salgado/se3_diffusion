import torch
from torch.utils.data import Dataset
import numpy as np
from typing import List, Dict, Optional
from data import utils as du

class IDPCFGDataset(Dataset):
    def __init__(
        self,
        p15_data_path: str,  # Path to p15 conformations
        ar_data_path: str,   # Path to AR conformations
        p15_embedding_path: str,  # Path to single p15 embedding txt file
        ar_embedding_path: str,   # Path to single AR embedding txt file
        cfg_dropout_prob: float = 0.1,
    ):
        """Dataset for classifier-free guidance training with IDP data.
        
        Args:
            p15_data_path: Path to p15 conformation data
            ar_data_path: Path to AR conformation data
            p15_embedding_path: Path to single p15 embedding txt file
            ar_embedding_path: Path to single AR embedding txt file
            cfg_dropout_prob: Probability of dropping condition during training
        """
        super().__init__()
        
        # Load conformations
        self.p15_data = du.load_structure_data(p15_data_path)
        self.ar_data = du.load_structure_data(ar_data_path)
        
        # Load single embeddings from txt files
        self.p15_embedding = self._load_single_embedding(p15_embedding_path)
        self.ar_embedding = self._load_single_embedding(ar_embedding_path)
        
        # Store parameters
        self.cfg_dropout_prob = cfg_dropout_prob
        
        # For unconditioned training, create length-matched indices
        self.p15_length = len(self.p15_data[0]['positions'])
        self.ar_length = len(self.ar_data[0]['positions'])

    def _load_single_embedding(self, path: str) -> torch.Tensor:
        """Load a single embedding vector from a txt file.
        
        Expected format:
        - Single line of space-separated floats
        """
        with open(path, 'r') as f:
            # Read the single line and split into floats
            values = [float(x) for x in f.readline().strip().split()]
            return torch.tensor(values)

    def __len__(self):
        return len(self.p15_data) + len(self.ar_data)

    def __getitem__(self, idx):
        # Determine if this sample is from p15 or AR dataset
        is_p15 = idx < len(self.p15_data)
        
        if is_p15:
            data = self.p15_data[idx]
            embedding = self.p15_embedding  # Use the single p15 embedding
        else:
            adj_idx = idx - len(self.p15_data)
            data = self.ar_data[adj_idx]
            embedding = self.ar_embedding  # Use the single AR embedding

        # Apply CFG dropout during training
        if self.training and torch.rand(1) < self.cfg_dropout_prob:
            # For unconditioned samples, zero out the embedding
            embedding = torch.zeros_like(embedding)
            
            # Use pretrained-generated structures matching the length
            if is_p15:
                data = self._get_pretrained_structure(self.p15_length)
            else:
                data = self._get_pretrained_structure(self.ar_length)

        # Need to match the expected input format
        return {
            # Fields required by ScoreNetwork
            'res_mask': data['mask'],
            'seq_idx': data['seq_idx'],
            'fixed_mask': torch.zeros_like(data['mask']),  # No fixed residues for now
            'torsion_angles_sin_cos': data['torsion_angles'],
            'sc_ca_t': data['ca_positions'],
            'rigids': data['rigids'],
            
            # Additional fields for CFG
            'sequence_embedding': embedding,
            'is_p15': torch.tensor(is_p15),
            
            # Any other fields needed by the diffusion model
            **{k:v for k,v in data.items() if k not in ['mask', 'torsion_angles', 'ca_positions', 'rigids']}
        }

    def _get_pretrained_structure(self, length: int) -> Dict:
        """Get a pretrained-generated structure of specified length.
        This should return a structure from your pretrained model's outputs
        matching the desired length."""
        # TODO: Implement this based on your pretrained model's outputs
        pass 