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
        p15_embedding_path: str,  # Path to p15 embedding txt file
        ar_embedding_path: str,   # Path to AR embedding txt file
        pretrained_p15_path: Optional[str] = None,  # Path to pretrained p15-length structures
        pretrained_ar_path: Optional[str] = None,   # Path to pretrained ar-length structures
        cfg_dropout_prob: float = 0.1,
        is_training: bool = True,  # Whether this is for training or validation
    ):
        """Dataset for classifier-free guidance training with IDP data.
        
        Args:
            p15_data_path: Path to p15 conformation data
            ar_data_path: Path to AR conformation data
            p15_embedding_path: Path to single p15 embedding txt file
            ar_embedding_path: Path to single AR embedding txt file
            pretrained_p15_path: Path to pretrained structures matching p15 length
            pretrained_ar_path: Path to pretrained structures matching ar length
            cfg_dropout_prob: Probability of dropping condition during training
            is_training: Whether this dataset is for training or validation
        """
        super().__init__()
        
        # Load structure data using correct function
        self.p15_data = du.load_ensemble_structure(p15_data_path)
        self.ar_data = du.load_ensemble_structure(ar_data_path)
        
        # Load pretrained structures if provided
        if pretrained_p15_path:
            self.pretrained_p15 = du.load_structure_dir(pretrained_p15_path)
        if pretrained_ar_path:
            self.pretrained_ar = du.load_structure_dir(pretrained_ar_path)
        
        # Load embeddings
        self.p15_embedding = self._load_single_embedding(p15_embedding_path)
        self.ar_embedding = self._load_single_embedding(ar_embedding_path)
        
        # Store parameters
        self.cfg_dropout_prob = cfg_dropout_prob
        self._is_training = is_training
        
        # Store lengths for convenience
        self.p15_length = len(self.p15_data[0]['positions'])
        self.ar_length = len(self.ar_data[0]['positions'])

    def _load_single_embedding(self, path):
        """Load embedding from file."""
        try:
            with open(path, 'r') as f:
                # Read the line and remove brackets
                line = f.readline().strip()
                line = line.strip('[]')
                # Split by comma and convert to float
                values = [float(x) for x in line.split(',')]
            return torch.from_numpy(np.array(values)).float()
        except Exception as e:
            print(f"Error loading embedding from {path}: {e}")
            return None

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
        if self._is_training and torch.rand(1) < self.cfg_dropout_prob:
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
        """Get a structure from the appropriate conformations directory.
        
        Args:
            length: Length of structure to return (either p15_length or ar_length)
            
        Returns:
            Dictionary containing structure data
        """
        # Choose the appropriate dataset based on length
        if length == self.p15_length:
            data = self.p15_data
            path = self.p15_data_path
        else:
            data = self.ar_data
            path = self.ar_data_path
            
        # Randomly select one structure from the dataset
        idx = np.random.randint(len(data))
        return data[idx]

class LengthBasedBatchSampler:
    """Sampler that creates batches of same-length structures."""
    
    def __init__(self, dataset: IDPCFGDataset, batch_size: int, drop_last: bool = False):
        """Initialize the sampler.
        
        Args:
            dataset: The IDPCFGDataset instance
            batch_size: Number of structures per batch
            drop_last: Whether to drop the last batch if it's incomplete
        """
        self.dataset = dataset
        self.batch_size = batch_size
        self.drop_last = drop_last
        
        # Group indices by structure length
        self.p15_indices = list(range(len(dataset.p15_data)))
        self.ar_indices = list(range(len(dataset.p15_data), len(dataset)))
        
        # Calculate number of batches
        n_p15_batches = len(self.p15_indices) // batch_size
        n_ar_batches = len(self.ar_indices) // batch_size
        
        if not drop_last:
            if len(self.p15_indices) % batch_size != 0:
                n_p15_batches += 1
            if len(self.ar_indices) % batch_size != 0:
                n_ar_batches += 1
                
        self.n_batches = n_p15_batches + n_ar_batches
    
    def __iter__(self):
        # Shuffle indices for each length
        p15_indices = self.p15_indices.copy()
        ar_indices = self.ar_indices.copy()
        np.random.shuffle(p15_indices)
        np.random.shuffle(ar_indices)
        
        # Create batches for each length
        p15_batches = [
            p15_indices[i:i + self.batch_size] 
            for i in range(0, len(p15_indices), self.batch_size)
        ]
        ar_batches = [
            ar_indices[i:i + self.batch_size] 
            for i in range(0, len(ar_indices), self.batch_size)
        ]
        
        # Drop last incomplete batches if requested
        if self.drop_last:
            if len(p15_batches[-1]) < self.batch_size:
                p15_batches.pop()
            if len(ar_batches[-1]) < self.batch_size:
                ar_batches.pop()
        
        # Combine and shuffle batches
        all_batches = p15_batches + ar_batches
        np.random.shuffle(all_batches)
        
        for batch in all_batches:
            yield batch
    
    def __len__(self):
        return self.n_batches 