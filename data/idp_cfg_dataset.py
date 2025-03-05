import torch
from torch.utils.data import Dataset
import numpy as np
from typing import List, Dict, Optional
from data import utils as du
import os
from data.utils import rigid_utils
import logging

class IDPCFGDataset(Dataset):
    # Class variables to store shared data
    _shared_data = {
        'p15_data': None,
        'ar_data': None,
        'p15_embedding': None,
        'ar_embedding': None,
        'pretrained_p15': None,
        'pretrained_ar': None,
        'p15_length': None,
        'ar_length': None
    }

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
        self._log = logging.getLogger(__name__)
        
        self._log.info(f"Initializing {'training' if is_training else 'validation'} dataset...")
        
        # Validate input paths
        self._log.info("Validating input paths...")
        if not os.path.exists(p15_data_path):
            raise ValueError(f"P15 data path does not exist: {p15_data_path}")
        if not os.path.exists(ar_data_path):
            raise ValueError(f"AR data path does not exist: {ar_data_path}")
        if not os.path.exists(p15_embedding_path):
            raise ValueError(f"P15 embedding path does not exist: {p15_embedding_path}")
        if not os.path.exists(ar_embedding_path):
            raise ValueError(f"AR embedding path does not exist: {ar_embedding_path}")
        
        # Load data only if not already loaded
        if IDPCFGDataset._shared_data['p15_data'] is None:
            self._log.info(f"Loading P15 structures from {p15_data_path}...")
            IDPCFGDataset._shared_data['p15_data'] = du.load_ensemble_structure(p15_data_path)
            self._log.info(f"Loading AR structures from {ar_data_path}...")
            IDPCFGDataset._shared_data['ar_data'] = du.load_ensemble_structure(ar_data_path)
            
            # Validate loaded data
            if len(IDPCFGDataset._shared_data['p15_data']) == 0:
                raise ValueError(f"No structures loaded from P15 data path: {p15_data_path}")
            if len(IDPCFGDataset._shared_data['ar_data']) == 0:
                raise ValueError(f"No structures loaded from AR data path: {ar_data_path}")
            
            # Load pretrained structures if provided
            if pretrained_p15_path:
                self._log.info(f"Loading pretrained P15 structures from {pretrained_p15_path}...")
                if not os.path.exists(pretrained_p15_path):
                    raise ValueError(f"Pretrained P15 path does not exist: {pretrained_p15_path}")
                IDPCFGDataset._shared_data['pretrained_p15'] = du.load_structure_dir(pretrained_p15_path)
                if len(IDPCFGDataset._shared_data['pretrained_p15']) == 0:
                    raise ValueError(f"No structures loaded from pretrained P15 path: {pretrained_p15_path}")
                    
            if pretrained_ar_path:
                self._log.info(f"Loading pretrained AR structures from {pretrained_ar_path}...")
                if not os.path.exists(pretrained_ar_path):
                    raise ValueError(f"Pretrained AR path does not exist: {pretrained_ar_path}")
                IDPCFGDataset._shared_data['pretrained_ar'] = du.load_structure_dir(pretrained_ar_path)
                if len(IDPCFGDataset._shared_data['pretrained_ar']) == 0:
                    raise ValueError(f"No structures loaded from pretrained AR path: {pretrained_ar_path}")
            
            # Load embeddings
            self._log.info("Loading sequence embeddings...")
            IDPCFGDataset._shared_data['p15_embedding'] = self._load_single_embedding(p15_embedding_path)
            IDPCFGDataset._shared_data['ar_embedding'] = self._load_single_embedding(ar_embedding_path)
            
            # Validate embeddings
            if IDPCFGDataset._shared_data['p15_embedding'] is None:
                raise ValueError(f"Failed to load P15 embedding from: {p15_embedding_path}")
            if IDPCFGDataset._shared_data['ar_embedding'] is None:
                raise ValueError(f"Failed to load AR embedding from: {ar_embedding_path}")
            
            # Validate embedding dimensions match
            if IDPCFGDataset._shared_data['p15_embedding'].shape != IDPCFGDataset._shared_data['ar_embedding'].shape:
                raise ValueError(f"Embedding dimension mismatch: P15 {IDPCFGDataset._shared_data['p15_embedding'].shape} vs AR {IDPCFGDataset._shared_data['ar_embedding'].shape}")
            
            # Store lengths
            IDPCFGDataset._shared_data['p15_length'] = len(IDPCFGDataset._shared_data['p15_data'][0]['positions'])
            IDPCFGDataset._shared_data['ar_length'] = len(IDPCFGDataset._shared_data['ar_data'][0]['positions'])
            
            # Log dataset statistics
            self._log.info("Dataset initialization complete:")
            self._log.info(f"- P15 structures: {len(IDPCFGDataset._shared_data['p15_data'])} with length {IDPCFGDataset._shared_data['p15_length']}")
            self._log.info(f"- AR structures: {len(IDPCFGDataset._shared_data['ar_data'])} with length {IDPCFGDataset._shared_data['ar_length']}")
            self._log.info(f"- Embedding dimension: {IDPCFGDataset._shared_data['p15_embedding'].shape}")
            if pretrained_p15_path:
                self._log.info(f"- Pretrained P15 structures: {len(IDPCFGDataset._shared_data['pretrained_p15'])}")
            if pretrained_ar_path:
                self._log.info(f"- Pretrained AR structures: {len(IDPCFGDataset._shared_data['pretrained_ar'])}")
        
        # Store parameters
        self.cfg_dropout_prob = cfg_dropout_prob
        self._is_training = is_training

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
        return len(IDPCFGDataset._shared_data['p15_data']) + len(IDPCFGDataset._shared_data['ar_data'])

    def __getitem__(self, idx):
        # Determine if this sample is from p15 or AR dataset
        is_p15 = idx < len(IDPCFGDataset._shared_data['p15_data'])
        
        if is_p15:
            data = IDPCFGDataset._shared_data['p15_data'][idx]
            embedding = IDPCFGDataset._shared_data['p15_embedding']  # Use the single p15 embedding
            length = IDPCFGDataset._shared_data['p15_length']
            self._log.debug(f"Loading P15 structure {idx}")
        else:
            adj_idx = idx - len(IDPCFGDataset._shared_data['p15_data'])
            data = IDPCFGDataset._shared_data['ar_data'][adj_idx]
            embedding = IDPCFGDataset._shared_data['ar_embedding']  # Use the single AR embedding
            length = IDPCFGDataset._shared_data['ar_length']
            self._log.debug(f"Loading AR structure {adj_idx}")

        # Apply CFG dropout during training
        if self._is_training and torch.rand(1) < self.cfg_dropout_prob:
            self._log.debug("Applying CFG dropout")
            # For unconditioned samples, zero out the embedding
            embedding = torch.zeros_like(embedding)
            
            # Use pretrained-generated structures matching the length
            if is_p15:
                data = self._get_pretrained_structure(IDPCFGDataset._shared_data['p15_length'])
                self._log.debug("Using pretrained P15 structure")
            else:
                data = self._get_pretrained_structure(IDPCFGDataset._shared_data['ar_length'])
                self._log.debug("Using pretrained AR structure")

        # Convert positions to tensor
        positions = torch.from_numpy(data['positions']).float()
        
        # Create mask (1 for residues with all backbone atoms)
        mask = torch.ones(length)
        
        # Create sequence indices (1-based)
        seq_idx = torch.arange(1, length + 1)
        
        # Create fixed mask (all zeros for now)
        fixed_mask = torch.zeros(length)
        
        # Calculate rigid body transforms from backbone atoms
        gt_bb_rigid = rigid_utils.Rigid.from_3_points(
            positions[:, 0],  # N
            positions[:, 1],  # CA
            positions[:, 2],  # C
        )
        
        # Calculate torsion angles (placeholder for now)
        torsion_angles = torch.zeros(length, 7, 2)  # 7 torsion angles, sin/cos for each
        
        # Sample time uniformly between 0 and 1 for training
        t = torch.rand(1).item() if self._is_training else 1.0
        
        # Create timestep tensor with correct shape [1]
        t_tensor = torch.tensor(t).reshape(-1)  # This ensures 1D shape
        
        # Expand sequence embedding to match residue dimension
        sequence_embedding = embedding.unsqueeze(0).expand(length, -1)  # [L, 1024]
        
        # Debug logging for tensor shapes
        self._log.debug(f"Tensor shapes:")
        self._log.debug(f"- seq_idx: {seq_idx.shape}")
        self._log.debug(f"- mask: {mask.shape}")
        self._log.debug(f"- fixed_mask: {fixed_mask.shape}")
        self._log.debug(f"- positions: {positions.shape}")
        self._log.debug(f"- torsion_angles: {torsion_angles.shape}")
        self._log.debug(f"- gt_bb_rigid: {gt_bb_rigid.shape}")
        self._log.debug(f"- sequence_embedding: {sequence_embedding.shape}")
        self._log.debug(f"- t: {t_tensor.shape}")
        
        # Need to match the expected input format
        return {
            # Fields required by ScoreNetwork
            'res_mask': mask,                    # [L]
            'seq_idx': seq_idx,                  # [L]
            'fixed_mask': fixed_mask,            # [L]
            'torsion_angles_sin_cos': torsion_angles,  # [L, 7, 2]
            'sc_ca_t': positions[:, 1],          # [L, 3] - CA positions
            'rigids': gt_bb_rigid.to_tensor_7(), # [L, 7]
            'rigids_t': gt_bb_rigid.to_tensor_7(), # [L, 7] - for self-conditioning
            
            # Additional fields for CFG
            'sequence_embedding': sequence_embedding,  # [L, 1024] - expanded to match residue dimension
            'is_p15': torch.tensor(is_p15),      # scalar
            
            # Time information for diffusion
            't': t_tensor,                       # [1] - 1D tensor for timestep
            
            # Any other fields needed by the diffusion model
            'positions': positions,               # [L, 3]
            'length': length,                    # scalar
            
            # Additional fields that might be needed
            'aatype': torch.zeros(length),       # [L] - amino acid types (placeholder)
            'chain_idx': torch.zeros(length),    # [L] - chain indices (placeholder)
            'chain_mask': torch.ones(length),    # [L] - chain mask (placeholder)
            'chain_encoding_all': torch.zeros(length),  # [L] - chain encoding (placeholder)
        }

    def _get_pretrained_structure(self, length: int) -> Dict:
        """Get a structure from the appropriate conformations directory.
        
        Args:
            length: Length of structure to return (either p15_length or ar_length)
            
        Returns:
            Dictionary containing structure data
        """
        # Choose the appropriate dataset based on length
        if length == IDPCFGDataset._shared_data['p15_length']:
            data = IDPCFGDataset._shared_data['pretrained_p15']
        else:
            data = IDPCFGDataset._shared_data['pretrained_ar']
            
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
        self.p15_indices = list(range(len(IDPCFGDataset._shared_data['p15_data'])))
        self.ar_indices = list(range(len(IDPCFGDataset._shared_data['p15_data']), len(dataset)))
        
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