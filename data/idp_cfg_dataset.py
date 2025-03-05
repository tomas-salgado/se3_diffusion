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
        """Initialize the dataset.
        
        Args:
            p15_data_path: Path to p15 conformations
            ar_data_path: Path to AR conformations
            p15_embedding_path: Path to p15 embedding txt file
            ar_embedding_path: Path to AR embedding txt file
            pretrained_p15_path: Path to pretrained p15-length structures
            pretrained_ar_path: Path to pretrained ar-length structures
            cfg_dropout_prob: Probability of dropping embeddings during training
            is_training: Whether this is for training or validation
        """
        # Call parent init
        super().__init__()
        
        # Initialize logger
        self._log = logging.getLogger(self.__class__.__name__)
        self._log.info(f"Initializing {'training' if is_training else 'validation'} dataset...")
        
        # Set parameters early to avoid AttributeError
        self.p15_embedding_path = p15_embedding_path
        self.ar_embedding_path = ar_embedding_path
        self.cfg_dropout_prob = cfg_dropout_prob
        self._is_training = is_training
        
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
            IDPCFGDataset._shared_data['p15_embedding'] = self._load_single_embedding(p15_embedding_path, is_p15=True)
            IDPCFGDataset._shared_data['ar_embedding'] = self._load_single_embedding(ar_embedding_path, is_p15=False)
            
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
        
        # Store paths for later use
        self.p15_data_path = p15_data_path
        self.ar_data_path = ar_data_path
        self.pretrained_p15_path = pretrained_p15_path
        self.pretrained_ar_path = pretrained_ar_path

    def _load_single_embedding(self, path, is_p15=True):
        """Load a single embedding from a file and apply CFG dropout during training.
        
        Args:
            path: Path to the embedding file
            is_p15: Whether this is a P15 embedding or not
            
        Returns:
            Embedding tensor with shape [embed_dim]
        """
        # Use shared embeddings if already loaded
        if is_p15 and 'p15_embedding' in IDPCFGDataset._shared_data and IDPCFGDataset._shared_data['p15_embedding'] is not None:
            embedding = IDPCFGDataset._shared_data['p15_embedding']
        elif not is_p15 and 'ar_embedding' in IDPCFGDataset._shared_data and IDPCFGDataset._shared_data['ar_embedding'] is not None:
            embedding = IDPCFGDataset._shared_data['ar_embedding']
        else:
            try:
                # Load embedding from file
                with open(path, 'r') as f:
                    embedding_str = f.read().strip()
                
                # Process the string to handle brackets and newlines
                embedding_str = embedding_str.replace('\n', '').replace(' ', '')
                if embedding_str.startswith('[') and embedding_str.endswith(']'):
                    embedding_str = embedding_str[1:-1]
                
                # Split by commas and convert to float
                embedding_values = []
                for val in embedding_str.split(','):
                    if val.strip():  # Skip empty strings
                        try:
                            embedding_values.append(float(val.strip()))
                        except ValueError as e:
                            self._log.error(f"Error parsing value '{val}': {str(e)}")
                            raise ValueError(f"Could not parse embedding value '{val}'")
                
                if not embedding_values:
                    raise ValueError("No valid embedding values found in file")
                
                # Convert to tensor
                embedding = torch.tensor(embedding_values, dtype=torch.float32)
                
                # Log success and dimensions
                self._log.info(f"Successfully loaded embedding from {path} with shape {embedding.shape}")
                
                # Cache in shared data
                if is_p15:
                    IDPCFGDataset._shared_data['p15_embedding'] = embedding
                else:
                    IDPCFGDataset._shared_data['ar_embedding'] = embedding
            except Exception as e:
                self._log.error(f"Failed to load embedding from {path}: {str(e)}")
                return None
                
        # During training, apply CFG dropout randomly
        # Note: This part should only be used in __getitem__, not during initialization
        # since we need valid embeddings during init
        if hasattr(self, '_is_training') and self._is_training and torch.rand(1) < self.cfg_dropout_prob:
            # Return zero embedding for unconditioned samples
            return torch.zeros_like(embedding)
        
        return embedding

    def __len__(self):
        return len(IDPCFGDataset._shared_data['p15_data']) + len(IDPCFGDataset._shared_data['ar_data'])

    def __getitem__(self, idx):
        """Get a single sample from the dataset.
        
        Args:
            idx: Index of the sample to get
            
        Returns:
            Dictionary containing the sample data
        """
        # Determine which dataset to use based on the index
        if idx < len(IDPCFGDataset._shared_data['p15_data']):
            is_p15 = True
            structure = IDPCFGDataset._shared_data['p15_data'][idx]
            embedding = self._load_single_embedding(self.p15_embedding_path, is_p15=True)
        else:
            is_p15 = False
            adjusted_idx = idx - len(IDPCFGDataset._shared_data['p15_data'])
            structure = IDPCFGDataset._shared_data['ar_data'][adjusted_idx]
            embedding = self._load_single_embedding(self.ar_embedding_path, is_p15=False)
        
        # Apply CFG dropout during training - use pretrained structure if dropout is applied
        use_pretrained = False
        if self._is_training and torch.rand(1) < self.cfg_dropout_prob:
            self._log.debug("Applying CFG dropout")
            # Zero out the embedding
            embedding = torch.zeros_like(embedding)
            
            # Use pretrained structures
            use_pretrained = True
            
        # Get pretrained structure if needed
        if use_pretrained:
            if is_p15:
                self._log.debug("Using pretrained P15 structure")
                structure = self._get_pretrained_structure(IDPCFGDataset._shared_data['p15_length'])
            else:
                self._log.debug("Using pretrained AR structure")
                structure = self._get_pretrained_structure(IDPCFGDataset._shared_data['ar_length'])
        
        # Extract required data
        positions = structure['positions']
        length = positions.shape[0]
        gt_bb_rigid = structure['rigids']
        torsion_angles = structure['torsion_angles']
        
        # Create amino acid sequence indices (0-indexed)
        seq_idx = torch.arange(length, dtype=torch.long)
        
        # Create mask (1 for valid residues)
        mask = torch.ones(length, dtype=torch.float32)
        
        # Create fixed mask (0 for residues that can move)
        fixed_mask = torch.zeros(length, dtype=torch.float32)
        
        # Sample a random timestep [0, 1]
        t = torch.rand(1, dtype=torch.float32)
        t_tensor = t.unsqueeze(0)  # Make it [1, 1] for batching
        
        # Need to match the expected input format
        return {
            # Required features used directly by ScoreNetwork's forward method
            'res_mask': mask,                    # [L]
            'seq_idx': seq_idx,                  # [L]
            'fixed_mask': fixed_mask,            # [L]
            'sc_ca_t': positions[:, 1],          # [L, 3] - CA positions for self-conditioning
            't': t_tensor,                       # [1] - 1D tensor for timestep
            
            # Features for the score model
            'torsion_angles_sin_cos': torsion_angles,  # [L, 7, 2]
            'rigids': gt_bb_rigid.to_tensor_7(), # [L, 7]
            'rigids_t': gt_bb_rigid.to_tensor_7(), # [L, 7] - for self-conditioning
            
            # Sequence embedding for conditioning
            'sequence': embedding,               # [1024] - Raw embedding tensor
            
            # Any other fields needed by the diffusion model
            'positions': positions,              # [L, 3, 3] - N, CA, C atom positions
            'length': torch.tensor(length, dtype=torch.long),  # scalar
            
            # Metadata
            'is_p15': torch.tensor(is_p15, dtype=torch.float32),  # scalar
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

    def collate_fn(self, batch):
        """Custom collate function for our dataset.
        
        This function handles the special case of sequence embeddings and timesteps.
        For sequence embeddings, we stack them into a batch tensor.
        For timesteps, we ensure they remain 1D after stacking.
        """
        result = {}
        
        # Process each key in the first sample
        for key in batch[0].keys():
            # Get all values for this key from all samples
            values = [sample[key] for sample in batch]
            
            # Special handling for timestep tensor
            if key == 't':
                # Stack timesteps and ensure 1D shape [B]
                result[key] = torch.stack(values).squeeze(-1)
            
            # Special handling for sequence embedding
            elif key == 'sequence':
                # Stack sequence embeddings into [B, D] tensor
                result[key] = torch.stack(values)
            
            # Handle other tensors
            elif isinstance(values[0], torch.Tensor):
                result[key] = torch.stack(values, dim=0)
            
            # Handle lists, strings, etc.
            else:
                result[key] = values
            
        return result
    
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