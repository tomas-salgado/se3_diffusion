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
        
        # Store parameters 
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
            # Load protein structures
            self._log.info(f"Loading P15 structures from {p15_data_path}...")
            IDPCFGDataset._shared_data['p15_data'] = du.load_ensemble_structure(p15_data_path)
            self._log.info(f"Loading AR structures from {ar_data_path}...")
            IDPCFGDataset._shared_data['ar_data'] = du.load_ensemble_structure(ar_data_path)
            
            # Validate loaded data
            if len(IDPCFGDataset._shared_data['p15_data']) == 0:
                raise ValueError(f"No structures loaded from P15 data path: {p15_data_path}")
            if len(IDPCFGDataset._shared_data['ar_data']) == 0:
                raise ValueError(f"No structures loaded from AR data path: {ar_data_path}")
                
            # Store lengths for pretrained structure generation
            IDPCFGDataset._shared_data['p15_length'] = IDPCFGDataset._shared_data['p15_data'][0]['positions'].shape[0]
            IDPCFGDataset._shared_data['ar_length'] = IDPCFGDataset._shared_data['ar_data'][0]['positions'].shape[0]
            
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
            
            # Load embeddings - this is critical for CFG
            self._log.info("Loading embeddings for conditioning...")
            try:
                # Load P15 embedding
                if IDPCFGDataset._shared_data['p15_embedding'] is None:
                    self._log.info(f"Loading P15 embedding from {p15_embedding_path}...")
                    IDPCFGDataset._shared_data['p15_embedding'] = self._load_single_embedding(p15_embedding_path, is_p15=True)
                    self._log.info(f"P15 embedding loaded with shape {IDPCFGDataset._shared_data['p15_embedding'].shape}")
                
                # Load AR embedding
                if IDPCFGDataset._shared_data['ar_embedding'] is None:
                    self._log.info(f"Loading AR embedding from {ar_embedding_path}...")
                    IDPCFGDataset._shared_data['ar_embedding'] = self._load_single_embedding(ar_embedding_path, is_p15=False)
                    self._log.info(f"AR embedding loaded with shape {IDPCFGDataset._shared_data['ar_embedding'].shape}")
                
                # Validate embedding dimensions match (should be the same size for both P15 and AR)
                if IDPCFGDataset._shared_data['p15_embedding'].shape != IDPCFGDataset._shared_data['ar_embedding'].shape:
                    p15_shape = IDPCFGDataset._shared_data['p15_embedding'].shape
                    ar_shape = IDPCFGDataset._shared_data['ar_embedding'].shape
                    self._log.warning(f"Embedding dimensions don't match: P15 {p15_shape} vs AR {ar_shape}")
                    self._log.warning("This is OK if using the dimension adapter in the sequence embedder")
                    
                # Store the embedding dimensions for reference
                self.embedding_dim = IDPCFGDataset._shared_data['p15_embedding'].shape[0]
                self._log.info(f"Using embedding dimension: {self.embedding_dim}")
                
            except Exception as e:
                self._log.error(f"Error loading embeddings: {str(e)}")
                raise
        
        self._log.info(f"Dataset initialization complete: {len(self)} samples available")
        
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
        try:
            # Load embedding from file
            with open(path, 'r') as f:
                embedding_str = f.read().strip()
                
                # Remove square brackets if present
                if embedding_str.startswith('[') and embedding_str.endswith(']'):
                    embedding_str = embedding_str[1:-1]
                
                # Clean the string (remove newlines, extra spaces)
                embedding_str = embedding_str.replace('\n', '').replace(' ', '')
                
                # Split by commas and convert to float
                embedding_values = []
                for val in embedding_str.split(','):
                    if val.strip():  # Skip empty strings
                        try:
                            embedding_values.append(float(val.strip()))
                        except ValueError as e:
                            self._log.warning(f"Skipping invalid value in embedding: {val}: {str(e)}")
                
                if not embedding_values:
                    raise ValueError("No valid embedding values found in file")
                
                # Convert to tensor
                embedding = torch.tensor(embedding_values, dtype=torch.float32)
                
                # Log success and dimensions
                self._log.info(f"Successfully loaded embedding from {path} with shape {embedding.shape}")
                self._log.debug(f"Embedding range: min={embedding.min().item():.4f}, max={embedding.max().item():.4f}, mean={embedding.mean().item():.4f}")
                
                # Store in shared data to avoid reloading
                if is_p15:
                    IDPCFGDataset._shared_data['p15_embedding'] = embedding
                else:
                    IDPCFGDataset._shared_data['ar_embedding'] = embedding
                
                return embedding
                
        except Exception as e:
            self._log.error(f"Error loading embedding from {path}: {str(e)}")
            # Rethrow with more context
            raise ValueError(f"Failed to load embedding from {path}: {str(e)}") from e

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
            structure_idx = idx
            structure = IDPCFGDataset._shared_data['p15_data'][structure_idx]
            embedding_path = self.p15_embedding_path
        else:
            is_p15 = False
            structure_idx = idx - len(IDPCFGDataset._shared_data['p15_data'])
            structure = IDPCFGDataset._shared_data['ar_data'][structure_idx]
            embedding_path = self.ar_embedding_path
        
        # Get the embedding (cached at init time to avoid reloading)
        if is_p15 and 'p15_embedding' in IDPCFGDataset._shared_data and IDPCFGDataset._shared_data['p15_embedding'] is not None:
            embedding = IDPCFGDataset._shared_data['p15_embedding']
        elif not is_p15 and 'ar_embedding' in IDPCFGDataset._shared_data and IDPCFGDataset._shared_data['ar_embedding'] is not None:
            embedding = IDPCFGDataset._shared_data['ar_embedding']
        else:
            # Load embedding if not cached
            embedding = self._load_single_embedding(embedding_path, is_p15=is_p15)
        
        # Apply CFG dropout during training
        use_pretrained = False
        if self._is_training and torch.rand(1) < self.cfg_dropout_prob:
            self._log.debug("Applying CFG dropout - using null embedding")
            # Zero out the embedding for CFG
            embedding = torch.zeros_like(embedding)
            
            # Use pretrained structures for unconditioned samples
            if is_p15 and 'pretrained_p15' in IDPCFGDataset._shared_data and IDPCFGDataset._shared_data['pretrained_p15']:
                # Randomly select a pretrained structure for p15 length
                pretrained = IDPCFGDataset._shared_data['pretrained_p15']
                pretrained_idx = np.random.randint(0, len(pretrained))
                structure = pretrained[pretrained_idx]
                self._log.debug(f"Using pretrained P15 structure (idx {pretrained_idx})")
            elif not is_p15 and 'pretrained_ar' in IDPCFGDataset._shared_data and IDPCFGDataset._shared_data['pretrained_ar']:
                # Randomly select a pretrained structure for AR length
                pretrained = IDPCFGDataset._shared_data['pretrained_ar']
                pretrained_idx = np.random.randint(0, len(pretrained))
                structure = pretrained[pretrained_idx]
                self._log.debug(f"Using pretrained AR structure (idx {pretrained_idx})")
        
        # Extract positions from the structure
        positions = structure['positions']  # Shape: [L, 4, 3] for N, CA, C, O
        
        # Convert to torch tensor if it's a numpy array
        if isinstance(positions, np.ndarray):
            positions = torch.tensor(positions, dtype=torch.float32)
            
        # Get the sequence length
        length = positions.shape[0]
        
        # Create backbone rigid transformations from N, CA, C positions
        try:
            # Extract the backbone atom positions
            n_xyz = positions[:, 0]   # N atoms
            ca_xyz = positions[:, 1]  # CA atoms
            c_xyz = positions[:, 2]   # C atoms
            
            # Create rigid transformations using openfold's utility
            gt_bb_rigid = rigid_utils.Rigid.from_3_points(
                p_neg_x_axis=n_xyz,  # N atoms
                origin=ca_xyz,       # CA atoms 
                p_xy_plane=c_xyz     # C atoms
            )
            
            # Convert to tensor format
            rigids_tensor = gt_bb_rigid.to_tensor_7()  # [L, 7]
        except Exception as e:
            self._log.error(f"Error creating rigid transformations: {str(e)}")
            # Create a simple fallback rigid transformation
            device = positions.device
            identity_rot = torch.eye(3, device=device).unsqueeze(0).repeat(length, 1, 1)
            rot_obj = rigid_utils.Rotation(rot_mats=identity_rot)
            gt_bb_rigid = rigid_utils.Rigid(rot_obj, positions[:, 1])  # Use CA as translation
            rigids_tensor = gt_bb_rigid.to_tensor_7()  # [L, 7]
        
        # Create placeholder torsion angles (7 angles, sin & cos values)
        torsion_angles = torch.zeros((length, 7, 2), dtype=torch.float32)
        torsion_angles[:, :, 0] = 0.0  # sin values
        torsion_angles[:, :, 1] = 1.0  # cos values
        
        # Create sequence indices and masks
        seq_idx = torch.arange(length, dtype=torch.long)
        res_mask = torch.ones(length, dtype=torch.float32)
        fixed_mask = torch.zeros(length, dtype=torch.float32)
        
        # Sample a random timestep [0, 1] - keep as 1D tensor
        t = torch.rand(1, dtype=torch.float32)
        
        # Return the sample data with all required fields
        return {
            # Required inputs for SE(3) diffusion model
            'res_mask': res_mask,                    # [L]
            'seq_idx': seq_idx,                      # [L]
            'fixed_mask': fixed_mask,                # [L]
            'sc_ca_t': ca_xyz,                       # [L, 3] - CA positions for self-conditioning
            't': t,                                  # [1] - 1D tensor for timestep (don't unsqueeze)
            
            # Features for the score model
            'torsion_angles_sin_cos': torsion_angles,  # [L, 7, 2]
            'rigids': rigids_tensor,                 # [L, 7]
            'rigids_t': rigids_tensor,               # [L, 7] - for self-conditioning
            
            # Conditioning information
            'sequence': embedding,                   # [embed_dim] - Raw embedding tensor
            
            # Additional data
            'positions': positions,                  # [L, 4, 3] - Atom positions
            'length': torch.tensor(length, dtype=torch.long),  # scalar
            'is_p15': torch.tensor(is_p15, dtype=torch.float32),  # Boolean flag
        }

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
        For timesteps, we ensure they maintain the correct shape.
        """
        result = {}
        
        # Process each key in the first sample
        for key in batch[0].keys():
            # Get all values for this key from all samples
            values = [sample[key] for sample in batch]
            
            # Special handling for timestep tensor
            if key == 't':
                # Stack timesteps - Each timestep is a 1D tensor of shape [1]
                # After stacking, result will be [B, 1]
                # The model expects the first dimension to be batch size
                result[key] = torch.cat(values, dim=0)
            
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