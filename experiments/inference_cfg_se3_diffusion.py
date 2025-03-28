"""Simple inference script for SE(3) diffusion with classifier-free guidance."""

import os
import logging
import torch
import numpy as np
from pathlib import Path
import argparse
from datetime import datetime
import json
import hydra
from omegaconf import DictConfig, OmegaConf
from experiments.train_cfg_se3_diffusion import CFGExperiment

from data import all_atom
from data import se3_diffuser
from model import score_network
from openfold.utils import rigid_utils

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def load_embedding(embedding_path):
    """Load sequence embedding from file."""
    logger.info(f"Loading embedding from {embedding_path}")
    
    if embedding_path.endswith('.pt'):
        embedding = torch.load(embedding_path)
    elif embedding_path.endswith('.npy'):
        embedding = torch.from_numpy(np.load(embedding_path))
    elif embedding_path.endswith('.txt'):
        # Handle text file with array of numbers
        try:
            with open(embedding_path, 'r') as f:
                content = f.read().strip()
                # Remove brackets if present
                content = content.strip('[]')
                # Split by commas and convert to float
                values = [float(x.strip()) for x in content.split(',')]
                embedding = torch.tensor(values, dtype=torch.float32)
        except Exception as e:
            logger.error(f"Error parsing embedding file: {e}")
            raise ValueError(f"Failed to parse embedding file: {embedding_path}")
    else:
        raise ValueError(f"Unsupported embedding format: {embedding_path}. Supported formats: .pt, .npy, .txt")
    
    # Ensure embedding is 2D (batch_size, embed_dim)
    if embedding.dim() == 1:
        embedding = embedding.unsqueeze(0)
    
    logger.info(f"Loaded embedding with shape: {embedding.shape}")
    
    return embedding

def save_structure(positions, output_dir, name="structure"):
    """Save generated structure to PDB file."""
    # This function would convert positions to PDB format
    # For now, just save the positions as a numpy array
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, f"{name}.npy")
    np.save(output_path, positions.cpu().numpy())
    logger.info(f"Saved structure to {output_path}")

@hydra.main(version_base=None, config_path="../config", config_name="inference_cfg")
def main(conf: DictConfig):
    """Main inference function."""
    # Use CUDA if available
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Extract parameters from config
    checkpoint_path = conf.inference.checkpoint_path
    embedding_path = conf.embedding_path
    output_dir = conf.output_dir
    cfg_scale = conf.cfg_scale
    num_samples = conf.num_samples
    min_t = conf.inference.min_t
    max_t = conf.inference.max_t
    num_t = conf.inference.num_t
    
    # Log configuration
    logger.info(f"Running inference with:")
    logger.info(f"  Checkpoint: {checkpoint_path}")
    logger.info(f"  Embedding: {embedding_path}")
    logger.info(f"  Output directory: {output_dir}")
    logger.info(f"  CFG scale: {cfg_scale}")
    logger.info(f"  Samples: {num_samples}")
    
    # Create experiment (this will load the model)
    experiment = CFGExperiment(conf)
    
    # Get model and diffuser from experiment
    model = experiment.exp.model
    diffuser = experiment.exp.diffuser
    
    # Load sequence embedding
    seq_embedding = load_embedding(embedding_path)
    logger.info(f"Original embedding shape: {seq_embedding.shape}")

    # Ensure embedding is 2D (batch_size, embed_dim)
    if seq_embedding.dim() == 1:
        seq_embedding = seq_embedding.unsqueeze(0)
    logger.info(f"Reshaped embedding: {seq_embedding.shape}")

    # Move to device
    seq_embedding = seq_embedding.to(device)
    
    # Set up sampling
    timesteps = torch.linspace(max_t, min_t, num_t, device=device)
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Generate samples
    for sample_idx in range(num_samples):
        logger.info(f"Generating sample {sample_idx+1}/{num_samples}")
        
        # Get sequence length from embedding
        sequence_length = seq_embedding.shape[1] if seq_embedding.dim() > 1 else 1
        
        # Sample from reference distribution
        sample_ref_output = diffuser.sample_ref(
            n_samples=sequence_length,
            as_tensor_7=True
        )

        # Move the tensor to the device
        rigids_t = sample_ref_output['rigids_t'].to(device)

        # Create input features - ensure all tensors are on the same device
        input_feats = {
            'rigids_t': rigids_t,
            't': timesteps[0],
            'res_mask': torch.ones(sequence_length, device=device).unsqueeze(0),
            'fixed_mask': torch.zeros(sequence_length, device=device).unsqueeze(0),
            'sc_ca_t': torch.zeros(1, sequence_length, 3, device=device),
            'seq_idx': torch.arange(sequence_length, device=device).unsqueeze(0),
            'seq_embedding': seq_embedding
        }
        
        # Run with conditioning
        with torch.no_grad():
            # Run model inference
            output = model(input_feats, cfg_scale=cfg_scale)
            
            # Update rigids with model prediction
            rigids = output['rigids_0_pred']
            
            # Save intermediate step if needed
            if (0 + 1) % 10 == 0:
                logger.info(f"  Step {0+1}/{len(timesteps)}")
            
            # Add to trajectory
            trajectory = [rigids.clone()]
        
        # Save final structure
        save_structure(rigids, output_dir, name=f"sample_{sample_idx}")
        
        # Optionally save trajectory
        # np.save(os.path.join(output_dir, f"trajectory_{sample_idx}.npy"), 
        #         torch.stack(trajectory).cpu().numpy())

if __name__ == "__main__":
    main()