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

@hydra.main(version_base=None, config_path="../config", config_name="finetune_cfg")
def main(conf: DictConfig):
    """Main inference function."""
    # Parse command line arguments
    parser = argparse.ArgumentParser(description="SE(3) diffusion inference")
    parser.add_argument("--checkpoint_path", type=str, default='weights/checkpoint.pth', help="Path to model checkpoint")
    parser.add_argument("--embedding_path", type=str, default='embeddings/ar_idr_embedding.txt', help="Path to sequence embedding")
    parser.add_argument("--output_dir", type=str, default='results_cfg/inference_results', help="Directory to save generated structures")
    parser.add_argument("--cfg_scale", type=float, default=1.0, help="Guidance scale")
    parser.add_argument("--num_samples", type=int, default=5, help="Number of samples to generate")
    parser.add_argument("--min_t", type=float, default=0.01, help="Minimum timestep")
    parser.add_argument("--max_t", type=float, default=1.0, help="Maximum timestep")
    parser.add_argument("--num_t", type=int, default=100, help="Number of timesteps")
    args = parser.parse_args()
    
    # Use CUDA if available
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Update config with checkpoint path
    conf.model.checkpoint = args.checkpoint_path
    
    # Create experiment (this will load the model)
    experiment = CFGExperiment(conf)
    
    # Get model and diffuser from experiment
    model = experiment.exp.model
    diffuser = experiment.exp.diffuser
    
    # Load sequence embedding
    seq_embedding = load_embedding(args.embedding_path)
    seq_embedding = seq_embedding.to(device)
    
    # Set up sampling
    timesteps = torch.linspace(args.max_t, args.min_t, args.num_t, device=device)
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Generate samples
    for sample_idx in range(args.num_samples):
        logger.info(f"Generating sample {sample_idx+1}/{args.num_samples}")
        
        # Get sequence length from embedding
        sequence_length = seq_embedding.shape[1] if seq_embedding.dim() > 1 else 1
        
        # Sample from reference distribution
        rigids = diffuser.sample_ref(
            n_samples=sequence_length,
            as_tensor_7=True
        ).to(device)
        
        # Generate trajectory
        trajectory = [rigids.clone()]
        
        # Iterative denoising
        for step_idx, t in enumerate(timesteps):
            t_batch = torch.full((1,), t, device=device)
            
            # Create input features
            input_feats = {
                'rigids_t': rigids,
                't': t_batch,
                'res_mask': torch.ones(sequence_length, device=device),
                'fixed_mask': torch.zeros(sequence_length, device=device),
                'sc_ca_t': torch.zeros(sequence_length, 3, device=device),
                'seq_idx': torch.arange(sequence_length, device=device),
                'seq_embedding': seq_embedding
            }
            
            # Run with conditioning
            with torch.no_grad():
                # Run model inference
                output = model(input_feats, cfg_scale=args.cfg_scale)
                
                # Update rigids with model prediction
                rigids = output['rigids_0_pred']
                
                # Save intermediate step if needed
                if (step_idx + 1) % 10 == 0:
                    logger.info(f"  Step {step_idx+1}/{len(timesteps)}")
                
                # Add to trajectory
                trajectory.append(rigids.clone())
        
        # Save final structure
        save_structure(rigids, args.output_dir, name=f"sample_{sample_idx}")
        
        # Optionally save trajectory
        # np.save(os.path.join(args.output_dir, f"trajectory_{sample_idx}.npy"), 
        #         torch.stack(trajectory).cpu().numpy())

if __name__ == "__main__":
    main()