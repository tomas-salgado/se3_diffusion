"""Simple inference script for SE(3) diffusion with classifier-free guidance."""

import os
import logging
import torch
import numpy as np
from pathlib import Path
import argparse
from datetime import datetime
import json

from data import all_atom
from data import se3_diffuser
from model import score_network
from openfold.utils import rigid_utils

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def load_embedding(embedding_path):
    """Load embedding from a text file."""
    logger.info(f"Loading embedding from {embedding_path}")
    
    try:
        with open(embedding_path, 'r') as f:
            lines = f.readlines()
            
        # Remove any brackets, newlines, or extra spaces
        embedding_str = ' '.join([line.strip() for line in lines])
        embedding_str = embedding_str.replace('[', '').replace(']', '')
        
        # Convert to tensor
        embedding = torch.tensor([float(x) for x in embedding_str.split()], 
                                 dtype=torch.float32)
        
        # Reshape to match expected format if needed
        if embedding.dim() == 1:
            embedding = embedding.unsqueeze(0)  # Add batch dimension
            
        logger.info(f"Loaded embedding with shape {embedding.shape}")
        return embedding
        
    except Exception as e:
        logger.error(f"Failed to load embedding: {e}")
        raise

def save_structure(positions, output_dir, name="structure"):
    """Save the generated structure to a PDB file."""
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, f"{name}.pdb")
    
    # Create a simple PDB with the backbone atoms
    with open(output_path, 'w') as f:
        for i, (n, ca, c, o) in enumerate(positions):
            # Write N atom
            f.write(f"ATOM  {i*4+1:5d}  N   ALA A{i+1:4d}    "
                    f"{n[0]:8.3f}{n[1]:8.3f}{n[2]:8.3f}"
                    f"  1.00  0.00           N\n")
            # Write CA atom
            f.write(f"ATOM  {i*4+2:5d}  CA  ALA A{i+1:4d}    "
                    f"{ca[0]:8.3f}{ca[1]:8.3f}{ca[2]:8.3f}"
                    f"  1.00  0.00           C\n")
            # Write C atom
            f.write(f"ATOM  {i*4+3:5d}  C   ALA A{i+1:4d}    "
                    f"{c[0]:8.3f}{c[1]:8.3f}{c[2]:8.3f}"
                    f"  1.00  0.00           C\n")
            # Write O atom
            f.write(f"ATOM  {i*4+4:5d}  O   ALA A{i+1:4d}    "
                    f"{o[0]:8.3f}{o[1]:8.3f}{o[2]:8.3f}"
                    f"  1.00  0.00           O\n")
        f.write("TER\nEND\n")
    
    logger.info(f"Saved structure to {output_path}")
    return output_path

def main():
    # Parse command line arguments
    parser = argparse.ArgumentParser(description="SE(3) diffusion with classifier-free guidance")
    parser.add_argument("--checkpoint_path", type=str, required=True, help="Path to model checkpoint")
    parser.add_argument("--embedding_path", type=str, required=True, help="Path to sequence embedding")
    parser.add_argument("--output_dir", type=str, required=True, help="Directory to save generated structures")
    parser.add_argument("--cfg_scale", type=float, default=1.0, help="Guidance scale")
    parser.add_argument("--num_samples", type=int, default=5, help="Number of samples to generate")
    parser.add_argument("--sequence_length", type=int, default=100, help="Length of protein sequence")
    parser.add_argument("--min_t", type=float, default=0.01, help="Minimum timestep")
    parser.add_argument("--max_t", type=float, default=1.0, help="Maximum timestep")
    parser.add_argument("--num_t", type=int, default=100, help="Number of timesteps")
    args = parser.parse_args()
    
    # Use CUDA if available
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Load checkpoint
    logger.info(f"Loading checkpoint from {args.checkpoint_path}")
    checkpoint = torch.load(args.checkpoint_path, map_location=device)
    
    # Get model state
    if 'model_state_dict' in checkpoint:
        model_state = checkpoint['model_state_dict']
    elif 'model' in checkpoint:
        model_state = checkpoint['model']
    else:
        raise KeyError("No model weights found in checkpoint")
    
    # Handle DataParallel saved models
    model_state = {k.replace('module.', ''):v for k,v in model_state.items()}
    
    # Determine input dimension from checkpoint
    input_dim = model_state['embedding_layer.node_embedder.0.weight'].shape[1]
    logger.info(f"Found input dimension in checkpoint: {input_dim}")
    
    # Create mock configurations
    from types import SimpleNamespace
    
    # Create diffuser configuration
    diff_conf = SimpleNamespace(
        diffuse_trans=True,
        diffuse_rot=True,
        r3=SimpleNamespace(
            min_b=0.1,
            max_b=20.0,
            coordinate_scaling=0.1
        ),
        so3=SimpleNamespace(
            num_omega=1000,
            num_sigma=1000,
            min_sigma=0.1,
            max_sigma=1.5,
            schedule="logarithmic",
            cache_dir=".cache/",
            use_cached_score=False
        )
    )
    
    # Create model configuration
    model_conf = SimpleNamespace(
        node_embed_size=256,
        edge_embed_size=128,
        use_sequence_conditioning=True,
        conditioning_method="cross_attention",
        sequence_embed=SimpleNamespace(
            embed_dim=256,
            adapt_dimensions=True
        ),
        embed=SimpleNamespace(
            index_embed_size=32,
            aatype_embed_size=64,
            embed_self_conditioning=True,
            num_bins=22,
            min_bin=1e-5,
            max_bin=20.0
        ),
        ipa=SimpleNamespace(
            c_s=256,
            c_z=128,
            c_hidden=256,
            c_skip=64,
            no_heads=8,
            no_qk_points=8,
            no_v_points=12,
            seq_tfmr_num_heads=4,
            seq_tfmr_num_layers=2,
            num_blocks=4,
            coordinate_scaling=0.1
        )
    )
    
    # Create diffuser
    diffuser = se3_diffuser.SE3Diffuser(diff_conf)
    
    # Create model constructor with custom initialization
    import torch.nn as nn
    class CustomScoreNetwork(score_network.ScoreNetwork):
        def __init__(self, model_conf, diffuser, input_dim):
            super(CustomScoreNetwork, self).__init__(model_conf, diffuser)
            # Replace the first layer of node_embedder to match checkpoint dimensions
            node_embed_size = model_conf.node_embed_size
            self.embedding_layer.node_embedder[0] = nn.Linear(input_dim, node_embed_size)
    
    # Create model with custom initialization
    model = CustomScoreNetwork(model_conf, diffuser, input_dim)
    
    # Load state dict
    model.load_state_dict(model_state, strict=False)
    model.to(device)
    model.eval()
    logger.info("Model loaded successfully")
    
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
        
        # Sample from reference distribution
        rigids = diffuser.sample_ref(
            n_samples=args.sequence_length,
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
                'res_mask': torch.ones(args.sequence_length, device=device),
                'fixed_mask': torch.zeros(args.sequence_length, device=device),
                'sc_ca_t': torch.zeros(args.sequence_length, 3, device=device),
                'seq_idx': torch.arange(args.sequence_length, device=device)
            }
            
            # Run with conditioning
            with torch.no_grad():
                # Run with conditioning
                cond_output = model(input_feats, cfg_scale=None)
                
                # For CFG, run without conditioning as well
                if args.cfg_scale != 1.0:
                    # Temporarily remove sequence embedding
                    original_embedding = seq_embedding.clone()
                    null_embedding = torch.zeros_like(seq_embedding)
                    # Run without conditioning
                    uncond_output = model(input_feats, cfg_scale=None)
                    
                    # Combine outputs with guidance scale
                    rot_score = uncond_output['rot_score'] + args.cfg_scale * (
                        cond_output['rot_score'] - uncond_output['rot_score'])
                    trans_score = uncond_output['trans_score'] + args.cfg_scale * (
                        cond_output['trans_score'] - uncond_output['trans_score'])
                else:
                    rot_score = cond_output['rot_score']
                    trans_score = cond_output['trans_score']
                
                # Update with the score
                rigids = diffuser.reverse_sample(
                    rigids_t=rigids,
                    rot_score=rot_score,
                    trans_score=trans_score,
                    t=t_batch,
                    dt=timesteps[0] - timesteps[-1] if step_idx == 0 else timesteps[step_idx-1] - timesteps[step_idx]
                )
            
            # Save current state
            trajectory.append(rigids.clone())
            
            if (step_idx + 1) % 10 == 0 or step_idx == len(timesteps) - 1:
                logger.info(f"  Step {step_idx+1}/{len(timesteps)}")
        
        # Save final structure
        final_rigid = trajectory[-1]
        frame_rigids = rigid_utils.Rigid.from_tensor_7(final_rigid)
        frame_ca_pos = frame_rigids.get_trans().cpu().numpy()
        frame_bb_pos = all_atom.compute_backbone(frame_ca_pos)
        
        sample_name = f"sample_{sample_idx+1}"
        save_structure(frame_bb_pos, args.output_dir, name=sample_name)
    
    # Save metadata
    metadata = {
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "cfg_scale": args.cfg_scale,
        "num_samples": args.num_samples,
        "embedding_path": args.embedding_path,
        "checkpoint_path": args.checkpoint_path,
        "sequence_length": args.sequence_length,
        "min_t": args.min_t,
        "max_t": args.max_t,
        "num_t": args.num_t,
    }
    
    with open(os.path.join(args.output_dir, "metadata.json"), 'w') as f:
        json.dump(metadata, f, indent=2)
    
    logger.info(f"Inference complete. Generated {args.num_samples} samples in {args.output_dir}")

if __name__ == "__main__":
    main()