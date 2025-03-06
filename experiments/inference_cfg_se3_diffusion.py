"""Script for running classifier-free guidance (CFG) inference.

This script implements inference with classifier-free guidance for the SE(3) diffusion model.
It allows generating protein structures conditioned on sequence embeddings with various guidance scales.

Sample command:
> python experiments/inference_cfg_se3_diffusion.py checkpoint_path=weights/checkpoint.pth embedding_path=embeddings/p15_idr_embedding.txt

"""

import os
import time
import logging
import argparse
import numpy as np
import torch
import hydra
import json
from pathlib import Path
from typing import Dict, Optional, List, Tuple
from datetime import datetime
from omegaconf import DictConfig, OmegaConf

from data import all_atom
from data import se3_diffuser
from data import residue_constants
from data import utils as du
from model import score_network
from openfold.utils import rigid_utils
from experiments import train_se3_diffusion

CA_IDX = residue_constants.atom_order['CA']


class CFGSampler:
    """Sampler that implements classifier-free guidance for protein structure generation."""

    def __init__(self, conf: DictConfig):
        """Initialize the CFG sampler.
        
        Args:
            conf: The configuration object.
        """
        self._log = logging.getLogger(__name__)
        self._conf = conf
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Create the diffuser first (same as in training)
        self._diffuser = se3_diffuser.SE3Diffuser(self._conf.diffuser, device=self.device)
        
        # Create and load the model (similar to training)
        self._load_model()
        
        # Set up inference parameters
        self.min_t = self._conf.inference.min_t
        self.max_t = self._conf.inference.max_t
        self.num_t = self._conf.inference.num_t
        self.cfg_scale = self._conf.inference.cfg_scale
        self.num_samples = self._conf.inference.num_samples

    def _load_model(self):
        """Load the model from a checkpoint, maintaining training configuration."""
        checkpoint_path = self._conf.inference.checkpoint_path
        self._log.info(f"Loading model checkpoint from {checkpoint_path}")
        
        # Load checkpoint
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        
        # Determine the appropriate state dict key
        if 'model_state_dict' in checkpoint:
            model_state = checkpoint['model_state_dict'] 
        elif 'model' in checkpoint:
            model_state = checkpoint['model']
        else:
            raise KeyError(f"No model weights found in checkpoint")
        
        # Handle DataParallel saved models
        model_state = {k.replace('module.', ''):v for k,v in model_state.items()}
        
        # Create the model with the same configuration as during training
        self._model = score_network.ScoreNetwork(self._conf.model, self._diffuser)
        
        # Load the state dict
        result = self._model.load_state_dict(model_state, strict=False)
        
        # Log any issues with loading
        if result.unexpected_keys:
            self._log.warning(f"Unexpected keys in checkpoint: {result.unexpected_keys}")
        if result.missing_keys:
            self._log.warning(f"Missing keys in model: {result.missing_keys}")
            
        self._model.to(self.device)
        self._model.eval()
        self._log.info(f"Model loaded successfully")

    def _load_embedding(self, embedding_path: str) -> torch.Tensor:
        """Load the sequence embedding from a file.
        
        Args:
            embedding_path: Path to the embedding file.
            
        Returns:
            torch.Tensor: The loaded embedding.
        """
        self._log.info(f"Loading embedding from {embedding_path}")
        
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
                
            self._log.info(f"Loaded embedding with shape {embedding.shape}")
            return embedding.to(self.device)
            
        except Exception as e:
            self._log.error(f"Failed to load embedding: {e}")
            raise

    def _sample_diffused_positions(
        self, 
        seq_embedding: torch.Tensor, 
        seq_length: int,
        num_samples: int = 1
    ) -> List[torch.Tensor]:
        """Sample positions using classifier-free guidance.
        
        Args:
            seq_embedding: The sequence embedding tensor.
            seq_length: The length of the protein sequence.
            num_samples: Number of samples to generate.
            
        Returns:
            List of generated trajectories.
        """
        # Setup timesteps
        timesteps = torch.linspace(
            self.max_t, self.min_t, self.num_t, device=self.device)
        
        trajectories = []
        
        for i in range(num_samples):
            self._log.info(f"Generating sample {i+1}/{num_samples}")
            
            # Sample from the reference distribution
            rigids = self._diffuser.sample_ref(
                n_samples=seq_length,
                as_tensor_7=True
            ).to(self.device)
            
            # Generate trajectory
            trajectory = [rigids.clone()]
            
            # Iterative denoising
            for j, t in enumerate(timesteps):
                t_batch = torch.full((1,), t, device=self.device)
                
                # Need to run twice for CFG - once with conditioning, once without
                with torch.no_grad():
                    # Run with conditioning
                    cond_output = self._model(
                        rigids, 
                        t_batch, 
                        seq_embedding=seq_embedding
                    )
                    
                    # Run without conditioning (null embedding)
                    uncond_output = self._model(
                        rigids, 
                        t_batch, 
                        seq_embedding=None
                    )
                    
                    # Combine outputs with guidance scale
                    rot_score = uncond_output['rot_score'] + self.cfg_scale * (
                        cond_output['rot_score'] - uncond_output['rot_score'])
                    trans_score = uncond_output['trans_score'] + self.cfg_scale * (
                        cond_output['trans_score'] - uncond_output['trans_score'])
                    
                    # Update with the score
                    rigids = self._diffuser.reverse_sample(
                        rigids_t=rigids,
                        rot_score=rot_score,
                        trans_score=trans_score,
                        t=t_batch,
                        dt=timesteps[0] - timesteps[-1] if j == 0 else timesteps[j-1] - timesteps[j]
                    )
                    
                # Save current state
                trajectory.append(rigids.clone())
                
                if (j + 1) % 10 == 0 or j == len(timesteps) - 1:
                    self._log.info(f"  Step {j+1}/{len(timesteps)}")
            
            trajectories.append(trajectory)
            
        return trajectories

    def save_structure(self, positions, output_dir, name="structure"):
        """Save the generated structure to a PDB file.
        
        Args:
            positions: The backbone atom positions.
            output_dir: Directory to save the output.
            name: Name of the output file.
        """
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
        
        self._log.info(f"Saved structure to {output_path}")
        return output_path

    def run_inference(self, embedding_path: str, output_dir: str, metadata: Optional[Dict] = None):
        """Run inference to generate protein structures.
        
        Args:
            embedding_path: Path to the sequence embedding file.
            output_dir: Directory to save the output.
            metadata: Optional metadata to save with the output.
        """
        self._log.info(f"Running inference with CFG scale {self.cfg_scale}")
        
        # Load the sequence embedding
        seq_embedding = self._load_embedding(embedding_path)
        
        # Determine sequence length (can be inferred from embedding or provided)
        seq_length = self._conf.inference.sequence_length if hasattr(self._conf.inference, 'sequence_length') else None
        if seq_length is None:
            # Try to infer from embedding properties or use a default
            seq_length = 100  # Default length if not specified
            self._log.warning(f"Sequence length not provided, using default: {seq_length}")
        
        # Create output directory
        os.makedirs(output_dir, exist_ok=True)
        
        # Save metadata
        run_metadata = {
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "cfg_scale": self.cfg_scale,
            "num_samples": self.num_samples,
            "embedding_path": embedding_path,
            "checkpoint_path": self._conf.inference.checkpoint_path,
            "sequence_length": seq_length,
            "min_t": self.min_t,
            "max_t": self.max_t,
            "num_t": self.num_t,
        }
        if metadata:
            run_metadata.update(metadata)
            
        with open(os.path.join(output_dir, "metadata.json"), 'w') as f:
            json.dump(run_metadata, f, indent=2)
        
        # Sample trajectories
        trajectories = self._sample_diffused_positions(
            seq_embedding, seq_length, self.num_samples)
        
        # Save the final structures
        for i, trajectory in enumerate(trajectories):
            final_rigid = trajectory[-1]
            frame_rigids = rigid_utils.Rigid.from_tensor_7(final_rigid)
            frame_ca_pos = frame_rigids.get_trans().cpu().numpy()
            frame_bb_pos = all_atom.compute_backbone(frame_ca_pos)
            
            sample_name = f"sample_{i+1}"
            self.save_structure(frame_bb_pos, output_dir, name=sample_name)
            
            # Optionally save trajectory frames if configured
            if self._conf.inference.save_trajectory:
                traj_dir = os.path.join(output_dir, f"trajectory_{i+1}")
                os.makedirs(traj_dir, exist_ok=True)
                
                # Save a subset of frames (e.g., every 10th)
                frames_to_save = range(0, len(trajectory), max(1, len(trajectory) // 10))
                for j in frames_to_save:
                    frame = trajectory[j]
                    frame_rigids = rigid_utils.Rigid.from_tensor_7(frame)
                    frame_ca_pos = frame_rigids.get_trans().cpu().numpy()
                    frame_bb_pos = all_atom.compute_backbone(frame_ca_pos)
                    
                    frame_name = f"frame_{j:03d}"
                    self.save_structure(frame_bb_pos, traj_dir, name=frame_name)
        
        self._log.info(f"Inference complete. Generated {self.num_samples} samples in {output_dir}")


@hydra.main(version_base=None, config_path="../config", config_name="inference_cfg")
def main(conf: DictConfig) -> None:
    """Main function for CFG inference.
    
    Args:
        conf: The configuration object.
    """
    # Set up logging
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__name__)
    
    # Update inference parameters from command line arguments if provided
    if hasattr(conf, 'cfg_scale'):
        conf.inference.cfg_scale = conf.cfg_scale
    if hasattr(conf, 'num_samples'):
        conf.inference.num_samples = conf.num_samples
    if hasattr(conf, 'checkpoint_path'):
        conf.inference.checkpoint_path = conf.checkpoint_path
    
    # Create the sampler and run inference
    sampler = CFGSampler(conf)
    
    sampler.run_inference(
        embedding_path=conf.embedding_path,
        output_dir=conf.output_dir
    )


if __name__ == "__main__":
    main() 