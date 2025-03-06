"""Script for running classifier-free guidance (CFG) inference.

This script implements inference with classifier-free guidance for the SE(3) diffusion model.
It allows generating protein structures conditioned on sequence embeddings with various guidance scales.

Sample command:
> python experiments/inference_cfg_se3_diffusion.py cfg_scale=7.5 embedding_path=embeddings/p15_idr_embedding.txt

"""

import os
import time
import tree
import numpy as np
import hydra
import torch
import logging
import json
from pathlib import Path
from typing import Dict, Optional, List, Tuple

from analysis import utils as au
from data import utils as du
from data import residue_constants
from data import so3_diffuser
from data import all_atom
from experiments import train_se3_diffusion
from omegaconf import DictConfig, OmegaConf
from openfold.utils import rigid_utils
from model import score_network
from data import se3_diffuser


CA_IDX = residue_constants.atom_order['CA']


class CFGSampler:
    """Sampler that implements classifier-free guidance for protein structure generation."""

    def __init__(
        self,
        conf: DictConfig,
        conf_overrides: Dict = None
    ):
        """Initialize the CFG sampler.
        
        Args:
            conf: The configuration object.
            conf_overrides: Dictionary of configuration overrides.
        """
        self._log = logging.getLogger(__name__)
        self._conf = OmegaConf.merge(conf, conf_overrides or {})
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Load model
        self._load_model()
        
        # Create diffuser
        self._diffuser = so3_diffuser.SE3Diffuser(
            self._conf.diffuser,
            device=self.device,
        )
        
        # Set up inference parameters
        self.min_t = self._conf.inference.min_t
        self.max_t = self._conf.inference.max_t
        self.num_t = self._conf.inference.num_t
        self.cfg_scale = self._conf.inference.cfg_scale

    def _load_model(self):
        """Load the model from a checkpoint."""
        self._log.info(f"Loading model checkpoint from {self._conf.inference.checkpoint_path}")
        checkpoint = torch.load(self._conf.inference.checkpoint_path, map_location=self.device)
        
        # Log available keys in checkpoint
        self._log.info(f"Checkpoint keys: {list(checkpoint.keys())}")
        
        # Store original inference settings
        inference_conf = self._conf.inference
        
        # Try to load configuration from checkpoint if available
        if 'conf' in checkpoint:
            self._log.info("Using model configuration from checkpoint")
            from omegaconf import OmegaConf
            
            # Get the checkpoint configuration
            checkpoint_conf = checkpoint['conf']
            
            # Log some key configuration values for debugging
            self._log.info(f"Checkpoint config model.node_embed_size: {checkpoint_conf.model.node_embed_size}")
            if hasattr(checkpoint_conf.model, 'sequence_embed'):
                self._log.info(f"Checkpoint config model.sequence_embed.embed_dim: {checkpoint_conf.model.sequence_embed.embed_dim}")
            
            # Create a merged configuration
            # Start with the inference configuration
            merged_conf = OmegaConf.create(OmegaConf.to_container(self._conf))
            
            # Update model, diffuser, and data sections from checkpoint
            merged_conf.model = checkpoint_conf.model
            merged_conf.diffuser = checkpoint_conf.diffuser
            if hasattr(checkpoint_conf, 'data'):
                merged_conf.data = checkpoint_conf.data
            
            # Use the merged configuration
            self._conf = merged_conf
        else:
            self._log.warning("Checkpoint does not contain configuration, using current config")
            self._log.warning("This might cause dimension mismatches if your current config differs from training config!")
        
        # Create model components with the configuration
        from model import score_network
        from data import se3_diffuser
        
        # Create diffuser
        self._diffuser = se3_diffuser.SE3Diffuser(self._conf.diffuser)
        
        # Create the model
        self._model = score_network.ScoreNetwork(
            self._conf.model, self._diffuser)
        
        # Load the state dict with proper handling
        if 'model_state_dict' in checkpoint:
            model_state = checkpoint['model_state_dict']
        elif 'model' in checkpoint:
            model_state = checkpoint['model']
        else:
            raise KeyError(f"No model weights found in checkpoint. Available keys: {list(checkpoint.keys())}")
        
        # Handle DataParallel saved models
        model_state = {k.replace('module.', ''):v for k,v in model_state.items()}
        
        # Load the state dict
        self._model.load_state_dict(model_state)
        
        self._model.to(self.device)
        self._model.eval()
        
        self._log.info(f"Model loaded successfully")

    def _load_embedding(self, embedding_path: str) -> torch.Tensor:
        """Load the embedding from a file.
        
        Args:
            embedding_path: Path to the embedding file.
            
        Returns:
            The embedding tensor.
        """
        self._log.info(f"Loading embedding from {embedding_path}")
        
        if embedding_path.endswith('.txt'):
            # Read embedding from text file (one float per line)
            with open(embedding_path, 'r') as f:
                embedding_text = f.read().strip()
                # Handle different formatting possibilities
                if '[' in embedding_text:
                    # JSON-like format
                    embedding_text = embedding_text.replace('[', '').replace(']', '')
                
                # Split by commas if present, otherwise by whitespace/newlines
                if ',' in embedding_text:
                    values = [float(x.strip()) for x in embedding_text.split(',') if x.strip()]
                else:
                    values = [float(x.strip()) for x in embedding_text.split() if x.strip()]
                
                embedding = torch.tensor(values, dtype=torch.float32)
        
        elif embedding_path.endswith('.pt') or embedding_path.endswith('.pth'):
            # Load PyTorch tensor directly
            embedding = torch.load(embedding_path, map_location=self.device)
        
        elif embedding_path.endswith('.json'):
            # Load from JSON file
            with open(embedding_path, 'r') as f:
                values = json.load(f)
                embedding = torch.tensor(values, dtype=torch.float32)
        
        else:
            raise ValueError(f"Unsupported embedding file format: {embedding_path}")
        
        self._log.info(f"Loaded embedding with shape {embedding.shape}")
        return embedding

    def _prepare_sample_inputs(self, length: int) -> Dict:
        """Prepare inputs for the sampling process.
        
        Args:
            length: The length of the protein to generate.
            
        Returns:
            Dictionary of model inputs.
        """
        # Create residue mask
        res_mask = torch.ones(length, dtype=torch.float32, device=self.device)
        
        # Create sequence indices (1-indexed)
        seq_idx = torch.arange(1, length + 1, dtype=torch.long, device=self.device)
        
        # Create fixed mask (all zeros for full generation)
        fixed_mask = torch.zeros(length, dtype=torch.float32, device=self.device)
        
        # Return the inputs
        return {
            'res_mask': res_mask.unsqueeze(0),  # Add batch dimension
            'seq_idx': seq_idx.unsqueeze(0),
            'fixed_mask': fixed_mask.unsqueeze(0),
        }

    def sample_with_cfg(self, embedding: torch.Tensor, length: int) -> Tuple[np.ndarray, List[np.ndarray]]:
        """Sample a protein structure with classifier-free guidance.
        
        Args:
            embedding: The embedding tensor to condition on.
            length: The length of the protein to generate.
            
        Returns:
            Tuple of (final structure, list of intermediate structures).
        """
        self._log.info(f"Generating protein with length {length} and CFG scale {self.cfg_scale}")
        
        # Prepare inputs
        inputs = self._prepare_sample_inputs(length)
        
        # Move embedding to device and ensure it has batch dimension
        embedding = embedding.to(self.device)
        if embedding.dim() == 1:
            embedding = embedding.unsqueeze(0)  # [1, embed_dim]
        
        # Compute timesteps
        ts = torch.linspace(self.max_t, self.min_t, self.num_t, device=self.device)
        
        # Initialize noise
        rigids_0 = rigid_utils.Rigid.identity(
            num_rigids=length, 
            device=self.device,
            requires_grad=False,
        )
        
        # Apply forward diffusion to get noised rigids at max_t
        diffusion_out = self._diffuser.forward_marginal(
            rigids_0=rigids_0,
            t=self.max_t,
            as_tensor_7=True,
        )
        
        rigids_t = torch.tensor(diffusion_out['rigids_t'], device=self.device).unsqueeze(0)
        
        # Initialize trajectory
        traj = [rigids_t.clone().cpu().numpy()]
        
        # Sampling loop
        for i in range(self.num_t - 1):
            t_i = ts[i]
            t_next = ts[i + 1]
            
            # Prepare model inputs
            curr_inputs = {**inputs}
            curr_inputs['rigids'] = rigids_t
            curr_inputs['t'] = torch.tensor([t_i], device=self.device)
            
            with torch.no_grad():
                # Run the model with conditioning
                curr_inputs['sequence'] = embedding
                cond_out = self._model(curr_inputs)
                
                # Run the model without conditioning (empty embedding)
                curr_inputs['sequence'] = torch.zeros_like(embedding)
                uncond_out = self._model(curr_inputs)
                
                # Apply classifier-free guidance
                rot_score = uncond_out['rot_score'] + self.cfg_scale * (cond_out['rot_score'] - uncond_out['rot_score'])
                trans_score = uncond_out['trans_score'] + self.cfg_scale * (cond_out['trans_score'] - uncond_out['trans_score'])
            
            # Use the diffuser to take a step
            rigids_t, _ = self._diffuser.reverse_sample(
                rigids_t=rigids_t.squeeze(0),
                rot_score=rot_score.squeeze(0),
                trans_score=trans_score.squeeze(0),
                t=t_i,
                dt=t_i - t_next,
                mask=inputs['res_mask'].squeeze(0),
            )
            
            # Add batch dimension back
            rigids_t = rigids_t.unsqueeze(0)
            
            # Add to trajectory
            traj.append(rigids_t.clone().cpu().numpy())
            
            if (i + 1) % 10 == 0:
                self._log.info(f"Sampling step {i + 1}/{self.num_t - 1}")
        
        # Convert final rigids to atom positions
        final_rigids = rigid_utils.Rigid.from_tensor_7(rigids_t.squeeze(0))
        ca_pos = final_rigids.get_trans().cpu().numpy()
        
        # Create full backbone using CA positions
        bb_pos = all_atom.compute_backbone(ca_pos)
        
        return bb_pos, traj

    def save_structure(self, positions: np.ndarray, output_path: str, name: str = "generated"):
        """Save the generated structure to a PDB file.
        
        Args:
            positions: The atomic positions.
            output_path: The directory to save the structure to.
            name: The name of the structure.
        """
        os.makedirs(output_path, exist_ok=True)
        
        # Create a mask for the atoms
        atom_mask = np.ones(positions.shape[:-1])
        
        # Create a dummy aatype (all alanines)
        aatype = np.zeros(positions.shape[0], dtype=np.int32)
        
        # Create a protein object
        protein = au.create_full_prot(positions, atom_mask, aatype)
        
        # Save as PDB
        pdb_path = os.path.join(output_path, f"{name}.pdb")
        with open(pdb_path, 'w') as f:
            f.write(au.to_pdb(protein))
        
        self._log.info(f"Saved structure to {pdb_path}")
        return pdb_path

    def run_inference(self, embedding_path: str, output_dir: str, num_samples: int = 10):
        """Run inference with the specified embedding.
        
        Args:
            embedding_path: Path to the embedding file.
            output_dir: Directory to save the generated structures.
            num_samples: Number of samples to generate.
        """
        # Load embedding
        embedding = self._load_embedding(embedding_path)
        
        # Determine protein length based on embedding name
        if 'p15' in embedding_path.lower():
            length = 110  # p15PAF length
            name_prefix = 'p15'
        elif 'ar' in embedding_path.lower():
            length = 56   # AR length
            name_prefix = 'ar'
        else:
            # Default to p15 length if can't determine
            length = 110
            name_prefix = 'unknown'
        
        # Create output directory
        path = Path(output_dir)
        path.mkdir(parents=True, exist_ok=True)
        
        # Save inference parameters
        with open(path / 'inference_params.json', 'w') as f:
            params = {
                'embedding_path': embedding_path,
                'cfg_scale': float(self.cfg_scale),
                'min_t': float(self.min_t),
                'max_t': float(self.max_t),
                'num_t': int(self.num_t),
                'num_samples': num_samples,
                'protein_length': length,
                'timestamp': time.strftime('%Y-%m-%d-%H-%M-%S')
            }
            json.dump(params, f, indent=2)
        
        # Generate samples
        for i in range(num_samples):
            self._log.info(f"Generating sample {i+1}/{num_samples}")
            
            # Sample with classifier-free guidance
            bb_pos, traj = self.sample_with_cfg(embedding, length)
            
            # Save the structure
            sample_name = f"{name_prefix}_cfg{self.cfg_scale:.1f}_sample{i+1}"
            self.save_structure(bb_pos, output_dir, name=sample_name)
            
            # Optionally save trajectory frames
            if self._conf.inference.save_trajectory:
                traj_dir = os.path.join(output_dir, f"{sample_name}_traj")
                os.makedirs(traj_dir, exist_ok=True)
                
                for j, frame in enumerate(traj):
                    # Convert rigid to atom positions
                    if j % self._conf.inference.traj_save_frequency == 0:
                        frame_rigids = rigid_utils.Rigid.from_tensor_7(frame.squeeze(0))
                        frame_ca_pos = frame_rigids.get_trans().cpu().numpy()
                        frame_bb_pos = all_atom.compute_backbone(frame_ca_pos)
                        
                        # Save the frame
                        frame_name = f"frame_{j:03d}"
                        self.save_structure(frame_bb_pos, traj_dir, name=frame_name)
        
        self._log.info(f"Inference complete. Generated {num_samples} samples in {output_dir}")


@hydra.main(version_base=None, config_path="../config", config_name="inference_cfg")
def main(conf: DictConfig) -> None:
    """Main function for CFG inference.
    
    Args:
        conf: The configuration object.
    """
    # Set up logging
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__name__)
    
    # Create the sampler
    conf_overrides = {
        'inference': {
            'cfg_scale': conf.cfg_scale,
            'num_samples': conf.num_samples,
        }
    }
    
    sampler = CFGSampler(conf, conf_overrides)
    
    # Run inference
    sampler.run_inference(
        embedding_path=conf.embedding_path,
        output_dir=conf.output_dir,
        num_samples=conf.num_samples
    )
    
    logger.info("Inference completed successfully!")


if __name__ == "__main__":
    main() 