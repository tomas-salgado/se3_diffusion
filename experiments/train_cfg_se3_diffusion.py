"""Training script for classifier-free guidance with SE(3) diffusion."""

import os
import hydra
import logging
import torch
import numpy as np
from omegaconf import DictConfig
from torch.nn import functional as F
from data.pdb_data_loader import IDPEnsembleDataset
from experiments import train_se3_diffusion
import glob
from Bio import PDB
from data import residue_constants
from openfold.utils import rigid_utils
import warnings
from data.idp_cfg_dataset import IDPCFGDataset, LengthBasedBatchSampler

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class CFGExperiment:
    """Extends base Experiment class for CFG training."""
    
    def __init__(self, conf: DictConfig):
        # Disable wandb if no API key is available
        if 'WANDB_API_KEY' not in os.environ:
            conf.experiment.use_wandb = False
            logger.info("Wandb disabled - no API key found")
        
        self.exp = train_se3_diffusion.Experiment(conf=conf)
        self._conf = conf
        self._log = logging.getLogger(__name__)
        
        # Override the dataset creation method
        self.exp.create_dataset = self.create_dataset
        
        # Log configuration
        self._log.info("Training configuration:")
        self._log.info(f"- Batch size: {conf.experiment.batch_size}")
        self._log.info(f"- Number of workers: {conf.experiment.num_loader_workers}")
        self._log.info(f"- CFG dropout probability: {conf.model.cfg_dropout_prob}")
        self._log.info(f"- Using sequence conditioning: {conf.model.use_sequence_conditioning}")
        self._log.info(f"- Conditioning method: {conf.model.conditioning_method}")
    
    def create_dataset(self):
        """Create datasets for CFG training."""
        self._log.info("Creating datasets...")
        
        # Create dataset with both real structures and pretrained structures
        self._log.info("Initializing training dataset...")
        train_dataset = IDPCFGDataset(
            # Real protein structures
            p15_data_path=self._conf.data.p15_data_path,
            ar_data_path=self._conf.data.ar_data_path,
            # Pretrained inference structures
            pretrained_p15_path=self._conf.data.pretrained_p15_dir,
            pretrained_ar_path=self._conf.data.pretrained_ar_dir,
            # Embeddings for conditioning
            p15_embedding_path=self._conf.data.p15_embedding_path,
            ar_embedding_path=self._conf.data.ar_embedding_path,
            cfg_dropout_prob=self._conf.model.cfg_dropout_prob,
            is_training=True  # Training dataset
        )
        
        # Create batch sampler
        self._log.info("Creating batch sampler...")
        train_sampler = LengthBasedBatchSampler(
            dataset=train_dataset,
            batch_size=self._conf.experiment.batch_size,
            drop_last=True  # Drop incomplete batches for training
        )
        
        # Create data loader with custom sampler
        self._log.info("Creating data loader...")
        train_loader = torch.utils.data.DataLoader(
            train_dataset,
            batch_sampler=train_sampler,
            num_workers=self._conf.experiment.num_loader_workers,
            collate_fn=train_sampler.collate_fn  # Use custom collate function
        )
        
        # Similar for validation, but with smaller batch size and no dropping
        self._log.info("Initializing validation dataset...")
        valid_dataset = IDPCFGDataset(
            p15_data_path=self._conf.data.p15_data_path,
            ar_data_path=self._conf.data.ar_data_path,
            p15_embedding_path=self._conf.data.p15_embedding_path,
            ar_embedding_path=self._conf.data.ar_embedding_path,
            pretrained_p15_path=self._conf.data.pretrained_p15_dir,
            pretrained_ar_path=self._conf.data.pretrained_ar_dir,
            cfg_dropout_prob=0.0,  # No dropout during validation
            is_training=False  # Validation dataset
        )
        
        valid_sampler = LengthBasedBatchSampler(
            dataset=valid_dataset,
            batch_size=self._conf.experiment.eval_batch_size,
            drop_last=False
        )
        
        valid_loader = torch.utils.data.DataLoader(
            valid_dataset,
            batch_sampler=valid_sampler,
            num_workers=self._conf.experiment.num_loader_workers,
            collate_fn=valid_sampler.collate_fn  # Use custom collate function
        )
        
        self._log.info("Dataset creation complete")
        return train_loader, valid_loader, None, None

    def train(self):
        """Delegate to base experiment's train method"""
        self._log.info("Starting training...")
        return self.exp.start_training()

    def train_step(self, batch):
        """Delegate to base experiment's train_step method"""
        return self.exp.train_step(batch)

class CombinedIDPDataset(torch.utils.data.Dataset):
    """Dataset that handles both P15 and AR PDB files."""
    
    def __init__(self, data_conf, diffuser, p15_paths, ar_paths, is_training):
        self._log = logging.getLogger(__name__)
        self._is_training = is_training
        self._data_conf = data_conf
        self._diffuser = diffuser
        
        # Store paths and verify they exist
        self.p15_paths = p15_paths
        self.ar_paths = ar_paths
        self.all_paths = p15_paths + ar_paths
        self.conditions = ['p15'] * len(p15_paths) + ['ar'] * len(ar_paths)
        
        for path in self.all_paths:
            if not os.path.exists(path):
                raise ValueError(f"PDB file not found: {path}")
                
        self._log.info(f"Dataset initialized with {len(p15_paths)} P15 files and {len(ar_paths)} AR files")
    
    def __len__(self):
        return len(self.all_paths)
    
    def __getitem__(self, idx):
        pdb_path = self.all_paths[idx]
        condition = self.conditions[idx]
        
        # Use BioPython's PDB parser
        parser = PDB.PDBParser(QUIET=True)
        structure = parser.get_structure('protein', pdb_path)
        chain = list(structure[0].get_chains())[0]
        
        # Get sequence and number of residues
        sequence = []
        for residue in chain:
            if 'CA' in residue:  # Only include residues with CA atoms
                res_name = residue.get_resname()
                sequence.append(res_name)
        
        n_residues = len(sequence)
        aatype = np.array([
            residue_constants.restype_order.get(
                residue_constants.restype_3to1.get(res, 'X'), 
                residue_constants.restype_num
            ) for res in sequence
        ])
        
        # Initialize features
        chain_feats = {
            'aatype': torch.tensor(aatype).long(),
            'seq_idx': torch.arange(1, n_residues + 1),  # 1-based indexing
            'chain_idx': torch.ones(n_residues),  # Single chain
            'sequence': ''.join([residue_constants.restype_3to1.get(res, 'X') for res in sequence]),
            'res_mask': torch.ones(n_residues),
            'atom37_pos': torch.zeros(n_residues, 37, 3),
            'atom37_mask': torch.zeros(n_residues, 37),
            'torsion_angles_sin_cos': torch.zeros(n_residues, 7, 2),  # Placeholder
            'condition': condition  # Add condition label
        }
        
        # Fill in backbone atoms (N, CA, C, O)
        for i, residue in enumerate(chain):
            if 'N' in residue:
                chain_feats['atom37_pos'][i, 0] = torch.tensor(residue['N'].get_coord())
                chain_feats['atom37_mask'][i, 0] = 1.0
            if 'CA' in residue:
                chain_feats['atom37_pos'][i, 1] = torch.tensor(residue['CA'].get_coord())
                chain_feats['atom37_mask'][i, 1] = 1.0
            if 'C' in residue:
                chain_feats['atom37_pos'][i, 2] = torch.tensor(residue['C'].get_coord())
                chain_feats['atom37_mask'][i, 2] = 1.0
            if 'O' in residue:
                chain_feats['atom37_pos'][i, 4] = torch.tensor(residue['O'].get_coord())
                chain_feats['atom37_mask'][i, 4] = 1.0

        # Calculate rigid body transforms from backbone atoms
        gt_bb_rigid = rigid_utils.Rigid.from_3_points(
            chain_feats['atom37_pos'][:, 0],  # N
            chain_feats['atom37_pos'][:, 1],  # CA
            chain_feats['atom37_pos'][:, 2],  # C
        )
        
        # Add diffusion-specific features
        chain_feats['rigids_0'] = gt_bb_rigid.to_tensor_7()
        chain_feats['fixed_mask'] = torch.zeros(n_residues)
        chain_feats['sc_ca_t'] = torch.zeros(n_residues, 3)

        # Add noise according to diffusion schedule
        if self._is_training and self._diffuser is not None:
            t = np.random.uniform(self._data_conf.min_t, 1.0)
            diff_feats = self._diffuser.forward_marginal(
                rigids_0=gt_bb_rigid,
                t=t,
                diffuse_mask=None
            )
            chain_feats.update(diff_feats)
            chain_feats['t'] = t
        else:
            t = 1.0
            diff_feats = self._diffuser.sample_ref(
                n_samples=gt_bb_rigid.shape[0],
                impute=gt_bb_rigid,
                diffuse_mask=None,
                as_tensor_7=True,
            )
            chain_feats.update(diff_feats)
            chain_feats['t'] = t
        
        if self._is_training:
            return chain_feats
        else:
            return chain_feats, os.path.basename(pdb_path)

@hydra.main(version_base=None, config_path="../config", config_name="finetune_cfg")
def main(conf: DictConfig) -> None:
    """Main training function."""
    # Add some debug prints
    logger.info(f"Config type: {type(conf)}")
    logger.info(f"Config contents: {conf}")
    experiment = CFGExperiment(conf)
    experiment.train()

if __name__ == "__main__":
    main() 