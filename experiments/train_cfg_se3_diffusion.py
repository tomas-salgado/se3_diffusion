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

class CFGExperiment:
    """Extends base Experiment class for CFG training."""
    
    def __init__(self, conf: DictConfig):
        # Disable wandb if no API key is available
        if 'WANDB_API_KEY' not in os.environ:
            conf.experiment.use_wandb = False
            logging.info("Wandb disabled - no API key found")
        
        self.exp = train_se3_diffusion.Experiment(conf=conf)
        self._conf = conf
        self._log = logging.getLogger(__name__)
        
        # Override the dataset creation method
        self.exp.create_dataset = self.create_dataset
    
    def create_dataset(self):
        """Create datasets for CFG training with both p15 and ar conditions"""
        # Get all PDB files for each condition
        p15_pdbs = glob.glob(os.path.join(self._conf.data.p15_data_path, "*.pdb"))
        ar_pdbs = glob.glob(os.path.join(self._conf.data.ar_data_path, "*.pdb"))
        
        self._log.info(f"Found {len(p15_pdbs)} P15 PDB files and {len(ar_pdbs)} AR PDB files")
        
        # Split into train/validation
        train_split = 0.8  # 80% for training
        p15_train = p15_pdbs[:int(len(p15_pdbs) * train_split)]
        p15_valid = p15_pdbs[int(len(p15_pdbs) * train_split):]
        ar_train = ar_pdbs[:int(len(ar_pdbs) * train_split)]
        ar_valid = ar_pdbs[int(len(ar_pdbs) * train_split):]
        
        # Create training datasets for both conditions
        train_datasets = []
        for pdb_path in p15_train + ar_train:
            dataset = IDPEnsembleDataset(
                data_conf=self._conf.data,
                diffuser=self.exp.diffuser,
                is_training=True,
                pdb_path=pdb_path
            )
            train_datasets.append(dataset)
        
        # Create validation datasets for both conditions
        valid_datasets = []
        for pdb_path in p15_valid + ar_valid:
            dataset = IDPEnsembleDataset(
                data_conf=self._conf.data,
                diffuser=self.exp.diffuser,
                is_training=False,
                pdb_path=pdb_path
            )
            valid_datasets.append(dataset)
        
        # Combine datasets for training and validation
        train_dataset = torch.utils.data.ConcatDataset(train_datasets)
        valid_dataset = torch.utils.data.ConcatDataset(valid_datasets)
        
        # Create data loaders
        train_loader = torch.utils.data.DataLoader(
            train_dataset,
            batch_size=self._conf.experiment.batch_size,
            shuffle=True,
            num_workers=self._conf.experiment.num_loader_workers
        )
        
        valid_loader = torch.utils.data.DataLoader(
            valid_dataset,
            batch_size=self._conf.experiment.eval_batch_size,
            shuffle=False,
            num_workers=self._conf.experiment.num_loader_workers
        )
        
        return train_loader, valid_loader, None, None

    def train(self):
        """Delegate to base experiment's train method"""
        return self.exp.start_training()

    def train_step(self, batch):
        """Delegate to base experiment's train_step method"""
        return self.exp.train_step(batch)

@hydra.main(version_base=None, config_path="../config", config_name="finetune_cfg")
def main(conf: DictConfig) -> None:
    """Main training function."""
    # Add some debug prints
    print(f"Config type: {type(conf)}")
    print(f"Config contents: {conf}")
    experiment = CFGExperiment(conf)
    experiment.train()

if __name__ == "__main__":
    main() 