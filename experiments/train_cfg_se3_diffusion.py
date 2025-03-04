"""Training script for classifier-free guidance with SE(3) diffusion."""

import os
import hydra
import logging
import torch
import numpy as np
from omegaconf import DictConfig
from torch.nn import functional as F
from data.idp_cfg_dataset import IDPCFGDataset
from experiments import train_se3_diffusion 

class CFGExperiment:
    """Extends base Experiment class for CFG training."""
    
    def __init__(self, conf: DictConfig):
        self.exp = train_se3_diffusion.Experiment(conf)
        self._conf = conf
        self._log = logging.getLogger(__name__)
        
    def train(self):
        """Delegate to base experiment's train method"""
        return self.exp.train()

    def train_step(self, batch):
        """Delegate to base experiment's train_step method"""
        return self.exp.train_step(batch)

    def _setup_data(self):
        """Override data setup to use IDPCFGDataset."""
        self._log.info("Setting up CFG datasets...")
        
        # Create training dataset
        self.train_dataset = IDPCFGDataset(
            p15_data_path=self._conf.data.p15_data_path,
            ar_data_path=self._conf.data.ar_data_path,
            p15_embedding_path=self._conf.model.sequence_embed.p15_embedding_path,
            ar_embedding_path=self._conf.model.sequence_embed.ar_embedding_path,
            cfg_dropout_prob=self._conf.model.cfg_dropout_prob
        )
        
        # Create data loader
        self.train_loader = torch.utils.data.DataLoader(
            self.train_dataset,
            batch_size=self._conf.training.batch_size,
            shuffle=True,
            num_workers=self._conf.experiment.num_loader_workers,
            prefetch_factor=self._conf.experiment.prefetch_factor,
        )
        
        self._log.info(
            f"Created dataset with {len(self.train_dataset)} samples "
            f"({len(self.train_dataset.p15_data)} p15, "
            f"{len(self.train_dataset.ar_data)} AR)"
        )

@hydra.main(version_base=None, config_path="../config", config_name="finetune_cfg")
def main(conf: DictConfig) -> None:
    """Main training function."""
    experiment = CFGExperiment(conf)
    experiment.train()

if __name__ == "__main__":
    main() 