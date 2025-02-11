"""Pytorch script for finetuning SE(3) protein diffusion on Androgen Receptor data.

To run:
> python experiments/finetune_ar.py

Without Wandb:
> python experiments/finetune_ar.py experiment.use_wandb=False
"""
import os
import hydra
import torch
import numpy as np
from omegaconf import DictConfig
from data.pdb_data_loader import MDEnhancedPdbDataset
from experiments.train_se3_diffusion import MDFineTuningExperiment
from data import pdb_data_loader
from data import utils as du


class ARFineTuningExperiment(MDFineTuningExperiment):
    """Extends MDFineTuningExperiment class specifically for Androgen Receptor finetuning."""
    
    def create_dataset(self):
        """Override dataset creation to use AR trajectory data."""
        # Load and validate AR data
        ar_data_path = self._data_conf.md_trajectory_path
        if not os.path.exists(ar_data_path):
            raise ValueError(f"AR trajectory file not found at: {ar_data_path}")
        
        try:
            ar_data = np.load(ar_data_path)
            self._log.info(f"Loaded AR trajectory with keys: {ar_data.files}")
            self._log.info(f"Positions shape: {ar_data['positions'].shape}")
        except Exception as e:
            raise ValueError(f"Failed to load AR trajectory: {str(e)}")

        # Create datasets with AR trajectory
        train_dataset = MDEnhancedPdbDataset(
            data_conf=self._data_conf,
            diffuser=self._diffuser,
            is_training=True,
            md_trajectory_path=ar_data_path
        )
        
        valid_dataset = MDEnhancedPdbDataset(
            data_conf=self._data_conf,
            diffuser=self._diffuser,
            is_training=False,
            md_trajectory_path=ar_data_path
        )

        if not self._use_ddp:
            train_sampler = pdb_data_loader.TrainSampler(
                data_conf=self._data_conf,
                dataset=train_dataset,
                batch_size=self._exp_conf.batch_size,
                sample_mode='time_batch'  # Using time_batch mode for MD data
            )
        else:
            train_sampler = pdb_data_loader.DistributedTrainSampler(
                data_conf=self._data_conf,
                dataset=train_dataset,
                batch_size=self._exp_conf.batch_size,
            )
        valid_sampler = None

        # Create data loaders
        train_loader = du.create_data_loader(
            train_dataset,
            sampler=train_sampler,
            np_collate=False,
            length_batch=True,
            batch_size=self._exp_conf.batch_size if not self._use_ddp else self._exp_conf.batch_size // self.ddp_info['world_size'],
            shuffle=False,
            num_workers=self._exp_conf.num_loader_workers,
            drop_last=False,
            max_squared_res=self._exp_conf.max_squared_res,
        )
        valid_loader = du.create_data_loader(
            valid_dataset,
            sampler=valid_sampler,
            np_collate=False,
            length_batch=False,
            batch_size=self._exp_conf.eval_batch_size,
            shuffle=False,
            num_workers=0,
            drop_last=False,
        )
        
        self._log.info(f"Created AR dataset with {len(train_dataset)} training frames")
        self._log.info(f"Validation set has {len(valid_dataset)} frames")
        
        return train_loader, valid_loader, train_sampler, valid_sampler


@hydra.main(version_base=None, config_path="../config", config_name="finetune_ar")
def run(conf: DictConfig) -> None:
    os.environ["WANDB_START_METHOD"] = "thread"
    exp = ARFineTuningExperiment(conf=conf)
    exp.start_training()


if __name__ == '__main__':
    run() 