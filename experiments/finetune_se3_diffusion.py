"""Pytorch script for finetuning SE(3) protein diffusion on IDP conformational ensembles.

To run:
> python experiments/finetune_se3_diffusion.py

Without Wandb:
> python experiments/finetune_se3_diffusion.py experiment.use_wandb=False
"""
import os
import hydra
from omegaconf import DictConfig
from experiments.train_se3_diffusion import Experiment
from data.pdb_data_loader import IDPEnsembleDataset
from data import pdb_data_loader
from data import utils as du
import torch

class IDPEnsembleFineTuningExperiment(Experiment):
    """Extends base Experiment class for fine-tuning on IDP conformational ensembles.
    
    Supports both experimental (multi-frame PDB) and computational (XTC+topology) ensemble formats.
    """
    
    def create_dataset(self):
        """Override dataset creation to use IDP ensemble data."""
        train_dataset = IDPEnsembleDataset(
            data_conf=self._data_conf,
            diffuser=self._diffuser,
            is_training=True,
            pdb_path=self._data_conf.pdb_path if hasattr(self._data_conf, 'pdb_path') else None,
            xtc_path=self._data_conf.xtc_path if hasattr(self._data_conf, 'xtc_path') else None,
            top_path=self._data_conf.top_path if hasattr(self._data_conf, 'top_path') else None
        )
        
        # Use a portion of conformers for validation
        valid_dataset = IDPEnsembleDataset(
            data_conf=self._data_conf,
            diffuser=self._diffuser,
            is_training=False,
            pdb_path=self._data_conf.pdb_path if hasattr(self._data_conf, 'pdb_path') else None,
            xtc_path=self._data_conf.xtc_path if hasattr(self._data_conf, 'xtc_path') else None,
            top_path=self._data_conf.top_path if hasattr(self._data_conf, 'top_path') else None
        )

        train_sampler = pdb_data_loader.TrainSampler(
            data_conf=self._data_conf,
            dataset=train_dataset,
            batch_size=self._exp_conf.batch_size,
            sample_mode='time_batch'
        )

        # Use create_data_loader instead of DataLoader directly
        train_loader = du.create_data_loader(
            train_dataset,
            sampler=train_sampler,
            np_collate=False,
            length_batch=True,
            batch_size=self._exp_conf.batch_size,
            shuffle=False,
            num_workers=self._exp_conf.num_loader_workers,
            drop_last=False,
            max_squared_res=self._exp_conf.max_squared_res,
        )

        valid_loader = du.create_data_loader(
            valid_dataset,
            sampler=None,
            np_collate=False,
            length_batch=False,
            batch_size=self._exp_conf.batch_size,
            shuffle=False,
            num_workers=self._exp_conf.num_loader_workers,
            drop_last=False,
        )

        return train_loader, valid_loader, train_sampler, None


@hydra.main(version_base=None, config_path="../config", config_name="finetune_ar")
def run(conf: DictConfig) -> None:
    os.environ["WANDB_START_METHOD"] = "thread"
    exp = IDPEnsembleFineTuningExperiment(conf=conf)
    exp.start_training()


if __name__ == '__main__':
    run() 