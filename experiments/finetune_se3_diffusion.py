"""Pytorch script for finetuning SE(3) protein diffusion on MD data.

To run:
> python experiments/finetune_se3_diffusion.py

Without Wandb:
> python experiments/finetune_se3_diffusion.py experiment.use_wandb=False
"""
import os
import hydra
from omegaconf import DictConfig
from experiments.train_se3_diffusion import Experiment
from data.pdb_data_loader import MDEnhancedPdbDataset
from data import pdb_data_loader

class MDFineTuningExperiment(Experiment):
    """Extends base Experiment class for MD fine-tuning."""
    
    def create_dataset(self):
        """Override dataset creation to use MD trajectory."""
        train_dataset = MDEnhancedPdbDataset(
            data_conf=self._data_conf,
            diffuser=self._diffuser,
            is_training=True,
            md_trajectory_path=self._data_conf.md_trajectory_path
        )
        
        # Use a portion of MD data for validation
        valid_dataset = MDEnhancedPdbDataset(
            data_conf=self._data_conf,
            diffuser=self._diffuser,
            is_training=False,
            md_trajectory_path=self._data_conf.md_trajectory_path
        )

        train_sampler = pdb_data_loader.TrainSampler(
            data_conf=self._data_conf,
            dataset=train_dataset,
            batch_size=self._exp_conf.batch_size,
            sample_mode='time_batch'
        )

        return self._create_loaders(train_dataset, valid_dataset, train_sampler)



@hydra.main(version_base=None, config_path="../config", config_name="finetune_ar")
def run(conf: DictConfig) -> None:
    os.environ["WANDB_START_METHOD"] = "thread"
    exp = MDFineTuningExperiment(conf=conf)
    exp.start_training()


if __name__ == '__main__':
    run() 