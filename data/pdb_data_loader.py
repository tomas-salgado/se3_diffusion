"""PDB dataset loader."""
import math
from typing import Optional
from collections import defaultdict

import torch
import torch.distributed as dist

import tree
import numpy as np
import torch
import pandas as pd
import logging
import random
import functools as fn

from torch.utils import data
from data import utils as du
from openfold.data import data_transforms
from openfold.np import residue_constants
from openfold.utils import rigid_utils

from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression


def _rog_quantile_curve(df, quantile, eval_x):
    y_quant = pd.pivot_table(
        df,
        values='radius_gyration', 
        index='modeled_seq_len',
        aggfunc=lambda x: np.quantile(x, quantile)
    )
    x_quant = y_quant.index.to_numpy()
    y_quant = y_quant.radius_gyration.to_numpy()

    # Fit polynomial regressor
    poly = PolynomialFeatures(degree=4, include_bias=True)
    poly_features = poly.fit_transform(x_quant[:, None])
    poly_reg_model = LinearRegression()
    poly_reg_model.fit(poly_features, y_quant)

    # Calculate cutoff for all sequence lengths
    pred_poly_features = poly.fit_transform(eval_x[:, None])
    # Add a little more.
    pred_y = poly_reg_model.predict(pred_poly_features) + 0.1
    return pred_y

class PdbDataset(data.Dataset):
    def __init__(
            self,
            *,
            data_conf,
            diffuser,
            is_training,
        ):
        self._log = logging.getLogger(__name__)
        self._is_training = is_training
        self._data_conf = data_conf
        self._init_metadata()
        self._diffuser = diffuser

    @property
    def is_training(self):
        return self._is_training

    @property
    def diffuser(self):
        return self._diffuser

    @property
    def data_conf(self):
        return self._data_conf

    def _init_metadata(self):
        """Initialize metadata."""

        # Process CSV with different filtering criterions.
        filter_conf = self.data_conf.filtering
        pdb_csv = pd.read_csv(self.data_conf.csv_path)
        self.raw_csv = pdb_csv
        if filter_conf.allowed_oligomer is not None and len(filter_conf.allowed_oligomer) > 0:
            pdb_csv = pdb_csv[pdb_csv.oligomeric_detail.isin(
                filter_conf.allowed_oligomer)]
        if filter_conf.max_len is not None:
            pdb_csv = pdb_csv[pdb_csv.modeled_seq_len <= filter_conf.max_len]
        if filter_conf.min_len is not None:
            pdb_csv = pdb_csv[pdb_csv.modeled_seq_len >= filter_conf.min_len]
        if filter_conf.max_helix_percent is not None:
            pdb_csv = pdb_csv[
                pdb_csv.helix_percent < filter_conf.max_helix_percent]
        if filter_conf.max_loop_percent is not None:
            pdb_csv = pdb_csv[
                pdb_csv.coil_percent < filter_conf.max_loop_percent]
        if filter_conf.min_beta_percent is not None:
            pdb_csv = pdb_csv[
                pdb_csv.strand_percent > filter_conf.min_beta_percent]
        if filter_conf.rog_quantile is not None \
            and filter_conf.rog_quantile > 0.0:
            prot_rog_low_pass = _rog_quantile_curve(
                pdb_csv, 
                filter_conf.rog_quantile,
                np.arange(filter_conf.max_len))
            row_rog_cutoffs = pdb_csv.modeled_seq_len.map(
                lambda x: prot_rog_low_pass[x-1])
            pdb_csv = pdb_csv[pdb_csv.radius_gyration < row_rog_cutoffs]
        if filter_conf.subset is not None:
            pdb_csv = pdb_csv[:filter_conf.subset]
        pdb_csv = pdb_csv.sort_values('modeled_seq_len', ascending=False)
        self._create_split(pdb_csv)

    def _create_split(self, pdb_csv):
        # Training or validation specific logic.
        if self.is_training:
            self.csv = pdb_csv
            self._log.info(
                f'Training: {len(self.csv)} examples')
        else:
            all_lengths = np.sort(pdb_csv.modeled_seq_len.unique())
            length_indices = (len(all_lengths) - 1) * np.linspace(
                0.0, 1.0, self._data_conf.num_eval_lengths)
            length_indices = length_indices.astype(int)
            eval_lengths = all_lengths[length_indices]
            eval_csv = pdb_csv[pdb_csv.modeled_seq_len.isin(eval_lengths)]
            # Fix a random seed to get the same split each time.
            eval_csv = eval_csv.groupby('modeled_seq_len').sample(
                self._data_conf.samples_per_eval_length, replace=True, random_state=123)
            eval_csv = eval_csv.sort_values('modeled_seq_len', ascending=False)
            self.csv = eval_csv
            self._log.info(
                f'Validation: {len(self.csv)} examples with lengths {eval_lengths}')
    # cache make the same sample in same batch 
    @fn.lru_cache(maxsize=100)
    def _process_csv_row(self, processed_file_path):
        processed_feats = du.read_pkl(processed_file_path)
        processed_feats = du.parse_chain_feats(processed_feats)

        # Only take modeled residues.
        modeled_idx = processed_feats['modeled_idx']
        min_idx = np.min(modeled_idx)
        max_idx = np.max(modeled_idx)
        del processed_feats['modeled_idx']
        processed_feats = tree.map_structure(
            lambda x: x[min_idx:(max_idx+1)], processed_feats)

        # Run through OpenFold data transforms.
        chain_feats = {
            'aatype': torch.tensor(processed_feats['aatype']).long(),
            'all_atom_positions': torch.tensor(processed_feats['atom_positions']).double(),
            'all_atom_mask': torch.tensor(processed_feats['atom_mask']).double()
        }
        chain_feats = data_transforms.atom37_to_frames(chain_feats)
        chain_feats = data_transforms.make_atom14_masks(chain_feats)
        chain_feats = data_transforms.make_atom14_positions(chain_feats)
        chain_feats = data_transforms.atom37_to_torsion_angles()(chain_feats)

        # Re-number residue indices for each chain such that it starts from 1.
        # Randomize chain indices.
        chain_idx = processed_feats["chain_index"]
        res_idx = processed_feats['residue_index']
        new_res_idx = np.zeros_like(res_idx)
        new_chain_idx = np.zeros_like(res_idx)
        all_chain_idx = np.unique(chain_idx).tolist()
        shuffled_chain_idx = np.array(
            random.sample(all_chain_idx, len(all_chain_idx))) - np.min(all_chain_idx) + 1
        for i,chain_id in enumerate(all_chain_idx):
            chain_mask = (chain_idx == chain_id).astype(int)
            chain_min_idx = np.min(res_idx + (1 - chain_mask) * 1e3).astype(int)
            new_res_idx = new_res_idx + (res_idx - chain_min_idx + 1) * chain_mask

            # Shuffle chain_index
            replacement_chain_id = shuffled_chain_idx[i]
            new_chain_idx = new_chain_idx + replacement_chain_id * chain_mask

        # To speed up processing, only take necessary features
        final_feats = {
            'aatype': chain_feats['aatype'],
            'seq_idx': new_res_idx,
            'chain_idx': new_chain_idx,
            'residx_atom14_to_atom37': chain_feats['residx_atom14_to_atom37'],
            'residue_index': processed_feats['residue_index'],
            'res_mask': processed_feats['bb_mask'],
            'atom37_pos': chain_feats['all_atom_positions'],
            'atom37_mask': chain_feats['all_atom_mask'],
            'atom14_pos': chain_feats['atom14_gt_positions'],
            'rigidgroups_0': chain_feats['rigidgroups_gt_frames'],
            'torsion_angles_sin_cos': chain_feats['torsion_angles_sin_cos'],
        }
        return final_feats

    def _create_diffused_masks(self, atom37_pos, rng, row):
        bb_pos = atom37_pos[:, residue_constants.atom_order['CA']]
        dist2d = np.linalg.norm(bb_pos[:, None, :] - bb_pos[None, :, :], axis=-1)

        # Randomly select residue then sample a distance cutoff
        # TODO: Use a more robust diffuse mask sampling method.
        diff_mask = np.zeros_like(bb_pos)
        attempts = 0
        while np.sum(diff_mask) < 1:
            crop_seed = rng.integers(dist2d.shape[0])
            seed_dists = dist2d[crop_seed]
            max_scaffold_size = min(
                self._data_conf.scaffold_size_max,
                seed_dists.shape[0] - self._data_conf.motif_size_min
            )
            scaffold_size = rng.integers(
                low=self._data_conf.scaffold_size_min,
                high=max_scaffold_size
            )
            dist_cutoff = np.sort(seed_dists)[scaffold_size]
            diff_mask = (seed_dists < dist_cutoff).astype(float)
            attempts += 1
            if attempts > 100:
                raise ValueError(
                    f'Unable to generate diffusion mask for {row}')
        return diff_mask

    def __len__(self):
        return len(self.csv)

    def __getitem__(self, idx):

        # Sample data example.
        example_idx = idx
        csv_row = self.csv.iloc[example_idx]
        if 'pdb_name' in csv_row:
            pdb_name = csv_row['pdb_name']
        elif 'chain_name' in csv_row:
            pdb_name = csv_row['chain_name']
        else:
            raise ValueError('Need chain identifier.')
        processed_file_path = csv_row['processed_path']
        chain_feats = self._process_csv_row(processed_file_path)

        # Use a fixed seed for evaluation.
        if self.is_training:
            rng = np.random.default_rng(None)
        else:
            rng = np.random.default_rng(idx)

        gt_bb_rigid = rigid_utils.Rigid.from_tensor_4x4(
            chain_feats['rigidgroups_0'])[:, 0]
        diffused_mask = np.ones_like(chain_feats['res_mask'])
        if np.sum(diffused_mask) < 1:
            raise ValueError('Must be diffused')
        fixed_mask = 1 - diffused_mask
        chain_feats['fixed_mask'] = fixed_mask
        chain_feats['rigids_0'] = gt_bb_rigid.to_tensor_7()
        chain_feats['sc_ca_t'] = torch.zeros_like(gt_bb_rigid.get_trans())

        # Sample t and diffuse.
        if self.is_training:
            t = rng.uniform(self._data_conf.min_t, 1.0)
            diff_feats_t = self._diffuser.forward_marginal(
                rigids_0=gt_bb_rigid,
                t=t,
                diffuse_mask=None
            )
        else:
            t = 1.0
            diff_feats_t = self.diffuser.sample_ref(
                n_samples=gt_bb_rigid.shape[0],
                impute=gt_bb_rigid,
                diffuse_mask=None,
                as_tensor_7=True,
            )
        chain_feats.update(diff_feats_t)
        chain_feats['t'] = t

        # Convert all features to tensors.
        final_feats = tree.map_structure(
            lambda x: x if torch.is_tensor(x) else torch.tensor(x), chain_feats)
        final_feats = du.pad_feats(final_feats, csv_row['modeled_seq_len'])
        if self.is_training:
            return final_feats
        else:
            return final_feats, pdb_name


class TrainSampler(data.Sampler):

    def __init__(
            self,
            *,
            data_conf,
            dataset,
            batch_size,
            sample_mode,
        ):
        self._log = logging.getLogger(__name__)
        self._data_conf = data_conf
        self._dataset = dataset
        self._data_csv = self._dataset.csv
        self._dataset_indices = list(range(len(self._data_csv)))
        self._data_csv['index'] = self._dataset_indices
        self._batch_size = batch_size
        self.epoch = 0
        self._sample_mode = sample_mode
        self.sampler_len = len(self._dataset_indices) * self._batch_size

        if self._sample_mode in ['cluster_length_batch', 'cluster_time_batch']:
            self._pdb_to_cluster = self._read_clusters()
            self._max_cluster = max(self._pdb_to_cluster.values())
            self._log.info(f'Read {self._max_cluster} clusters.')
            self._missing_pdbs = 0
            def cluster_lookup(pdb):
                pdb = pdb.upper()
                if pdb not in self._pdb_to_cluster:
                    self._pdb_to_cluster[pdb] = self._max_cluster + 1
                    self._max_cluster += 1
                    self._missing_pdbs += 1
                return self._pdb_to_cluster[pdb]
            self._data_csv['cluster'] = self._data_csv['pdb_name'].map(cluster_lookup)
            num_clusters = len(set(self._data_csv['cluster']))
            self.sampler_len = num_clusters * self._batch_size
            self._log.info(
                f'Training on {num_clusters} clusters. PDBs without clusters: {self._missing_pdbs}'
            )

    def _read_clusters(self):
        pdb_to_cluster = {}
        with open(self._data_conf.cluster_path, "r") as f:
            for i,line in enumerate(f):
                for chain in line.split(' '):
                    pdb = chain.split('_')[0]
                    pdb_to_cluster[pdb.upper()] = i
        return pdb_to_cluster

    def __iter__(self):
        if self._sample_mode == 'length_batch':
            # Each batch contains multiple proteins of the same length.
            sampled_order = self._data_csv.groupby('modeled_seq_len').sample(
                self._batch_size, replace=True, random_state=self.epoch)
            return iter(sampled_order['index'].tolist())
        elif self._sample_mode == 'time_batch':
            # Each batch contains multiple time steps of the same protein.
            random.shuffle(self._dataset_indices)
            repeated_indices = np.repeat(self._dataset_indices, self._batch_size)
            return iter(repeated_indices)
        elif self._sample_mode == 'cluster_length_batch':
            # Each batch contains multiple clusters of the same length.
            sampled_clusters = self._data_csv.groupby('cluster').sample(
                1, random_state=self.epoch)
            sampled_order = sampled_clusters.groupby('modeled_seq_len').sample(
                self._batch_size, replace=True, random_state=self.epoch)
            return iter(sampled_order['index'].tolist())
        elif self._sample_mode == 'cluster_time_batch':
            # Each batch contains multiple time steps of a protein from a cluster.
            sampled_clusters = self._data_csv.groupby('cluster').sample(
                1, random_state=self.epoch)
            dataset_indices = sampled_clusters['index'].tolist()
            repeated_indices = np.repeat(dataset_indices, self._batch_size)
            return iter(repeated_indices.tolist())
        else:
            raise ValueError(f'Invalid sample mode: {self._sample_mode}')

    def set_epoch(self, epoch):
        self.epoch = epoch

    def __len__(self):
        return self.sampler_len


# modified from torch.utils.data.distributed.DistributedSampler
# key points: shuffle of each __iter__ is determined by epoch num to ensure the same shuffle result for each proccessor
class DistributedTrainSampler(data.Sampler):
    r"""Sampler that restricts data loading to a subset of the dataset.

    modified from torch.utils.data.distributed import DistributedSampler

    .. note::
        Dataset is assumed to be of constant size and that any instance of it always
        returns the same elements in the same order.

    Args:
        dataset: Dataset used for sampling.
        num_replicas (int, optional): Number of processes participating in
            distributed training. By default, :attr:`world_size` is retrieved from the
            current distributed group.
        rank (int, optional): Rank of the current process within :attr:`num_replicas`.
            By default, :attr:`rank` is retrieved from the current distributed
            group.
        shuffle (bool, optional): If ``True`` (default), sampler will shuffle the
            indices.
        seed (int, optional): random seed used to shuffle the sampler if
            :attr:`shuffle=True`. This number should be identical across all
            processes in the distributed group. Default: ``0``.
        drop_last (bool, optional): if ``True``, then the sampler will drop the
            tail of the data to make it evenly divisible across the number of
            replicas. If ``False``, the sampler will add extra indices to make
            the data evenly divisible across the replicas. Default: ``False``.

    .. warning::
        In distributed mode, calling the :meth:`set_epoch` method at
        the beginning of each epoch **before** creating the :class:`DataLoader` iterator
        is necessary to make shuffling work properly across multiple epochs. Otherwise,
        the same ordering will be always used.

    Example::

        >>> # xdoctest: +SKIP
        >>> sampler = DistributedSampler(dataset) if is_distributed else None
        >>> loader = DataLoader(dataset, shuffle=(sampler is None),
        ...                     sampler=sampler)
        >>> for epoch in range(start_epoch, n_epochs):
        ...     if is_distributed:
        ...         sampler.set_epoch(epoch)
        ...     train(loader)
    """

    def __init__(self, 
                *,
                data_conf,
                dataset,
                batch_size,
                num_replicas: Optional[int] = None,
                rank: Optional[int] = None, shuffle: bool = True,
                seed: int = 0, drop_last: bool = False) -> None:
        if num_replicas is None:
            if not dist.is_available():
                raise RuntimeError("Requires distributed package to be available")
            num_replicas = dist.get_world_size()
        if rank is None:
            if not dist.is_available():
                raise RuntimeError("Requires distributed package to be available")
            rank = dist.get_rank()
        if rank >= num_replicas or rank < 0:
            raise ValueError(
                "Invalid rank {}, rank should be in the interval"
                " [0, {}]".format(rank, num_replicas - 1))
        self._data_conf = data_conf
        self._dataset = dataset
        self._data_csv = self._dataset.csv
        self._dataset_indices = list(range(len(self._data_csv)))
        self._data_csv['index'] = self._dataset_indices
        # _repeated_size is the size of the dataset multiply by batch size
        self._repeated_size = batch_size * len(self._data_csv)
        self._batch_size = batch_size
        self.num_replicas = num_replicas
        self.rank = rank
        self.epoch = 0
        self.drop_last = drop_last
        # If the dataset length is evenly divisible by # of replicas, then there
        # is no need to drop any data, since the dataset will be split equally.
        if self.drop_last and self._repeated_size % self.num_replicas != 0:  # type: ignore[arg-type]
            # Split to nearest available length that is evenly divisible.
            # This is to ensure each rank receives the same amount of data when
            # using this Sampler.
            self.num_samples = math.ceil(
                (self._repeated_size - self.num_replicas) / self.num_replicas  # type: ignore[arg-type]
            )
        else:
            self.num_samples = math.ceil(self._repeated_size / self.num_replicas)  # type: ignore[arg-type]
        self.total_size = self.num_samples * self.num_replicas
        self.shuffle = shuffle
        self.seed = seed

    def __iter__(self) :
        if self.shuffle:
            # deterministically shuffle based on epoch and seed
            g = torch.Generator()
            g.manual_seed(self.seed + self.epoch)
            indices = torch.randperm(len(self._data_csv), generator=g).tolist()  # type: ignore[arg-type]
        else:
            indices = list(range(len(self._data_csv)))  # type: ignore[arg-type]

        # indices is expanded by self._batch_size times
        indices = np.repeat(indices, self._batch_size)
        
        if not self.drop_last:
            # add extra samples to make it evenly divisible
            padding_size = self.total_size - len(indices)
            if padding_size <= len(indices):
                indices = np.concatenate((indices, indices[:padding_size]), axis=0)
            else:
                indices = np.concatenate((indices, np.repeat(indices, math.ceil(padding_size / len(indices)))[:padding_size]), axis=0)

        else:
            # remove tail of data to make it evenly divisible.
            indices = indices[:self.total_size]
        assert len(indices) == self.total_size

        # subsample
        indices = indices[self.rank:self.total_size:self.num_replicas]
        # 
        assert len(indices) == self.num_samples

        return iter(indices)

    def __len__(self) -> int:
        return self.num_samples

    def set_epoch(self, epoch: int) -> None:
        r"""
        Sets the epoch for this sampler. When :attr:`shuffle=True`, this ensures all replicas
        use a different random ordering for each epoch. Otherwise, the next iteration of this
        sampler will yield the same ordering.

        Args:
            epoch (int): Epoch number.
        """
        self.epoch = epoch

class IDPEnsembleDataset(PdbDataset):
    """Dataset for fine-tuning using IDP conformational ensembles.
    
    This dataset can load IDP conformational ensembles from either:
    1. A multi-frame PDB file (e.g. from NMR or other experimental sources)
    2. An XTC trajectory with its topology file (e.g. from MD simulations)
    """
    
    def __init__(self, data_conf, diffuser, is_training, pdb_path=None, xtc_path=None, top_path=None):
        # Basic initialization
        self._log = logging.getLogger(__name__)
        self._is_training = is_training
        self._data_conf = data_conf
        self._diffuser = diffuser
        
        # Validate input format
        if pdb_path is not None and (xtc_path is not None or top_path is not None):
            raise ValueError("Provide either pdb_path OR (xtc_path + top_path), not both")
        if (xtc_path is not None and top_path is None) or (xtc_path is None and top_path is not None):
            raise ValueError("Both xtc_path and top_path must be provided together")
        if pdb_path is None and xtc_path is None:
            raise ValueError("Must provide either pdb_path or (xtc_path + top_path)")
        
        # Load conformational ensemble based on provided format
        import mdtraj as md
        if pdb_path is not None:
            self._log.info(f"Loading multi-frame PDB ensemble from {pdb_path}")
            self.trajectory = md.load(pdb_path)
            self.topology = self.trajectory.topology
        else:
            self._log.info(f"Loading topology from {top_path}")
            self.topology = md.load(top_path).topology
            self._log.info(f"Loading MD trajectory from {xtc_path}")
            self.trajectory = md.load(xtc_path, top=self.topology)
            
        # Pre-select backbone atoms
        self.backbone_indices = []
        residues = defaultdict(dict)
        for atom in self.topology.atoms:
            if atom.name in ['N', 'CA', 'C']:
                residues[atom.residue.index][atom.name] = atom.index
        
        # Only use residues that have all three atoms
        self.complete_residues = []
        for res_idx, atoms in residues.items():
            if all(name in atoms for name in ['N', 'CA', 'C']):
                self.complete_residues.append(res_idx)
                self.backbone_indices.extend([atoms['N'], atoms['CA'], atoms['C']])
        
        self.n_residues = len(self.complete_residues)
        if self.n_residues == 0:
            raise ValueError("No complete residues found with all backbone atoms!")
            
        self._log.info(f"Found {self.n_residues} residues with complete backbone atoms")
        
        # Get sequence information
        try:
            self.aatype = []
            for res_idx in self.complete_residues:
                residue = self.topology.residue(res_idx)
                # Convert 3-letter code to 1-letter code, then to index
                res_shortname = residue_constants.restype_3to1.get(residue.name, 'X')
                restype_idx = residue_constants.restype_order.get(
                    res_shortname, residue_constants.restype_num)
                self.aatype.append(restype_idx)
            self.aatype = np.array(self.aatype)
        except Exception as e:
            self._log.warning(f"Could not extract sequence information: {str(e)}")
            self.aatype = np.zeros(self.n_residues, dtype=np.int64)
        
        # Create metadata for the frames
        self._init_metadata()

    def _init_metadata(self):
        """Create metadata with train/validation split."""
        n_frames = self.trajectory.n_frames
        self._log.info(f"Found {n_frames} conformers in ensemble")
        
        # Create DataFrame with all frames
        self.csv = pd.DataFrame({
            'pdb_name': [f'conformer_{i}' for i in range(n_frames)],
            'modeled_seq_len': self.n_residues,
            'frame_idx': range(n_frames)
        })
        
        # Split for validation if not training
        if not self.is_training:
            n_eval = self._data_conf.num_eval_lengths * self._data_conf.samples_per_eval_length
            step = len(self.csv) // n_eval
            eval_indices = np.arange(0, len(self.csv), step)[:n_eval]
            self.csv = self.csv.iloc[eval_indices]
        
        self._log.info(f"{'Training' if self.is_training else 'Validation'}: {len(self.csv)} conformers with {self.n_residues} residues")

    def __getitem__(self, idx):
        """Get a single frame and prepare it for training."""
        frame_idx = self.csv.iloc[idx]['frame_idx']
        
        # Get coordinates for this frame
        frame_coords = self.trajectory.xyz[frame_idx][self.backbone_indices]
        # Convert to angstroms if needed (XTC is in nm, PDB already in angstroms)
        if not hasattr(self, '_is_pdb'):
            self._is_pdb = self.trajectory.unitcell_lengths is None
        if not self._is_pdb:
            frame_coords = frame_coords * 10  # Convert nm to angstroms
        
        # Initialize features needed by the model
        chain_feats = {
            'aatype': torch.tensor(self.aatype).long(),
            'seq_idx': torch.arange(1, self.n_residues + 1),  # 1-based indexing
            'chain_idx': torch.ones(self.n_residues),  # Single chain
            'res_mask': torch.ones(self.n_residues),
            'atom37_pos': torch.zeros(self.n_residues, 37, 3),
            'atom37_mask': torch.zeros(self.n_residues, 37),
            'torsion_angles_sin_cos': torch.zeros(self.n_residues, 7, 2),  # Placeholder
        }
        
        # Fill in backbone atoms (N, CA, C)
        for i in range(self.n_residues):
            # Add backbone atoms to their correct positions in atom37
            chain_feats['atom37_pos'][i, 0] = torch.from_numpy(frame_coords[i * 3 + 0])  # N
            chain_feats['atom37_pos'][i, 1] = torch.from_numpy(frame_coords[i * 3 + 1])  # CA
            chain_feats['atom37_pos'][i, 2] = torch.from_numpy(frame_coords[i * 3 + 2])  # C
            
            # Set mask for backbone atoms
            chain_feats['atom37_mask'][i, 0:3] = 1.0  # Mask for N, CA, C

        # Calculate rigid body transforms from backbone atoms
        gt_bb_rigid = rigid_utils.Rigid.from_3_points(
            chain_feats['atom37_pos'][:, 0],  # N
            chain_feats['atom37_pos'][:, 1],  # CA
            chain_feats['atom37_pos'][:, 2],  # C
        )
        
        # Add diffusion-specific features
        chain_feats['rigids_0'] = gt_bb_rigid.to_tensor_7()
        chain_feats['fixed_mask'] = torch.zeros(self.n_residues)
        chain_feats['sc_ca_t'] = torch.zeros(self.n_residues, 3)

        # Add noise according to diffusion schedule
        if self.is_training:
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
            diff_feats = self.diffuser.sample_ref(
                n_samples=gt_bb_rigid.shape[0],
                impute=gt_bb_rigid,
                diffuse_mask=None,
                as_tensor_7=True,
            )
            chain_feats.update(diff_feats)
            chain_feats['t'] = t
        
        if self.is_training:
            return chain_feats
        else:
            return chain_feats, self.csv.iloc[idx]['pdb_name']

def calc_dihedral(p1, p2, p3, p4):
    """Calculate dihedral angle between 4 points."""
    b1 = p2 - p1
    b2 = p3 - p2
    b3 = p4 - p3
    
    n1 = np.cross(b1, b2)
    n2 = np.cross(b2, b3)
    
    m1 = np.cross(n1, b2/np.linalg.norm(b2))
    
    x = np.dot(n1, n2)
    y = np.dot(m1, n2)
    
    return np.arctan2(y, x)
