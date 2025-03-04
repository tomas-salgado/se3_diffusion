"""Score network module."""
import torch
import math
from torch import nn
from torch.nn import functional as F
from data import utils as du
from data import all_atom
from model import ipa_pytorch
import functools as fn

Tensor = torch.Tensor


def get_index_embedding(indices, embed_size, max_len=2056):
    """Creates sine / cosine positional embeddings from a prespecified indices.

    Args:
        indices: offsets of size [..., N_edges] of type integer
        max_len: maximum length.
        embed_size: dimension of the embeddings to create

    Returns:
        positional embedding of shape [N, embed_size]
    """
    K = torch.arange(embed_size//2, device=indices.device)
    pos_embedding_sin = torch.sin(
        indices[..., None] * math.pi / (max_len**(2*K[None]/embed_size))).to(indices.device)
    pos_embedding_cos = torch.cos(
        indices[..., None] * math.pi / (max_len**(2*K[None]/embed_size))).to(indices.device)
    pos_embedding = torch.cat([
        pos_embedding_sin, pos_embedding_cos], axis=-1)
    return pos_embedding


def get_timestep_embedding(timesteps, embedding_dim, max_positions=10000):
    # Code from https://github.com/hojonathanho/diffusion/blob/master/diffusion_tf/nn.py
    assert len(timesteps.shape) == 1
    timesteps = timesteps * max_positions
    half_dim = embedding_dim // 2
    emb = math.log(max_positions) / (half_dim - 1)
    emb = torch.exp(torch.arange(half_dim, dtype=torch.float32, device=timesteps.device) * -emb)
    emb = timesteps.float()[:, None] * emb[None, :]
    emb = torch.cat([torch.sin(emb), torch.cos(emb)], dim=1)
    if embedding_dim % 2 == 1:  # zero pad
        emb = F.pad(emb, (0, 1), mode='constant')
    assert emb.shape == (timesteps.shape[0], embedding_dim)
    return emb

class Embedder(nn.Module):

    def __init__(self, model_conf):
        super(Embedder, self).__init__()
        self._model_conf = model_conf
        self._embed_conf = model_conf.embed
        
        # Time step embedding
        index_embed_size = self._embed_conf.index_embed_size
        t_embed_size = index_embed_size
        node_embed_dims = t_embed_size + 1
        edge_in = (t_embed_size + 1) * 2

        # Sequence index embedding
        node_embed_dims += index_embed_size
        edge_in += index_embed_size
        
        # Add sequence embedding conditioning if enabled
        self.use_sequence_conditioning = False
        if hasattr(self._model_conf, 'use_sequence_conditioning') and self._model_conf.use_sequence_conditioning:
            self.use_sequence_conditioning = True
            # Add sequence embedding dimension to node features
            node_embed_dims += self._model_conf.sequence_embed.embed_dim

        node_embed_size = self._model_conf.node_embed_size
        self.node_embedder = nn.Sequential(
            nn.Linear(node_embed_dims, node_embed_size),
            nn.ReLU(),
            nn.Linear(node_embed_size, node_embed_size),
            nn.ReLU(),
            nn.Linear(node_embed_size, node_embed_size),
            nn.LayerNorm(node_embed_size),
        )

        if self._embed_conf.embed_self_conditioning:
            edge_in += self._embed_conf.num_bins
        edge_embed_size = self._model_conf.edge_embed_size
        self.edge_embedder = nn.Sequential(
            nn.Linear(edge_in, edge_embed_size),
            nn.ReLU(),
            nn.Linear(edge_embed_size, edge_embed_size),
            nn.ReLU(),
            nn.Linear(edge_embed_size, edge_embed_size),
            nn.LayerNorm(edge_embed_size),
        )

        self.timestep_embedder = fn.partial(
            get_timestep_embedding,
            embedding_dim=self._embed_conf.index_embed_size
        )
        self.index_embedder = fn.partial(
            get_index_embedding,
            embed_size=self._embed_conf.index_embed_size
        )
        
        # Sequence embedding conditioning
        if self.use_sequence_conditioning:
            # Create cross-attention for sequence conditioning
            if self._model_conf.conditioning_method == 'cross_attention':
                self.conditioning_attn = nn.MultiheadAttention(
                    embed_dim=node_embed_size,
                    num_heads=4,
                    batch_first=True
                )

    def _cross_concat(self, feats_1d, num_batch, num_res):
        return torch.cat([
            torch.tile(feats_1d[:, :, None, :], (1, 1, num_res, 1)),
            torch.tile(feats_1d[:, None, :, :], (1, num_res, 1, 1)),
        ], dim=-1).float().reshape([num_batch, num_res**2, -1])

    def forward(
            self,
            *,
            seq_idx,
            t,
            fixed_mask,
            self_conditioning_ca,
            sequence_embedding=None,
        ):
        """Embeds a set of inputs

        Args:
            seq_idx: [..., N] Positional sequence index for each residue.
            t: Sampled t in [0, 1].
            fixed_mask: mask of fixed (motif) residues.
            self_conditioning_ca: [..., N, 3] Ca positions of self-conditioning
                input.
            sequence_embedding: Optional sequence embedding for conditioning.

        Returns:
            node_embed: [B, N, D_node]
            edge_embed: [B, N, N, D_edge]
            
        Key CFG components:
        1. During training:
           - Randomly drop embeddings with probability cfg_dropout_prob
           - When dropped, use zero embedding tensor
           - Target conformations come from pretrained model outputs
           
        2. During inference:
           - If cfg_scale is provided, run both conditioned and unconditioned
           - Interpolate between outputs using cfg_scale
           - cfg_scale=0 gives unconditioned output
           - cfg_scale>0 strengthens conditioning (typical values: 3-7)
        """
        num_batch, num_res = seq_idx.shape
        node_feats = []

        # Set time step to epsilon=1e-5 for fixed residues.
        fixed_mask = fixed_mask[..., None]
        prot_t_embed = torch.tile(
            self.timestep_embedder(t)[:, None, :], (1, num_res, 1))
        prot_t_embed = torch.cat([prot_t_embed, fixed_mask], dim=-1)
        node_feats = [prot_t_embed]
        pair_feats = [self._cross_concat(prot_t_embed, num_batch, num_res)]

        # Positional index features.
        node_feats.append(self.index_embedder(seq_idx))
        rel_seq_offset = seq_idx[:, :, None] - seq_idx[:, None, :]
        rel_seq_offset = rel_seq_offset.reshape([num_batch, num_res**2])
        pair_feats.append(self.index_embedder(rel_seq_offset))
        
        # Add sequence embedding for conditioning if provided
        if self.use_sequence_conditioning and sequence_embedding is not None:
            # During training, randomly drop embeddings with cfg_dropout_prob
            if self.training and torch.rand(1) < self._model_conf.cfg_dropout_prob:
                sequence_embedding = torch.zeros_like(sequence_embedding)
            
            # Expand sequence embedding to match residue dimension
            seq_embed_expanded = sequence_embedding.unsqueeze(1).expand(-1, num_res, -1)
            node_feats.append(seq_embed_expanded)

        # Self-conditioning distogram.
        if self._embed_conf.embed_self_conditioning:
            sc_dgram = du.calc_distogram(
                self_conditioning_ca,
                self._embed_conf.min_bin,
                self._embed_conf.max_bin,
                self._embed_conf.num_bins,
            )
            pair_feats.append(sc_dgram.reshape([num_batch, num_res**2, -1]))

        node_embed = self.node_embedder(torch.cat(node_feats, dim=-1).float())
        edge_embed = self.edge_embedder(torch.cat(pair_feats, dim=-1).float())
        edge_embed = edge_embed.reshape([num_batch, num_res, num_res, -1])
        
        # Apply conditioning via cross-attention if enabled
        if self.use_sequence_conditioning and sequence_embedding is not None and self._model_conf.conditioning_method == 'cross_attention':
            # Use sequence embedding as query for cross-attention
            seq_embed_query = sequence_embedding.unsqueeze(1)  # [B, 1, D]
            node_embed_context = node_embed  # [B, N, D]
            
            # Apply cross-attention
            attn_output, _ = self.conditioning_attn(
                query=seq_embed_query,
                key=node_embed_context,
                value=node_embed_context
            )
            
            # Use attention output to modulate node embeddings
            attn_output = attn_output.expand(-1, num_res, -1)
            node_embed = node_embed + attn_output
            
        return node_embed, edge_embed


class ScoreNetwork(nn.Module):

    def __init__(self, model_conf, diffuser):
        super(ScoreNetwork, self).__init__()
        self._model_conf = model_conf

        self.embedding_layer = Embedder(model_conf)
        self.diffuser = diffuser
        self.score_model = ipa_pytorch.IpaScore(model_conf, diffuser)
        
        # Initialize sequence embedder if conditioning is enabled
        self.use_sequence_conditioning = False
        if hasattr(self._model_conf, 'use_sequence_conditioning') and self._model_conf.use_sequence_conditioning:
            self.use_sequence_conditioning = True
            from model.sequence_embedder import SequenceEmbedder
            self.sequence_embedder = SequenceEmbedder(model_conf)
            
            # Add dropout for classifier-free guidance
            self.cfg_dropout_prob = self._model_conf.cfg_dropout_prob if hasattr(self._model_conf, 'cfg_dropout_prob') else 0.1

    def _apply_mask(self, aatype_diff, aatype_0, diff_mask):
        return diff_mask * aatype_diff + (1 - diff_mask) * aatype_0

    def forward(self, input_feats, cfg_scale=None):
        """Forward pass with optional classifier-free guidance"""
        bb_mask = input_feats['res_mask'].type(torch.float32)
        fixed_mask = input_feats['fixed_mask'].type(torch.float32)
        edge_mask = bb_mask[..., None] * bb_mask[..., None, :]
        
        # Process sequence embeddings if conditioning is enabled
        sequence_embedding = None
        if self.use_sequence_conditioning:
            if 'sequence' in input_feats:
                sequence_embedding = self.sequence_embedder(
                    input_feats['sequence'], 
                    input_feats['t'].device
                )
                
                # During training, randomly drop embeddings for CFG
                if self.training and torch.rand(1) < self.cfg_dropout_prob:
                    sequence_embedding = torch.zeros_like(sequence_embedding)
                
                # During inference with CFG, run both conditioned and unconditioned
                if not self.training and cfg_scale is not None:
                    # Unconditioned forward pass
                    uncond_embedding = torch.zeros_like(sequence_embedding)
                    uncond_node_embed, uncond_edge_embed = self.embedding_layer(
                        seq_idx=input_feats['seq_idx'],
                        t=input_feats['t'],
                        fixed_mask=fixed_mask,
                        self_conditioning_ca=input_feats['sc_ca_t'],
                        sequence_embedding=uncond_embedding,
                    )
                    uncond_out = self.score_model(
                        uncond_node_embed * bb_mask[..., None],
                        uncond_edge_embed * edge_mask[..., None],
                        input_feats
                    )
                    
                    # Conditioned forward pass
                    cond_node_embed, cond_edge_embed = self.embedding_layer(
                        seq_idx=input_feats['seq_idx'],
                        t=input_feats['t'],
                        fixed_mask=fixed_mask,
                        self_conditioning_ca=input_feats['sc_ca_t'],
                        sequence_embedding=sequence_embedding,
                    )
                    cond_out = self.score_model(
                        cond_node_embed * bb_mask[..., None],
                        cond_edge_embed * edge_mask[..., None],
                        input_feats
                    )
                    
                    # Interpolate between conditioned and unconditioned outputs
                    for key in cond_out:
                        if isinstance(cond_out[key], torch.Tensor):
                            cond_out[key] = uncond_out[key] + cfg_scale * (cond_out[key] - uncond_out[key])
                    
                    return cond_out

        # Regular forward pass (either training or non-CFG inference)
        init_node_embed, init_edge_embed = self.embedding_layer(
            seq_idx=input_feats['seq_idx'],
            t=input_feats['t'],
            fixed_mask=fixed_mask,
            self_conditioning_ca=input_feats['sc_ca_t'],
            sequence_embedding=sequence_embedding,
        )
        edge_embed = init_edge_embed * edge_mask[..., None]
        node_embed = init_node_embed * bb_mask[..., None]
        
        model_out = self.score_model(node_embed, edge_embed, input_feats)
        
        # Psi angle prediction
        gt_psi = input_feats['torsion_angles_sin_cos'][..., 2, :]
        psi_pred = self._apply_mask(
            model_out['psi'], gt_psi, 1 - fixed_mask[..., None])

        pred_out = {
            'psi': psi_pred,
            'rot_score': model_out['rot_score'],
            'trans_score': model_out['trans_score'],
        }
        rigids_pred = model_out['final_rigids']
        pred_out['rigids'] = rigids_pred.to_tensor_7()
        bb_representations = all_atom.compute_backbone(rigids_pred, psi_pred)
        pred_out['atom37'] = bb_representations[0].to(rigids_pred.device)
        pred_out['atom14'] = bb_representations[-1].to(rigids_pred.device)
        return pred_out
