import torch
import torch.nn as nn
from model._abstract_model import SequentialRecModel
from model._modules import LayerNorm, TransformerEncoder
from torch.nn import Parameter
import copy
from model._modules import LayerNorm, FeedForward, MultiHeadAttention
import matplotlib
import matplotlib.pyplot as plt
import numpy as np

import os

matplotlib.use('Agg')  # GUI不要な描画バックエンドに切り替え
"""
[Paper]
Author: Ruihong Qiu et al.
Title: "Contrastive Learning for Representation Degeneration Problem in Sequential Recommendation."
Conference: WSDM 2022

[Code Reference]
https://github.com/RuihongQiu/DuoRec
"""

class SICLRecModel(SequentialRecModel):
    def __init__(self, args):
        super(SICLRecModel, self).__init__(args)
        self.args = args
        self.item_embeddings = nn.Embedding(args.item_size, args.hidden_size, padding_idx=0)
        self.position_embeddings = nn.Embedding(args.max_seq_length, args.hidden_size)
        self.LayerNorm = LayerNorm(args.hidden_size, eps=1e-12)
        self.dropout = nn.Dropout(args.hidden_dropout_prob)
        self.item_encoder = BSARecEncoder(args)
        self.batch_size = args.batch_size
        self.gamma = 1e-10

        self.aug_nce_fct = nn.CrossEntropyLoss()
        self.mask_default = self.mask_correlated_samples(batch_size=self.batch_size)
        self.tau = args.tau
        self.ssl = args.ssl
        self.sim = args.sim
        self.lmd_sem = args.lmd_sem
        self.lmd = args.lmd
        self.fredom = args.fredom
        self.fredom_type = args.fredom_type
        self.bce_loss = nn.BCEWithLogitsLoss()
        

        self.apply(self.init_weights)
        

    @staticmethod
    def mean_cosine_similarity(seq_emb, aug_emb):
        seq_n = seq_emb / (seq_emb.norm(dim=1, keepdim=True) + 1e-8)
        aug_n = aug_emb / (aug_emb.norm(dim=1, keepdim=True) + 1e-8)
        cos = (seq_n * aug_n).sum(dim=1)
        return cos.mean().item()


    def info_nce(self, z_i, z_j, temp, batch_size, sim='dot'):
        """
        We do not sample negative examples explicitly.
        Instead, given a positive pair, similar to (Chen et al., 2017), we treat the other 2(N − 1) augmented examples within a minibatch as negative examples.
        """
        N = 2 * batch_size

        z = torch.cat((z_i, z_j), dim=0)
        # z = z[:, -1, :]
    
        if sim == 'cos':
            sim = nn.functional.cosine_similarity(z.unsqueeze(1), z.unsqueeze(0), dim=2) / temp
        elif sim == 'dot':
            sim = torch.mm(z, z.T) / temp
    
        sim_i_j = torch.diag(sim, batch_size)
        sim_j_i = torch.diag(sim, -batch_size)
    
        positive_samples = torch.cat((sim_i_j, sim_j_i), dim=0).reshape(N, 1)
        if batch_size != self.batch_size:
            mask = self.mask_correlated_samples(batch_size)
        else:
            mask = self.mask_default
        negative_samples = sim[mask].reshape(N, -1)
    
        labels = torch.zeros(N).to(positive_samples.device).long()
        logits = torch.cat((positive_samples, negative_samples), dim=1)
        return logits, labels

    def forward(self, input_ids, user_ids=None, all_sequence_output=False,freq_flag:bool = False, mask_flag: bool = False,epoch:int = 0):
        extended_attention_mask = self.get_attention_mask(input_ids)
        sequence_emb = self.add_position_embedding(input_ids)

        item_encoded_layers = self.item_encoder(sequence_emb,
                                                extended_attention_mask,
                                                freq_flag=freq_flag,
                                                mask_flag=mask_flag,
                                                output_all_encoded_layers=True,
                                                epoch=epoch,
                                                )
        if all_sequence_output:
            sequence_output = item_encoded_layers
        else:
            sequence_output = item_encoded_layers[-1]

        return sequence_output

    def calculate_loss(self, input_ids, answers, neg_answers, same_target, user_ids, epoch:int):
        seq_output = self.forward(input_ids,freq_flag=False,mask_flag=False, epoch=epoch)
        seq_output = seq_output[:, -1, :]

        #! start binary cross-entropy loss
        pos_ids, neg_ids = answers, neg_answers
        pos_emb = self.item_embeddings(pos_ids)      # [batch, hidden]
        neg_emb = self.item_embeddings(neg_ids)      # [batch, hidden]

        # Sequence embedding
        seq_emb = seq_output                        # [batch, hidden]

        # Compute logits
        pos_logits = torch.sum(pos_emb * seq_emb, dim=-1)  # [batch]
        neg_logits = torch.sum(neg_emb * seq_emb, dim=-1)  # [batch]

        # Create labels
        pos_labels = torch.ones_like(pos_logits)
        neg_labels = torch.zeros_like(neg_logits)

        # Mask out padding positions (where pos_id == 0)
        valid_idx = (pos_ids != 0)

        # Compute BCE loss on valid positions
        loss = self.bce_loss(pos_logits[valid_idx], pos_labels[valid_idx])
        loss += self.bce_loss(neg_logits[valid_idx], neg_labels[valid_idx])
        #! end binary cross-entropy loss

        #! Unsupervised NCE: original vs noise
        if self.ssl in ['us', 'un']:
            aug_seq_output_noise = self.forward(input_ids, freq_flag=True,mask_flag=False, epoch=epoch)
            
            aug_seq_output_noise = aug_seq_output_noise[:, -1, :]
            cosine_sim = self.mean_cosine_similarity(seq_output, aug_seq_output_noise)
            print(f"Cosine Similarity between original and noise: {cosine_sim:.4f}")
            nce_logits, nce_labels = self.info_nce(seq_output, aug_seq_output_noise, temp=self.tau,
                                                batch_size=input_ids.shape[0], sim=self.sim)
            loss += self.lmd * self.aug_nce_fct(nce_logits, nce_labels)

        return loss


class BSARecEncoder(nn.Module):
    def __init__(self, args):
        super(BSARecEncoder, self).__init__()
        self.args = args
        block = BSARecBlock(args)
        self.blocks = nn.ModuleList([copy.deepcopy(block) for _ in range(args.num_hidden_layers)])

    def forward(self, hidden_states, attention_mask, freq_flag, mask_flag,output_all_encoded_layers=False,epoch:int = 0):
        all_encoder_layers = [hidden_states]
        for layer_module in self.blocks:
            hidden_states = layer_module(hidden_states, attention_mask,freq_flag,mask_flag,epoch=epoch)
            if output_all_encoded_layers:
                all_encoder_layers.append(hidden_states)
        if not output_all_encoded_layers:
            all_encoder_layers.append(hidden_states)
        return all_encoder_layers


class BSARecBlock(nn.Module):
    def __init__(self, args):
        super(BSARecBlock, self).__init__()
        self.layer = BSARecLayer(args)
        self.feed_forward = FeedForward(args)

    def forward(self, hidden_states, attention_mask, freq_flag, mask_flag, epoch:int = 0):
        layer_output = self.layer(hidden_states, attention_mask, freq_flag,mask_flag,epoch=epoch)
        return self.feed_forward(layer_output)


class BSARecLayer(nn.Module):
    def __init__(self, args):
        super(BSARecLayer, self).__init__()
        self.args = args
        self.filter_layer = FrequencyLayer(args)
        self.attention_layer = MultiHeadAttention(args)
        self.alpha = args.alpha

    def forward(self, input_tensor, attention_mask, freq_flag,mask_flag, epoch:int = 0):

        filtered_output = self.filter_layer(input_tensor, freq_flag, mask_flag, epoch=epoch)

        return filtered_output

class FrequencyLayer(nn.Module):
    def __init__(self, args,alpha=0.5,switch_epoch=0):
        super(FrequencyLayer, self).__init__()
        self.out_dropout = nn.Dropout(args.hidden_dropout_prob)
        self.LayerNorm = LayerNorm(args.hidden_size, eps=1e-12)
        len_sw = args.max_seq_length
        n_fourier_comp = len_sw // 2 + 1
        self.n_fourier_comp = n_fourier_comp
        self.weight = Parameter(torch.empty((n_fourier_comp, 2)))
        self.reset_parameters()
        # noise injection strength
        self.alpha = alpha
        # epoch threshold to switch noise mode
        self.temperature = args.temperature
        self.data_name = args.data_name
        self.noise_scope = "all"
        self.noise_scale = getattr(args, 'noise_scale', 0.1)


    def get_sampling(self, weight, bias=0.0):
        if self.training:
            bias = bias + 1e-4
            eps = (bias - (1 - bias)) * torch.rand(weight.size(), device=weight.device) + (1 - bias)
            gate = torch.log(eps) - torch.log(1 - eps)
            gate = (gate + weight) / self.temperature
            return torch.sigmoid(gate)
        else:
            return torch.sigmoid(weight)

    def reset_parameters(self):
        nn.init.normal_(self.weight, mean=0.0, std=0.10)

    def forward(self, input_tensor, freq_flag, mask_flag, epoch: int = 0):
        para = self.get_sampling(self.weight)
        self.para = para

            
        # compute noise scaling
        noise_para = self.weight.detach().clone() * -1
        noise_para[noise_para < max(0, noise_para[:, 0].mean())] = 0.0

        scaling = 1.0 / noise_para[:, 0][noise_para[:, 0] != 0].mean()

        batch, seq_len, hidden = input_tensor.shape
        x = torch.fft.rfft(input_tensor, dim=1, norm='ortho')


        if freq_flag:

            gate = para[:, 0]

            # deterministic noise_para -> sample-wise stochastic noise
            noise_std_vec = (noise_para[:, 0] * scaling * 0.1).clamp(min=0.0)  # [n_freq]
            # fallback if all zeros
            if (noise_std_vec == 0).all():
                noise_std_vec = torch.ones_like(noise_std_vec) * (0.01)

            # sample-wise gaussian noise, then clip mask into [0,1]
            noise = torch.randn(batch, self.n_fourier_comp, device=gate.device) * noise_std_vec.unsqueeze(0)
            mask = gate.unsqueeze(0) + noise  # [batch, n_freq]
            mask = torch.clamp(mask, 0.0, 1.0)
            x_ft = x * mask.unsqueeze(-1)

        else:
    
            # Apply the mask without noise scaling
            x_ft = x * para[:, 0].unsqueeze(-1)

        seq_fft = torch.fft.irfft(x_ft, n=seq_len, dim=1, norm='ortho')
        out = self.out_dropout(seq_fft)
        return self.LayerNorm(out + input_tensor)



