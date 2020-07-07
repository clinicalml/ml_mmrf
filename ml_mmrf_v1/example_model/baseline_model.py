import torch 
import sys, math
import numpy as np
from base import Model
from pyro.distributions import Normal, Independent, Categorical, LogNormal
import torch.nn.init as init
import torch.nn as nn
from typing import List, Tuple
from torch import Tensor
import torch.jit as jit
import warnings 
from collections import namedtuple
from typing import List, Tuple
from torch.autograd import Variable

np.random.seed(0)

class FOMM(Model):
    def __init__(self, dim_hidden, dim_base, dim_data, dim_treat, mtype = 'linear', \
            C =0., reg_all = True, reg_type = 'l1', clock_ablation = False):
        super(FOMM, self).__init__()
        self.reg_all = reg_all
        self.reg_type= reg_type
        self.C     = C
        self.dh    = dim_hidden
        self.dbase = dim_base
        self.ddata = dim_data
        self.dtreat= dim_treat
        self.mtype = mtype 
        self.clock_ablation = clock_ablation
        if mtype   == 'linear':
            self.model_mu   = nn.Linear(dim_data, dim_data)
            self.model_sig  = nn.Linear(dim_treat+dim_base+dim_data, dim_data) 

    def p_X(self, X, A, B):
        base_cat = B[:,None,:].repeat(1, max(1, X.shape[1]-1), 1)
        if self.mtype == 'linear':
            p_x_mu   = self.model_mu(X[:,:-1,:])
        cat      = torch.cat([X[:,:-1,:], A[:,:-1,:], base_cat],-1)
        p_x_sig  = torch.nn.functional.softplus(self.model_sig(cat))
        return p_x_mu, p_x_sig 
    
    def get_loss(self,B, X, A, M, Y, CE, anneal = 1., return_reconstruction = False):
        _, _, lens         = self.get_masks(M)
        B, X, A, M, Y, CE  = B[lens>1], X[lens>1], A[lens>1], M[lens>1], Y[lens>1], CE[lens>1]
        p_x_mu, p_x_std    = self.p_X(X, A, B)
        masked_nll = self.masked_gaussian_nll_3d(X[:,1:,:], p_x_mu, p_x_std, M[:,1:,:])
        full_masked_nll = masked_nll
        nll        = masked_nll.sum(-1).sum(-1)
        if return_reconstruction:
            return (nll, p_x_mu*M[:,1:,:], p_x_std*M[:,1:,:])
        else:
            return (nll,)
    
    def forward_unsupervised(self, B, X, A, M, Y, CE, anneal = 1.):
        if self.clock_ablation: 
            A = A[...,1:]
        (nll,)     = self.get_loss(B, X, A, M, Y, CE, anneal = anneal)
        reg_loss   = torch.mean(nll)
        for name,param in self.named_parameters():
            if self.reg_all:
                reg_loss += self.C*self.apply_reg(param, reg_type=self.reg_type)
            else:
                if 'weight' in name:
                    reg_loss += self.C*self.apply_reg(param, reg_type=self.reg_type)
        return (torch.mean(nll), torch.mean(nll), torch.tensor(0), torch.tensor(0)), torch.mean(reg_loss) 
    
    def sample(self, T_forward, X, A, B):
        with torch.no_grad():
            base           = B[:,None,:]
            obs_list       = [X[:,[0],:]]
            for t in range(1, T_forward):
                x_prev     = obs_list[-1]
                if self.mtype == 'linear': 
                    p_x_mu     = self.model_mu(torch.cat([x_prev, A[:,[t-1],:], base], -1))
                obs_list.append(p_x_mu) 
        return torch.cat(obs_list, 1)
    
    def inspect(self, T_forward, T_condition, B, X, A, M, Y, CE, restrict_lens = False):
        self.eval()
        if restrict_lens: 
            m_t, m_g_t, lens         = self.get_masks(M[:,1:,:])
            B, X, A, M, Y, CE  = B[lens>1], X[lens>1], A[lens>1], M[lens>1], Y[lens>1], CE[lens>1]
        p_x_mu, p_x_std = self.p_X(X, A, B)
        masked_nll      = self.masked_gaussian_nll_3d(X[:,1:,:], p_x_mu, p_x_std, M[:,1:,:])
        masked_nll_s    = masked_nll.sum(-1).sum(-1)
        nll             = torch.mean(masked_nll_s)
        per_feat_nll    = masked_nll.sum(1).mean(0)
        
        # calculate MSE instead
        mse = (((p_x_mu-X[:,1:])**2)*M[:,1:]).sum(0).sum(0)
        vals=M[:,1:].sum(0).sum(0)
        per_feat_nll   = mse/vals
        
        # Sample forward unconditionally
        inp_x      = self.sample(T_forward, X, A, B)
        inp_x_post = self.sample(T_forward+1, X[:,T_condition-1:], A[:,T_condition-1:], B)
        inp_x_post = torch.cat([X[:,:T_condition], inp_x_post[:,1:]], 1) 
        empty      = torch.ones(X.shape[0], 3)
        return nll, per_feat_nll, empty, empty, inp_x_post, inp_x
    
    def predict(self,**kwargs):
        raise NotImplemented()
    def forward(self,**kwargs):
        raise NotImplemented()