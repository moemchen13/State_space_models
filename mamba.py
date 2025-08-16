from dataclasses import dataclass
from typing import Optional, Tuple

import torch
import torch.nn as nn

@dataclass
class S4Config:
    channels: int
    hidden_state: int = 64
    min_delta: float = 1e-3
    eps: float = 1e-6

class S4(nn.Module):
    def __init__(self,channels: int,hidden_state: int = 64,min_delta: float = 1e-3,eps: float = 1e-6):
        super().__init__()
        D,N = channels,hidden_state
        self.D,self.N = D,N
        self.eps = eps
        self.min_delta = min_delta
        
        #register kernel cache to not compute it every time new
        self.register_buffer("_cached_kernel",None,persistent=False)
        self._cached_len=None

        self.log_A = nn.Parameter(torch.randn(D,N))
        self.B = nn.Parameter(torch.randn(D,N)*0.02)
        self.C = nn.Parameter(torch.randn(D,N)*0.02)

        self.log_delta = nn.Parameter(torch.ones(D))
        self.skip_D = nn.Parameter(torch.ones(D))
        self.out = nn.Linear(D,D)


    def discretize(self):
        #returns A und B discrete
        A = -nn.functional.softplus(self.log_A)
        dt = nn.functional.softplus(self.log_delta) + self.min_delta
        discrete_A = torch.exp(A*dt.unsqueeze(-1))
        discrete_B = (discrete_A - 1.0)/(A+self.eps)*self.B
        return discrete_A,discrete_B
    
    
    def _kernel(self,L:int,device:torch.device,dtype=torch.dtype):
        if self._cached_kernel is not None and self._cached_len == L and self._cached_kernel.device==device and self._cached_kernel.dtype==dtype:
            return self._cached_kernel
        discrete_A,discrete_B = self.discretize()
        D,N = discrete_A.shape
        n = torch.arange(L,device=device,dtype=dtype)
        lam_pows =discrete_A.to(dtype).unsqueeze(-1) ** n.unsqueeze(0).unsqueeze(0)
        gamma = (self.C*discrete_B).to(dtype).unqueeze(-1)
        k = (gamma * lam_pows).sum(dim=1)
        self._cached_kernel, self._cached_len = k,L
        return k
    
    def forward(self,x:torch.Tensor):
        B,L,D = x.shape
        if D != self.N:
            raise Exception(f"Dimensions are not the same D:{D} unequal expected hidden dim:{self.N}")
        k = self._kernel(L,x.type)
        x_T = x.transpose(1,2)
        w = k.flip(-1).unsqueeze(1)
        y = nn.functional.conv1d(x_T,w,padding=L-1,groups=D)
        y = y[:,:,:L]
        y = y.transpose(1,2)
        y = self.out(y + x*self.skip_D)
        return y
    

class S4Regressor(nn.Module):
    def __init__(self,channels:int = 32,hidden_dim:int=64,depth:int=2,out_dim:int=1):
        super().__init__()
        assert depth>0
        assert channels>0
        assert hidden_dim>0
        self.layers = nn.ModuleList([nn.ModuleDict({
            "s4block":  S4(channels=channels,hidden_state=hidden_dim),
            "norm":     nn.LayerNorm(channels)
        })
        for i in range(depth)])
        self.head = nn.Linear(channels,out_dim)

    def forward(self,x):
        for layer in self.layers:
            residual = x
            x = layer["s4block"](x)
            x = layer["norm"](x+residual)
        return self.head(x)




class MambaBlock:
    def __init__(self,input_dim,hidden_dim,output_dim,use_real):
        self.delta = torch.tensor((hidden_dim,hidden_dim),dtype=float)
        self.A = torch.tensor((hidden_dim,hidden_dim),dtype=float)
        self.B = torch.tensor((hidden_dim,1),dtype=float)
        self.C = torch.tensor((1,hidden_dim),dtype=float)

    def forward(input):
        pass
