from dataclasses import dataclass
from typing import Optional, Tuple

import torch
import torch.nn as nn

class S4(nn.Module):
    def __init__(self,channels: int,hidden_state: int = 64,min_delta: float = 1e-3,eps: float = 1e-6,use_hippo:bool=True):
        super().__init__()
        D,N = channels,hidden_state
        self.D,self.N = D,N
        self.eps = eps
        self.min_delta = min_delta
        
        #register kernel cache to not compute it every time new
        self.register_buffer("_cached_kernel",None,persistent=False)
        self._cached_len=None

        if use_hippo:
            A = self.hippo_init(N)
            log_A = torch.log(torch.expm1(-A))
            self.log_A = nn.Parameter(log_A)
        else:
            self.log_A = nn.Parameter(-torch.randn(N,N)) #reparametrization for stability
        self.B = nn.Parameter(torch.randn(D,N)*0.02)
        self.C = nn.Parameter(torch.randn(D,N)*0.02)

        self.log_delta = nn.Parameter(torch.ones(D))
        self.skip_D = nn.Parameter(torch.ones(D))
        self.out = nn.Linear(D,D)

        #trying to avoid going multiple times through the computation graph 
        for p in (self.log_A, self.B, self.C, self.log_delta):
            p.register_hook(lambda grad, _self=self: _self._clear_cache())

    def _clear_cache(self):
        self._cached_kernel = None
        self._cached_len = None


    @staticmethod
    def hippo_init(N:int):
        A = torch.zeros((N, N), dtype=torch.float32)
        #set diag
        idx = torch.arange(N) 
        A[idx,idx]= idx.float()+1
        #set lower triangle
        i = torch.arange(N, dtype=torch.float32).view(-1, 1)  #(N,)-> (N,1)
        v = torch.sqrt(2 * i + 1)                             
        prod = v @ v.T  #(N,N)
        lower_mask = torch.tril(torch.ones_like(A, dtype=torch.bool), diagonal=-1)
        A[lower_mask] = prod[lower_mask]
        return A


    def discretize(self):
        # Reconstruct continuous-time A
        A = nn.functional.softplus(self.log_A) 
        dt = nn.functional.softplus(self.log_delta) + self.min_delta  # (D,)
        A_batch = A.unsqueeze(0) * dt.view(-1, 1, 1)                  # (D,N,N)
        discrete_A = torch.matrix_exp(A_batch)                        # (D,N,N)
        N = A.shape[0]
        I = torch.eye(N, device=A.device, dtype=A.dtype)
        A_reg = A + self.eps * I                                      # (N,N)
        #discrete_B= A⁻¹(discrete_A-I)B
        X = torch.linalg.solve(A_reg, self.B.T)                       # (N,D)
        Xv = X.T.unsqueeze(-1)                                        # (D,N,1)
        # (D,N,N) @ (D,N,1) -> (D,N,1) -> (D,N)
        discrete_B = torch.matmul(discrete_A - I, Xv).squeeze(-1) 
        return discrete_A, discrete_B


    def _kernel(self,L:int,device:torch.device,dtype=torch.dtype):
        if self._cached_kernel is not None and self._cached_len == L and self._cached_kernel.device==device and self._cached_kernel.dtype==dtype:
            return self._cached_kernel
        discrete_A,discrete_B = self.discretize()
        discrete_A = discrete_A.to(device=device, dtype=dtype)
        discrete_B = discrete_B.to(device=device, dtype=dtype)
        
        D,N,_ = discrete_A.shape
        A_pows = []
        I = torch.eye(N, device=device, dtype=dtype).expand(D, N, N)
        A_pows.append(I)
        for _ in range(1, L):
            A_pows.append(discrete_A @ A_pows[-1])
        A_pows = torch.stack(A_pows, dim=0)  # (L,D,N,N)
        # Multiply: (L,D,N,N) @ (D,N,1) -> (L,D,N,1)
        Bd_col = discrete_B.unsqueeze(-1)            # (D,N,1)
        v_all = torch.matmul(A_pows, Bd_col) # (L,D,N,1)
        # Contract with C: (D,N) · (L,D,N) -> (D,L)
        k = torch.einsum('dn,ldn->dl', self.C, v_all.squeeze(-1))
        if self.training:
            self._cached_kernel, self._cached_len = k, L      # graph-carrying, valid within this step
        else:
            self._cached_kernel, self._cached_len = k.detach(), L
        self._cached_kernel, self._cached_len = k, L
        return k


    def forward(self, x: torch.Tensor):
        B, L, D = x.shape
        if D != self.D:
            raise ValueError(f"Channel mismatch: got D={D}, expected {self.D}")

        # get kernel on correct device/dtype
        k = self._kernel(L, x.device, x.dtype)   # (D, L)
        x_T = x.transpose(1, 2).contiguous()     # (B, D, L)
        w = k.flip(-1).unsqueeze(1)              # (D, 1, L)
        y = nn.functional.conv1d(x_T, w, padding=L-1, groups=D)  # (B, D, 2L-1)
        y = y[:, :, :L]
        y = y.transpose(1, 2)                                    # (B, L, D)

        y = self.out(y + x * self.skip_D)                        # (B, L, D)
        return y


class S4Regressor(nn.Module):
    def __init__(self,channels:int = 32,hidden_dim:int=64,depth:int=2,out_dim:int=1,sequence_to_one:bool=False):
        super().__init__()
        assert depth>0
        assert channels>0
        assert hidden_dim>0
        self.sequence_to_one = sequence_to_one
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

        if self.sequence_to_one:
            x = x.mean(dim=1)
        return self.head(x)


class MambaBlock:
    def __init__(self,input_dim,hidden_dim,output_dim,use_real):
        self.delta = torch.tensor((hidden_dim,hidden_dim),dtype=float)
        self.A = torch.tensor((hidden_dim,hidden_dim),dtype=float)
        self.B = torch.tensor((hidden_dim,1),dtype=float)
        self.C = torch.tensor((1,hidden_dim),dtype=float)

    def forward(input):
        pass
