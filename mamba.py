from dataclasses import dataclass
from typing import Optional, Tuple

import torch
import torch.nn as nn


class S4_with_shared_A(nn.Module):
    #TODO: Implement version of model with shared matrix instead of multiple per channel
    #Advantage less params mixture of signal in latent space possible
    def __init__(self,channels: int,hidden_state: int = 64,min_delta: float = 1e-3,
                max_delta:float=0.1,kernel_max_size:int=1,eps: float = 1e-6,mode:str="Conv_FFT",
                use_hippo:bool=True,seed=42,persist_cache=True):
        super().__init__()
        D,N = channels,hidden_state
        self.D,self.N = D,N
        self.eps = eps
        self.min_delta = torch.tensor(min_delta)
        self.max_delta = torch.tensor(max_delta)
        self.seed=seed
        self.kernel_max_size = kernel_max_size
        self.mode = mode

        if seed:
            torch.manual_seed(seed)

        self.A = nn.Parameter(torch.empty((N,N)))
        self.B = nn.Parameter(torch.empty((N,D)))
        self.C = nn.Parameter(torch.empty((D,N)))
        self.D = nn.Parameter(torch.ones(D))
        self.log_delta = nn.Parameter(torch.empty(1,))
        #init matrices
        self.reset_matrices()
        #kernel for convolutional view/training
        #self.K = self.kernel()
        #buffer for really long sequences and chunk editing
        self.register_buffer("cache_h_k", torch.zeros(N), persistent=persist_cache)


    def reset_matrices(self):
        nn.init.kaiming_normal_(self.A,nonlinearity="linear") #He init
        nn.init.normal_(self.B,std=(1.0/self.N)**0.5)
        nn.init.normal_(self.C,std=1.0)
        nn.init.ones_(self.D)
        nn.init.uniform_(self.log_delta,a=torch.log(self.min_delta),b=torch.log(self.max_delta))


    def reset_hidden_state(self, batch_size: int | None = None, *, device=None, dtype=None):
        #reset hidden state after large new sequence
        device = device or self.A.device
        dtype = dtype or self.A.dtype
        if batch_size is None:
            new = torch.zeros(self.N, device=device, dtype=dtype)
        else:
            new = torch.zeros(batch_size, self.N, device=device, dtype=dtype)
        with torch.no_grad():
            if self.cache_h_k.shape != new.shape:
                self.cache_h_k = new
            else:
                self.cache_h_k.copy_(new)


    def skip_connection(self,y,X):
        if y.dim() == 1:
            skip = self.D * X
        else:
            skip = self.D[:,torch.newaxis] * X
        return skip


    def propagate_RNN(self,X,reset_hidden_state:bool=True):
        
        is_sequence = False
        if X.dim() == 1:
            is_sequence = True
            X = X[torch.newaxis,:] #(D,L)

        L = X.shape[1]
        discrete_A,discrete_B = self.discretize()

        def propagate_time_step(h_k_1,x_k):
            h_k =  discrete_A @ h_k_1 + discrete_B @ x_k
            y_k = self.C @ h_k
            return h_k,y_k 
        
        #propagagate_sequence
        pred = []
        hidden_state = self.cache_h_k

        for t in range(X.shape[1]):
            x_t = X[:,t]
            hidden_state,y = propagate_time_step(hidden_state,x_t)
            pred.append(y)
        x_final = torch.stack(pred,dim=-1)
        if is_sequence:
            x_final = x_final.squeeze()

        if reset_hidden_state:
            self.reset_hidden_state()
        return x_final
    

    def propagate_convolution_filter(self,X,use_fourier=True):
        is_sequence = False
        if X.dim() == 1:
            X = X[torch.newaxis,:]
            is_sequence = True
        L = X.shape[1]
        
        K = self.K # -> (D,D,N)

        assert K.shape[-1] <= L

        if use_fourier:
            end_size = L+self.kernel_max_size-1
            X_pad = nn.functional.pad(X,(0,end_size-L)) #needs padding for kernel length
            K_pad = nn.functional.pad(K,(0,end_size-self.kernel_max_size)) # needs padding for seq length
            Xd = torch.fft.rfft(X_pad)
            Kd = torch.fft.rfft(K_pad)
            prod = torch.einsum('odf,df->of',Kd,Xd)
            y = torch.fft.irfft(prod,n=end_size)[:,:L]
        else:
            K_rev = torch.flip(K,dims=[2])
            y = nn.functional.conv1d(X,K_rev,bias=None,padding=self.kernel_max_size-1)[:,:L]
        
        if is_sequence:
            y = y.squeeze()

        return y
    


class S4_base(nn.Module):
    def __init__(self,hidden_state: int = 64,min_delta: float = 1e-3,
                max_delta:float=0.1,kernel_max_size:int=1,eps: float = 1e-6,mode:str="Conv_FFT",
                use_hippo:bool=True,seed=42,persist_cache=True):
        super().__init__()
        N = hidden_state
        self.N = N
        self.eps = eps
        self.min_delta = torch.tensor(min_delta)
        self.max_delta = torch.tensor(max_delta)
        self.seed=seed
        self.kernel_max_size = kernel_max_size
        self.mode = mode
        self.use_hippo = use_hippo

        if seed:
            torch.manual_seed(seed)

        self.A = nn.Parameter(torch.empty((N,N)))
        self.B = nn.Parameter(torch.empty((N,1)))
        self.C = nn.Parameter(torch.empty((1,N)))
        self.D = nn.Parameter(torch.ones(1))
        self.log_delta = nn.Parameter(torch.empty(1,))
        #init matrices
        self.reset_matrices()
        #buffer for really long sequences and chunk editing
        self.register_buffer("cache_h_k", torch.zeros((N,1)), persistent=persist_cache)
    

    @property
    def delta(self):
        return torch.exp(self.log_delta.clamp(self.min_delta.log(),self.max_delta.log()))
    
    
    @staticmethod
    def HiPPO_init(tensor):
        with torch.no_grad():
            N = tensor.shape[-1]
            idx = torch.arange(N)
            init = torch.sqrt(2*idx+1).unsqueeze(1)
            HiPPO = init @ init.T
            row, col = torch.triu_indices(HiPPO.size(0), HiPPO.size(1), offset=1)
            HiPPO[row,col] = 0
            HiPPO[torch.arange(N),torch.arange(N)] -= idx
            tensor.copy_(HiPPO)
        return tensor


    def reset_matrices(self):
        
        if self.use_hippo:
            self.HiPPO_init(self.A)
        else:#He init
            nn.init.kaiming_normal_(self.A,nonlinearity="linear") 

        nn.init.normal_(self.B,std=(1.0/self.N)**0.5)
        nn.init.normal_(self.C,std=1.0)
        nn.init.ones_(self.D)
        nn.init.uniform_(self.log_delta,a=torch.log(self.min_delta),b=torch.log(self.max_delta))
        
        with torch.no_grad():
            I = torch.eye(self.N, device=self.A.device, dtype=self.A.dtype)
            # scale to avoid huge norms
            anorm = torch.linalg.matrix_norm(self.A, ord=2)
            self.A.div_(anorm.clamp_min(1e-6))
            # shift eigenvalues to Re(λ) < 0
            self.A.add_(-0.1 * I) 
        #Hurwitz for numerical stability for high N and L values


    def reset_hidden_state(self, batch_size: int | None = None, *, device=None, dtype=None):
        #reset hidden state after large new sequence
        device = device or self.A.device
        dtype = dtype or self.A.dtype
        if batch_size is None:
            new = torch.zeros(self.N, device=device, dtype=dtype)
        else:
            new = torch.zeros(batch_size, self.N, device=device, dtype=dtype)
        with torch.no_grad():
            if self.cache_h_k.shape != new.shape:
                self.cache_h_k = new
            else:
                self.cache_h_k.copy_(new)


    def forward(self,X):
        if self.mode=="RNN":
            y = self.propagate_RNN(X)
        elif self.mode=="Conv":
            y = self.propagate_convolution_filter(X,use_fourier=False)
        else:
            y = self.propagate_convolution_filter(X,use_fourier=True)
        
        skip = self.D * X

        return y + skip


    def discretize(self):
        I = torch.eye(self.N,device=self.A.device,dtype=self.A.dtype)
        delta = self.delta.to(device=self.A.device,dtype=self.A.dtype)
        A1 = I - delta*0.5 *self.A
        A2 = I + delta*0.5 *self.A
        #A1_inv = torch.linalg.inv(A1)
        #discrete_A = A1_inv @ A2
        #discrete_B = A1_inv @ (self.delta * self.B)
        discrete_A = torch.linalg.solve(A1,A2)
        discrete_B = torch.linalg.solve(A1,self.delta*self.B)
        return discrete_A,discrete_B


    def propagate_RNN(self,X,reset_hidden_state:bool=True):
        
        L = len(X)
        discrete_A,discrete_B = self.discretize()

        def propagate_time_step(h_k_1,x_k):
            h_k =  discrete_A @ h_k_1 + discrete_B * x_k
            y_k = self.C @ h_k
            return h_k,y_k 
        
        #propagagate_sequence
        pred = []
        hidden_state = self.cache_h_k

        for t in range(L):
            x_t = X[t]
            hidden_state,y = propagate_time_step(hidden_state,x_t)
            pred.append(y)
        pred = torch.stack(pred).squeeze()
        
        if reset_hidden_state:
            self.reset_hidden_state()
        return pred
    

    def kernel(self,kernel_length):
        discrete_A,discrete_B = self.discretize()
        ks = []
        v = discrete_B
        for _ in range(kernel_length):
            ks.append(self.C@v)
            v = discrete_A@v
        return torch.stack(ks,dim=-1)


    def propagate_convolution_filter(self,X,use_fourier=True):
        X_in = X
        if X.dim() == 1:
            X = X.unsqueeze(0)
        B,L = X.shape

        K = self.kernel(min(L,self.kernel_max_size))

        K_len = K.shape[-1]
        end_size = L+ K_len -1

        X_pad = nn.functional.pad(X,(0,end_size-L))
        K_pad = nn.functional.pad(K.squeeze(0),(0,end_size - K_len))

        if use_fourier:
            Xd = torch.fft.rfft(X_pad,n=end_size,dim=-1)
            Kd = torch.fft.rfft(K_pad,n=end_size,dim=-1)
            Yd = Xd*Kd
            y = torch.fft.irfft(Yd,n=end_size,dim=-1)[...,:L]
        else:
            X3 = X.unsqueeze(1)
            y = nn.functional.conv1d(X3,K,padding=K_len).squeeze(1)[...,:L]

        return y.squeeze(0) if X_in.dim() == 1 else y
    

class MultiChannelS4(nn.Module):
    """
    Wrap a S4 module into D parallel, indepent channels 
    This is not the most efficient as we dont parallelize D
    """
    def __init__(self, D, *base_args, **base_kwargs):
        super().__init__()
        self.D = D
        # D independent copies (each has its own params)
        self.channels = nn.ModuleList([S4_base(*base_args, **base_kwargs) for _ in range(self.D)])

    def forward(self, x):
        
        if x.dim() == 1:
            # (L,)
            X = x.unsqueeze(0).unsqueeze(0) #(B,D,L)

        elif x.dim() == 2:
            #(D,L)
            X = x.unsqueeze(0) #(B,D,L)

        elif x.dim() == 3:
            # (B, D, L) -> feed channel i to module i
            X = x
        B, D, L = X.shape

        assert D == self.D, f"Expected {self.D} channels, got {D}"
        y = torch.empty_like(X)
        for d,channel in enumerate(self.channels):
            y[:,d,:]= channel(X[:,d,:])

        if x.dim()<2:
            y = y.squeeze(0)
        if x.dim()==1:
            y = y.squeeze(0)
        return y



class MambaBlock:
    def __init__(self,input_dim,hidden_dim,output_dim,use_real):
        self.delta = torch.tensor((hidden_dim,hidden_dim),dtype=float)
        self.A = torch.tensor((hidden_dim,hidden_dim),dtype=float)
        self.B = torch.tensor((hidden_dim,1),dtype=float)
        self.C = torch.tensor((1,hidden_dim),dtype=float)

    def forward(input):
        pass
