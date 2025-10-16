from dataclasses import dataclass
from typing import Optional, Tuple

import torch
import torch.nn as nn


class S4_with_shared_A(nn.Module):
    #TODO: Implement version of model with shared matrix instead of multiple per channel
    #Advantage less params mixture of signal in latent space possible
    def __init__(self,channels: int,hidden_state: int = 64,min_delta: float = 1e-3,
                max_delta:float=0.1,kernel_max_size:int=1,eps: float = 1e-6,mode:str="Conv_FFT",
                use_hippo:bool=True,seed=42,persist_cache=True,**kwargs):
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
    


class S4_naive(nn.Module):
    def __init__(self,hidden_state: int = 64,min_delta: float = 1e-3,
                max_delta:float=0.1,kernel_max_size:int=1,eps: float = 1e-6,mode:str="Conv_FFT",
                use_hippo:bool=True,seed=42,persist_cache=True,**kwargs):
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
            #print(HiPPO)
            #print("/n")
            row, col = torch.triu_indices(HiPPO.size(0), HiPPO.size(1), offset=1)
            HiPPO[row,col] = 0
            HiPPO[torch.arange(N),torch.arange(N)] -= idx
            HiPPO = -HiPPO
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
        
        """
        #Hurwitz for numerical stability for high N and L values -still needed
        with torch.no_grad():
            I = torch.eye(self.N, device=self.A.device, dtype=self.A.dtype)
            # scale to avoid huge norms
            anorm = torch.linalg.matrix_norm(self.A, ord=2)
            self.A.div_(anorm.clamp_min(1e-6))
            # shift eigenvalues to Re(λ) < 0
            self.A.add_(-0.1 * I) 
        """


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
    

class S4_fast(S4_naive):
    def __init__(self,hidden_state:int=64,use_ortho_pq:bool=True,rank:int=1,**kwargs):
        super().__init__(hidden_state=hidden_state,**kwargs)
        self.r = rank
        N = hidden_state

        self.theta = nn.Parameter(torch.empty(N)) #real valued part of D
        self.omega = nn.Parameter(torch.empty(N)) #imaginary part of D
        self.Q = nn.Parameter(torch.empty(N,self.r))
        self.P = nn.Parameter(torch.empty(N,self.r))

        self.init_matrices(use_ortho_pq=use_ortho_pq)
        #self.A = self.full_A()

    def init_matrices(self,use_ortho_pq:bool):
        #initializing the sparse matrices for A and B,C
        with torch.no_grad():
            n = torch.arange(self.N,dtype=torch.float32)
            base = torch.sqrt(2*n+1).unsqueeze(-1)
            self.theta.copy_(torch.log(n+1.0))
            self.omega.zero_()

            P0 = base.repeat(1,self.r) #not restricted in original S4 model to same 
            Q0 = base.repeat(1,self.r) #P!=Q possible

            if self.r>1 and use_ortho_pq:
                P0[:,1:] += 1e-3*torch.randn(self.N,self.r-1)
                Q0[:,1:] += 1e-3*torch.randn(self.N,self.r-1)
                
            self.P.copy_(P0/(self.r **0.5))
            self.Q.copy_(Q0/(self.r **0.5))
            nn.init.normal_(self.B,std=(1.0/self.N)**0.5)
            nn.init.normal_(self.C,std=1.0)

    @property
    def D_diag(self):
        #similar to delta but now with constructed diagonal unitary D matrix
        return -torch.exp(self.theta)+ 1j* self.omega
    
    @property
    def A(self):
        D = torch.diag(self.D_diag.to(self.P.dtype).to(self.P.device))
        Q = self.Q
        if Q.dim>2:
            #else throws error
            Q = Q.T
        A = D+(self.P @ Q.to(D.dtype))
        return A
    

    def kernel(self,L:int):
        #fast computation of the kernel filter via sampling the generating function
        #L=kernel_length >=1
        assert L>0
        import torch
        device = self.B.device
        dtypeR = self.B.dtype #either 32 or 64 
        dtypeC = torch.complex64 if dtypeR == torch.float32 else torch.complex128
        
        #1)frequencies z_m
        k = torch.arange(L,device=device,dtype=dtypeR)
        w = torch.pi*2 *k/L
        z = torch.exp(1j * w.to(dtypeC)) #frequency samples on the complex unit circle

        #2)bilinear map: s_k
        delta = self.delta.to(device=device,dtype=dtypeR)
        s = (2.0/delta).to(dtypeC) * (z-1.0)/(z+1.0) #each frequency z has continous freq s
        

        #3)create D and shapes complex and on correct device
        D = self.D_diag.to(device=device,dtype=dtypeC)
        B = self.B.to(device=device,dtype=dtypeC)
        C = self.C.to(device=device,dtype=dtypeC)
        P = self.P.to(device=device,dtype=dtypeC)
        Q = self.Q.to(device=device,dtype=dtypeC)
        
        #4)compute resolvent (sI-A)⁻¹B efficient
        R = (s[:,None]-D[None,:]) #R = sI-D #shape: [L,N]
        
        #Sherman-Morrison-Woodbury (sI-(D+PQ.T))⁻¹ = R⁻¹-R⁻¹P(I+Q.T R⁻¹P)⁻¹Q.T R⁻¹
        RB = (B.squeeze(-1)[None,:]/R) #nominator B #shape: [L,N]
        RP = (P[None,:,:]/R[...,None]) #nominator P #shape: [L,N,r]
        
        QT_RP = torch.einsum('bi,jbk->jki',Q,RP) #shape: [L,r,r] might need 'bi,jbk->jik'
        I = torch.eye(self.r,device=device,dtype=dtypeC).expand(L,-1,-1)#shape: [L,r,r]
        S = I+QT_RP #shape: [L,r,r]

        
        #Solve system t= S⁻¹Q.T RB (one for each freq)
        QT_RB = torch.einsum('bi,jb->ji',Q,RB) #shape: [L,r]
        t = torch.linalg.solve(S,QT_RB.unsqueeze(-1)) #low-rank correction term #shape [L,r,1] 
        
        RP_t = torch.einsum('lnk,lri->ln',RP,t) # shape: [L,N]
        x = RB-RP_t # shape: [L,N]
        
        #4)K(z_k) = C*x_m
        Kz = torch.einsum('rn,ln->l',C,x) #shape: [L]
        #5)iFFT into time domain kernel K
        K_time = torch.fft.ifft(Kz,n=L).real
        #print(K_time.shape)
        return K_time.view(1,1,L).to(self.B.dtype)


    def discretize(self):
        #only used for the RNN way therefore dense representation of A
        A = self.A  # complex
        I = torch.eye(self.N, device=A.device, dtype=A.dtype)
        delta = self.delta.to(device=A.device, dtype=A.dtype)
        A1 = I - delta * 0.5 * A
        A2 = I + delta * 0.5 * A
        Ad = torch.linalg.solve(A1, A2)                 #[N,N] (complex)
        Bd = torch.linalg.solve(A1, delta * self.B.to(A.dtype))  #[N,1] (complex)
        if Ad.is_complex():
            Ad_r = Ad.real
            Bd_r = Bd.real
            return Ad_r, Bd_r
        return Ad, Bd


class S4D(S4_naive):
    def __init__(self,hidden_state:int=64,fixed_B:bool=True,**kwargs):
        super().__init__(hidden_state=hidden_state,**kwargs)
        N = hidden_state

        self.theta = nn.Parameter(torch.empty(N))
        self.omega = nn.Parameter(torch.empty(N))
        self.init_diagonal_params()
        if fixed_B:
            self.B = nn.Parameter(torch.ones_like(self.B,requires_grad=False))

    @property
    def A(self):
        dtype = torch.complex64 if self.B.dtype == torch.float32 else torch.complex128 
        return (-torch.exp(self.theta) + 1j*self.omega).to(device=self.B.device,dtype=dtype)


    def init_diagonal_params(self):
        with torch.no_grad():
            N = self.N
            n = torch.arange(N,dtype=torch.float32,device=self.B.device)
            self.theta.copy_(torch.log1p(n))
            self.omega.copy_(torch.linspace(0.0,torch.pi * (1-1.0/(N+1)),N,device=self.B.device))

            #noise against symmetry
            self.theta.add_(1e-3 * torch.randn_like(self.theta))
            self.omega.add_(1e-3 * torch.randn_like(self.omega))


    def discretize(self):
        delta = self.delta.to(device=self.B.device,dtype=self.B.dtype)
        A = self.A
        B = self.B.squeeze(-1).to(A.dtype)  #shapes all [N] because diag
        denom = (1- 0.5*delta.to(A.dtype)*A)
        Ad = (1+0.5*delta.to(A.dtype)*A)/denom
        Bd = (delta.to(A.dtype)/denom)*B

        Ad_r = torch.diag(Ad.real)
        Bd_r = Bd.real.unqueeze(-1)
        return Ad_r,Bd_r 
    
    def kernel(self,L:int):
        assert L>0
        device = self.B.device
        dtypeC = torch.complex64 if self.B.dtype == torch.float32 else torch.complex128

        #1) same as in S4fast
        k = torch.arange(L, device=device, dtype=self.B.dtype)
        w = 2 * torch.pi * k / L
        z = torch.exp(1j * w.to(dtypeC)) #shape: [L]
        
        #2)
        delta = self.delta.to(device=device, dtype=self.B.dtype)
        s = (2.0 / delta).to(dtypeC) * (z - 1.0) / (z + 1.0) #shape: [L]
        
        #3)
        D = self.D
        D.to(device=device, dtype=dtypeC) #shape: [N]
        B = self.B.squeeze(-1).to(device=device, dtype=dtypeC)#shape: [N]
        C = self.C.squeeze(0).to(device=device, dtype=dtypeC) #shape: [N]
        
        #4) simpler than S4_fast because diagonal
        R = s[:,None]-D[None,:]
        RB = B[None,:]/R
        Kz = (RB*C[None,:]).sum(dim=-1)
        
        #5)
        K_time = torch.fft.ifft(Kz,n=L).real
        return K_time.view(1,1,L).to(self.B.dtype)
        


class MultiChannelS4(nn.Module):
    """
    Wrap a S4 module into D parallel, indepent channels 
    This is not the most efficient as we dont parallelize D
    """
    def __init__(self, n_channels:int=1,implementation:str= "naive", *base_args, **base_kwargs):
        super().__init__()
        self.n_channels = n_channels
        # D independent copies (each has its own params)
        if implementation=="naive":
            self.channels = nn.ModuleList([S4_naive(*base_args, **base_kwargs) for _ in range(self.n_channels)])
        elif implementation == "fast":
            self.channels = nn.ModuleList([S4_fast(*base_args, **base_kwargs) for _ in range(self.n_channels)])
        elif implementation == "diagonal":
            self.channels = nn.ModuleList([S4D(*base_args, **base_kwargs) for _ in range(self.n_channels)])
        else: 
            raise NameError(f"No implementation {implementation} found.")
        

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

        assert D == self.n_channels, f"Expected {self.n_channels} channels, got {D}"
        y = torch.empty_like(X)
        for d,channel in enumerate(self.channels):
            y[:,d,:]= channel(X[:,d,:])

        if x.dim()<2:
            y = y.squeeze(0)
        if x.dim()==1:
            y = y.squeeze(0)
        return y



class S6(nn.Module):
    def __init__(self,input_dim:int,hidden_dim:int,eps:float=1e-6,**kwargs):
        super().__init__()
        D = input_dim
        N = hidden_dim
        self.N = N
        self.eps = eps

        self.A_log = nn.Parameter(torch.empty(N,))

        self.to_B = nn.Linear(D,N,bias=True)
        self.to_C = nn.Linear(D,N,bias=True)
        self.to_delta = nn.Linear(D,1,bias=True)

        self.W_out = nn.Linear(N,D,bias=True)

    

    @property
    def A(self):
        return -torch.nn.functional.softplus(self.A_log)

    def init_matrices(self):
        #does the He initialization make sense here?
        nn.init.kaiming_normal_(self.A_log,nonlinearity="linear")

    def forward(self,input):
        y_t,x = self.conv_parallel(input)
        return y_t
    

    def context_dependent_params(self,u):
        B_t = self.to_B(u)
        C_t = self.to_C(u)
        Delta_t = torch.nn.functional.softplus(self.to_delta(u))
        
        return B_t,C_t,Delta_t


    def recurrent_one_step(self,X:torch.tensor,x_init:torch.tensor=None):
        B,D,L = X.shape
        device = X.device
        #init_state
        x = torch.zeros(B,self.N,device=device) if x_init is None else x_init
        
        u_t = X[:,:,-1]
        B_t, C_t, Delta_t =self.context_dependent_params(u_t)
        
        #discretize A,B
        dA = torch.exp(self.A*Delta_t)
        denom = self.A+self.eps
        dB = ((dA-1)/denom)*B_t

        #new_state
        x = dA*x + dB
        
        y_t = self.W_out(C_t * x)
        return y_t,x


    def recurrent_multi_step(self,X:torch.tensor,L:int,x_init:torch.Tensor=None):
        B,D,L = X.shape
        device = X.device
        
        #optional normalize
        #X = nn.LayerNorm(D)(X)
        
        #init_state
        x = torch.zeros(B,self.N,device=device) if x_init is None else x_init
        #prep out
        Y = torch.empty(B,L,D,device=device)
        
        for t in range(L):
            u_t = X[:,:,t] #B,N

            B_t, C_t, Delta_t = self.context_dependent_params(u_t)

            #discretize A,B
            dA = torch.exp(self.A*Delta_t)
            denom = self.A+self.eps
            dB = ((dA-1)/denom)*B_t

            #new_state
            x = dA*x + dB 
            y_t = self.W_out(C_t * x)
            Y[:,t,:] = y_t

        Y = nn.LayerNorm(D)(Y)

        return Y,X
    
    

    def conv_parallel(self,X:torch.Tensor,x_init:torch.Tensor=None):
        B,_,_ = X.shape #(B,L,D)
        device = X.device
        
        u_t = X
        B_t, C_t, Delta_t = self.context_dependent_params(u_t) 
        #discretize A,B
        dA = torch.exp(self.A*Delta_t)
        denom = self.A+self.eps
        dB = ((dA-1)/denom)*B_t
        
        #for long train output necessary
        x = torch.zeros(B,self.N,device=device) if x_init is None else x_init
        conv = torch.cumprod(dA,dim=1)
        ones = torch.ones(B,1,self.N,device=device) 
        C_t = torch.cat([ones,conv[:,:-1,:]],dim=1)

        z_t = dB/(C_t+self.eps)
        s_t = torch.cumsum(z_t,dim=1)
        x_init_exp = x.unsqueeze(1)
        X_state = C_t * (x_init_exp+s_t) #(B,L,N)
        Y = self.W_out(C_t * X_state)
        return Y,X_state



class MambaBlock(nn.Module):
    def __init__(self, d_model: int, d_state: int, ffn_multiplier: int = 4,
                 dropout: float = 0.0,**kwargs):
        """This is a Mamba block with feedforward MLP and S6 layer 
        please ensure to not use the canonical format in forward (B,D,L)
        use the Transformer like pass of B,L,D."""
        super().__init__()
        self.s6 = S6(d_model, d_state,**kwargs)
        self.dropout = nn.Dropout(dropout)

        hidden = ffn_multiplier * d_model
        self.ffn = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, d_model),
        )

    def forward(self,x:torch.Tensor):
        #Ensure format of X.shape = (B,L,D)
        y = self.s6(x)
        x = x + self.dropout(y)
        z = self.ffn(x)
        x = x + self.dropout(z)


        return x
    

class Mamba(nn.Module):
    def __init__(self, n_layers,canonical_format:bool=True, *base_args, **base_kwargs):
        super().__init__()
        self.n_layers = n_layers
        self.canonical_format = canonical_format
        self.layers = nn.Sequential(*[MambaBlock(*base_args, **base_kwargs) for _ in range(self.n_layers)])
        

    def correct_format(self,X):
        if self.canonical_format:
            #B,D,L -> B,L,D
            X = X.permute(0,2,1)
        return X
    

    def forward(self,X:torch.tensor):
        X = self.correct_format(X)
        y = self.layers(X)
        y = self.correct_format(y)
        return y

