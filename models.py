import torch
from torch import nn
from torch.nn.parameter import Parameter
from torch.utils.data import DataLoader

ln = 3
laplace_filter = -1 * torch.ones([ln])
laplace_filter[ln // 2] = -1 * laplace_filter[ln // 2] * (ln - 1)
laplace_filter = laplace_filter / ln
laplace_filter = laplace_filter.unsqueeze(0).unsqueeze(0)
        
class RidgeTorch(nn.Module):
    def __init__(self, samples, n=41, device='cpu'):
        super(RidgeTorch, self).__init__()
        self.device = device
        w = torch.rand(samples,n,1)
        self.weights = Parameter(w)
        # self.weights.uniform_(0,1)

    def forward(self, IFS, y_in):
        y = IFS@self.weights
        return y
    
    def R(self, IFS):
        return self.weights, None
        

class GaussianMixtureTorch(nn.Module):
    def __init__(self, samples, n=10, device='cpu'):
        super(GaussianMixtureTorch, self).__init__()
        mean = torch.linspace(0, 1, n, device=device).unsqueeze(0).unsqueeze(-1).tile((samples,1,1)) #torch.ones((samples,n,1))
        std = torch.ones((samples,n,1), device=device) * 0.1
        a = torch.ones((samples,n,1), device=device)  * 0.1
        self.mean = Parameter(mean)
        self.mean.data.uniform_(0,1)
        self.std = Parameter(std)
        self.std.data.uniform_(0.1,0.3)
        self.a = Parameter(a)
        self.a.data.uniform_(0.1,0.15)
        
        self.device = device
    
    def forward(self, IFS, y_in):
        gm, _ = self.R(IFS)
        y = IFS@gm
        return y
    
    def R(self, IFS):
        x = torch.linspace(0, 1, IFS.shape[-1], device=self.device).unsqueeze(0).unsqueeze(0).tile((self.mean.shape[0],self.mean.shape[1],1))
        gauss = torch.abs(self.a) * torch.exp(-(x-self.mean)**2/(2*self.std**2))
        gm = torch.sum(gauss,dim=1).unsqueeze(-1)
        return gm, gauss

class InitParams():
    def __init__(self, mean, std, a):
        self.mean = mean.detach()
        self.std = std.detach()
        self.a = a.detach()

class GaussianCNNTorch(nn.Module):

    def __init__(self, samples, n=10, size=(256,512,45), device='cpu', init_params=None):
        super(GaussianCNNTorch, self).__init__()
        ks = (5,5)
        self.seq = nn.Sequential(
            nn.Conv2d(size[-1], 128, kernel_size=ks, stride=(1,1), padding='same'),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 3*n, kernel_size=(1,1), padding='same'),
        )
        if init_params is None:
            mean = torch.linspace(0, 1, n, device=device).unsqueeze(0).unsqueeze(-1).tile((samples,1,1)) #torch.ones((samples,n,1))
            std = torch.ones((samples,n,1), device=device) * 0.2
            a = torch.ones((samples,n,1), device=device) * (1/n)
        else:
            mean = init_params.mean.detach()
            std = init_params.std.detach()
            a = init_params.a.detach()
        self.mean = Parameter(mean)
        self.std = Parameter(std)
        self.a = Parameter(a)
        
        self.device = device
        self.size = size
        self.n = n
    
    def forward(self, IFS, y_in):
        gm, gauss = self.R(IFS, y_in)
        y = IFS@gm
        return y
    
    def conv(self, x):
        n = self.n
        x = x.reshape((1,*self.size)).permute(0, 3, 1, 2)
        params = self.seq(x)
        params = params.permute(0, 2, 3, 1).reshape((self.size[0]*self.size[1],3*self.n,1))
        mean, bandwidth, a = params[:,:n,:], params[:,n:2*n,:], params[:,2*n:,:]
        return self._normalize(mean), self._normalize(bandwidth), self._normalize(a)
    
    def _normalize(self,outmap):
        outmap_min, _ = torch.min(outmap, dim=1, keepdim=True)
        outmap_max, _ = torch.max(outmap, dim=1, keepdim=True)
        outmap = (outmap - outmap_min) / (outmap_max - outmap_min + 1e-7)
        return outmap
    
    def R(self, IFS, x):
        mean, bandwidth, a = self.conv(x)
        x = torch.linspace(0, 1, IFS.shape[-1], device=self.device).unsqueeze(0).unsqueeze(0).tile((mean.shape[0],mean.shape[1],1))
        gauss = torch.abs(a + self.a) * torch.exp(-(x-(self.mean + mean))**2/(2 * (self.std + bandwidth)**2 + 1e-4))
        gm = torch.sum(gauss,dim=1).unsqueeze(-1)
        return gm, gauss
    
class RefineCNN(nn.Module):
    def __init__(self, samples, init_params, n=41, size=(256,512,45), device='cpu'):
        super(RefineCNN, self).__init__()
        ks = (5,5)
        self.seq = nn.Sequential(
            nn.Conv2d(n, 128, kernel_size=ks, stride=(1,1), padding='same'),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, n, kernel_size=(1,1), padding='same'),
            nn.ReLU(inplace=True),
        )
        if init_params is None:
            mean = torch.linspace(0, 1, n, device=device).unsqueeze(0).unsqueeze(-1).tile((samples,1,1))
            std = torch.ones((samples,n,1), device=device) * 0.2
            a = torch.ones((samples,n,1), device=device)  * 0.1
        else:
            mean = init_params.mean.detach().to(device)
            std = init_params.std.detach().to(device)
            a = init_params.a.detach().to(device)
        
        self.device = device
        self.size = size
        self.n = n
        
        self.init_spds = self.R(mean, std, a,(size[0]*size[1],size[2],n))[0]
        self.spds = self.init_spds
    
    def forward(self, IFS, y_in):
        gm = self.conv(self.spds)
        self.spds = gm.detach()
        y = IFS@gm
        return y
    
    def conv(self, x):
        x = x.reshape((1,self.size[0], self.size[1], self.n)).permute(0, 3, 1, 2)
        params = self.seq(x)
        params = params.permute(0, 2, 3, 1).reshape((self.size[0]*self.size[1],self.n,1))
        params = params / (torch.max(params, dim=1, keepdim=True)[0] + 1e-7)
        return params
    
    def R(self, mean, bandwidth, a, shape):
        x = torch.linspace(0, 1, shape[-1], device=self.device).unsqueeze(0).unsqueeze(0).tile((mean.shape[0],mean.shape[1],1))
        gauss = torch.abs(a) * torch.exp(-(x-(mean))**2/(2 * (bandwidth)**2 + 1e-4))
        gm = torch.sum(gauss,dim=1).unsqueeze(-1)
        return gm, gauss

class DirectGaussianCNNTorch(nn.Module):

    def __init__(self, n=10, size=(256,512,45), device='cpu'):
        super(DirectGaussianCNNTorch, self).__init__()
        ks = (5,5)
        self.seq = nn.Sequential(
            nn.Conv2d(size[-1], 128, kernel_size=ks, stride=(1,1), padding='same'),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 256, kernel_size=(3,3), stride=(1,1), padding='same'),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 128, kernel_size=(3,3), stride=(1,1), padding='same'),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 3*n, kernel_size=(1,1), padding='same'),
        )
        
        self.device = device
        self.size = size
        self.n = n
    
    def forward(self, IFS, y_in):
        gm, gauss = self.R(IFS, y_in)
        y = IFS@gm
        return y
    
    def conv(self, x):
        n = self.n
        x = x.reshape((1,*self.size)).permute(0, 3, 1, 2)
        params = self.seq(x)
        params = params.permute(0, 2, 3, 1).reshape((self.size[0]*self.size[1],3*self.n,1))
        mean, bandwidth, a = params[:,:n,:], params[:,n:2*n,:], params[:,2*n:,:]
        return self._normalize(mean), self._normalize(bandwidth), self._normalize(a)
    
    def _normalize(self,outmap):
        outmap_min, _ = torch.min(outmap, dim=1, keepdim=True)
        outmap_max, _ = torch.max(outmap, dim=1, keepdim=True)
        outmap = (outmap - outmap_min) / (outmap_max - outmap_min + 1e-7)
        return outmap
    
    def R(self, IFS, x):
        mean, bandwidth, a = self.conv(x)
        x = torch.linspace(0, 1, IFS.shape[-1], device=self.device).unsqueeze(0).unsqueeze(0).tile((mean.shape[0],mean.shape[1],1))
        gauss = torch.abs(a) * torch.exp(-(x-(mean))**2/(2 * (bandwidth)**2 + 1e-4))
        gm = torch.sum(gauss,dim=1).unsqueeze(-1)
        return gm, gauss

def fit(model, X, y, optim, loss_fn, epochs, reg, verbose=0):
    f = torch.zeros(1)

    for e in range(epochs):
        optim.zero_grad()
        ygm = model(X, y)
        
        loss = loss_fn(ygm, y)
        if reg > 0:
            f = torch.nn.functional.conv1d(ygm[None,:,:,0].transpose(1,2), laplace_filter.tile(1, ygm.shape[1], 1)).abs().mean()
            loss1 = loss + reg * f
        else: 
            loss1 = loss
        loss1.backward()
        optim.step()
        
        if verbose > 0 and e%verbose==0:
            if reg > 0:
                print(f"epoch {e}: loss {loss.cpu().detach().numpy().mean()} + {f.cpu().detach().numpy()}")
            else:
                print(f"epoch {e}: loss {loss.cpu().detach().numpy().mean()}")

def fit_dataset(model, dataset, optim, loss_fn, epochs, reg, device, batch_size=1, verbose=0):
    f = torch.zeros(1)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    for e in range(epochs):
        for data in iter(dataloader):
            X,y = data
            X,y = X.to(device), y.to(device)
            optim.zero_grad()
            ygm = model(X, y)
        
            loss = loss_fn(ygm, y)
            if reg > 0:
                f = torch.nn.functional.conv1d(ygm[None,:,:,0].transpose(1,2), laplace_filter.tile(1, ygm.shape[1], 1)).abs().mean()
                loss1 = loss + reg * f
            else: 
                loss1 = loss
            loss1.backward()
            optim.step()
        
            if verbose > 0 and e%verbose==0:
                if reg > 0:
                    print(f"epoch {e}: loss {loss.cpu().detach().numpy().mean()} + {f.cpu().detach().numpy()}")
                else:
                    print(f"epoch {e}: loss {loss.cpu().detach().numpy().mean()}")