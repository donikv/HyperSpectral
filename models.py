import torch
import colour
from torch import nn
from torch.nn.parameter import Parameter
from torch.utils.data import DataLoader
from tqdm import tqdm
from copy import deepcopy

ln = 3
__laplace_filter = -1 * torch.ones([ln])
__laplace_filter[ln // 2] = -1 * __laplace_filter[ln // 2] * (ln - 1)
__laplace_filter = __laplace_filter / ln
__laplace_filter = __laplace_filter.unsqueeze(0).unsqueeze(0)
        
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
        self.regularization = torch.zeros(1, device=device)
    
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
        self.regularization = torch.zeros(1, device=device)
    
    def forward(self, IFS, y_in):
        gm, gauss = self.R(IFS, y_in)
        y = IFS@gm
        return y
    
    def conv(self, x):
        n = self.n
        x = x.reshape((1,*self.size)).permute(0, 3, 1, 2)
        params = self.seq(x)
        params = params.permute(0, 2, 3, 1).reshape((self.size[0]*self.size[1],3*self.n,1))
        mean, std, a = params[:,:n,:], params[:,n:2*n,:], params[:,2*n:,:]
        mean, std, a = self._normalize(mean), self._normalize(std), self._normalize(a)
        self.regularization = (0.1 - std)
        return mean, std, a
    
    def _normalize(self,outmap):
        outmap_min, _ = torch.min(outmap, dim=1, keepdim=True)
        outmap_max, _ = torch.max(outmap, dim=1, keepdim=True)
        outmap = (outmap - outmap_min) / (outmap_max - outmap_min + 1e-7)
        return outmap
    
    def R(self, IFS, x):
        mean, std, a = self.conv(x)
        x = torch.linspace(0, 1, IFS.shape[-1], device=self.device).unsqueeze(0).unsqueeze(0).tile((mean.shape[0],mean.shape[1],1))
        gauss = torch.abs(a + self.a) * torch.exp(-(x-(self.mean + mean))**2/(2 * (self.std + std)**2 + 1e-4))
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
        self.regularization = torch.zeros(1, device=device)
        
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

    def __init__(self, n=10, size=(256,512,45), fixed_means=False, device='cpu', std_regularization=0.1):
        super(DirectGaussianCNNTorch, self).__init__()
        ks = (5,5)
        self.out_channels = 2*n if fixed_means else 3*n
        self.seq = nn.Sequential(
            nn.Conv2d(size[-1], 128, kernel_size=ks, stride=(1,1), padding='same'),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 256, kernel_size=(3,3), stride=(1,1), padding='same'),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 128, kernel_size=(3,3), stride=(1,1), padding='same'),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, self.out_channels, kernel_size=(1,1), padding='same'),
            nn.ReLU(inplace=True),
        )
        
        self.device = device
        self.size = size
        self.n = n
        self.means = torch.linspace(0, 1, n, device=device) if fixed_means else None
        self.training = True
        self.regularization = torch.zeros(1, device=device)
        self.std_regularization = std_regularization
    
    def forward(self, IFS, y_in):
        gm, gauss = self.R(IFS, y_in)
        y = IFS@gm
        return y
        # return {'output':y, 'regularization':gauss}
    
    def conv(self, x):
        n = self.n
        x = x.reshape((1,*self.size)).permute(0, 3, 1, 2)
        params = self.seq(x)
        params = params.permute(0, 2, 3, 1).reshape((self.size[0]*self.size[1],self.out_channels,1))
        if self.means is not None:
            std, a = params[:,:n,:], params[:,n:,:]
            mean = self.means.unsqueeze(0).unsqueeze(-1).tile((std.shape[0],1,1))
            if self.training:
                mean = mean + torch.normal(torch.zeros_like(mean), torch.zeros_like(mean) + 0.01)
                self.regularization = torch.abs(self.std_regularization - std)
            return self._normalize(mean), std, a
        else:
            mean, std, a = params[:,:n,:], params[:,n:2*n,:], params[:,2*n:,:]
            self.regularization = torch.abs(self.std_regularization - std)
            return self._normalize(mean), std, a
    
    def _normalize(self,outmap):
        outmap_min, _ = torch.min(outmap, dim=1, keepdim=True)
        outmap_max, _ = torch.max(outmap, dim=1, keepdim=True)
        outmap = (outmap - outmap_min) / (outmap_max - outmap_min + 1e-7)
        return outmap
    
    def R(self, IFS, x):
        mean, std, a = self.conv(x)
        x = torch.linspace(0, 1, IFS.shape[-1], device=self.device).unsqueeze(0).unsqueeze(0).tile((mean.shape[0],mean.shape[1],1))
        gauss = torch.abs(a) * torch.exp(-(x-(mean))**2/(2 * (std)**2 + 1e-4))
        gm = torch.sum(gauss,dim=1).unsqueeze(-1)
        return gm, gauss

class SRGBBasisCNNTorch(nn.Module):

    def __init__(self, msds=colour.recovery.MSDS_BASIS_FUNCTIONS_sRGB_MALLETT2019, size=(256,512,45), fixed_means=False, device='cpu', std_regularization=0.1):
        super(SRGBBasisCNNTorch, self).__init__()
        ks = (5,5)

        self.msds = msds.values
        n = self.msds.shape[1]

        self.out_channels = n
        self.seq = nn.Sequential(
            nn.Conv2d(size[-1], 128, kernel_size=ks, stride=(1,1), padding='same'),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 256, kernel_size=(3,3), stride=(1,1), padding='same'),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 128, kernel_size=(3,3), stride=(1,1), padding='same'),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, self.out_channels, kernel_size=(1,1), padding='same'),
            nn.ReLU(inplace=True),
        )
        
        self.device = device
        self.size = size
        self.n = n
        self.regularization = torch.zeros(1, device=device)
        self.std_regularization = std_regularization
        self.msds = torch.tensor(self.msds, device=device, dtype=torch.float32)
        self.msds = self.msds.transpose(1,0).unsqueeze(0)
        self.training = True
    
    def forward(self, IFS, y_in):
        gm, gauss = self.R(IFS, y_in)
        y = IFS@gm
        return y
        # return {'output':y, 'regularization':gauss}
    
    def conv(self, x):
        n = self.out_channels
        x = x.reshape((1,*self.size)).permute(0, 3, 1, 2)
        params = self.seq(x)
        params = params.permute(0, 2, 3, 1).reshape((self.size[0]*self.size[1],self.out_channels,1))
        
        # b = 1 - torch.norm(params, dim=-2, keepdim=True)
        # weights = torch.concat([params,b], dim=-2)
        weights = params
        return weights
    
    def R(self, IFS, x):
        weights = self.conv(x)
        gauss = weights * self.msds
        gm = torch.sum(gauss,dim=1).unsqueeze(-1)
        return gm, gauss

def fit(model, X, y, optim, loss_fn, epochs, reg, valid_loss, X_valid=None, y_valid=None, validate=False, verbose=0):
    f = torch.zeros(1)
    min_loss = None
    train_losses = []
    valid_losses = []
    best_params = model.state_dict()

    for e in range(epochs):
        optim.zero_grad()
        ygm = model(X, y)
        
        loss = loss_fn(ygm, y)
        if reg > 0:
            f = model.regularization.mean()
            loss1 = loss + reg * f
        else: 
            loss1 = loss
        loss1.backward()
        optim.step()
        train_losses.append(loss1.detach().cpu().item())
        if validate:
            ygm = model(X_valid, y)
            loss1 = valid_loss(ygm, y_valid)
            valid_losses.append(loss1.detach().cpu().item())

        if min_loss is None or loss1 < min_loss:
            best_params = deepcopy(model.state_dict())
        
        if verbose > 0 and e%verbose==0:
            if reg > 0:
                print(f"epoch {e}: loss {loss.cpu().detach().numpy().mean()} + {f.cpu().detach().numpy()} loss1 {loss1.cpu().detach().numpy().mean()}")
            else:
                print(f"epoch {e}: loss {loss.cpu().detach().numpy().mean()}")
    return model.state_dict(), best_params, train_losses, valid_losses

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
                f = model.regularization.mean()
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

def create_Gcnn(samples, device, n, size, ridge, reg=0.0, basis=None):
    cnn = GaussianCNNTorch(samples=samples, device=device, n=n, size=size, init_params=ridge)
    def f(X,y):
        Rs, _ = cnn.R(X,y)
        return Rs
    return cnn, f, "gcnn"

def create_DGcnn(samples, device, n, size, ridge, reg=0.0, basis=None):
    cnn = DirectGaussianCNNTorch(device=device, n=n, size=size, std_regularization=reg)
    def f(X,y):
        Rs, _ = cnn.R(X,y)
        return Rs
    return cnn, f, "direct_gcnn"

def create_DGcnn_fixed(samples, device, n, size, ridge, reg=0.0, basis=None):
    cnn = DirectGaussianCNNTorch(device=device, n=n, size=size, fixed_means=True, std_regularization=reg)
    def f(X,y):
        cnn.training = False
        Rs, _ = cnn.R(X,y)
        cnn.training = True
        return Rs
    return cnn, f, "direct_gcnn_fixed"

def create_SRGBCNN(samples, device, n, size, ridge, reg=0.0, basis=colour.recovery.MSDS_BASIS_FUNCTIONS_sRGB_MALLETT2019):
    cnn = SRGBBasisCNNTorch(device=device, msds=basis, size=size, fixed_means=True, std_regularization=reg)  # type: ignore
    def f(X,y):
        cnn.training = False
        Rs, _ = cnn.R(X,y)
        cnn.training = True
        return Rs
    return cnn, f, "srgb_basis_cnn"

def create_Rcnn(samples, device, n, size, ridge, reg=0.0, basis=None):
    cnn = RefineCNN(samples=samples, device=device, n=n, size=size, init_params=ridge)
    def f(X,y):
        Rs = cnn.spds
        return Rs
    return cnn, f, "refined"