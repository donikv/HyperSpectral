import numpy as np
import matplotlib.pyplot as plt
from colour.colorimetry import SpectralShape
from utilities import *
from models import *
from loader import *

if __name__ == '__main__':
    image_dir = 'test_nikon_ured'
    image_size = (512, 256)
    result_spectral_shape = SpectralShape(380, 780, 10)
    
    collection = get_images_filters_camera_and_illumination(image_dir, image_size, result_spectral_shape)
    data = get_formated_data(collection=collection)
    IFScs_flat, target = data["x"], data["y"]

    camspecs = np.load('./filter_measurements/camspecs.npy', allow_pickle=True).item()

    device = 'cuda:0'
    n = 6
    ridge = GaussianMixtureTorch(samples=target.shape[0], device=device, n=n)

    try:
        ridge.load_state_dict(torch.load(f'./filter_measurements/{image_dir}/ridge.model'))
    except Exception:
        pass

    ridge.to(device)
    X = torch.tensor(IFScs_flat.astype(np.float32), requires_grad=True, device=device)
    y = torch.tensor(target.astype(np.float32), requires_grad=True, device=device)

    o = torch.optim.SGD(params=ridge.parameters(), lr=1)
    l = torch.nn.MSELoss()

    laplace_filter = laplace_filter.to(device)
    fit(ridge, X, y, o, l, 20000, 0, verbose=100)

    ridge.to('cpu')
    torch.save(ridge.state_dict(), f'./filter_measurements/{image_dir}/ridge.model')