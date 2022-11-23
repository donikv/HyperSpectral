import colour
import cv2
import numpy as np
import glob
import os
from scipy.ndimage import median_filter
from skimage.filters.rank import mean, median, minimum
import matplotlib.pyplot as plt
from colour.colorimetry import SpectralDistribution, MultiSpectralDistributions, SpectralShape
from utilities import *
from models import *
from loader import *
from visualize import show, visualize_set_program
visualize_set_program(os.path.basename(__file__))

def visualize_fitted(Rs, result_spectral_shape, camspecs):
    Rs = Rs.cpu().detach().numpy()
    print(Rs.shape)
    R = SpectralDistribution(Rs[256*512 // 2, :, 0], [x for x in result_spectral_shape])
    plt.plot(R.domain, R.values)
    R = SpectralDistribution(Rs[256*512 - 43, :, 0], [x for x in result_spectral_shape])
    plt.plot(R.domain, R.values)
    show()

    cmfs = colour.colorimetry.MSDS_CMFS_RGB
    cmf = cmfs['Wright & Guild 1931 2 Degree RGB CMFs']
    cmf = cmf.extrapolate(result_spectral_shape)
    cmf = cmf.interpolate(result_spectral_shape)
    cmf = cmf.to_sds()

    flat_gray = colour.SpectralDistribution({k:0.5 for k in range(380,780)})
    ill = flat_gray.extrapolate(result_spectral_shape).interpolate(result_spectral_shape)

    ills = colour.SDS_ILLUMINANTS
    ill_name = 'ISO 7589 Studio Tungsten'
    camera_name = 'Nikon D700'

    # cmf = camspecs[camera_name].extrapolate(result_spectral_shape).interpolate(result_spectral_shape).to_sds()
    # ill = ills[ill_name].extrapolate(result_spectral_shape).interpolate(result_spectral_shape)

    r = Rs.transpose(0,2,1)@np.expand_dims(np.array([(cmf[0]*ill).values]), axis=-1)
    r = r.reshape((*size[0:2], 1))
    g = Rs.transpose(0,2,1)@np.expand_dims(np.array([(cmf[1]*ill).values]), axis=-1)
    g = g.reshape((*size[0:2], 1))
    b = Rs.transpose(0,2,1)@np.expand_dims(np.array([(cmf[2]*ill).values]), axis=-1)
    b = b.reshape((*size[0:2], 1))
    rgb = np.concatenate([r,g,b], axis=-1)
    rgb = rgb / rgb.max()
    rgb = (rgb.clip(0,1) ** (1/2.2))
    plt.imshow(rgb)
    show()
    cv2.imwrite(f'./filter_measurements/{image_dir}/generated/human_cnn_{ill_name}.png', (np.stack([rgb[...,2],rgb[...,1],rgb[...,0]], axis=-1) * 255).astype(np.uint8))

if __name__ == '__main__':
    image_dir = 'test_nikon_papers'
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
    
    size = (image_size[1], image_size[0], target.shape[1])

    X = torch.tensor(IFScs_flat.astype(np.float32), requires_grad=True, device=device)
    y = torch.tensor(target.astype(np.float32), requires_grad=True, device=device)

    cnn, get_Rs, name = create_DGcnn_fixed(target.shape[0], device, n, size, ridge, reg=0.15)
    try:
        cnn.load_state_dict(torch.load(f'./filter_measurements/{image_dir}/{name}.model'))
    except Exception:
        pass
    # cnn, get_Rs, name = create_Rcnn(target.shape[0], device, len(result_spectral_shape), size, ridge)
    cnn.to(device)

    o = torch.optim.Adam(params=cnn.parameters(), lr=0.0001)
    l1 = torch.nn.MSELoss() 
    l2 = torch.nn.CosineSimilarity()
    def l(x,y):
        a = 1 - l2(x,y)
        b = l1(x,y)
        return 0.1 * a.mean() + b

    
    best_params = fit(cnn, X, y, o, l, 5000, 0.5, verbose=100)
    torch.save(best_params, f'./filter_measurements/{image_dir}/{name}.model')

    cnn.load_state_dict(torch.load(f'./filter_measurements/{image_dir}/{name}.model'))

    Rs = get_Rs(X, y)
    visualize_fitted(Rs, result_spectral_shape, camspecs)

    exit(0)