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

def visualize_fitted(Rs, result_spectral_shape, camspecs, model_name, image_dir):
    Rs = Rs.cpu().detach().numpy()
    cmfs = colour.colorimetry.MSDS_CMFS_RGB
    cmf = cmfs['Wright & Guild 1931 2 Degree RGB CMFs']
    camspecs['Human'] = cmf
    cmf = cmf.extrapolate(result_spectral_shape)
    cmf = cmf.interpolate(result_spectral_shape)
    cmf = cmf.to_sds()

    flat_gray = colour.SpectralDistribution({k:0.5 for k in range(380,780)})
    ill = flat_gray.extrapolate(result_spectral_shape).interpolate(result_spectral_shape)

    ills = colour.SDS_ILLUMINANTS
    ill_name = 'ISO 7589 Studio Tungsten'
    camera_name = 'Human'
    cmf = camspecs[camera_name].extrapolate(result_spectral_shape).interpolate(result_spectral_shape).to_sds()

    pth = f'./filter_measurements/{image_dir}/generated/{model_name}/{camera_name}'
    os.makedirs(pth, exist_ok=True)

    ill_names = ['D50','D65','D75','FL3','FL8','FL11','HP1',
           'LED-B1','LED-B3','LED-B5','LED-RGB1','LED-V1','LED-V2','ISO 7589 Studio Tungsten']
    for ill_name in ill_names:
        print(ill_name)
        ill = ills[ill_name].extrapolate(result_spectral_shape).interpolate(result_spectral_shape)

        r = Rs.transpose(0,2,1)@np.expand_dims(np.array([(cmf[0]*ill).values]), axis=-1)
        r = r.reshape((*size[0:2], 1))
        g = Rs.transpose(0,2,1)@np.expand_dims(np.array([(cmf[1]*ill).values]), axis=-1)
        g = g.reshape((*size[0:2], 1))
        b = Rs.transpose(0,2,1)@np.expand_dims(np.array([(cmf[2]*ill).values]), axis=-1)
        b = b.reshape((*size[0:2], 1))
        rgb = np.concatenate([r,g,b], axis=-1)
        rgb = rgb / rgb.max()
        rgb = (rgb.clip(0,1) ** (1/2.2))
        # plt.imshow(rgb)
        # show()
        cv2.imwrite(f'./filter_measurements/{image_dir}/generated/{model_name}/{camera_name}/cnn_{ill_name}.png', (np.stack([rgb[...,2],rgb[...,1],rgb[...,0]], axis=-1) * 255).astype(np.uint8))

if __name__ == '__main__':
    image_dir = 'test_nikon_papers'
    image_size = (512, 256)
    result_spectral_shape = SpectralShape(380, 780, 10)
    
    camspecs = np.load('./filter_measurements/camspecs.npy', allow_pickle=True).item()
    mobile_camspecs = np.load('./filter_measurements/mobile_camspecs.npy', allow_pickle=True).item()

    collection = get_images_filters_camera_and_illumination(image_dir, image_size, result_spectral_shape)
    data = get_formated_data(collection=collection)
    IFScs_flat, target = data["x"], data["y"]

    device = 'cuda:0'
    n = 6
    ridge = GaussianMixtureTorch(samples=target.shape[0], device=device, n=n)
    ridge.load_state_dict(torch.load(f'./filter_measurements/{image_dir}/ridge.model'))
    
    size = (image_size[1], image_size[0], target.shape[1])

    X = torch.tensor(IFScs_flat.astype(np.float32), requires_grad=True, device=device)
    y = torch.tensor(target.astype(np.float32), requires_grad=True, device=device)

    cnn, get_Rs, name = create_DGcnn_fixed(target.shape[0], device, n, size, ridge)
    cnn.load_state_dict(torch.load(f'./filter_measurements/{image_dir}/{name}.model'))
    cnn.to(device)

    Rs = get_Rs(X, y)
    visualize_fitted(Rs, result_spectral_shape, mobile_camspecs, name, image_dir=image_dir)

    exit(0)