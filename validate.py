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

if __name__ == '__main__':
    image_dir = 'test_nikon_papers'
    image_size = (512, 256)
    result_spectral_shape = SpectralShape(380, 780, 10)
    regions = load_regions(f'./filter_measurements/{image_dir}/validation_regions.txt')
    
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
    Rs = Rs.cpu().detach().numpy().transpose((0,2,1))
    spectral_image = Rs.reshape((*size[0:2], -1))
    avgs, stds, squares = inspect_regions(spectral_image, regions)


    for i in range(avgs.shape[0]):
        avg_R = avgs[i]
        R = SpectralDistribution(avg_R, [x for x in result_spectral_shape])
        plt.plot(R.domain, R.values)
    show()

    plt.figure()
    
    for i in range(stds.shape[0]):
        R = stds[i]
        R = SpectralDistribution(R, [x for x in result_spectral_shape])
        plt.plot(R.domain, R.values)
    show()

    exit(0)