import colour
import cv2
import numpy as np
import glob
import sys
import os
from datetime import datetime
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
    n = int(sys.argv[1])
    initial_lr = float(sys.argv[2])
    lam = float(sys.argv[3])
    std_reg = float(sys.argv[4])
    reg_val = float(sys.argv[5])
    device = sys.argv[6]
    image_dir = sys.argv[7]

    # image_dir = 'test_nikon_outdoors1'
    image_size = (512, 256)
    result_spectral_shape = SpectralShape(380, 780, 10)
    
    collection = get_images_filters_camera_and_illumination(image_dir, image_size, result_spectral_shape)

    valid_names = ['transparent_h', 'day_3', 'night_3']
    train_names = list(filter(lambda x: x not in valid_names, collection.images.keys()))

    train_collection = collection.get_subset(train_names)
    valid_collection = collection.get_subset(valid_names)

    train_data = get_formated_data(collection=train_collection)
    IFScs_flat, target = train_data["x"], train_data["y"]

    data = get_formated_data(collection=valid_collection)
    valid_IFScs_flat, valid_target = data["x"], data["y"]

    camspecs = np.load('./filter_measurements/camspecs.npy', allow_pickle=True).item()
    
    device = f'cuda:{device}'
    n = 6
    size = (image_size[1], image_size[0], target.shape[1])

    X = torch.tensor(IFScs_flat.astype(np.float32), requires_grad=True, device=device)
    y = torch.tensor(target.astype(np.float32), requires_grad=True, device=device)

    X_valid = torch.tensor(valid_IFScs_flat.astype(np.float32), requires_grad=True, device=device)
    y_valid = torch.tensor(valid_target.astype(np.float32), requires_grad=True, device=device)

    # basis = load_basis('./measurements/surfaces')
    basis = colour.recovery.MSDS_BASIS_FUNCTIONS_sRGB_MALLETT2019
    basis = basis.extrapolate(result_spectral_shape).interpolate(result_spectral_shape)
    cnn, get_Rs, name = create_DGcnn_fixed(target.shape[0], device, n, size, None, reg=std_reg, basis=basis)
    try:
        cnn.load_state_dict(torch.load(f'./filter_measurements/{image_dir}/{name}.model'))
    except Exception:
        pass
    cnn.to(device)

    o = torch.optim.Adam(params=cnn.parameters(), lr=initial_lr)
    l1 = torch.nn.MSELoss() 
    l2 = torch.nn.CosineSimilarity()
    def l(x,y):
        a = 1 - l2(x,y)
        b = l1(x,y)
        return lam * a.mean() + b

    params, best_params, train_losses, valid_losses = fit(cnn, X, y, o, l, 5000, reg_val, verbose=100, X_valid=X_valid, y_valid=y_valid, validate=True) 

    save_folder = f'./filter_measurements/{image_dir}/{name}'
    os.makedirs(save_folder, exist_ok=True)
    save_experiment(save_folder, {"p": params, "bp":best_params, "tl":train_losses, "vl":valid_losses})

    torch.save(best_params, f'{save_folder}/best.model')
    torch.save(params, f'{save_folder}/{name}.model')

    cnn.load_state_dict(torch.load(f'./filter_measurements/{image_dir}/{name}.model'))

    Rs = get_Rs(X, y)
    visualize_fitted(Rs, result_spectral_shape, camspecs)