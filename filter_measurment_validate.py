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
import models as models
visualize_set_program(os.path.basename(__file__))

def visualize_fitted(Rs, result_spectral_shape, camspecs, size, cwd, image_dir):
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
    ill_name = 'D75'
    camera_name = 'Nikon D700'

    # cmf = camspecs[camera_name].extrapolate(result_spectral_shape).interpolate(result_spectral_shape).to_sds()
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
    plt.imshow(rgb)
    show()
    plt.figure()
    cv2.imwrite(f'{cwd}/{image_dir}/generated/human_cnn_{ill_name}.png', (np.stack([rgb[...,2],rgb[...,1],rgb[...,0]], axis=-1) * 255).astype(np.uint8))

def get_model(load_from, cwd, model_fn, result_spectral_shape, image_dir, device, n, size, std_reg, target):
    basis = colour.recovery.MSDS_BASIS_FUNCTIONS_sRGB_MALLETT2019
    basis = basis.extrapolate(result_spectral_shape).interpolate(result_spectral_shape)
    model_fn = getattr(models, model_fn)
    cnn, get_Rs, name = model_fn(target.shape[0], device, n, size, None, reg=std_reg, basis=result_spectral_shape)
    save_folder = f'{cwd}/{image_dir}/{name}'
    if load_from != "":
        exp = load_experiment(load_from)
        cnn.load_state_dict(exp['p'])
    cnn.to(device)
    return cnn, get_Rs, save_folder

def main(model_fn, n, initial_lr, lam, std_reg, reg_val, device, num_epochs, cwd, image_dir, camera_sensitivity_path, load_from, run_params):
    # image_dir = 'test_nikon_outdoors1'
    image_size = (1024, 512)
    result_spectral_shape = SpectralShape(380, 780, 10)
    
    collection = get_images_filters_camera_and_illumination(cwd, image_dir, camera_sensitivity_path, image_size, result_spectral_shape)

    valid_names = ['gray', 'day_3', 'night_3']
    train_names = list(filter(lambda x: x not in valid_names, collection.images.keys()))

    train_collection = collection.get_subset(train_names)
    valid_collection = collection.get_subset(valid_names)

    train_data = get_formated_data(collection=train_collection)
    IFScs_flat, target = train_data["x"], train_data["y"]

    data = get_formated_data(collection=valid_collection)
    valid_IFScs_flat, valid_target = data["x"], data["y"]

    camspecs = np.load(f'{cwd}/camspecs.npy', allow_pickle=True).item()
    
    device = f'cuda:{device}'
    size = (image_size[1], image_size[0], target.shape[1])

    X = torch.tensor(IFScs_flat.astype(np.float32), requires_grad=True, device=device)
    y = torch.tensor(target.astype(np.float32), requires_grad=True, device=device)

    X_valid = torch.tensor(valid_IFScs_flat.astype(np.float32), requires_grad=True, device=device)
    y_valid = torch.tensor(valid_target.astype(np.float32), requires_grad=True, device=device)

    cnn, get_Rs, save_folder = get_model(load_from, cwd, model_fn, result_spectral_shape, image_dir, device, n, size, std_reg, target)

    o = torch.optim.Adam(params=cnn.parameters(), lr=initial_lr)
    l1 = torch.nn.MSELoss() 
    l2 = torch.nn.CosineSimilarity()
    def l(x,y):
        a = 1 - l2(x,y)
        b = l1(x,y)
        return lam * a.mean() + b
    def vl(x,y):
        a = torch.arccos(l2(x,y))
        return a.mean() * 180 / 3.14

    params, best_params, train_losses, valid_losses = fit(cnn, X, y, o, l, num_epochs, reg_val, valid_loss=vl, verbose=10, X_valid=X_valid, y_valid=y_valid, validate=True) 

    os.makedirs(save_folder, exist_ok=True)
    # run_params = f'{n}_{initial_lr}_{lam}_{std_reg}_{reg_val}'
    experiment_data = {"p": params, "bp":best_params, "tl":train_losses, "vl":valid_losses, "run":run_params}
    saved_experiment = save_experiment(save_folder, experiment_data)

    torch.save(best_params, f'{save_folder}/best.model')
    torch.save(params, f'{save_folder}/final.model')

    Rs = get_Rs(X, y)
    visualize_fitted(Rs, result_spectral_shape, camspecs, size, cwd, image_dir)
    
    exp = load_experiment(saved_experiment)
    cnn.load_state_dict(exp['bp'])

    Rs = get_Rs(X, y)
    visualize_fitted(Rs, result_spectral_shape, camspecs, size, cwd, image_dir)
    return cnn, get_Rs, experiment_data, save_folder

def validate(model_fn, cwd, image_dir, validate_dir, camera_sensitivity_path, device, load_from, std_reg, n):
    image_size = (512, 256)
    result_spectral_shape = SpectralShape(380, 780, 10)
    
    collection = get_images_filters_camera_and_illumination(cwd, validate_dir, camera_sensitivity_path, image_size, result_spectral_shape)

    valid_names = ['gray', 'day_3', 'night_3']
    train_names = list(filter(lambda x: x not in valid_names, collection.images.keys()))

    train_collection = collection.get_subset(train_names)
    valid_collection = collection.get_subset(valid_names)

    train_data = get_formated_data(collection=train_collection)
    IFScs_flat, target = train_data["x"], train_data["y"]

    data = get_formated_data(collection=valid_collection)
    valid_IFScs_flat, valid_target = data["x"], data["y"]

    camspecs = np.load(f'{cwd}/camspecs.npy', allow_pickle=True).item()
    
    device = f'cuda:{device}'
    size = (image_size[1], image_size[0], target.shape[1])

    images = torch.tensor(target.astype(np.float32), requires_grad=True, device=device)

    X_valid = torch.tensor(valid_IFScs_flat.astype(np.float32), requires_grad=True, device=device)
    y_valid = torch.tensor(valid_target.astype(np.float32), requires_grad=True, device=device)

    cnn, get_Rs, save_folder = get_model(load_from, cwd, model_fn, result_spectral_shape, image_dir, device, n, size, std_reg, target)

    params = torch.load(f'{save_folder}/final.model', map_location='cpu')
    cnn.load_state_dict(params)
    
    cnn.to(device)

    l2 = torch.nn.CosineSimilarity()
    def vl(x,y):
        a = torch.arccos(l2(x,y))
        return a.mean() * 180 / 3.14

    valid_loss = models.validate(cnn, images, vl, X_valid, y_valid)

    Rs = get_Rs(X_valid, images)
    visualize_fitted(Rs, result_spectral_shape, camspecs, size, cwd, image_dir)
    return valid_loss

if __name__ == '__main__':
    args = {'model_fn': 'create_SRGBCNN_Combined', 'initial_lr': 1e-05, 'std_reg': 0.1, 'lam': 2.0, 'n': 4, 'reg_val': 0.5, 'num_epochs': 2500, 'cwd': './filter_measurements', 'image_dir': 'test_nikon_papers', 'valid_dir': 'test_nikon_outdoors1', 'camera_sensitivity_path': './measurements/nikon_d7000.txt', 'load_from': '', 'device': 1} 
    
    model, get_Rs, exp, save_folder = main(**args, run_params=args)
    validate_dir = 'test_nikon_ured'
    valid_loss = validate(model_fn=args['model_fn'], cwd=args['cwd'], image_dir=args['image_dir'], validate_dir=validate_dir, camera_sensitivity_path=args['camera_sensitivity_path'], load_from="", device=args['device'], n=args['n'], std_reg=args['std_reg'])
    print(valid_loss)