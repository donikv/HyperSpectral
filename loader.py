import cv2
import numpy as np
import os
from utilities import *
from torch.utils.data import Dataset
import torch

def get_images_filters_camera_and_illumination(cwd, image_dir, camera_sensitivity_path, image_size, result_spectral_shape) -> FilteredImageCollection:
    filtered_directory = f'{cwd}/{image_dir}/filtered/'
    image_names = os.listdir(filtered_directory)
    filtered_names = list(filter(lambda f: not f.startswith('original') and not f.startswith('.'), image_names))
    filtered_names = [os.path.splitext(x)[0] for x in filtered_names]

    illumination = load_spec(f'{cwd}/{image_dir}/light.npy')
    camera_sensitivity = load_camera_spec(camera_sensitivity_path)
    
    illumination = illumination.extrapolate(result_spectral_shape)
    illumination = illumination.interpolate(result_spectral_shape)

    camera_sensitivity = camera_sensitivity.extrapolate(result_spectral_shape)
    camera_sensitivity = camera_sensitivity.interpolate(result_spectral_shape)
    camera_sensitivity = camera_sensitivity.to_sds()

    images = {}
    filter_specs = {}
    for image_name in filtered_names:
        image = load_png(filtered_directory+image_name+'.png')
        image = cv2.resize(image, image_size)
        images[image_name] = image
        filter_spec = load_spec(f'{cwd}/set1/{image_name}/final.npy')
        filter_spec = filter_spec.extrapolate(result_spectral_shape)
        filter_spec = filter_spec.interpolate(result_spectral_shape)
        filter_specs[image_name] = filter_spec

    return FilteredImageCollection(images, filter_specs, camera_sensitivity, illumination)

def get_formated_data(collection: FilteredImageCollection):
    images = collection.images
    filter_specs = collection.filter_specs
    camera_sensitivity = collection.camera_sensitivity
    illumination = collection.illumination

    IFS = {}
    for image_name in images.keys():
        IFScs = []
        for c in (0,1,2):
            IFSc = camera_sensitivity[c] * illumination * filter_specs[image_name]
            IFScs.append(IFSc.values)
        IFS[image_name] = np.array(IFScs) # 3xN N=41

    imgs = []
    IFScs = []
    for image_name in images.keys():
        imgs.append(images[image_name]) # img_names x h x w x 3
        IFScs.append(IFS[image_name]) # img_names x 3 x N
    IFScs = np.array(IFScs)
    imgs = np.array(imgs)
    IFScs_flat = IFScs.transpose((1,0,2)).reshape((-1, IFScs.shape[-1])) # 3 * img_names x N
    IFScs_flat = np.expand_dims(IFScs_flat, 0) # 1 x 3 * img_names x N
    imgs_flat = imgs.reshape(imgs.shape[0], -1, imgs.shape[-1]).transpose((1,2,0)) # hw x 3 x img_names
    target = imgs_flat.reshape((imgs_flat.shape[0],-1, 1)) # hw x 3 * img_names, [[c1,c1,c1...,c2,c2,c2...,c3,c3,c3...], ...]
    IFScs_flat = np.tile(IFScs_flat, (target.shape[0], 1, 1))

    return {"x":IFScs_flat, "y":target}

# def get_formated_validation_data(collection: FilteredImageCollection):
#     images = collection.images
#     filter_specs = collection.filter_specs
#     camera_sensitivity = collection.camera_sensitivity
#     illumination = collection.illumination

#     IS = {}
#     for image_name in images.keys():
#         IScs = []
#         for c in (0,1,2):
#             ISc = camera_sensitivity[c] * illumination
#             IScs.append(ISc.values)
#         IS[image_name] = np.array(IScs) # 3xN N=41

#     IFS = {}
#     for image_name in images.keys():
#         IFScs = []
#         IFSc = filter_specs[image_name]
#         IFScs.append(IFSc.values)
#         IFS[image_name] = np.array(IFScs) # 3xN N=41

#     imgs = []
#     IFScs = []
#     IScs
#     for image_name in images.keys():
#         imgs.append(images[image_name]) # img_names x h x w x 3
#         IFScs.append(IFS[image_name]) # img_names x 1 x N
#     IFScs = np.array(IFScs)
#     imgs = np.array(imgs)
#     IFScs_flat = IFScs.transpose((1,0,2)).reshape((-1, IFScs.shape[-1])) # 1 * img_names x N
#     IFScs_flat = np.expand_dims(IFScs_flat, 0) # 1 x 1 * img_names x N
#     imgs_flat = imgs.reshape(imgs.shape[0], -1, imgs.shape[-1]).transpose((1,2,0)) # hw x 3 x img_names
#     target = imgs_flat.reshape((imgs_flat.shape[0],-1, 1)) # hw x 3 * img_names, [[c1,c1,c1...,c2,c2,c2...,c3,c3,c3...], ...]
#     IFScs_flat = np.tile(IFScs_flat, (target.shape[0], 1, 1))

#     return {"x":IFScs_flat, "y":target}

class SequentialDataset(Dataset):
    def __init__(self, IFScs_flat, target, device='cpu') -> None:
        super(SequentialDataset, self).__init__()

        self.X = torch.tensor(IFScs_flat.astype(np.float32), requires_grad=True)[0]
        self.y = torch.tensor(target.astype(np.float32), requires_grad=True)[0]
        self.device = device
    
    def __len__(self):
        return 1
    
    def __getitem__(self, idx):
        return self.X, self.y

class ImageDataset(Dataset):
    def __init__(self, IFScs_flat, target, size, device='cpu') -> None:
        super(ImageDataset, self).__init__()

        X = torch.tensor(IFScs_flat.astype(np.float32), requires_grad=True)
        y = torch.tensor(target.astype(np.float32), requires_grad=True)

        self.X = X.reshape(self.size)
        self.y = y.reshape((self.size[0], self.size[1], 3))

        self.device = device
        self.size = size
    
    def __len__(self):
        return 1
    
    def __getitem__(self, idx):
        return self.X, self.y
