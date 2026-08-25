from dataclasses import dataclass, field
from os.path import join as pjoin
from typing import List
import numpy as np
import torchvision.transforms as transforms
# from pytorch_dataset_loaders.pytorch_datasets import Ecoset
import torch
import h5py

from . import decision_mappings, info_mappings
from .base import Dataset, ecosetDataset
from .dataloaders import PytorchLoader
from .registry import register_dataset
from .. import constants as c
from ..evaluation import metrics as m


hyp = {
    'dataset': {
        'name': 'ecoset_square256_chunked.h5',
        'image_size': 256, # determines which dataset to load
        'dataset_path': '/share/klab/datasets/', # Folder where dataset exists (end with '/')
        'augment': {'rotate','flip','grayscale','normalize'}, # Mention augmentations to be used here - trivialaug, autoaugment, randaugment, normalize
    },
    
    'optimizer': {
        'dataloader': {
            'num_workers_train': 10, # number of cpu workers processing the batches 
            'prefetch_factor_train': 4, # number of batches kept in memory by each worker (providing quick access for the gpu)
            'num_workers_val_test': 3, # do not need lots of workers for val/test
            'prefetch_factor_val_test': 4, # do not need lots of workers for val/test
        
        }
    },
    'misc': {
        'use_amp': True, # use automatic mixed precision during training - forward pass .half(), backward full
        'batch_size_val_test': 256, # issue: {a single A100 could not handle 1k batch size for val}
        'save_logs': 5, # after how many epochs should we save a copy of the logs
        'save_net': 5, # after how many epochs should we save a copy of the net
    }
    }

def get_transform(aug_str,hyp=None):
    # Returns a transform compose function given the transforms listed in "aug_str"

    transform_list = []
    if hyp and hyp['dataset']['image_size']:
        transform_list.append(transforms.Resize((hyp['dataset']['image_size'],hyp['dataset']['image_size']), antialias=True))
    if 'randomrotation' in aug_str:
        transform_list.append(transforms.RandomRotation(degrees=45))
    if 'randomflip' in aug_str:
        transform_list.append(transforms.RandomHorizontalFlip(p=0.5))
    if 'grayscale' in aug_str:
        transform_list.append(transforms.RandomGrayscale(p=0.5))
    transform_list.append(transforms.ConvertImageDtype(torch.float))
    # if 'normalize' in aug_str:
    #     transform_list.append(transforms.Normalize(mean = hyp['dataset']['train_img_mean_channels']/255., std = hyp['dataset']['train_img_std_channels']/255.))

    transform = transforms.Compose(transform_list)
    return transform

class MiniEcoset(torch.utils.data.Dataset):
    """A PyTorch dataset for MiniEcoset."""

    def __init__(self, split, dataset_path, transform=None):
        """
        Args:
            dataset_path (string): Path to the .h5 file
            transform (callable, optional): Optional transforms to be applied
                on a sample.
        """
        self.root_dir = dataset_path
        self.transform = transform

        with h5py.File(dataset_path, "r") as f:
            self.imgs = torch.from_numpy(f[split]['data'][()]).permute((0, 3, 1, 2)) # to match the CHW expectation of pytorch
            self.labels = torch.from_numpy(f[split]['labels'][()])
            # self.imgs = torch.from_numpy(f[split]['data'][()].astype(np.int8)).permute((0, 3, 1, 2))
            # self.labels = torch.from_numpy(f[split]['labels'][()].astype(np.int8))

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx): 
        # accepts ids and returns the imgs and labels transformed to the Dataloader
        if torch.is_tensor(idx):
            idx = idx.tolist()

        imgs = self.imgs[idx]
        labels = self.labels[idx]
        paths = ['None'] * len(labels) # * manually set to None

        if self.transform:
            imgs = self.transform(imgs)

        return imgs, labels, paths

class Ecoset(Dataset):
    def __init__(self, split, dataset_path, transform=None, label_transform = None, in_memory= True, channels = 3):
        """
        Args:
            split (string): 'train', 'val' or 'test'
            dataset_path (string): Path to the .h5 file
            transform (callable, optional): Optional transforms to be applied
                on a sample.

        Note:
            The original data is stored in the Height-Width-Channel (HWC) format.
            We convert the data to a PyTorch tensor and reorder the dimensions to
            match the Channel-Height-Width (CHW) format expected by PyTorch.
        """
        self.dataset_path = dataset_path
        self.transform = transform
        self.label_transform = label_transform
        self.in_memory = in_memory
        self.channels = channels
        if self.in_memory:
            with h5py.File(dataset_path, "r") as f:
                
                self.images = torch.from_numpy(
                    f[split]['data'][()]).permute((0, 3, 1, 2))
                if not self.label_transform:
                    print(f'No label transform {self.label_transform}')
                    self.labels = torch.from_numpy(f[split]['labels'][()]).long()
                else:
                    self.labels = torch.from_numpy(f[split]['labels'].astype(np.int64)[()])
        else:
            
            self.image_label = h5py.File(dataset_path, "r")[split]
            self.images = self.image_label['data']
            self.labels = self.image_label['labels']

    def __len__(self):
        
        return len(self.labels)

    def __getitem__(self, idx):
        
       
        if self.in_memory:
            image = self.images[idx]
            label = self.labels[idx]
            
        else:
            image = torch.from_numpy(np.asarray(self.images[idx])).permute((2,0,1))    
            label = torch.from_numpy(np.asarray(self.labels[idx].astype(np.int64)))
        path = 'None' # * manually set to None, see it stored in h5

        if self.transform:
            image = self.transform(image)
           

        return image, label, path

@dataclass
class miniecosetParams:
    path: str
    image_size: int = 224
    metrics: list = field(default_factory=lambda: [m.Accuracy(topk=1)]) #TODO! using m.PresentationSimilarity()
    decision_mapping: object = decision_mappings.miniecosetZejinProbabilitiesTo112ClassesMapping() #decision_mappings.ImageNetProbabilitiesTo16ClassesMapping()#
    info_mapping: object = info_mappings.ecosetInfoMapping() #* ???
    experiments: List = field(default_factory=list)
    contains_sessions: bool = False

@dataclass
class miniecoset64pxParams:
    path: str
    image_size: int = 64
    metrics: list = field(default_factory=lambda: [m.Accuracy(topk=1)]) #TODO! using m.PresentationSimilarity()
    decision_mapping: object = decision_mappings.miniecosetZejinProbabilitiesTo112ClassesMapping() #decision_mappings.ImageNetProbabilitiesTo16ClassesMapping()#
    info_mapping: object = info_mappings.ecosetInfoMapping() #* ???
    experiments: List = field(default_factory=list)
    contains_sessions: bool = False

@dataclass
class ecosetParams:
    path: str
    image_size: int = 224
    metrics: list = field(default_factory=lambda: [m.Accuracy(topk=1)]) #TODO! using m.PresentationSimilarity()
    decision_mapping: object =decision_mappings.ecosetProbabilitiesTo565ClassesMapping() #!decision_mappings.ImageNetProbabilitiesTo16ClassesMapping()#
    info_mapping: object = info_mappings.ecosetInfoMapping() #* ???
    experiments: List = field(default_factory=list)
    contains_sessions: bool = False

@register_dataset(name='ecoset_testset')
def ecoset_testset(*args, **kwargs):

    hyp['dataset']['image_size'] =256
    hyp['dataset']['name'] = 'ecoset_square256_proper_chunks.h5' # though it same as default
    dataset_path = hyp['dataset']['dataset_path'] + hyp['dataset']['name'] 
    hyp['dataset']['train_img_mean_channels'] = np.array([125.1569, 121.0330, 113.6814])/255.
    hyp['dataset']['train_img_std_channels'] = np.array([72.0660, 71.1811, 73.2189])/255.

    transform_val_test = get_transform(['normalize'],hyp)
    test_data = Ecoset('test',dataset_path,  transform=transform_val_test, label_transform=True)

    test_loader = torch.utils.data.DataLoader(test_data, batch_size=hyp['misc']['batch_size_val_test'],
                                                num_workers=hyp['optimizer']['dataloader']['num_workers_val_test'],
                                                prefetch_factor=hyp['optimizer']['dataloader']['prefetch_factor_val_test'])
    params = ecosetParams(image_size=hyp['dataset']['image_size'],
                          path = None)
    return ecosetDataset(name="ecoset_testset",
                   params=params,
                   predefined_loader=test_loader,
                   *args,
                   **kwargs)
    

@register_dataset(name='miniecoset_zejin_testset')
def miniecoset_zejin_testset(*args, **kwargs):

    hyp['dataset']['image_size'] =256
    hyp['dataset']['name'] = f"texture2shape_miniecoset_{hyp['dataset']['image_size']}px.h5" # though it same as default
    dataset_path = hyp['dataset']['dataset_path'] + hyp['dataset']['name'] 
    hyp['dataset']['train_img_mean_channels'] = np.array([125.1569, 121.0330, 113.6814])/255.
    hyp['dataset']['train_img_std_channels'] = np.array([72.0660, 71.1811, 73.2189])/255.

    transform_val_test = get_transform(['normalize'],hyp)
    test_data = Ecoset('test',dataset_path,  transform=transform_val_test, label_transform=True)

    test_loader = torch.utils.data.DataLoader(test_data, batch_size=hyp['misc']['batch_size_val_test'],
                                                num_workers=hyp['optimizer']['dataloader']['num_workers_val_test'],
                                                prefetch_factor=hyp['optimizer']['dataloader']['prefetch_factor_val_test'])
    params = ecosetParams(image_size=hyp['dataset']['image_size'],
                        path = None)
    return ecosetDataset(name="miniecoset_zejin_testset",
                   params=params,
                   predefined_loader=test_loader,
                   *args,
                   **kwargs)

@register_dataset(name='miniecoset64px_testset')
def miniecoset64px_testset(*args, **kwargs):
    hyp['dataset']['image_size'] =64
    hyp['dataset']['name'] = f"texture2shape_miniecoset_{hyp['dataset']['image_size']}px.h5" # though it same as default
    dataset_path = hyp['dataset']['dataset_path'] + hyp['dataset']['name'] 
    hyp['dataset']['train_img_mean_channels'] = np.array([125.1569, 121.0330, 113.6814])/255.
    hyp['dataset']['train_img_std_channels'] = np.array([72.0660, 71.1811, 73.2189])/255.

    transform_val_test = get_transform(['normalize'],hyp)
    test_data = Ecoset('test',dataset_path,  transform=transform_val_test, label_transform=True)

    test_loader = torch.utils.data.DataLoader(test_data, batch_size=hyp['misc']['batch_size_val_test'],
                                                num_workers=hyp['optimizer']['dataloader']['num_workers_val_test'],
                                                prefetch_factor=hyp['optimizer']['dataloader']['prefetch_factor_val_test'])
    params = ecosetParams(image_size=hyp['dataset']['image_size'],
                            path = None)
    return ecosetDataset(name="miniecoset64px_testset",
                   params=params,
                   predefined_loader=test_loader,
                   *args,
                   **kwargs)

# @register_dataset(name='miniecoset_dkl')
# def miniecoset_dkl_testset(*args, **kwargs):
#     hyp['dataset']['image_size'] = 256
#     hyp['dataset']['name'] = "texture2shape_miniecoset_256px_dkl.h5" 
#     dataset_path = hyp['dataset']['dataset_path'] + hyp['dataset']['name'] 

#     transform_val_test = get_transform([''], hyp)

#     test_data = MiniEcoset('test', dataset_path, transforms=transform_val_test)

#     test_loader = torch.utils.data.DataLoader(test_data, batch_size=hyp['misc']['batch_size_val_test'],
#                                                 num_workers=hyp['optimizer']['dataloader']['num_workers_val_test'],
#                                                 prefetch_factor=hyp['optimizer']['dataloader']['prefetch_factor_val_test'])
    
#     params = miniecosetParams(image_size=hyp['dataset']['image_size'],
#                         path = None)
    
#     return ecosetDataset(
#         name="miniecoset_dkl_testset",
#         params=params,
#         predefined_loader=test_loader,
#         *args,
#         **kwargs)
