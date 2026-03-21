import os
import glob
import torch
import torch.utils.data as data
from . import ISTD_transforms
from PIL import Image
import random
import numpy as np
from torchvision import transforms


def add_gaussian_noise(image, noise_level):
    """Add Gaussian noise to a PIL image"""
    img_array = np.array(image).astype(np.float32)
    noise = np.random.randn(*img_array.shape) * noise_level
    noisy_array = img_array + noise
    noisy_array = np.clip(noisy_array, 0, 255).astype(np.uint8)
    return Image.fromarray(noisy_array)


def make_shadow_datapath_list(phase="train", rate=0.8):
    """Shadow removal dataset (AISTD)"""
    random.seed(44)
    rootpath = './dataset/' + phase + '/'
    files_name = os.listdir(rootpath + phase + '_A')

    if phase == 'train':
        random.shuffle(files_name)
    elif phase == 'test':
        files_name.sort()

    path_A, path_B, path_C = [], [], []
    for name in files_name:
        path_A.append(rootpath + phase + '_A/' + name)
        path_B.append(rootpath + phase + '_B/' + name)
        path_C.append(rootpath + phase + '_C/' + name)

    num = len(path_A)

    if phase == 'train':
        split_idx = int(num * rate)
        path_list = {'path_A': path_A[:split_idx], 'path_B': path_B[:split_idx], 'path_C': path_C[:split_idx]}
        path_list_val = {'path_A': path_A[split_idx:], 'path_B': path_B[split_idx:], 'path_C': path_C[split_idx:]}
        return path_list, path_list_val
    else:
        return {'path_A': path_A, 'path_B': path_B, 'path_C': path_C}


def make_denoise_datapath_list(phase="train", rate=0.8, noise_level=None):
    """Denoising dataset (BSD400/CBSD68)"""
    random.seed(44)

    if phase == 'train':
        rootpath = './BSDdataset/train/'
        files = glob.glob(os.path.join(rootpath, '*.jpg'))
        if len(files) == 0:
            files = glob.glob(os.path.join(rootpath, '*.JPG'))
        files.sort()
        random.shuffle(files)

        path_A, path_B, path_C = [], [], []
        for file_path in files:
            path_A.append(file_path)
            path_B.append(None)
            path_C.append(file_path)

        num = len(path_A)
        split_idx = int(num * rate)
        path_list = {'path_A': path_A[:split_idx], 'path_B': [None] * len(path_A[:split_idx]), 'path_C': path_C[:split_idx]}
        path_list_val = {'path_A': path_A[split_idx:], 'path_B': [None] * len(path_A[split_idx:]), 'path_C': path_C[split_idx:]}
        return path_list, path_list_val
    else:
        if noise_level is None:
            raise ValueError("noise_level must be specified for test phase")
        
        noisy_path = f'./BSDdataset/CBSD68/noisy{noise_level}/'
        clean_path = './BSDdataset/CBSD68/original/'
        
        noisy_files = glob.glob(os.path.join(noisy_path, '*.png'))
        if len(noisy_files) == 0:
            noisy_files = glob.glob(os.path.join(noisy_path, '*.PNG'))
        noisy_files.sort()

        path_A, path_B, path_C = [], [], []
        for noisy_file in noisy_files:
            filename = os.path.basename(noisy_file)
            clean_file = os.path.join(clean_path, filename)
            if os.path.exists(clean_file):
                path_A.append(noisy_file)
                path_B.append(None)
                path_C.append(clean_file)
        
        return {'path_A': path_A, 'path_B': path_B, 'path_C': path_C}


def make_lol_datapath_list(phase="train", rate=0.8):
    """Low-light enhancement dataset (LOL)"""
    random.seed(44)
    rootpath = './lol/' + phase + '/'
    files_name = os.listdir(rootpath + phase + '_A')

    if phase == 'train':
        random.shuffle(files_name)
    elif phase == 'test':
        files_name.sort()

    path_A, path_B, path_C = [], [], []
    for name in files_name:
        path_A.append(rootpath + phase + '_A/' + name)
        path_B.append(None)
        path_C.append(rootpath + phase + '_B/' + name)

    num = len(path_A)

    if phase == 'train':
        split_idx = int(num * rate)
        path_list = {'path_A': path_A[:split_idx], 'path_B': [None] * len(path_A[:split_idx]), 'path_C': path_C[:split_idx]}
        path_list_val = {'path_A': path_A[split_idx:], 'path_B': [None] * len(path_A[split_idx:]), 'path_C': path_C[split_idx:]}
        return path_list, path_list_val
    else:
        return {'path_A': path_A, 'path_B': [None] * len(path_A), 'path_C': path_C}


def make_rain_datapath_list(phase="train", rate=0.8):
    """Deraining dataset (Rain100L)"""
    random.seed(44)
    rootpath = './rain100L/' + phase + '/'
    files_name = os.listdir(rootpath + phase + '_A')

    if phase == 'train':
        random.shuffle(files_name)
    elif phase == 'test':
        files_name.sort()

    path_A, path_B, path_C = [], [], []
    for name in files_name:
        path_A.append(rootpath + phase + '_A/' + name)
        path_B.append(None)
        path_C.append(rootpath + phase + '_B/' + name)

    num = len(path_A)

    if phase == 'train':
        split_idx = int(num * rate)
        path_list = {'path_A': path_A[:split_idx], 'path_B': [None] * len(path_A[:split_idx]), 'path_C': path_C[:split_idx]}
        path_list_val = {'path_A': path_A[split_idx:], 'path_B': [None] * len(path_A[split_idx:]), 'path_C': path_C[split_idx:]}
        return path_list, path_list_val
    else:
        return {'path_A': path_A, 'path_B': [None] * len(path_A), 'path_C': path_C}


class ImageTransform:
    """Preprocessing images for all tasks"""
    def __init__(self, size=286, crop_size=256, mean=(0.5,), std=(0.5,)):
        self.data_transform = {
            'train': ISTD_transforms.Compose([
                ISTD_transforms.Scale(size=size),
                ISTD_transforms.RandomCrop(size=crop_size),
                ISTD_transforms.RandomHorizontalFlip(p=0.5),
                ISTD_transforms.RandomVerticalFlip(p=0.5),
                ISTD_transforms.ToTensor(),
                ISTD_transforms.Normalize(mean, std)
            ]),
            'val': ISTD_transforms.Compose([
                ISTD_transforms.Resize([crop_size, crop_size]),
                ISTD_transforms.ToTensor(),
                ISTD_transforms.Normalize(mean, std)
            ]),
            'test': ISTD_transforms.Compose([
                ISTD_transforms.Scale(size=size),
                ISTD_transforms.RandomCrop(size=crop_size),
                ISTD_transforms.ToTensor(),
                ISTD_transforms.Normalize(mean, std)
            ]),
            'test_no_crop': ISTD_transforms.Compose([
                ISTD_transforms.Resize([256, 256]),
                ISTD_transforms.ToTensor(),
                ISTD_transforms.Normalize(mean, std)
            ])
        }
    
    def __call__(self, phase, img):
        return self.data_transform[phase](img)


class TaskDataset(data.Dataset):
    """Single task dataset"""
    def __init__(self, img_list, img_transform, phase, task_name, noise_levels=[15, 25, 50]):
        self.img_list = img_list
        self.img_transform = img_transform
        self.phase = phase
        self.task_name = task_name
        self.noise_levels = noise_levels
    
    def __len__(self):
        return len(self.img_list['path_A'])
    
    def __getitem__(self, index):
        if self.task_name == 'shadow':
            img = Image.open(self.img_list['path_A'][index]).convert('RGB')
            gt_shadow = Image.open(self.img_list['path_B'][index])
            gt = Image.open(self.img_list['path_C'][index]).convert('RGB')
            img, gt_shadow, gt = self.img_transform(self.phase, [img, gt_shadow, gt])
        
        elif self.task_name == 'denoise':
            if self.phase in ['train', 'val']:
                clean_img = Image.open(self.img_list['path_A'][index]).convert('RGB')
                noise_level = random.choice(self.noise_levels)
                img = add_gaussian_noise(clean_img, noise_level)
                gt = clean_img
            else:
                img = Image.open(self.img_list['path_A'][index]).convert('RGB')
                gt = Image.open(self.img_list['path_C'][index]).convert('RGB')
            
            gt_shadow = Image.new('L', img.size, 0)
            img, gt_shadow, gt = self.img_transform(self.phase, [img, gt_shadow, gt])
        
        elif self.task_name in ['lol', 'rain']:
            img = Image.open(self.img_list['path_A'][index]).convert('RGB')
            gt = Image.open(self.img_list['path_C'][index]).convert('RGB')
            gt_shadow = Image.new('L', img.size, 0)
            img, gt_shadow, gt = self.img_transform(self.phase, [img, gt_shadow, gt])
        
        return img, gt_shadow, gt, self.task_name


class MultiTaskDataset(data.Dataset):
    """Combined multi-task dataset"""
    def __init__(self, datasets):
        self.datasets = datasets
        self.task_names = list(datasets.keys())
        self.lengths = {name: len(ds) for name, ds in datasets.items()}
        self.total_length = sum(self.lengths.values())
        
        self.cumulative_lengths = []
        cumsum = 0
        for name in self.task_names:
            cumsum += self.lengths[name]
            self.cumulative_lengths.append(cumsum)
    
    def __len__(self):
        return self.total_length
    
    def __getitem__(self, index):
        for i, cumlen in enumerate(self.cumulative_lengths):
            if index < cumlen:
                task_name = self.task_names[i]
                task_idx = index - (self.cumulative_lengths[i-1] if i > 0 else 0)
                return self.datasets[task_name][task_idx]
        
        raise IndexError("Index out of range")


def create_multitask_dataloaders(batch_size=8, num_workers=4, size=286, crop_size=256, rate=0.95):
    """Create dataloaders for all tasks"""
    
    img_transform = ImageTransform(size=size, crop_size=crop_size, mean=(0.5,), std=(0.5,))
    
    # Load all datasets
    datasets_train = {}
    datasets_val = {}
    datasets_test = {}
    
    # Shadow removal
    try:
        shadow_train, shadow_val = make_shadow_datapath_list(phase='train', rate=rate)
        shadow_test = make_shadow_datapath_list(phase='test')
        
        datasets_train['shadow'] = TaskDataset(shadow_train, img_transform, 'train', 'shadow')
        datasets_val['shadow'] = TaskDataset(shadow_val, img_transform, 'test_no_crop', 'shadow')
        datasets_test['shadow'] = TaskDataset(shadow_test, img_transform, 'test_no_crop', 'shadow')
        print(f" Shadow: Train={len(datasets_train['shadow'])}, Val={len(datasets_val['shadow'])}, Test={len(datasets_test['shadow'])}")
    except Exception as e:
        print(f" Shadow dataset not found: {e}")
    
    # Denoising
    try:
        denoise_train, denoise_val = make_denoise_datapath_list(phase='train', rate=rate)
        
        datasets_train['denoise'] = TaskDataset(denoise_train, img_transform, 'train', 'denoise', noise_levels=[15, 25, 50])
        datasets_val['denoise'] = TaskDataset(denoise_val, img_transform, 'val', 'denoise', noise_levels=[15, 25, 50])
        
        datasets_test['denoise'] = {}
        for noise_level in [15, 25, 50]:
            denoise_test = make_denoise_datapath_list(phase='test', noise_level=noise_level)
            datasets_test['denoise'][noise_level] = TaskDataset(denoise_test, img_transform, 'test_no_crop', 'denoise')
        
        print(f" Denoise: Train={len(datasets_train['denoise'])}, Val={len(datasets_val['denoise'])}, Test=[15:{len(datasets_test['denoise'][15])}, 25:{len(datasets_test['denoise'][25])}, 50:{len(datasets_test['denoise'][50])}]")
    except Exception as e:
        print(f" Denoise dataset not found: {e}")
    
    # Low-light (LOL)
    try:
        lol_train, lol_val = make_lol_datapath_list(phase='train', rate=rate)
        lol_test = make_lol_datapath_list(phase='test')
        
        datasets_train['lol'] = TaskDataset(lol_train, img_transform, 'train', 'lol')
        datasets_val['lol'] = TaskDataset(lol_val, img_transform, 'test_no_crop', 'lol')
        datasets_test['lol'] = TaskDataset(lol_test, img_transform, 'test_no_crop', 'lol')
        print(f" LOL: Train={len(datasets_train['lol'])}, Val={len(datasets_val['lol'])}, Test={len(datasets_test['lol'])}")
    except Exception as e:
        print(f" LOL dataset not found: {e}")
    
    # Deraining (Rain100L)
    try:
        rain_train, rain_val = make_rain_datapath_list(phase='train', rate=rate)
        rain_test = make_rain_datapath_list(phase='test')
        
        datasets_train['rain'] = TaskDataset(rain_train, img_transform, 'train', 'rain')
        datasets_val['rain'] = TaskDataset(rain_val, img_transform, 'test_no_crop', 'rain')
        datasets_test['rain'] = TaskDataset(rain_test, img_transform, 'test_no_crop', 'rain')
        print(f" Rain: Train={len(datasets_train['rain'])}, Val={len(datasets_val['rain'])}, Test={len(datasets_test['rain'])}")
    except Exception as e:
        print(f" Rain dataset not found: {e}")
    
    # Create multi-task datasets
    multitask_train = MultiTaskDataset(datasets_train)
    multitask_val = MultiTaskDataset(datasets_val)
    
    # Create dataloaders
    train_loader = data.DataLoader(multitask_train, batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=True)
    val_loader = data.DataLoader(multitask_val, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True)
    
    # Test loaders for each task
    test_loaders = {}
    for task_name, dataset in datasets_test.items():
        if task_name == 'denoise':
            test_loaders[task_name] = {}
            for noise_level, ds in dataset.items():
                test_loaders[task_name][noise_level] = data.DataLoader(ds, batch_size=1, shuffle=False, num_workers=num_workers)
        else:
            test_loaders[task_name] = data.DataLoader(dataset, batch_size=1, shuffle=False, num_workers=num_workers)
    
    return train_loader, val_loader, test_loaders, datasets_train, datasets_val