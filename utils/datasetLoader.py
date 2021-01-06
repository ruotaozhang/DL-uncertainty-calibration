import torch
import torchvision.transforms as transforms
import numpy as np
import random
import os
from torch.utils.data import DataLoader
from PIL import Image


class DRDataset(torch.utils.data.Dataset):
    def __init__(self, file_list, img_path, label_path, transform=None, is_binary=True):
        self.file_list = file_list
        self.img_path = img_path
        self.name2label = dict(np.loadtxt(os.path.join(label_path, 'trainLabels.csv'), dtype=str, delimiter=','))
        self.transform = transform
        self.is_binary = is_binary

    def __len__(self):
        return len(self.file_list)

    def __getitem__(self, idx):
        img_file_name = self.file_list[idx]  # contains .jpeg
        img_name = img_file_name[:img_file_name.index('.jpeg')]  # does not contain .jpeg
        img = Image.open(os.path.join(self.img_path, img_file_name))
        label = int(self.name2label[img_name])
        if self.is_binary:
            label = 1 if label >= 2 else 0
        if self.transform is not None:
            img = self.transform(img)
        return img, label


def DR_data_loader(img_path, label_path, subset_size=5000, seed=666, batch_size=64, b_seed=None,
                   num_workers=8, verbose=0) -> dict:
    # negative subset_size means use the whole dataset. 
    data_files = sorted(os.listdir(img_path))
    assert subset_size < len(data_files), 'subset_size must be smaller than the total number of images'

    # split data into train/val/test
    if subset_size > 0:
        random.seed(seed)
        train_val_files = sorted(random.sample(data_files, subset_size))  # random sampling without replacement
        val_size = 0.2
        num_train_val = len(train_val_files)
        num_val = int(np.floor(val_size * num_train_val))
    else:
        random.seed(seed)
        train_val_files = sorted(random.sample(data_files, 32626))
        num_val = 2500

    test_files = sorted(list(filter(lambda x: x not in train_val_files, data_files)))

    random.seed(seed * 666)
    val_files = sorted(random.sample(train_val_files, num_val))
    train_files = sorted(list(filter(lambda x: x not in val_files, train_val_files)))
    num_train = len(train_files)

    # define transformations
    # normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],std=[0.229, 0.224, 0.225])
    normalize = transforms.Normalize(mean=[0.42606387, 0.29752496, 0.21309826], std=[0.27662534, 0.20280295, 0.1687619])
    train_transform = transforms.Compose([transforms.RandomAffine(degrees=180, translate=(0.05, 0.05))
                                             , transforms.Resize((550, 550))
                                             , transforms.RandomCrop(512)
                                             , transforms.RandomHorizontalFlip()
                                             , transforms.RandomVerticalFlip()
                                             , transforms.ToTensor()
                                             , normalize
                                          ])
    val_transform = transforms.Compose([transforms.Resize((550, 550))
                                           , transforms.CenterCrop(512)
                                           , transforms.ToTensor()
                                           , normalize
                                        ])
    # training dataset
    train_aug_set = DRDataset(file_list=train_files, img_path=img_path, label_path=label_path,
                              transform=train_transform)
    train_set = DRDataset(file_list=train_files, img_path=img_path, label_path=label_path, transform=val_transform)

    # bootstrap dataset
    if b_seed is not None:
        random.seed(b_seed)
    else:
        random.seed()
    b_indices = random.choices(range(num_train), k=num_train)
    oob_indices = list(filter(lambda x: x not in b_indices, range(num_train)))
    train_b_aug_dataset = torch.utils.data.Subset(train_aug_set, b_indices)
    train_oob_dataset = torch.utils.data.Subset(train_set, oob_indices)
    train_oob_aug_dataset = torch.utils.data.Subset(train_aug_set, oob_indices)

    # val and test datasets
    val_set = DRDataset(file_list=val_files, img_path=img_path, label_path=label_path, transform=val_transform)
    val_aug_set = DRDataset(file_list=val_files, img_path=img_path, label_path=label_path, transform=train_transform)
    test_set = DRDataset(file_list=test_files, img_path=img_path, label_path=label_path, transform=val_transform)
    test_aug_set = DRDataset(file_list=test_files, img_path=img_path, label_path=label_path, transform=train_transform)

    # define dataloaders
    pin_memory = torch.cuda.is_available()
    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=False, num_workers=num_workers,
                              pin_memory=pin_memory)
    train_augment_loader = DataLoader(train_aug_set, batch_size=batch_size, shuffle=True,
                                      num_workers=num_workers, pin_memory=pin_memory)
    train_b_aug_loader = DataLoader(train_b_aug_dataset, batch_size=batch_size, shuffle=True,
                                    num_workers=num_workers, pin_memory=pin_memory)
    train_oob_loader = DataLoader(train_oob_dataset, batch_size=batch_size, shuffle=False,
                                  num_workers=num_workers, pin_memory=pin_memory)
    train_oob_aug_loader = DataLoader(train_oob_aug_dataset, batch_size=batch_size, shuffle=False,
                                      num_workers=num_workers, pin_memory=pin_memory)
    val_loader = DataLoader(val_set, batch_size=batch_size, shuffle=False, num_workers=num_workers,
                            pin_memory=pin_memory)
    val_aug_loader = DataLoader(val_aug_set, batch_size=batch_size, shuffle=False,
                                num_workers=num_workers, pin_memory=pin_memory)
    test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=False, num_workers=num_workers,
                             pin_memory=pin_memory)
    test_aug_loader = DataLoader(test_aug_set, batch_size=batch_size, shuffle=False, num_workers=num_workers,
                                 pin_memory=pin_memory)

    assert len(train_oob_loader.sampler) == len(oob_indices), 'oob data length wrong'

    if verbose == 1:
        print('DRD training data have {} images'.format(len(train_augment_loader.sampler)))
        print('DRD val data have {} images'.format(len(val_loader.sampler)))
        print('DRD test data have {} images'.format(len(test_loader.sampler)))
        print('DRD train bootstrap data have {} unique images'.format(len(set(b_indices))))
        print('DRD train oob data have {} images'.format(len(train_oob_loader.sampler)))

    return {'train': train_loader, 'train_aug': train_augment_loader, 'train_b_aug': train_b_aug_loader,
            'train_oob': train_oob_loader, 'train_oob_aug': train_oob_aug_loader, 'val': val_loader,
            'test': test_loader, 'val_aug': val_aug_loader, 'test_aug': test_aug_loader, 'train_files': train_files,
            'val_files': val_files, 'test_files': test_files, 'data_files': data_files}


class AptosDataset(torch.utils.data.Dataset):
    def __init__(self, file_list, img_path, label_path, transform=None, is_binary=True):
        self.file_list = file_list
        self.img_path = img_path
        self.name2label = dict(np.loadtxt(os.path.join(label_path, 'trainLabels.csv'), dtype=str, delimiter=','))
        self.transform = transform
        self.is_binary = is_binary

    def __len__(self):
        return len(self.file_list)

    def __getitem__(self, idx):
        img_file_name = self.file_list[idx]  # contains .png
        img_name = img_file_name[:img_file_name.index('.png')]  # does not contain .png
        img = Image.open(os.path.join(self.img_path, img_file_name))
        label = int(self.name2label[img_name])
        if self.is_binary:
            label = 1 if label >= 2 else 0
        if self.transform:
            img = self.transform(img)
        return img, label


def Aptos_data_loader(img_path, label_path, seed=666, batch_size=64, num_workers=8, verbose=0) -> dict:
    data_files = sorted(os.listdir(img_path))

    # split into train/test
    random.seed(seed)
    train_files = sorted(random.sample(data_files, 2930))  # random sampling without replacement
    test_files = sorted(list(set(data_files) - set(train_files)))

    train_idx = [data_files.index(file) for file in train_files]
    test_idx = [data_files.index(file) for file in test_files]

    # normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],std=[0.229, 0.224, 0.225])
    normalize = transforms.Normalize(mean=[0.42606387, 0.29752496, 0.21309826], std=[0.27662534, 0.20280295, 0.1687619])
    train_transform = transforms.Compose([transforms.RandomAffine(degrees=180, translate=(0.05, 0.05))
                                             , transforms.Resize((550, 550))
                                             , transforms.RandomCrop(512)
                                             , transforms.RandomHorizontalFlip()
                                             , transforms.RandomVerticalFlip()
                                             , transforms.ToTensor()
                                             , normalize
                                          ])
    val_transform = transforms.Compose([transforms.Resize((550, 550))
                                           , transforms.CenterCrop(512)
                                           , transforms.ToTensor()
                                           , normalize
                                        ])

    # datasets
    all_aug_dataset = AptosDataset(file_list=data_files, img_path=img_path, label_path=label_path,
                                   transform=train_transform)
    all_dataset = AptosDataset(file_list=data_files, img_path=img_path, label_path=label_path, transform=val_transform)
    train_dataset = AptosDataset(file_list=train_files, img_path=img_path, label_path=label_path,
                                 transform=val_transform)
    train_aug_dataset = AptosDataset(file_list=train_files, img_path=img_path, label_path=label_path,
                                     transform=train_transform)
    test_dataset = AptosDataset(file_list=test_files, img_path=img_path, label_path=label_path, transform=val_transform)

    # define dataloaders
    pin_memory = torch.cuda.is_available()
    all_loader = DataLoader(all_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers,
                            pin_memory=pin_memory)
    all_aug_loader = DataLoader(all_aug_dataset, batch_size=batch_size, shuffle=True,
                                num_workers=num_workers, pin_memory=pin_memory)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=False,
                              num_workers=num_workers, pin_memory=pin_memory)
    train_aug_loader = DataLoader(train_aug_dataset, batch_size=batch_size, shuffle=True,
                                  num_workers=num_workers, pin_memory=pin_memory)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers,
                             pin_memory=pin_memory)

    if verbose == 1:
        print('APTOS data have {} images'.format(len(all_loader.sampler)))
        print('APTOS train set has {} images'.format(len(train_loader.sampler)))
        print('APTOS test set has {} images'.format(len(test_loader.sampler)))

    return {'all': all_loader, 'all_aug': all_aug_loader, 'train': train_loader, 'test': test_loader,
            'train_aug': train_aug_loader, 'data_files': data_files, 'train_files': train_files,
            'test_files': test_files, 'train_idx': train_idx, 'test_idx': test_idx}
