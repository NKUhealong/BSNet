import cv2
import torch
import os
import random
import numpy as np
from torch.utils.data import Dataset,DataLoader
from torchvision import transforms
import torch.nn as nn
import torch.nn.functional as F
import itertools
from scipy import ndimage
from torch.utils.data.sampler import Sampler


def random_rot_flip(image, label):
    k = np.random.randint(0, 4)
    image = np.rot90(image, k)
    label = np.rot90(label, k)
    axis = np.random.randint(0, 2)
    image = np.flip(image, axis=axis).copy()
    label = np.flip(label, axis=axis).copy()
    return image, label

    
################################  semi train dataset  ###########################
class vessel_BaseDataSets(Dataset):
    def __init__(self, base_dir, list_name,image_size,dataset,transform):
        self.base_dir = base_dir
        self.sample_list = []
        self.h, self.w = image_size
        self.dataset = dataset
        self.transform = transform
        with open(self.base_dir + list_name, 'r') as f1:
            self.sample_list = f1.readlines()
        self.sample_list = [item.replace('\n', '')  for item in self.sample_list]
        #self.sample_list = self.sample_list[0:700]
        print("total {} samples".format(len(self.sample_list)))

    def __len__(self):
        return len(self.sample_list)

    def __getitem__(self, idx):
        name = self.sample_list[idx]
        if self.dataset == 'skin':
            image = cv2.imread(self.base_dir + '/images/'+name+'.jpg', cv2.IMREAD_COLOR)
        else:
            image = cv2.imread(self.base_dir + '/images/'+name+'.png', cv2.IMREAD_COLOR)
            
        label = cv2.imread(self.base_dir + '/masks/'+name+'.png', cv2.IMREAD_GRAYSCALE)
        image = cv2.resize(image,(self.w, self.h),interpolation = cv2.INTER_NEAREST)
        label = cv2.resize(label,(self.w, self.h),interpolation = cv2.INTER_NEAREST)/255
        '''
        if random.random() > 0.5:
            image, label = random_rot_flip(image, label)
        '''
        image = np.asarray(image, np.float32)
        image = image.transpose((2, 0, 1))
        sample = {'image': image, 'label': label}
        sample = self.transform(sample)
        sample["idx"] = name
        return sample

class TwoStreamBatchSampler(Sampler):

    def __init__(self, primary_indices, secondary_indices, batch_size, secondary_batch_size):
        self.primary_indices = primary_indices
        self.secondary_indices = secondary_indices
        self.secondary_batch_size = secondary_batch_size
        self.primary_batch_size = batch_size - secondary_batch_size

        assert len(self.primary_indices) >= self.primary_batch_size > 0
        assert len(self.secondary_indices) >= self.secondary_batch_size > 0

    def __iter__(self): 
        primary_iter = iterate_once(self.primary_indices) 
        secondary_iter = iterate_eternally(self.secondary_indices)
        return (
            #secondary_batch + primary_batch
            primary_batch + secondary_batch
            for (primary_batch, secondary_batch)
            in zip(grouper(primary_iter, self.primary_batch_size),
                   grouper(secondary_iter, self.secondary_batch_size))
        )

    def __len__(self):
        return len(self.primary_indices) // self.primary_batch_size


def iterate_once(iterable):
    return np.random.permutation(iterable) 


def iterate_eternally(indices):
    def infinite_shuffles():
        while True:
            yield np.random.permutation(indices)
    return itertools.chain.from_iterable(infinite_shuffles())

def grouper(iterable, n):
    args = [iter(iterable)] * n
    return zip(*args)

################################  test dataset  ###########################
class testBaseDataSets(Dataset):
    def __init__(self, base_dir, list_name,image_size,dataset,transform):
        self.base_dir = base_dir
        self.sample_list = []
        self.h, self.w = image_size
        self.dataset = dataset
        self.transform = transform
        with open(self.base_dir + list_name, 'r') as f1:
            self.sample_list = f1.readlines()
        self.sample_list = [item.replace('\n', '')  for item in self.sample_list]
        print("total {} samples".format(len(self.sample_list)))

    def __len__(self):
        return len(self.sample_list)

    def __getitem__(self, idx):
        name = self.sample_list[idx]
        if self.dataset == 'skin':
            image = cv2.imread(self.base_dir + '/images/'+name+'.jpg', cv2.IMREAD_COLOR)
        else:
            image = cv2.imread(self.base_dir + '/images/'+name+'.png', cv2.IMREAD_COLOR)
            
        label = cv2.imread(self.base_dir + '/masks/'+name+'.png', cv2.IMREAD_GRAYSCALE)
        image = cv2.resize(image,(self.w, self.h),interpolation = cv2.INTER_NEAREST)
        label = cv2.resize(label,(self.w, self.h),interpolation = cv2.INTER_NEAREST)/255
        image = np.asarray(image, np.float32)
        image = image.transpose((2, 0, 1))
        sample = {'image': image, 'label': label}
        sample = self.transform(sample)
        return sample

################################  fully supvised  ######################################       

class MyDataSet(Dataset):
    def __init__(self, root, list_name, train_num, image_size,dataset):
        self.root = root
        self.list_path = self.root + list_name
        self.h, self.w = image_size
        self.dataset = dataset
        self.img_ids = [i_id.strip() for i_id in open(self.list_path)]
        if train_num>700:
            self.img_ids = self.img_ids[:]
        else:
            self.img_ids = self.img_ids[:train_num]
        self.files = []
        for name in self.img_ids:
            if self.dataset == 'skin':
                img_file = os.path.join(self.root, "images/%s.jpg" % name)
            else:
                img_file = os.path.join(self.root, "images/%s.png" % name)
            label_file = os.path.join(self.root, "masks/%s.png" % name)
            self.files.append({"img": img_file,"label": label_file, "name": name})
        #np.random.shuffle(self.files)
        print("total {} samples".format(len(self.files)))

    def __len__(self):
        return len(self.files)

    def __getitem__(self, index):
        datafiles = self.files[index]
        image = cv2.imread(datafiles["img"], cv2.IMREAD_COLOR)
        label = cv2.imread(datafiles["label"], cv2.IMREAD_GRAYSCALE)
        
        image = cv2.resize(image,(self.w, self.h),interpolation = cv2.INTER_NEAREST)
        label = cv2.resize(label,(self.w, self.h),interpolation = cv2.INTER_NEAREST)/255
        
        
        if random.random() > 0.5:
            image, label = random_rot_flip(image, label)
            
        name = datafiles["name"]

        image = np.asarray(image, np.float32)
        image = image.transpose((2, 0, 1))
        image = torch.from_numpy(image.astype(np.float32))
        label = torch.from_numpy(label.astype(np.uint8)).long()
        return image, label

################################  patch  ######################################  
def paint_border_overlap(images,patch_size,s):
    
    h = images.shape[0]  
    w = images.shape[1] 
    p_h = (h-patch_size)%s  
    p_w = (w-patch_size)%s  
        
    new_images = np.zeros((h+(s-p_h),w+(s - p_w),3))
    new_images[0:h,0:w,:] = images
    #print(new_images.shape)
    return new_images

def extract_ordered_overlap(new_images, patch_size,s):
    
    image_h = new_images.shape[0]  
    image_w = new_images.shape[1] 
    N_patches_tot = ((image_h-patch_size)//s+1)*((image_w-patch_size)//s+1) 
    
    patches = np.empty((N_patches_tot,patch_size,patch_size,3))
    iter_tot = 0   
    for h in range((image_h-patch_size)//s+1):
        for w in range((image_w-patch_size)//s+1):
            patch = new_images[h*s:(h*s)+patch_size,w*s:(w*s)+patch_size,:]
            patches[iter_tot]=patch
            iter_tot =iter_tot+1                   
    return patches
        
class RandomCrop(object):

    def __init__(self, output_size):
        assert isinstance(output_size, (int, tuple))
        if isinstance(output_size, int):
            self.output_size = (output_size, output_size)
        else:
            assert len(output_size) == 2
            self.output_size = output_size

    def __call__(self, sample):
        image, label = sample['image'], sample['label']

        h, w = image.shape[1:3]
        new_h, new_w = self.output_size
        #print(h,new_h)
        top = np.random.randint(0, h - new_h)
        left = np.random.randint(0, w - new_w)

        image = image[:,top: top + new_h, left: left + new_w]
        label = label[top: top + new_h, left: left + new_w]
        
        return {'image': image, 'label': label}
        
class RandomGenerator(object):
    def __init__(self):
        self.k = 9
    def __call__(self, sample):
        image, label = sample['image'], sample['label']
        image = torch.from_numpy(image.astype(np.float32))
        label = torch.from_numpy(label.astype(np.uint8)).long()
        sample = {'image': image, 'label': label}
        return sample