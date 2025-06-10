from PIL import Image, ImageEnhance
import numpy as np
import torch
from torchvision import transforms
import time
import random
from albumentations.augmentations.transforms import CLAHE
import torch.functional as F

IMG_SIZE = 640
image_options = {'resize': True, 'resize_size': IMG_SIZE}

class Resize(object):

    def __init__(self, output_size):
        assert isinstance(output_size, (int))
        self.output_size = output_size

    def __call__(self, sample):
        image, masks = sample['image'], sample['masks']
        image = Image.fromarray(image)
        image = transforms.Resize(self.output_size)(image)
        image = np.array(image)
        new_masks = []
        for mask in masks:
            img = Image.fromarray(mask)
            img = transforms.Resize(self.output_size)(img)
            new_masks.append(np.array(img))
        new_masks = np.array(new_masks)
        new_sample = {'image': image, 'masks': new_masks}
        return new_sample

class Resize_val_test(object):

    def __init__(self, output_size):
        assert isinstance(output_size, (int))
        self.output_size = output_size

    def __call__(self, sample):
        image, masks = sample['image'], sample['masks']
        image = Image.fromarray(image)
        image = transforms.Resize([self.output_size, self.output_size])(image)
        image = np.array(image)
        new_masks = []
        for mask in masks:
            img = Image.fromarray(mask)
            img = transforms.Resize([self.output_size, self.output_size])(img)
            new_masks.append(np.array(img))
        new_masks = np.array(new_masks)
        new_sample = {'image': image, 'masks': new_masks}
        return new_sample

class CenterCrop(object):

    def __init__(self, output_size):
        self.output_size = output_size

    def __call__(self, sample):
        image, masks = sample['image'], sample['masks']
        image = Image.fromarray(image)
        image = transforms.CenterCrop(self.output_size)(image)
        image = np.array(image)
        new_masks = []
        for mask in masks:
            img = Image.fromarray(mask)
            img = transforms.CenterCrop(self.output_size)(img)
            new_masks.append(np.array(img))
        new_masks = np.array(new_masks)
        new_sample = {'image': image, 'masks': new_masks}
        return new_sample

class RandomResizedCrop(object):

    def __init__(self, output_size):
        assert isinstance(output_size, (int))
        self.output_size = output_size

    def __call__(self, sample):
        image, masks = sample['image'], sample['masks']
        image = Image.fromarray(image)
        image = transforms.RandomResizedCrop(self.output_size)(image)
        image = np.array(image)
        new_masks = []
        for mask in masks:
            img = Image.fromarray(mask)
            img = transforms.RandomResizedCrop(self.output_size)(img)
            new_masks.append(np.array(img))
        new_masks = np.array(new_masks)
        new_sample = {'image': image, 'masks': new_masks}
        return new_sample

class Normalize(object):

    def __init__(self, mean, std):
        self.mean = mean
        self.std = std

    def __call__(self, sample):
        image, masks = sample['image'], sample['masks']
        image = Image.fromarray(image)
        image = F.normalize(image, self.mean, self.std)
        image = np.array(image)
        new_masks = []
        for mask in masks:
            img = Image.fromarray(mask)
            img = F.normalize(img, self.mean, self.std)
            new_masks.append(np.array(img))
        new_masks = np.array(new_masks)
        new_sample = {'image': image, 'masks': new_masks}
        return new_sample

class RandomHorizontalFlip(object):

    def __init__(self, flip_prob):
        self.flip_prob = flip_prob

    def __call__(self, sample):
        image, masks = sample['image'], sample['masks']
        if random.random() < self.flip_prob:
            img = Image.fromarray(image)
            image = transforms.RandomHorizontalFlip(p=1.0)(img)
            image = np.array(image)
            new_masks = []
            for mask in masks:
                img = Image.fromarray(mask)
                img = transforms.RandomHorizontalFlip(p=1.0)(img)
                new_masks.append(np.array(img))
            new_masks = np.array(new_masks)
            sample = {'image': image, 'masks': new_masks}
        return sample

class RandomVerticalFlip(object):

    def __init__(self, flip_prob):
        self.flip_prob = flip_prob

    def __call__(self, sample):
        image, masks = sample['image'], sample['masks']
        if random.random() < self.flip_prob:
            img = Image.fromarray(image)
            image = transforms.RandomVerticalFlip(p=1.0)(img)
            image = np.array(image)
            new_masks = []
            for mask in masks:
                img = Image.fromarray(mask)
                img = transforms.RandomVerticalFlip(p=1.0)(img)
                new_masks.append(np.array(img))
            new_masks = np.array(new_masks)
            sample = {'image': image, 'masks': new_masks}
        return sample

class RandomRotate90:

    def __init__(self, num_rot=(1, 2, 3, 4)):
        self.num_rot = num_rot
        self.axes = (0, 1)

    def __call__(self, sample):
        image, masks = sample['image'], sample['masks']
        n = np.random.choice(self.num_rot)
        image_rotate = np.ascontiguousarray(np.rot90(image, n, self.axes))
        new_masks = []
        for (i, mask) in enumerate(masks):
            new_masks.append(np.rot90(mask, n, self.axes))
        new_sample = {'image': image_rotate, 'masks': new_masks}
        return new_sample

class RandomRotate180:

    def __init__(self, num_rot=(1, 2, 3, 4)):
        self.num_rot = num_rot
        self.axes = (0, 1)

    def __call__(self, sample):
        image, masks = sample['image'], sample['masks']
        n = np.random.choice(self.num_rot)
        image_rotate = np.ascontiguousarray(np.rot180(image, n, self.axes))
        new_masks = []
        for (i, mask) in enumerate(masks):
            new_masks.append(np.rot180(mask, n, self.axes))
        new_sample = {'image': image_rotate, 'masks': new_masks}
        return new_sample

class ApplyCLAHE(object):

    def __init__(self, green=False):
        self.green = green

    def __call__(self, sample):
        light = CLAHE(clip_limit=2, p=1)
        image, masks = sample['image'], sample['masks']
        image = np.uint8(image)
        image = light(image=image, clip_limit=2)['image']
        if self.green:
            image = [image[:, :, 1]]
        new_sample = {'image': image, 'masks': masks}
        return new_sample

class ImageEnhencer(object):

    def __init__(self, color_jitter=False, green=False):
        self.color_jitter = color_jitter
        self.green = green

    def __call__(self, sample, color_jitter=False):
        t1 = time.time()
        image, masks = sample['image'], sample['masks']
        image = np.uint8(image)
        image = Image.fromarray(image)
        if self.color_jitter:
            image = transforms.ColorJitter(brightness=0.4, contrast=0.2, saturation=0, hue=0.05)(image)
            image = np.array(image)
        else:
            enh_bri = ImageEnhance.Brightness(image)
            brightness = 1.3
            image = enh_bri.enhance(brightness)
            enh_col = ImageEnhance.Color(image)
            color = 1.0
            image = enh_col.enhance(color)
            enh_con = ImageEnhance.Contrast(image)
            contrast = 1.2
            image = enh_con.enhance(contrast)
            image = np.array(image)
        if self.green:
            image = [image[:, :, 1]]
        masks = masks
        new_sample = {'image': image, 'masks': masks}
        return new_sample

class RandomCrop(object):

    def __init__(self, output_size):
        assert isinstance(output_size, (int, tuple))
        if isinstance(output_size, int):
            self.output_size = (output_size, output_size)
        else:
            assert len(output_size) == 2
            self.output_size = output_size

    def __call__(self, sample):
        image, masks = sample['image'], sample['masks']
        h, w = image.shape[:2]
        new_h, new_w = self.output_size
        top = np.random.randint(0, h - new_h)
        left = np.random.randint(0, w - new_w)
        image = image[top: top + new_h,
                left: left + new_w]
        new_masks = []
        for mask in masks:
            mask_crop = mask[top: top + new_h, left: left + new_w]
            new_masks.append(mask_crop)
        new_masks = np.array(new_masks)
        return {'image': image, 'masks': new_masks}

class ToTensor(object):

    def __init__(self, green=False):
        self.green = green

    def __call__(self, sample):
        image, masks = sample['image'], sample['masks']
        image = np.array(image)
        masks = torch.from_numpy(np.array(masks))
        if not self.green:
            image = np.rollaxis(image, 2, 0)
        image = torch.from_numpy(image)
        return {'image': image, 'masks': masks}