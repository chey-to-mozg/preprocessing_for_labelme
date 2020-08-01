import os
import matplotlib.pyplot as plt
import cv2
import tensorflow.keras as keras
import segmentation_models as sm
import albumentations as A
sm.set_framework('tf.keras')
import random
import re
import glob
import os.path as osp
import json
import labelme.utils.shape as transform

import sys
sys.path.append('super_resolution')
from model.edsr import edsr
from model import resolve_single
import numpy as np


def change_hist(img):
    hist,bins = np.histogram(img.flatten(),(2**16 - 1),[0,(2**16 - 1)])
    cdf = hist.cumsum()
    cdf_m = np.ma.masked_equal(cdf,0)
    cdf_m = (cdf_m - cdf_m.min())*(2**16 - 1)/(cdf_m.max()-cdf_m.min())
    cdf = np.ma.filled(cdf_m,0).astype('uint16')
    img2 = cdf[img]

    return img2


def change_hist_v2(img, bits=8):
    min = np.min(img)
    max = np.max(img)
    dif = max - min

    step = (2**bits - 1) / dif

    img2 = (img - min) * step
    img2 = np.asarray(img2, dtype=f'uint{bits}')

    return img2


def erode_dilate(mask, mask_size=(5,5), iter=2):
    kernel = np.ones(mask_size, np.uint8)
    erosion = cv2.erode(mask, kernel, iterations = iter)
    dilation = cv2.dilate(erosion, kernel, iterations = iter)

    return dilation


def change_hist_v3(img):
    img2 = img - np.min(img)
    img2[img2 > 255] = 255

    return img2.astype('uint8')


def mask_applying(img, mask, bits=8, depth=4):
    img2 = np.zeros((img.shape[0], img.shape[1], 3))
    img2[..., 1] = img
    img2[..., 2] = img

    mask = mask / (2**bits - 1)
    mask = 1 - mask
    mask = ((mask ** depth) * (2**bits - 1))
    img2[..., 0] = mask

    return img2.astype(f'uint{bits}')


def super_resolution_x4(img):

    weights_dir = f'super_resolution/weights/edsr-16-x4'
    weights_file = os.path.join(weights_dir, 'weights.h5')
    depth = 16
    scale = 4

    model = edsr(scale=scale, num_res_blocks=depth)
    model.load_weights(weights_file)

    if len(img.shape) < 3:
        lr = np.zeros((img.shape[0], img.shape[1], 3), dtype=img.dtype)
        lr[:, :, 0] = img
        lr[:, :, 1] = img
        lr[:, :, 2] = img
    else:
        lr = img
    sr = resolve_single(model, lr)
    return np.asarray(sr)


def split_set(set_, k_train, k_val, k_test):
    quantity = len(set_)
    random.shuffle(set_)
    train_quantity = round(quantity*k_train)
    val_quatity= round(quantity*k_val)

    train = set_[0:train_quantity]
    val = set_[train_quantity:train_quantity+val_quatity]
    test= set_[-1:train_quantity+val_quatity-1:-1]

    return train, val, test


def sync_test_redundant_images(path_to_images, images, annotations):
    images_sync = []
    for annot_image in annotations:
        annot_image = re.split('[./\\\]', annot_image)[-2]
        im_name = os.path.join(path_to_images,'train',annot_image + '.tif')
        if os.path.exists(im_name):
            images_sync.append(im_name)
        else:
            images_sync.append(os.path.join(path_to_images,'train',annot_image + '.png'))
    return images_sync, annotations


def load_train_annot_dataset(path_to_images, k_train=0.85, k_val=0.1, k_test=0.05):
    train_paths = glob.glob(os.path.join(path_to_images, 'train', '*'))
    annot_paths = glob.glob(os.path.join(path_to_images, 'anot', '*'))

#     train_paths =  os.listdir(os.path.join(path_to_images, 'train'))
#     annot_paths = os.listdir(os.path.join(path_to_images, 'annot'))

    train_annotations, val_annotations, test_annotations = split_set(annot_paths, k_train, k_val, k_test)
    train_images, val_images, test_images = split_set(train_paths, k_train, k_val, k_test)

    train_images, train_annotations = sync_test_redundant_images(path_to_images, train_images, train_annotations)
    test_images, test_annotations = sync_test_redundant_images(path_to_images,test_images, test_annotations)
    val_images, val_annotations = sync_test_redundant_images(path_to_images,val_images, val_annotations)

    assert len(annot_paths) == len(train_annotations) + len(val_annotations) + len(test_annotations)




    return train_images, train_annotations, val_images, val_annotations, test_images, test_annotations


def visualize(**images):
    """PLot images in one row."""
    n = len(images)
    plt.figure(figsize=(30, 30))
    for i, (name, image) in enumerate(images.items()):
        plt.subplot(1, n, i + 1)
        plt.xticks([])
        plt.yticks([])
        plt.title(' '.join(name.split('_')).title())
        plt.imshow(image)
    plt.show()

# helper function for data visualization
def denormalize(x):
    """Scale image to range 0..1 for correct plot"""
    x_max = np.percentile(x, 98)
    x_min = np.percentile(x, 2)
    x = (x - x_min) / (x_max - x_min)
    x = x.clip(0, 1)
    return x


# classes for data loading and preprocessing
class Dataset:
    """CamVid Dataset. Read images, apply augmentation and preprocessing transformations.

    Args:
        images_dir (str): path to images folder
        masks_dir (str): path to segmentation masks folder
        class_values (list): values of classes to extract from segmentation mask
        augmentation (albumentations.Compose): data transfromation pipeline
            (e.g. flip, scale, etc.)
        preprocessing (albumentations.Compose): data preprocessing
            (e.g. noralization, shape manipulation, etc.)

    """

    CLASSES = ['leaf']

    def __init__(
            self,
            images,
            masks,
            classes=None,
            augmentation=None,
            preprocessing=None,
    ):

        self.masks_fps = masks
        self.images_fps = images
        self.ids = len(self.masks_fps) #if len(self.masks_fps) == len(self.images_fps)
        # convert str names to class values on masks4
        self.class_values = [self.CLASSES.index(cls.lower()) for cls in classes]

        self.augmentation = augmentation
        self.preprocessing = preprocessing

    def __getitem__(self, i):

        # read data
        image = cv2.imread(self.images_fps[i], -1)
        #image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        mask = cv2.imread(self.masks_fps[i], 0)
        # extract certain classes from mask (e.g. cars)
        masks = [(mask == v) for v in self.class_values]
        mask = np.stack(masks, axis=-1).astype('float')

        # add background if mask is not binary
        if mask.shape[-1] != 1:
            background = 1 - mask.sum(axis=-1, keepdims=True)
            mask = np.concatenate((mask, background), axis=-1)

        # apply augmentations
        if self.augmentation:

            sample = self.augmentation(image=image, mask=mask)
            image, mask = sample['image'], sample['mask']

        # apply preprocessing
        if self.preprocessing:
            sample = self.preprocessing(image=image, mask=mask)
            image, mask = sample['image'], sample['mask']

        return image, mask

    def __len__(self):
        return self.ids


class Dataloder(keras.utils.Sequence):
    """Load data from dataset and form batches

    Args:
        dataset: instance of Dataset class for image loading and preprocessing.
        batch_size: Integet number of images in batch.
        shuffle: Boolean, if `True` shuffle image indexes each epoch.
    """

    def __init__(self, dataset, batch_size=1, shuffle=False):
        self.dataset = dataset
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.indexes = np.arange(len(dataset))

        self.on_epoch_end()

    def __getitem__(self, i):

        # collect batch data
        start = i * self.batch_size
        stop = (i + 1) * self.batch_size
        X, Y = [], []
        for j in range(start, stop):
            data = self.dataset[j]
            X.append(data[0])
            Y.append(data[1])
        # transpose list of lists
        X = np.array(X)
        Y = np.array(Y)
        ######
        if len(X.shape) < 4:
            X = np.expand_dims(X, axis=3)
        return X, Y

    def __len__(self):
        """Denotes the number of batches per epoch"""
        return len(self.indexes) // self.batch_size

    def on_epoch_end(self):
        """Callback function to shuffle indexes each epoch"""
        if self.shuffle:
            self.indexes = np.random.permutation(self.indexes)


def round_clip_0_1(x, **kwargs):
    return x.round().clip(0, 1)

# define heavy augmentations
def get_training_augmentation():
    train_transform = [
        # A.ToFloat(always_apply=True),
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.1),
        A.ShiftScaleRotate(scale_limit=(-0.2,0.2), rotate_limit=(-15,15), shift_limit=(-0.01,0.01), p=0.7, border_mode=cv2.BORDER_CONSTANT, mask_value=1),
        # A.RandomCrop(height=640, width=960, always_apply=True),
        # A.RandomCrop(height=320, width=320, p=0.3),
        # A.PadIfNeeded(min_height=640, min_width=960, always_apply=True, border_mode=cv2.BORDER_CONSTANT, mask_value=1),
        # A.CLAHE(p=0.2),

        # A.OneOf(
        #     [
        #         # A.CLAHE(p=1),
        #         A.RandomBrightnessContrast(p=0.2),
        #         A.RandomGamma(p=0.2),
        #     ],
        #     p=0.3,
        # ),
        #
        # A.OneOf(
        #     [
        #         A.Blur(blur_limit=3, p=0.5),
        #         A.MotionBlur(blur_limit=3, p=0.5),
        #     ],
        #     p=0.3,
        # ),

        A.Lambda(mask=round_clip_0_1)
    ]
    return A.Compose(train_transform)


def get_validation_augmentation():
    """Add paddings to make image shape divisible by 32"""
    test_transform = [
        # A.ToFloat(always_apply=True),
        # A.RandomCrop(height=640, width=960, always_apply=True)
    ]
    return A.Compose(test_transform)

def get_preprocessing(preprocessing_fn):
    """Construct preprocessing transform

    Args:
        preprocessing_fn (callbale): data normalization function
            (can be specific for each pretrained neural network)
    Return:
        transform: albumentations.Compose

    """

    _transform = [
        A.Lambda(image=preprocessing_fn),
    ]
    return A.Compose(_transform)


def json2mask(json_path, mask_path_save, show_masks=False):

    jsons = os.listdir(json_path)

    for js in jsons:
        print(f'{js} started')
        with open(osp.join(json_path, js), encoding='utf-8') as f:
            data = json.load(f)

        label_name_to_value = {'_background_': 0}
        for shape in sorted(data['shapes'], key=lambda x: x['label']):
            label_name = shape['label']
            if label_name in label_name_to_value:
                label_value = label_name_to_value[label_name]
            else:
                label_value = len(label_name_to_value)
                label_name_to_value[label_name] = label_value

        mask = transform.shapes_to_label([data['imageHeight'], data['imageWidth']], data['shapes'], label_name_to_value)
        if show_masks:
            plt.imshow(mask[0])
        cv2.imwrite(osp.join(mask_path_save, js.split('.')[0] + '.png'), mask[0])