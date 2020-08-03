
import os
import matplotlib.pyplot as plt
import cv2


from data import DIV2K
from model.edsr import edsr
from train import EdsrTrainer
import numpy as np

# os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

weights_dir = f'weights/edsr-16-x4'
weights_file = os.path.join(weights_dir, 'weights.h5')

depth = 16

# Super-resolution factor
scale = 4


model = edsr(scale=scale, num_res_blocks=depth)
model.load_weights(weights_file)


from model import resolve_single
from utils import load_image, plot_sample

def resolve_and_plot(lr_image_path):

    lr = load_image(lr_image_path)
    print(f'{lr.shape[0]} {lr.shape[1]}')
    sr = resolve_single(model, lr)
    plot_sample(lr, sr)
    print(f'{sr.shape[0]} {sr.shape[1]}')
    return np.asarray(sr)
    plt.show()

resolve_and_plot('img1.png')
resolve_and_plot('img2.png')

