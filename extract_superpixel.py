import os
import numpy as np
import cv2
from skimage.color import rgb2gray
from skimage.filters import sobel
from skimage.segmentation import slic, watershed, felzenszwalb
from skimage.segmentation import mark_boundaries
from skimage.util import img_as_float
from skimage import io

# graph
from skimage.future import graph
from skimage.color import gray2rgb
import networkx as nx


from concurrent.futures import ProcessPoolExecutor
from multiprocessing import cpu_count
import tqdm
from functools import partial

import matplotlib.pyplot as plt

import argparse


parser = argparse.ArgumentParser()
parser.add_argument('--data_path', type=str,
                    help='path to nyu data',
                    required=True)
parser.add_argument('--output_dir', type=str,
                    help='where to store extracted segment',
                    required=True)
args = parser.parse_args()

data_path = args.data_path
output_dir = args.output_dir

if not os.path.exists(output_dir):
    os.makedirs(output_dir)

def undistort(image):
    k1 =  2.0796615318809061e-01
    k2 = -5.8613825163911781e-01
    p1 = 7.2231363135888329e-04
    p2 = 1.0479627195765181e-03
    k3 = 4.9856986684705107e-01

    fx = 5.1885790117450188e+02
    fy = 5.1946961112127485e+02
    cx = 3.2558244941119034e+02
    cy = 2.5373616633400465e+02

    kmat = np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]])
    dist = np.array([[k1, k2, p1, p2, k3]])
    image = cv2.undistort(image, kmat, dist)
    return image


def extract_superpixel(filename, index):
    CROP = 16
    scales = [1]
    markers = [400]
    filename = os.path.join(data_path, filename)
    image = cv2.imread(filename)
    # undistort
    image = undistort(image)
    image = image[CROP:-CROP, CROP:-CROP, :]

    segments = []
    Adjs = []
    Adjs_color = []

    for s, m in zip(scales, markers):
        image = cv2.resize(image, (384//s, 288//s))
        image = img_as_float(image)

        gradient = sobel(rgb2gray(image))
        # segment = watershed(gradient, markers=m, compactness=0.001)
        segment = felzenszwalb(image, scale=100, sigma=0.5, min_size=50)
        segments.append(segment)

    return segments[0].astype(np.int16)

def images2seg(filenames, index):
    assert(len(filenames) == 9)
    # [0, -4, -3, -2, -1, 1, 2, 3, 4]
    # [0, 1, 2, 3, 4, 5, 6, 7, 8]

    segments = {i: extract_superpixel(filenames[i], index) for i in [0]}
 
    np.savez(os.path.join(output_dir, "seg_%d.npz"%(index)), 
             segment_0 = segments[0])
    return
             


# multi processing fitting
executor = ProcessPoolExecutor(max_workers=cpu_count())
futures = []

lines = open("./splits//nyu_train_0_10_20_30_40.txt").readlines()
all_files = [line.split() for line in lines]

for index, files in enumerate(all_files):
    # if index >= 10000:
    #     continue

    task = partial(images2seg, files, index)
    futures.append(executor.submit(task))

results = []
[results.append(future.result()) for future in tqdm.tqdm(futures)]
