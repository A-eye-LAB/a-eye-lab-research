import glob
import os
import random
from collections import Counter

import numpy as np
from PIL import Image
from scipy import stats


def get_all_paths(path_exp):
    return [p for p in glob.glob(path_exp) if os.path.isdir(p)]


def get_all_image_paths(all_paths):
    all_images = []
    for path in all_paths:
        all_images += glob.glob(path + "/*")
    return all_images


def get_image_w_h(all_image_paths):
    all_widths, all_heights = [], []

    for image_path in all_image_paths:
        img = Image.open(image_path)
        w, h = img.size
        all_widths.append(w)
        all_heights.append(h)

    return all_widths, all_heights


def get_stats(props):
    print("min:", min(props))
    print("max:", max(props))
    print(f"mean: {np.mean(props):.1f}")
    print(f"median: {np.median(props):.0f}")
    mode = stats.mode(props)
    print(f"mode: {mode.mode} ({mode.count / len(props) * 100:.1f} %)")


def count_files(all_paths):
    for path in all_paths:
        print(path, ":", len(os.listdir(path)))


def count_formats(all_image_paths):
    return Counter([_file.split(".")[-1] for _file in all_image_paths])


def count_image_sizes(all_widths, all_heights):
    return Counter(zip(all_widths, all_heights))


def show_random_image(path):
    filename = random.sample(os.listdir(path), 1)[0]
    print(filename)
    filepath = os.path.join(path, filename)
    img = Image.open(filepath)
    return img
