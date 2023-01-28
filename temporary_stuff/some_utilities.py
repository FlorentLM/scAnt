import numpy as np
from scipy.ndimage import convolve
from skimage import filters, io

import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt

##

# os.system(f"align_image_stack -x -m -g 5 -c 10 -s 4 -f {hfov_degrees} -v -a {aligned_folder / stack} --use-given-order {inputs_string}")
# os.system(f"align_image_stack -vvv -x -m -s 2 -f {full_frame_equiv_horiz_fov} -a {aligned_folder / stack} {inputs_string}")


##

def compute_contrast(image, use_blur=True):
    laplace_kernel = np.array([[0, 1, 0],
                               [1, -4, 1],
                               [0, 1, 0]])

    mono = image.mean(axis=2)
    if use_blur:
        mono = filters.gaussian(mono, sigma=3)

    convolved = convolve(mono, laplace_kernel)
    return np.abs(convolved)


def compute_saturation(image):
    return image.std(axis=2)


def compute_exposition(image, sigma=0.2):
    ideal_exp = 0.5
    image = image/255
    exposition = np.exp(- ((image - ideal_exp)**2) / (2 * (sigma**2)))
    return exposition.prod(axis=2)


def compute_weightmap(contrast, saturation, exposition, omega_c=1.0, omega_s=1.0, omega_e=1.0):
    arrays_prod = np.stack([
        contrast ** omega_c,
        saturation ** omega_s,
        exposition ** omega_e
    ])
    W = np.prod(arrays_prod, axis=0)
    return W

