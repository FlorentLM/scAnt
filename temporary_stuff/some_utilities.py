import numpy as np
from scipy.ndimage import convolve
from skimage import filters, io

import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt

from scAnt.post_processing import *
from scAnt.files_io import lookup_bin

import yaml


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

def write_exif_to_img(img_path, custom_exif_dict, verbose=0):

    which_exiftool = lookup_bin('exiftool')
    img_path = Path(img_path)
    complete_command = [which_exiftool.as_posix(), img_path.as_posix(), "-overwrite_original_in_place"]
    for key in custom_exif_dict:
        write_str = f"-{key}={custom_exif_dict[key]}"
        print(write_str)
        complete_command.append(write_str)

    if verbose > 0:
        print(complete_command)

    process = subprocess.Popen(complete_command)
    process.wait()

def invert_masks(in_path, ext='png'):
    ext = ext.strip(".")

    out_path = in_path / 'inverted'
    out_path.mkdir(parents=True, exist_ok=True)

    nb_masks = len((list(in_path.glob(f'*.{ext}'))))

    i = 0
    for mask_file in in_path.glob(f'*.{ext}'):
        print(f'{i + 1}/{nb_masks}')

        input_mask = cv2.imread(mask_file.as_posix())
        mask = np.array(~input_mask.astype(bool) * 255, dtype=np.uint8)

        filepath = out_path / f'{mask_file.stem}.{ext}'.replace('__', '_')

        cv2.imwrite(filepath.as_posix(), mask)
        i += 1
    print('Done.')


def fix_metadata(path, cfg_path=None, ext='tif'):
    ext = ext.strip(".")

    if cfg_path is None:
        cfg_path = path.parent / f'{path.stem}_config.yaml'

    with open(cfg_path, 'r') as file:
        config = yaml.safe_load(file)

    nb_files = len((list(path.glob(f'*.{ext}'))))

    i = 0
    for img in path.glob(f'*.{ext}'):
        print(f'{i + 1}/{nb_files}')

        write_exif_to_img(img, config['exif_data'])
        i += 1
    print('Done.')

##

path = Path('F:\scans\messor_2\stacked')

fix_metadata(path, cfg_path=path.parent)
# invert_masks(in_path)

