import numpy as np
from scAnt.post_processing import *
from scAnt.files_io import lookup_bin
import yaml


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

# path = Path('F:\scans\messor_2')
# in_path = Path("D:\scans\cataglyphis_velox_2\masks_2")

# fix_metadata(path)
# invert_masks(in_path)

