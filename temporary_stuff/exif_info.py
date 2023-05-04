import math
import exiftool
import yaml
from yaml.loader import SafeLoader
from pathlib import Path


def get_diag(horiz_size, vert_size):
    """
    Simply returns the diagonal length, given horizontal and vertical dimensions
    """
    return math.sqrt(horiz_size**2 + vert_size**2)


def load_metadata(project_folder):
    """
    Loads available metadata
    """

    probe_img = next((project_folder / 'RAW').glob("*.tif"))

    with exiftool.ExifToolHelper() as et:
        file_exif_data = et.get_metadata(probe_img)[0]

    # Open the file and load the file
    with open(project_folder / f'{project_folder.stem}_config.yaml') as f:
        yaml_data = yaml.load(f, Loader=SafeLoader)['exif_data']

    metadata = {**yaml_data, **file_exif_data}
    return metadata


def compute_optics(project_folder):
    """
    Compute some optics values useful for Hugin image alignment
    """
    metadata = load_metadata(project_folder)

    sensor_px_size = 2.4e-3  # the current sensor is 2.4 micrometers per pixels
    # THIS IS SPECIFIC TO THE CAMERA USED AND MUST BE CHANGED ACCORDINGLY - TODO: add this to yaml

    h = metadata["EXIF:ImageWidth"] * sensor_px_size
    v = metadata["EXIF:ImageHeight"] * sensor_px_size

    sensor = (h, v)  # the current sensor dimensions in millimetres
    full_frame = (36, 24)  # the full frame standard dimensions in millimetres

    crop_factor = get_diag(*full_frame) / get_diag(*sensor)

    print(f"Crop factor: {crop_factor:.4f}")

    focal_length = float(metadata['FocalLength'])
    hfov_rad = 2 * math.atan(h / (2 * focal_length))
    hfov_degrees = math.degrees(hfov_rad)

    fframe_equiv_hfov = focal_length * crop_factor

    print(f"Sensor horizontal FOV (actual): {hfov_degrees:.4f}°")
    print(f"Full frame equivalent horizontal FOV: {fframe_equiv_hfov:.4f}°")

    return crop_factor, hfov_degrees, fframe_equiv_hfov


######

if __name__ == '__main__':

    project_folder = Path("/Users/florent/Desktop/atta_vollenweideri_000139_2/test_project")

    crop_factor, hfov, fframe_hfov = compute_optics(project_folder)