import math
import exiftool
import yaml
from yaml.loader import SafeLoader


raw_images_folder = Path("/home/florent/Desktop/atta_vollenweideri_000139_2/test_project/RAW")

img_files = raw_images_folder.glob("*.tif")
test_img = next(img_files)

with exiftool.ExifToolHelper() as et:
    metadata = et.get_metadata(test_img)

mtdt_dict = metadata[0]


# Open the file and load the file
with open(raw_images_folder.parent / 'raw_atta_vollenweideri_config.yaml') as f:
    exif_data = yaml.load(f, Loader=SafeLoader)['exif_data']


sensor_px_size = 2.4e-3  # 2.4 micrometers per pixels
            # ^ THIS IS SPECIFIC TO THE CAMERA USED AND MUST BE CHANGED ACCORDINGLY - TODO add this to default EXIF

horiz_size = mtdt_dict["EXIF:ImageWidth"] * sensor_px_size
vert_size = mtdt_dict["EXIF:ImageHeight"] * sensor_px_size
focal_length = float(exif_data['FocalLength'])

horiz_fov_rad = 2 * math.atan(horiz_size / (2 * focal_length))
horiz_fov_degrees = math.degrees(horiz_fov_rad)

full_frame_horiz_size = 36  # millimeters
full_frame_vert_size = 24   # millimeters

def get_sensor_diag(horiz_size, vert_size):
    return math.sqrt(horiz_size**2 + vert_size**2)

crop_factor = get_sensor_diag(full_frame_horiz_size, full_frame_vert_size) / get_sensor_diag(horiz_size, vert_size)

full_frame_equiv_horiz_fov = focal_length * crop_factor
