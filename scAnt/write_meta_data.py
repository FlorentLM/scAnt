import subprocess
import time
from pathlib import Path
import platform
try:
    from scAnt.project_manager import read_config_file
except ModuleNotFoundError:
    from scAnt.project_manager import read_config_file
import os
from files_io import lookup_bin

# follow installation guide for Ubuntu or use executable directly under Windows (located in "/external")
# sudo apt install libimage-exiftool-perl

def show_me_what_you_got(img_path):
    # if platform.system() == "Linux":
    #     exifToolPath = "exiftool"
    # else:
    #     exifToolPath = str(Path.cwd().joinpath("external", "exiftool.exe"))
    #     # for Windows user have to specify the Exif tool exe path for metadata extraction.

    which_exiftool = lookup_bin('exiftool')

    infoDict = {}  # Creating the dict to get the metadata tags
    ''' use Exif tool to get the metadata '''
    process = subprocess.Popen([which_exiftool, img_path],
                               stdout=subprocess.PIPE,
                               stderr=subprocess.STDOUT,
                               universal_newlines=True)

    """ get the tags in dict """
    for tag in process.stdout:
        line = tag.strip().split(':')
        infoDict[line[0].strip()] = line[-1].strip()

    for k, v in infoDict.items():
        print(k, ':', v)


def write_exif_to_img(img_path, custom_exif_dict):
    # if platform.system() == "Linux":
    #     exifToolPath = "exiftool"
    # else:
    #     exifToolPath = str(Path.cwd().joinpath("external", "exiftool.exe"))
    #     # for Windows user have to specify the Exif tool exe path for metadata extraction.
    #     if not os.path.isfile(exifToolPath):
    #         exifToolPath = str(Path.cwd().parent.joinpath("external", "exiftool.exe"))

    which_exiftool = lookup_bin('exiftool')

    complete_command = [which_exiftool, img_path, "-overwrite_original_in_place"]
    for key in custom_exif_dict:
        write_str = f"-{key}={custom_exif_dict[key]}"
        print(write_str)
        complete_command.append(write_str)

    print(complete_command)

    subprocess.Popen(complete_command)


def get_default_values():
    # WARNING! THESE SETTINGS ARE SPECIFIC TO THE CAMERA USED DURING DEVELOPMENT
    # OF THE SCANNER AND WILL LIKELY NOT APPLY TO YOUR SETUP
    exif = {"Make": "FLIR",
            "Model": "BFS-U3-200S6C-C",
            "SerialNumber": "18382947",
            "Lens": "MPZ",
            "CameraSerialNumber": "18382947",
            "LensManufacturer": "Computar",
            "LensModel": "35.0 f / 2.2",
            "FocalLength": "35.0",
            "FocalLengthIn35mmFormat": "95.0"}
    return exif


if __name__ == '__main__':

    root_folder = Path.cwd().parent

    img_path = (root_folder.parent / "Downloads") / "_x_00000_y_00000__cutout.tif"

    print("\nOriginal file:")
    show_me_what_you_got(img_path)

    config = read_config_file(root_folder / "example_config.yaml")
    custom_exif_dict = config["exif_data"]

    write_exif_to_img(img_path=img_path, custom_exif_dict=custom_exif_dict)

    # wait for file to be updated before opening it again
    time.sleep(1)

    print("\nUpdated file:")
    show_me_what_you_got(img_path)
