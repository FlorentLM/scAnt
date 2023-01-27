from pathlib import Path
import threading
import cv2
import numpy as np
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor
import time
import os
from natsort import natsorted, ns
import platform
import shutil
import subprocess


def measure_focus(image, display=False):
    """
    Computes the Laplacian image using the following 3x3 convolutional kernel:
        [0   1   0]
        [1  -4   1]
        [0   1   0]

    And returns the focus measure, which is simply the variance of the Laplacian image.

    Pech-Pacheco et al., "Diatom autofocusing in brightfield microscopy: a comparative study", ICPR-2000,
    pp. 314-317 vol.3, doi: 10.1109/ICPR.2000.903548.

    """
    # Apply median blur to suppress noise in RAW files
    blurred_image = cv2.medianBlur(image, 3)

    lap_image = cv2.Laplacian(blurred_image, cv2.CV_64F)

    if display:
        cv2.imshow("Laplacian of Image", lap_image)
        cv2.waitKey(1)

    return lap_image.var()


def neutral_grayscale(colour_image):
    """
    Computes the grayscale version of an image without the human-perception bias
    """
    summed_channels = colour_image.sum(axis=2)
    normalised_lower = summed_channels - summed_channels.min()
    mono = (normalised_lower / normalised_lower.max() * 255).astype(np.uint8)
    return mono


def focus_check_worker(image_path):

    if VERBOSE:
        print(f"Thread {str(threading.get_ident())[-5:]} (PID {threading.get_native_id()}) processing {image_path.stem}{image_path.suffix}")

    image = cv2.imread(image_path.as_posix())

    target_vsize = 600      # in pixels
    scale = image.shape[1]/target_vsize
    new_dims = round(image.shape[1]/scale),  round(image.shape[0]/scale)

    resized = cv2.resize(image, new_dims, interpolation=cv2.INTER_AREA)

    # gray = cv2.cvtColor(resized, cv2.COLOR_BGR2GRAY)      # isn't the human-perception-centered algorithm too biased?
    gray = neutral_grayscale(resized)

    focus_score = measure_focus(gray, display=DISPLAY)

    # If the focus measure is less than the supplied threshold, then the image should be considered "blurry"
    if focus_score < FOCUS_THRESH:
        text = "BLURRY"
        col = (0, 0, 255)
        is_focused = False

    else:
        text = "NOT blurry"
        col = (255, 0, 0)
        is_focused = True

    if VERBOSE:
        print(f"{image_path.stem}{image_path.suffix} is {text} (score: {focus_score:.2f})")

    if DISPLAY:
        cv2.putText(resized, f"{text}: {focus_score:.2f}", (10, 30), cv2.FONT_HERSHEY_DUPLEX, 0.8, col, 3)
        cv2.imshow("Image", resized)
        cv2.waitKey(1)

    return image_path, is_focused, focus_score


def multiprocessed_focus_check(paths_list):

    thread_executor = ThreadPoolExecutor()
    thread_results = thread_executor.map(focus_check_worker, paths_list)

    sharp = []
    blurry = []

    focused_scores = []

    for image, is_focused, score in thread_results:
        if is_focused:
            sharp.append(image)
            focused_scores.append(score)
        else:
            blurry.append(image)

    sharp = natsorted(sharp, alg=ns.IGNORECASE)

    # sharp = [x for _, x in sorted(zip(focused_scores, sharp))]
    # sharp.reverse()
    blurry = natsorted(blurry, alg=ns.IGNORECASE)

    return sharp, blurry


def get_stacks_names(raw_folder_path):
    """
    Returns a naturally sorted list of all stacks names present in the RAW folder
    """
    raw_folder_path = Path(raw_folder_path)
    uniq_names = set([img_path.stem.split('step')[0] for img_path in raw_folder_path.glob("*.tif")])
    list_stacks_names = list(uniq_names)
    return natsorted(list_stacks_names, alg=ns.IGNORECASE)


def get_paths_by_stack(stacks_names):
    """
    Returns all the raw images paths, grouped by stack (in a list of lists)
    """
    all_stacks = []
    for stack in stacks_names:
        images_current_stack = list(raw_images_folder.glob(f"{stack}*.tif"))
        all_stacks.append(natsorted(images_current_stack, alg=ns.IGNORECASE))
    return all_stacks


def lookup_external_executables():

    if platform.system().lower().startswith('win'):
        # Check if the executables are in the PATH
        which_hugin = shutil.which("align_image_stack.exe")
        which_enfuse = shutil.which("enfuse.exe")

        if which_hugin is not None and which_enfuse is not None:
            path_hugin = Path(which_hugin)
            path_enfuse = Path(which_enfuse)

        else:
            cwd = Path().cwd()

            local_build = cwd / 'external'
            if not local_build.exists():
                local_build = cwd.parent / 'external'

            path_hugin = Path(local_build) / "align_image_stack.exe"
            path_enfuse = Path(local_build) / "enfuse.exe"

    else:
        path_hugin = Path(shutil.which("align_image_stack"))
        path_enfuse = Path(shutil.which("enfuse"))

    return path_hugin, path_enfuse


def multiprocessed_alignment(images_paths):

    global aligned_images_folder

    prefix = images_paths[0].stem.split('step')[0]
    inputs = [p.as_posix() for p in images_paths]

    subprocess.run([HUGIN_PATH.as_posix(),
                    "-v",
                    "-m",
                    "-x",
                    "-s 1",
                    "-c 50",
                    # "--use-given-order",
                    # "--align-to-first",
                    # "--gpu",              # Should speed up the process - TODO: try it
                    f"-a {prefix}_ALIGNED",
                    *inputs
                    ],
                   cwd=aligned_images_folder)


def multiprocessed_fuse(images_paths):

    global stacked_images_folder

    inputs = [p.as_posix() for p in images_paths]
    inputs.reverse()

    subprocess.run([ENFUSE_PATH.as_posix(),
                    " --exposure-weight=0",
                    " --saturation-weight=0",
                    " --contrast-weight=1",
                    " --hard-mask",
                    " --contrast-edge-scale=1",
                    # "--save-masks",               # to save soft and hard masks
                    # "--gray-projector=l-star",    # alternative stacking method
                    f" --output={stacked_images_folder.as_posix()}",
                    *inputs
                    ],
                   cwd=stacked_images_folder)


########

raw_images_folder = Path("/home/florent/Desktop/atta_vollenweideri_000139_2/test_project/RAW")

DISPLAY = False
VERBOSE = False

FOCUS_THRESH = 5

NUM_CPU_CORES = os.cpu_count()

HUGIN_PATH, ENFUSE_PATH = lookup_external_executables()

##

stacks_names = get_stacks_names(raw_images_folder)
paths_by_stack = get_paths_by_stack(stacks_names)

##

start = time.time()

with ProcessPoolExecutor() as executor:
    focus_proc_results = executor.map(multiprocessed_focus_check, paths_by_stack)

end = time.time()

focus_results = list(focus_proc_results)

for stack_name, stack_result in zip(stacks_names, focus_results):
    sharp_images, blurry_images = stack_result
    print(f"Found {len(sharp_images)} focused images and {len(blurry_images)} blurry images for stack {stack_name}.")

print(f"Focus check took {end - start} seconds.")

##

aligned_images_folder = raw_images_folder.parent / 'aligned'
aligned_images_folder.mkdir(parents=True, exist_ok=True)

##

only_focused_by_stack = [group[0] for group in focus_results]

start = time.time()

with ProcessPoolExecutor() as executor:
    executor.map(multiprocessed_alignment, only_focused_by_stack)

end = time.time()

print(f"Alignment took {end - start} seconds.")

##

stacked_images_folder = raw_images_folder.parent / 'stacked'
stacked_images_folder.mkdir(parents=True, exist_ok=True)

start = time.time()

with ProcessPoolExecutor() as executor:
    executor.map(multiprocessed_fuse, paths_by_stack)

end = time.time()

print(f"Fuse took {end - start} seconds.")