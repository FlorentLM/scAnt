from pathlib import Path
import threading
import cv2
import numpy as np
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor
import time
from natsort import natsorted, ns
import platform
import shutil
import subprocess
from itertools import repeat
from skimage.transform import resize


def get_stacks_names(project_folder):
    """
    Returns a naturally sorted list of all stacks names present in the RAW folder
    """
    raw_images_folder = Path(project_folder / 'RAW')

    uniq_names = set([img_path.stem.split('step')[0] for img_path in raw_images_folder.glob("*.tif")])
    list_stacks_names = list(uniq_names)

    return natsorted(list_stacks_names, alg=ns.IGNORECASE)


def get_paths_by_stack(stacks_names):
    """
    Returns all the raw images paths, grouped by stack (in a list of lists)
    """
    all_stacks = []
    for stack in stacks_names:
        images_current_stack = list((project_folder / 'RAW').glob(f"{stack}*.tif"))
        all_stacks.append(natsorted(images_current_stack, alg=ns.IGNORECASE))
    return all_stacks


def lookup_external_executable(bin_name):
    """
    On macOS and Linux, fetches the absolute path for an executable. On Windows, it tries first the PATH and
    then looks it up in the local project folder.
    """
    # TODO: add pre-compiled binaries for Linux and macOS like for Windows?

    if platform.system().lower().startswith('win'):
        # Check if the executable is in the PATH
        which_res = shutil.which(f"{bin_name}.exe")

        if which_res is not None:
            path_executable = Path(which_res)

        else:
            cwd = Path().cwd()

            local_build = cwd / 'external'
            if not local_build.exists():
                local_build = cwd.parent / 'external'

            path_executable = Path(local_build) / f"{bin_name}.exe"

    else:
        path_executable = Path(shutil.which("bin_name"))

    return path_executable


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
    # TODO: is there any other quick way to do it without opencv?

    lap_image = cv2.Laplacian(blurred_image, cv2.CV_64F)

    if display:
        cv2.imshow("Laplacian of Image", lap_image)
        cv2.waitKey(1)

    return lap_image.var()


def resize_img(image, scale=0.5, use_opencv=False):
    new_shape = np.floor(np.array(image.shape[:2]) * scale).astype(int)
    if use_opencv:
        resized = cv2.resize(image, new_shape, interpolation=cv2.INTER_AREA)
    else:
        resized = resize(image, new_shape, anti_aliasing=True)
    return resized


def neutral_grayscale(colour_image):
    """
    Computes the grayscale version of an image without the human-perception bias
    """
    summed_channels = colour_image.sum(axis=2)
    normalised_lower = summed_channels - summed_channels.min()
    mono = (normalised_lower / normalised_lower.max() * 255).astype(np.uint8)
    return mono


def focus_check_single(image_path, threshold, display, verbose):
    """
    Performs the focus check on a single image and decides if it is blurry or not, based on the passed threshold value
    """
    if verbose:
        print(f"Thread {str(threading.get_ident())[-5:]} (PID {threading.get_native_id()}) processing {image_path.stem}{image_path.suffix}")

    image = cv2.imread(image_path.as_posix())

    target_vsize = 600      # in pixels
    scale = image.shape[1]/target_vsize

    resized = resize_img(image, scale=scale, use_opencv=True)

    # gray = cv2.cvtColor(resized, cv2.COLOR_BGR2GRAY)      # isn't the human-perception-centered algorithm too biased?
    gray = neutral_grayscale(resized)

    focus_score = measure_focus(gray, display=display)

    # If the focus measure is less than the supplied threshold, then the image should be considered "blurry"
    if focus_score < threshold:
        text = "BLURRY"
        col = (0, 0, 255)
        is_focused = False

    else:
        text = "NOT blurry"
        col = (255, 0, 0)
        is_focused = True

    if verbose:
        print(f"{image_path.stem}{image_path.suffix} is {text} (score: {focus_score:.2f})")

    if display:
        cv2.putText(resized, f"{text}: {focus_score:.2f}", (10, 30), cv2.FONT_HERSHEY_DUPLEX, 0.8, col, 3)
        cv2.imshow("Image", resized)
        cv2.waitKey(1)

    return image_path, is_focused, focus_score


def focus_check_multi(paths_list, threshold, display=False, verbose=False):
    """
    Performs the focus check on a list of images, and sorts them into two lists: sharp and blurry
    """
    thread_executor = ThreadPoolExecutor()
    thread_results = thread_executor.map(focus_check_single,
                                         paths_list,
                                         repeat(threshold),
                                         repeat(display),
                                         repeat(verbose))

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


def alignment(images_paths, output_folder):
    """
    Aligns a list of images using Hugin, and saves the files to disk
    """
    prefix = images_paths[0].stem.split('step')[0]
    inputs = [p.as_posix() for p in images_paths]

    output_folder = Path(output_folder)
    hugin_path = lookup_external_executable('align_image_stack')

    subprocess.run([hugin_path.as_posix(),
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
                   cwd=output_folder)


def fuse(images_paths, output_folder):
    """
    Fuses multiple images of different focus into a focus stack, using Hugin's enfuse
    """
    inputs = [p.as_posix() for p in images_paths]
    inputs.reverse()

    output_folder = Path(output_folder)
    enfuse_path = lookup_external_executable('enfuse')

    subprocess.run([enfuse_path.as_posix(),
                    " --exposure-weight=0",
                    " --saturation-weight=0",
                    " --contrast-weight=1",
                    " --hard-mask",
                    " --contrast-edge-scale=1",
                    # "--save-masks",               # to save soft and hard masks
                    # "--gray-projector=l-star",    # alternative stacking method
                    f" --output={output_folder.as_posix()}",
                    *inputs
                    ],
                   cwd=output_folder)


########

DISPLAY = False
VERBOSE = False
FOCUS_THRESH = 5


if __name__ == '__main__':

    project_folder = Path("/Users/florent/Desktop/atta_vollenweideri_000139_2/test_project")

    stacks_names = get_stacks_names(project_folder)
    paths_by_stack = get_paths_by_stack(stacks_names)

    ##

    print('Checking focus...')
    start = time.time()

    with ProcessPoolExecutor() as executor:

        # Note:
        # It is necessary to explicitly pass the global args FOCUS_THRESH, DISPLAY and VERBOSE to the map() call here,
        # despite them being common to each child process (unlike the image path values) - this is due to the fact
        # that the 'spawn' start method (see multiprocessing.get_context()) does not pass the global-level variables
        # to the child processes.
        # With the 'fork' start method (the default on Linux), the child process *do* inherit the globals, and it is
        # thus not necessary to pass them to map() - only the iterable containing the image paths is needed.
        # However, the 'fork' start method is unsafe on macOS, and does not exist on Windows.

        focus_proc_results = executor.map(focus_check_multi,
                                          paths_by_stack,          # 1 value of the iterable for 1 child process
                                          repeat(FOCUS_THRESH),    # the arg is repeated, same for each child process
                                          repeat(DISPLAY),
                                          repeat(VERBOSE))
    end = time.time()

    focus_results = list(focus_proc_results)

    for stack_name, stack_result in zip(stacks_names, focus_results):
        sharp_images, blurry_images = stack_result
        print(f"Found {len(sharp_images)} focused images and {len(blurry_images)} blurry images for stack {stack_name}.")

    print(f"Focus check took {end - start} seconds.")

    ##

    aligned_images_folder = project_folder / 'aligned'
    aligned_images_folder.mkdir(parents=True, exist_ok=True)

    only_focused_by_stack = [group[0] for group in focus_results]

    print('Aligning...')
    start = time.time()

    with ProcessPoolExecutor() as executor:
        executor.map(alignment, only_focused_by_stack,
                     repeat(aligned_images_folder))

    end = time.time()

    print(f"Alignment took {end - start} seconds.")

    ##

    stacked_images_folder = project_folder / 'stacked'
    stacked_images_folder.mkdir(parents=True, exist_ok=True)

    print('Fusing...')
    start = time.time()

    with ProcessPoolExecutor() as executor:
        executor.map(fuse, paths_by_stack,
                     repeat(stacked_images_folder))

    end = time.time()

    print(f"Fuse took {end - start} seconds.")