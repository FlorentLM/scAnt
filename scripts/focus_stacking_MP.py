# blur detection approach implemented based on
# pyimagesearch.com/2015/09/07/blur-detection-with-opencv/

# Use in the console with
"""
$ python Focus_stacking_MP.py --images "images_folder_path" --threshold "float"
"""

import argparse
import cv2
import os
from PIL import Image, ImageEnhance
import queue
import threading
import time
from pathlib import Path
import platform
import shutil

import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt

class myThread(threading.Thread):
    def __init__(self, threadID, name, q):
        threading.Thread.__init__(self)
        self.threadID = threadID
        self.name = name
        self.q = q

    def run(self):
        print("Starting " + self.name)
        process_data(self.name, self.q)
        print("Exiting " + self.name)


class myThread_Stacking(threading.Thread):
    def __init__(self, threadID, name, q):
        threading.Thread.__init__(self)
        self.threadID = threadID
        self.name = name
        self.q = q

    def run(self):
        print("Starting " + self.name)
        process_stack(self.name, self.q)
        print("Exiting " + self.name)


def process_data(threadName, q):
    while not exitFlag:
        queueLock.acquire()
        if not workQueue.empty():
            data = q.get()
            queueLock.release()
            print(f"{threadName} processing {data}")
            checkFocus((input_images_folder / data))
        else:
            queueLock.release()


def createThreadList(num_threads):
    threadNames = []
    for t in range(num_threads):
        threadNames.append("Thread_" + str(t))

    return threadNames


def variance_of_laplacian(image):
    # compute the Laplacian of the image and then return the focus
    # measure, which is simply the valirance of the Laplacian
    # using the following 3x3 convolutional kernel
    """
    [0   1   0]
    [1  -4   1]
    [0   1   0]

    as recommenden by Pech-Pacheco et al. in their 2000 ICPR paper,
    Diatom autofocusing in brightfield microscopy: a comparative study.
    """
    # apply median blur to image to suppress noise in RAW files
    blurred_image = cv2.medianBlur(image, 3)
    lap_image = cv2.Laplacian(blurred_image, cv2.CV_64F)
    lap_var = lap_image.var()

    print(args["display"])

    if args["display"]:
        cv2.imshow("Laplacian of Image", lap_image)

        cv2.waitKey(1)
    return lap_var


def checkFocus(image_path):
    image = cv2.imread(image_path.as_posix())

    # original window size (due to input image)
    # = 2448 x 2048 -> time to size it down!
    scale_percent = 15  # percent of original size
    width = int(image.shape[1] * scale_percent / 100)
    height = int(image.shape[0] * scale_percent / 100)
    dim = (width, height)
    # resize image
    resized = cv2.resize(image, dim, interpolation=cv2.INTER_AREA)

    gray = cv2.cvtColor(resized, cv2.COLOR_BGR2GRAY)
    fm = variance_of_laplacian(gray)

    # if the focus measure is less than the supplied threshold,
    # then the image should be considered "blurry"
    if fm < args["threshold"]:
        text = "BLURRY"
        color_text = (0, 0, 255)
        rejected_images.append(image_path)
    else:
        text = "NOT Blurry"
        color_text = (255, 0, 0)
        usable_images.append(image_path)

    print(image_path, "is", text)

    if args["display"]:
        # show the image
        cv2.putText(resized, f"{text}: {fm:.2f}", (10, 30),
                    cv2.FONT_HERSHEY_DUPLEX, 0.8, color_text, 3)
        cv2.imshow("Image", resized)

        cv2.waitKey(1)


def process_stack(threadName, q):
    while not exitFlag_stacking:
        queueLock.acquire()
        if not workQueue_stacking.empty():
            data = q.get()
            queueLock.release()

            first_img_name = data[0].stem
            current_stack = first_img_name.split('step')[0]

            temp_output_folder = output_folder / current_stack

            print(f"{threadName} is processing stack {current_stack}")

            if platform.system().lower().startswith('win'):
                HUGIN_PATH = (path_to_external / 'align_image_stack').as_posix()
            else:
                HUGIN_PATH = 'align_image_stack'

            INPUTS = ' '.join([imgpath.as_posix() for imgpath in data])

            os.system(f"{HUGIN_PATH} -m -x -c 100 -l -a {temp_output_folder / current_stack}OUT {INPUTS} -v")

            # TODO: try the additional modifier " --gpu " to hopefully speed up the process!!!

            print("\nFocus stacking...")

            # go through list in reverse order (better results of focus stacking)
            paths = [temp_file.as_posix() for temp_file in temp_output_folder.glob(f"{current_stack}OUT*.tif")]
            paths.sort()
            paths.reverse()

            image_str_focus = ' '.join(paths)

            output_path = output_folder / f"{stack_name}.tif"
            print(output_path)

            print(f"generating: {image_str_focus}\n")

            if platform.system().lower().startswith('win'):
                ENFUSE_PATH = (path_to_external / 'enfuse').as_posix()
            else:
                ENFUSE_PATH = 'enfuse'

            os.system(f"{ENFUSE_PATH}"
                            f" --exposure-weight=0"
                            f" --saturation-weight=0"
                            f" --contrast-weight=1"
                            f" --hard-mask"
                            f" --contrast-edge-scale=1"
                            f" --output={output_path}"
                            f" {image_str_focus}")

            # --save-masks     to save soft and hard masks
            # --gray-projector=l-star alternative stacking method

            print(f"Stacked image saved as {output_path}")

            if args["sharpen"]:
                stacked = Image.open(output_path)
                enhancer = ImageEnhance.Sharpness(stacked)
                sharpened = enhancer.enhance(1.5)
                sharpened.save(output_path)

                print("Sharpened", output_path)

        else:
            queueLock.release()


if __name__ == '__main__':

    """
    ### Loading image paths into queue from disk ###
    """

    # construct the argument parse and parse the arguments
    ap = argparse.ArgumentParser()
    ap.add_argument("-i", "--images", required=True,
                    help="path to input directory of images")
    ap.add_argument("-t", "--threshold", type=float, default=10.0,
                    help="focus measures that fall below this value will be considered 'blurry'")
    ap.add_argument("-s", "--sharpen", type=bool, default=False,
                    help="apply sharpening to final result [True / False]")
    ap.add_argument("-d", "--display", type=bool, default=False,
                    help="show images with displayed focus score [True / False]")
    args = vars(ap.parse_args())

    print("Using a laplacian variance threshold of", args["threshold"], "for discarding out-of-focus images")

    # parsing in boolean arguments
    if args["display"] == "False" or not args["display"]:
        args["display"] = False
        print("Images will NOT be displayed during out-of-focus check")
    elif args["display"]:
        args["display"] = True
        print("Images will be displayed during out-of-focus check")

    if args["sharpen"] == "True":
        args["sharpen"] = True
        print("Output images will be additionally sharpened")
    else:
        args["sharpen"] = False
        print("Output images will NOT be additionally sharpened")

    # convert input str of file location into path object
    input_images_folder = Path(args["images"])

    input_images_folder = Path("/Users/florent/Desktop/atta_vollenweideri_000139 2/test_project/RAW")

    blurry_removed = input("Have you removed blurry images already? [y/n] default n")

    # setup as many threads as there are (virtual) CPUs
    exitFlag = 0
    num_cores = os.cpu_count()
    print(f"Found {num_cores} cores: running {num_cores * 2} threads...")
    threadList = createThreadList(num_cores * 2)
    queueLock = threading.Lock()

    # define paths to all images and set the maximum number of items in the queue equivalent to the number of images
    all_image_paths = list(input_images_folder.glob('*.tif'))

    nb_imgs = len(all_image_paths)

    workQueue = queue.Queue(nb_imgs)
    threads = []
    threadID = 1

    # create list of image paths classified as in-focus or blurry
    usable_images = []
    rejected_images = []

    """
    ### extracting "in-focus" images for further processing ###
    """

    start = time.time()

    if blurry_removed != "y":

        # Create new threads
        for tName in threadList:
            thread = myThread(threadID, tName, workQueue)
            thread.start()
            threads.append(thread)
            threadID += 1

        cv2.ocl.setUseOpenCL(True)

        # Fill the queue
        queueLock.acquire()
        for path in all_image_paths:
            workQueue.put(path)
        queueLock.release()

        # Wait for queue to empty
        while not workQueue.empty():
            pass

        # Notify threads it's time to exit
        exitFlag = 1

        # Wait for all threads to complete
        for t in threads:
            t.join()
        print("Exiting Main Thread")

        cv2.destroyAllWindows()
    else:
        # if blurry images have been discarded already add all paths to "usable_images"
        usable_images = all_image_paths

    # as threads may terminate at different times the file list needs to be sorted
    usable_images.sort()

    if len(usable_images) > 1:
        print("\nThe following images are sharp enough for focus stacking:\n")
        for path in usable_images:
            print(path)
    else:
        print("No images suitable for focus stacking found!")
        exit()

    path_to_external = Path.cwd().parent.joinpath("external")

    output_folder = input_images_folder.parent / f"{input_images_folder.stem}_stacked"

    try:
        output_folder.mkdir(parents=True, exist_ok=False)
    except FileExistsError:
        print("Folder is already there")
    else:
        print("made folder!")

    # group images of each stack together
    nb_usable_imgs = len(usable_images)

    print("\nSorting in-focus images into stacks...")
    stacks_names = set([imgpath.stem.split('step')[0] for imgpath in usable_images])
    nb_stacks = len(stacks_names)

    for stack_name in stacks_names:
        try:
            (output_folder / stack_name).mkdir(parents=True, exist_ok=False)
        except FileExistsError:
            print("corresponding temporary folder already exists!")
        else:
            print("made corresponding temporary folder!")

    # sort stacks in ascending order
    stacks_names = list(stacks_names)
    stacks_names.sort()

    """
    ### Alignment and stacking of images ###
    """

    # setup as many threads as there are (virtual) CPUs
    exitFlag_stacking = 0
    # only use a fourth of the number of CPUs for stacking as hugin and enfuse utilise multi core processing in part
    threadList_stacking = createThreadList(min(nb_stacks, int(num_virtual_cores / 4)))
    print(f"Using {len(threadList_stacking)} threads for stacking...")
    queueLock = threading.Lock()

    # define paths to all images and set the maximum number of items in the queue equivalent to the number of images
    workQueue_stacking = queue.Queue(nb_stacks)
    threads = []
    threadID = 1

    # Create new threads
    for tName in threadList_stacking:
        thread = myThread_Stacking(threadID, tName, workQueue_stacking)
        thread.start()
        threads.append(thread)
        threadID += 1

    # Fill the queue with stacks
    queueLock.acquire()
    for stack_name in stacks_names:
        images_current_stack = list(input_images_folder.glob(f"{stack_name}*.tif"))
        # revert the order of images to begin with the image the furthest away
        # -> maximise field of view during alignment and leads to better blending results with less ghosting
        images_current_stack.sort()
        images_current_stack.reverse()

        workQueue_stacking.put(images_current_stack)
    queueLock.release()

    # Wait for queue to empty
    while not workQueue_stacking.empty():
        pass

    # Notify threads it's time to exit
    exitFlag_stacking = 1

    # Wait for all threads to complete
    for t in threads:
        t.join()
    print("Exiting Main Stacking Thread")
    print("Deleting temporary folders")

    for stack_name in stacks_names:
        shutil.rmtree((output_folder / stack_name))
        print(f"removed {stack_name}")

    print("Stacking finalised!")
    print(f"Time elapsed: {time.time() - start}")
