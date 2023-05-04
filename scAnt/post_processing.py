from pathlib import Path
import threading
import cv2
import numpy as np
from concurrent.futures import ThreadPoolExecutor
from natsort import natsorted, ns
import subprocess
from itertools import repeat
from skimage.transform import resize
from scAnt.files_io import lookup_bin
from os.path import commonprefix


#######################################################################################################################
#                                                 Stacking Section                                                    #
#######################################################################################################################


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


def focus_check_single(image_path, threshold, display=False, verbose=0):
    """
    Performs the focus check on a single image and decides if it is blurry or not, based on the passed threshold value
    """
    if verbose == 2:
        print(f"Thread {str(threading.get_ident())[-5:]} (PID {threading.get_native_id()}) processing {image_path.stem}{image_path.suffix}")

    image = cv2.imread(image_path.as_posix())

    target_vsize = 600      # in pixels
    scale = target_vsize/image.shape[1]

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

    if verbose > 0:
        print(f"{image_path.stem}{image_path.suffix} is {text} (score: {focus_score:.2f})")

    if display:
        cv2.putText(resized, f"{text}: {focus_score:.2f}", (10, 30), cv2.FONT_HERSHEY_DUPLEX, 0.8, col, 3)
        cv2.imshow("Image", resized)
        cv2.waitKey(1)

    return image_path, is_focused, focus_score


def focus_check_multi(paths_list, threshold, display=False, verbose=0):
    """
    Performs the focus check on a list of images, and sorts them into two lists: sharp and blurry
    """

    thread_executor = ThreadPoolExecutor(max_workers=10)        # 10 threads per process
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


def alignment(images_paths, output_folder, verbose=0):
    """
    Aligns a list of images using Hugin, and saves the files to disk
    """
    prefix = images_paths[0].stem.split('step')[0]
    inputs = [p.as_posix() for p in images_paths]

    output_folder = Path(output_folder)
    hugin_path = lookup_bin('align_image_stack')

    if verbose > 2:
        stdout = subprocess.STDOUT
    else:
        stdout = subprocess.DEVNULL
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
                   cwd=output_folder,
                   stdout=stdout,
                   stderr=subprocess.STDOUT)


def fuse(images_paths, output_folder, verbose=0):
    """
    Fuses multiple images of different focus into a focus stack, using Hugin's enfuse
    """
    inputs = [p.as_posix() for p in images_paths]
    inputs.reverse()

    output_folder = Path(output_folder)
    enfuse_path = lookup_bin('enfuse')

    if verbose > 1:
        stdout = subprocess.STDOUT
    else:
        stdout = subprocess.DEVNULL
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
                   cwd=output_folder,
                   stdout=stdout,
                   stderr=subprocess.STDOUT)

# from os.path import commonprefix
# from pathlib import Path
#
# images_paths = [Path('/Users/florent/Desktop/abc_673.tif'),
#                 Path('/Users/florent/Desktop/abc_494.tif')]

def focus_stack_2(images_paths, output_folder, verbose=0):
    inputs = [p.as_posix() for p in images_paths]

    stack_name = commonprefix([file.stem for file in images_paths]).replace('_step', '')

    output_folder = Path(output_folder)
    focusstack_path = lookup_bin('focus-stack')

    if verbose > 1:
        stdout = subprocess.STDOUT
    else:
        stdout = subprocess.DEVNULL
    subprocess.run([focusstack_path.as_posix(),
                    " --nocrop",
                    f" --output={(output_folder / (stack_name + '.tif')).as_posix()}",
                    *inputs
                    ],
                   cwd=output_folder)


#######################################################################################################################
#                                                Masking Section                                                      #
#######################################################################################################################

def filterOutSaltPepperNoise(edgeImg):
    # Get rid of salt & pepper noise.
    count = 0
    lastMedian = edgeImg
    median = cv2.medianBlur(edgeImg, 3)
    while not np.array_equal(lastMedian, median):
        # get those pixels that gets zeroed out
        zeroed = np.invert(np.logical_and(median, edgeImg))
        edgeImg[zeroed] = 0

        count = count + 1
        if count > 70:
            break
        lastMedian = median
        median = cv2.medianBlur(edgeImg, 3)


def findSignificantContour(edgeImg):
    try:
        image, contours, hierarchy = cv2.findContours(
            edgeImg,
            cv2.RETR_TREE,
            cv2.CHAIN_APPROX_SIMPLE)
    except ValueError:
        contours, hierarchy = cv2.findContours(
            edgeImg,
            cv2.RETR_TREE,
            cv2.CHAIN_APPROX_SIMPLE)
    # Find level 1 contours (i.e. largest contours)
    level1Meta = []
    for contourIndex, tupl in enumerate(hierarchy[0]):
        # Each array is in format (Next, Prev, First child, Parent)
        # Filter the ones without parent
        if tupl[3] == -1:
            tupl = np.insert(tupl.copy(), 0, [contourIndex])
            level1Meta.append(tupl)
            #  # From among them, find the contours with large surface area.
    contoursWithArea = []
    for tupl in level1Meta:
        contourIndex = tupl[0]
        contour = contours[contourIndex]
        area = cv2.contourArea(contour)
        contoursWithArea.append([contour, area, contourIndex])

    contoursWithArea.sort(key=lambda meta: meta[1], reverse=True)
    largestContour = contoursWithArea[0][0]
    return largestContour


def remove_holes(img, min_num_pixel):
    cleaned_img = np.zeros(shape=(img.shape[0], img.shape[1]))

    unique, counts = np.unique(img, return_counts=True)
    print("\nunique values:", unique)
    print("counted:", counts)

    for label in range(len(counts)):
        if counts[label] > min_num_pixel:
            if unique[label] != 0:
                cleaned_img[img == unique[label]] = 1

    return cleaned_img


def apply_local_contrast(img, grid_size=(7, 7)):
    """
    ### CLAHE (Contrast limited Adaptive Histogram Equilisation) ###

    Advanced application of local contrast. Adaptive histogram equalization is used to locally increase the contrast,
    rather than globally, so bright areas are not pushed into over exposed areas of the histogram. The image is tiled
    into a fixed size grid. Noise needs to be removed prior to this process, as it would be greatly amplified otherwise.
    Similar to Adobe's "Clarity" option which also amplifies local contrast and thus pronounces edges, reduces haze.
    """
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    blurred_gray = cv2.GaussianBlur(gray, (5, 5), 0)

    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=grid_size)
    cl1 = clahe.apply(blurred_gray)

    # convert to PIL format to apply laplacian sharpening
    img_pil = Image.fromarray(cl1)

    enhancer = ImageEnhance.Sharpness(img_pil)
    sharpened = enhancer.enhance(31)

    return cv2.cvtColor(np.array(sharpened), cv2.COLOR_GRAY2RGB)


def apply_local_contrastB(img, grid_size=(7, 7)):
    """
    ### CLAHE (Contrast limited Adaptive Histogram Equilisation) ###

    Advanced application of local contrast. Adaptive histogram equalization is used to locally increase the contrast,
    rather than globally, so bright areas are not pushed into over exposed areas of the histogram. The image is tiled
    into a fixed size grid. Noise needs to be removed prior to this process, as it would be greatly amplified otherwise.
    Similar to Adobe's "Clarity" option which also amplifies local contrast and thus pronounces edges, reduces haze.
    """
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    blurred_gray = cv2.GaussianBlur(gray, (3, 3), 0)

    clahe = cv2.createCLAHE(clipLimit=4.2, tileGridSize=grid_size)
    cl1 = clahe.apply(blurred_gray)

    # convert to PIL format to apply laplacian sharpening
    img_pil = Image.fromarray(cl1)

    enhancer = ImageEnhance.Sharpness(img_pil)
    sharpened = enhancer.enhance(25)

    # sharpened.save(source + "\\enhanced.png")

    return cv2.cvtColor(np.array(sharpened), cv2.COLOR_GRAY2RGB)

def createAlphaMask(data, edgeDetector, min_rgb=108, max_rgb=144, min_bl=500, min_wh=500, create_cutout=True):
    """
    create alpha mask for the image located in path
    :img_path: image location
    :create_cutout: additionally save final image with as the stacked image with the mask as an alpha layer
    :return: writes image to same location as input
    """
    src = cv2.imread(data, 1)

    img_enhanced = apply_local_contrast(src)

    # reduce noise in the image before detecting edges
    blurred = cv2.GaussianBlur(img_enhanced, (5, 5), 0)

    # turn image into float array
    blurred_float = blurred.astype(np.float32) / 255.0
    edges = edgeDetector.detectEdges(blurred_float) * 255.0

    # required as the contour finding step is susceptible to noise
    edges_8u = np.asarray(edges, np.uint8)
    filterOutSaltPepperNoise(edges_8u)

    contour = findSignificantContour(edges_8u)
    # Draw the contour on the original image
    contourImg = np.copy(src)
    cv2.drawContours(contourImg, [contour], 0, (0, 255, 0), 2, cv2.LINE_AA, maxLevel=1)
    # cv2.imwrite(data[:-4] + '_contour.png', contourImg)

    mask = np.zeros_like(edges_8u)
    cv2.fillPoly(mask, [contour], 255)

    # calculate sure foreground area by dilating the mask
    mapFg = cv2.erode(mask, np.ones((5, 5), np.uint8), iterations=10)

    # mark inital mask as "probably background"
    # and mapFg as sure foreground
    trimap = np.copy(mask)
    trimap[mask == 0] = cv2.GC_BGD
    trimap[mask == 255] = cv2.GC_PR_BGD
    trimap[mapFg == 255] = cv2.GC_FGD

    # visualize trimap
    trimap_print = np.copy(trimap)
    trimap_print[trimap_print == cv2.GC_PR_BGD] = 128
    trimap_print[trimap_print == cv2.GC_FGD] = 255
    # cv2.imwrite(data[:-4] + '_trimap.png', trimap_print)

    # run grabcut
    bgdModel = np.zeros((1, 65), np.float64)
    fgdModel = np.zeros((1, 65), np.float64)
    rect = (0, 0, mask.shape[0] - 1, mask.shape[1] - 1)
    cv2.grabCut(src, trimap, rect, bgdModel, fgdModel, 5, cv2.GC_INIT_WITH_MASK)

    # create mask again
    mask2 = np.where(
        (trimap == cv2.GC_FGD) | (trimap == cv2.GC_PR_FGD),
        255,
        0
    ).astype('uint8')

    contour2 = findSignificantContour(mask2)
    mask3 = np.zeros_like(mask2)
    cv2.fillPoly(mask3, [contour2], 255)

    # blended alpha cut-out
    mask3 = np.repeat(mask3[:, :, np.newaxis], 3, axis=2)
    mask4 = cv2.GaussianBlur(mask3, (3, 3), 0)
    alpha = mask4.astype(float) * 1.1  # making blend stronger
    alpha[mask3 > 0] = 255
    alpha[alpha > 255] = 255
    alpha = alpha.astype(float)

    foreground = np.copy(src).astype(float)
    foreground[mask4 == 0] = 0
    background = np.ones_like(foreground, dtype=float) * 255

    # Normalize the alpha mask to keep intensity between 0 and 1
    alpha = alpha / 255.0
    # Multiply the foreground with the alpha matte
    foreground = cv2.multiply(alpha, foreground)
    # Multiply the background with ( 1 - alpha )
    background = cv2.multiply(1.0 - alpha, background)
    # Add the masked foreground and background.
    cutout = cv2.add(foreground, background)

    cv2.imwrite(data[:-4] + '_contour.png', cutout)
    cutout = cv2.imread(data[:-4] + '_contour.png')

    used_platform = platform.system()

    if used_platform == "Linux":
        os.system("rm " + data[:-4] + '_contour.png')
    else:
        os.system("del " + data[:-4] + '_contour.png')

    # cutout = cv2.imread(source, 1)  # TEMPORARY

    # cutout_blurred = cv2.GaussianBlur(cutout, (5, 5), 0)
    cutout_blurred = cv2.GaussianBlur(cutout, (7, 7), 0)

    gray = cv2.cvtColor(cutout_blurred, cv2.COLOR_BGR2GRAY)
    # threshed = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
    #                                  cv2.THRESH_BINARY_INV, blockSize=501,C=2)

    # front and back light
    # lower_gray = np.array([175, 175, 175])  # [R value, G value, B value]
    # upper_gray = np.array([215, 215, 215])
    # front light only
    lower_gray = np.array([min_rgb, min_rgb, min_rgb])  # [R value, G value, B value]
    upper_gray = np.array([max_rgb, max_rgb, max_rgb])

    mask = cv2.bitwise_not(cv2.inRange(cutout_blurred, lower_gray, upper_gray) + cv2.inRange(gray, 254, 255))

    # binarise
    ret, image_bin = cv2.threshold(mask, 127, 255, cv2.THRESH_BINARY_INV)
    image_bin[image_bin < 127] = 0
    image_bin[image_bin > 127] = 1

    #cv2.imwrite(data[:-4] + '_threshed.png', 1 - image_bin, [cv2.IMWRITE_PNG_BILEVEL, 1])

    print("cleaning up thresholding result, using connected component labelling of %s"
          % (data.split("\\")[-1]))

    # remove black artifacts
    blobs_labels = measure.label(cv2.GaussianBlur(image_bin, (5, 5), 0), background=0)

    image_cleaned = remove_holes(blobs_labels, min_num_pixel=min_bl)

    image_cleaned_inv = 1 - image_cleaned

    # cv2.imwrite(data[:-4] + "_extracted_black_.png", image_cleaned_inv, [cv2.IMWRITE_PNG_BILEVEL, 1])

    # remove white artifacts
    blobs_labels_white = measure.label(image_cleaned_inv, background=0)

    image_cleaned_white = remove_holes(blobs_labels_white, min_num_pixel=min_wh)

    cv2.imwrite(data[:-4] + "_masked.png", image_cleaned_white, [cv2.IMWRITE_PNG_BILEVEL, 1])

    if create_cutout:
        image_cleaned_white = cv2.imread(data[:-4] + "_masked.png")
        cutout = cv2.imread(data)
        # create the image with an alpha channel
        # smooth masks prevent sharp features along the outlines from being falsely matched
        """
        smooth_mask = cv2.GaussianBlur(image_cleaned_white, (11, 11), 0)
        smooth_mask = cv2.GaussianBlur(image_cleaned_white, (5, 5), 0)
        rgba = cv2.cvtColor(cutout, cv2.COLOR_RGB2RGBA)
        # assign the mask to the last channel of the image
        rgba[:, :, 3] = smooth_mask
        # save as lossless png
        cv2.imwrite(data[:-4] + '_cutout.tif', rgba)
        """

        _, mask = cv2.threshold(cv2.cvtColor(image_cleaned_white, cv2.COLOR_BGR2GRAY), 240, 255, cv2.THRESH_BINARY)
        print(cutout.shape)
        img_jpg = cv2.bitwise_not(cv2.bitwise_not(cutout[:, :, :3], mask=mask))

        print(img_jpg.shape)
        img_jpg[np.where((img_jpg == [255, 255, 255]).all(axis=2))] = [0, 0, 0]
        cv2.imwrite(data[:-4] + '_cutout.jpg', img_jpg)



def mask_images(input_paths, min_rgb, max_rgb, min_bl, min_wh, create_cutout=False):
    # load pre-trained edge detector model
    edgeDetector = cv2.ximgproc.createStructuredEdgeDetection(str(Path.cwd().joinpath("scripts", "model.yml")))
    print("loaded edge detector...")

    for img in input_paths:
        createAlphaMask(img, edgeDetector, min_rgb, max_rgb, min_bl, min_wh, create_cutout)
