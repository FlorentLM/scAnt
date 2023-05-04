import argparse
import time
from os import cpu_count
import multiprocessing as mp
from concurrent.futures import ProcessPoolExecutor, as_completed
from collections.abc import Iterable
from scAnt import files_io
from scAnt.post_processing import *


def str2bool(s):
    if isinstance(s, bool):
        return s
    if s.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif s.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')

def str2list(s):
    if not type(s) is str and isinstance(s, Iterable):
        return s
    else:
        return s.split(',')


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Focus stack images.')

    parser.add_argument("-i", "--images",
                        type=str2list,
                        required=True,
                        help="Path to input directory of images, or list of image paths")
    parser.add_argument("-t", "--threshold",
                        type=float,
                        default=10.0,
                        help="Focus measures that fall below this value will be considered 'blurry'")
    parser.add_argument("-s", "--sharpen",
                        type=str2bool,
                        default=False,
                        help="Apply sharpening to final result [True / False]")
    parser.add_argument("-d", "--display",
                        type=str2bool,
                        default=False,
                        help="Show images with displayed focus score [True / False]")
    parser.add_argument("-v", "--verbose",
                        action='count',
                        default=0,
                        help="Verbose mode [levels 0, 1 or 2]")
    # parser.add_argument("-b", "--single_stack",
    #                     type=str2bool,
    #                     default=False,
    #                     help="Process all images in the specified folder [True / False]")
    parser.add_argument("-f", "--focus_check",
                        type=str2bool,
                        default=True,
                        help="Check whether out-of-focus images should be discarded before stacking")
    parser.add_argument("-m", "--method",
                        type=str,
                        default="Default",
                        help="Blending method (Default, 1-Star, Masks)")
    parser.add_argument("-x", "--experimental_stacking",
                        type=str2bool,
                        default=True,
                        help="Use experimental stacking method")

    args = vars(parser.parse_args())

    if args['verbose'] > 0:
        print("[INFO]:\n",
                f"  - Verbosity level: {args['verbose']}\n",
              f"""  - Out of focus images will {f'be discarded using a laplacian variance threshold of {args["threshold"]}' if args["focus_check"] else 'NOT be discarded'}\n""",
                f"  - {'Previews' if args['display'] else 'NO previews'} will be displayed during focus check\n",
                f"  - Output images {'will be' if args['sharpen'] else 'will NOT be'} additionally sharpened\n",
                f"  - Images in target directory {'will be treated as a single stack' if args['single_stack'] else 'will be processed stack by stack'}\n",
              )

    max_processes = max(1, cpu_count()-2)

    inputs = files_io.get_paths(args['images'], verbose=args['verbose'])
    output_dir = files_io.mk_outputdir(inputs, verbose=args['verbose'])

##

    # /!\ Note:
    # The 'spawn' start method (see multiprocessing.get_context()) does not pass the global-level variables
    # to the child processes, so it is necessary to explicitly repeat the global args FOCUS_THRESH, DISPLAY, VERBOSE
    # to the map() calls below.
    # With the 'fork' start method (the default on Linux), the child process *do* inherit the globals, so this would
    # not be needed. Unfortunately, the 'fork' start method is unsafe on macOS, and does not exist on Windows, so we
    # default to 'spawn'.
    mp.set_start_method('spawn')

    if args['focus_check']:
        print('Checking focus...', end='', flush=True)
        start = time.time()

        with ProcessPoolExecutor(max_workers=max_processes) as executor:
            focus_results = executor.map(focus_check_multi,
                                              inputs,                       # 1 value of the iterable per child process
                                              repeat(args['threshold']),    # same arg repeated in all child processes
                                              repeat(args['display']),
                                              repeat(args['verbose']))

        focus_results = list(focus_results)
        inputs = [group[0] for group in focus_results]
        nb_stacks = len(inputs)

        print(' Done.')
        end = time.time()

        if args['verbose'] > 0:
            # Print the results in a pretty way
            stacks_names = [s[0].stem.split("step_")[0] for s in inputs]
            for stack_name, stack_result in zip(stacks_names, focus_results):
                sharp_images, blurry_images = stack_result
                print(f"Found {len(sharp_images)} focused images and {len(blurry_images)} blurry images for stack {stack_name}.")
            print(f"Focus check took {end - start} seconds.")

##

    if args['experimental_stacking']:

        start = time.time()
        if args['verbose'] > 0:
            print('Stacking ...')
        else:
            print('Stacking ... ', end='', flush=True)  # If not verbose, still print this but without returning

        nb_stacks = len(inputs)
        stacks_done = 0
        with ProcessPoolExecutor(max_workers=1) as executor:
            for r in as_completed(
                    [executor.submit(focus_stack_2, s, output_dir) for s in inputs]
            ):
                stacks_done += 1
                if args['verbose'] == 1:
                    print(f"Processed stack [{stacks_done}/{nb_stacks}]")

        print('Done.')

        end = time.time()
        if args['verbose'] > 0:
            print(f"Stacking took {end - start} seconds.")

    else:

        start = time.time()
        if args['verbose'] > 0:
            print('Aligning ...')
        else:
            print('Aligning ... ', end='', flush=True)  # If not verbose, still print this but without returning

        stacks_done = 0
        with ProcessPoolExecutor(max_workers=max_processes) as executor:
            for r in as_completed(
                    [executor.submit(alignment, s, output_dir) for s in inputs]
            ):
                stacks_done += 1
                if args['verbose'] == 1:
                    print(f"Aligned stack [{stacks_done}/{nb_stacks}]")

        print('Done.')

        end = time.time()
        if args['verbose'] > 0:
            print(f"Alignment took {end - start} seconds.")

        start = time.time()
        if args['verbose'] > 0:
            print('Fusing ...')
        else:
            print('Fusing ... ', end='', flush=True)  # If not verbose, still print this but without returning

        stacks_done = 0
        with ProcessPoolExecutor(max_workers=max_processes) as executor:
            for r in as_completed(
                    [executor.submit(fuse, s, output_dir) for s in inputs]
            ):
                stacks_done += 1
                if args['verbose'] == 1:
                    print(f"Fused stack [{stacks_done}/{nb_stacks}]")
        print('Done.')

        end = time.time()
        if args['verbose'] > 0:
            print(f"Fuse took {end - start} seconds.")
