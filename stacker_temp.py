import argparse
import time
from os import cpu_count
import multiprocessing as mp
from concurrent.futures import ProcessPoolExecutor, as_completed
from collections.abc import Iterable
from scAnt import files_io
from scAnt.post_processing import *


base = Path("D:\scans")
all_folders = [
    base / 'cataglyphis_velox_2',
    base / 'cataglyphis_velox_3',
    base / 'cataglyphis_velox_4',
    base / 'messor_1',
    base / 'messor_2',
    base / 'messor_head_1',
    base / 'cataglyphis_head_1',
    base / 'cataglyphis_head_2'
]

for folder in all_folders:

    time.sleep(10)

    args = {'images': folder.as_posix(),
            'threshold': 10.0,
            'sharpen': False,
            'display': False,
            'verbose': 2,
            'single_stack': False,
            'focus_check': False,
            'method': 'Default',
            'experimental': True,
            }

    if args['verbose'] > 0:
        print("\n[INFO]:\n",
                f"  - Verbosity level: {args['verbose']}\n",
              f"""  - Out of focus images will {f'be discarded using a laplacian variance threshold of {args["threshold"]}' if args["focus_check"] else 'NOT be discarded'}\n""",
                f"  - {'Previews' if args['display'] else 'NO previews'} will be displayed during focus check\n",
                f"  - Output images {'will be' if args['sharpen'] else 'will NOT be'} additionally sharpened\n",
                f"  - Images in target directory {'will be treated as a single stack' if args['single_stack'] else 'will be processed stack by stack'}\n",
                f"  - Stacking using the "
              )

    max_processes = max(1, cpu_count()-2)

    inputs = files_io.get_paths(args['images'], force_single_stack=args['single_stack'], verbose=args['verbose'])
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

    if args['experimental']:

        start = time.time()
        if args['verbose'] > 0:
            print('Stacking ...')
        else:
            print('Stacking ... ', end='', flush=True)  # If not verbose, still print this but without returning

        nb_stacks = len(inputs)
        stacks_done = 0
        with ProcessPoolExecutor(max_workers=1) as executor:   # TODO - Maybe use more processes when not using the GPU?
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
