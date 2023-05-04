from pathlib import Path
from os.path import commonpath
from collections.abc import Iterable
from natsort import natsorted, ns
import shutil


def to_paths(*args):
    """ Accepts strings, Path objects, or any form of iterable thereof, and returns a list of Path objects. """

    if len(args) == 1:
        if type(args[0]) is str or isinstance(args[0], Path):
            list_of_paths = [Path(args[0])]
        elif isinstance(args[0], Iterable):
            list_of_paths = [Path(p) for p in args[0]]
        else:
            raise AttributeError(f"Cannot parse {args[0]} as a path")
    else:
        list_of_paths = [Path(p) for p in args]

    return list_of_paths

def check_files_paths(list_of_paths, verbose=0):
    """ Checks if files in a list of paths exist or not. Skips paths that are directories.
    Returns a list of (de-duplicated) existing files paths. """

    # extensions = ['.tif', '.tiff', '.jpg', '.jpeg', '.png', '.bmp']
    extensions = ['.tif', '.tiff']
    checked = set()
    for p in list_of_paths:
        imgpath = Path(p)   # It doesn't hurt to be careful
        if imgpath.exists() and imgpath.is_file() and imgpath.suffix in extensions:
            checked.add(imgpath)
        elif imgpath.exists() and imgpath.is_dir():
            if verbose > 0:
                print(f"{imgpath} is a directory; skipping")
        else:
            if verbose > 0:
                print(f"File {imgpath} not found; skipping")
    return natsorted(checked, alg=ns.IGNORECASE)

def find_RAW_folder(folder_path, verbose=0):
    """ Finds the RAW folder containing images to post-process. """

    folder_path = Path(folder_path)

    # The wildcard is redundant in this case, so strip it
    if '*' in folder_path.name:
        folder_path = folder_path.parent

    if folder_path.is_file():
        raise FileNotFoundError(f"{folder_path} is a file, not a folder!")
    elif not folder_path.exists():
        raise FileNotFoundError(f"{folder_path} not found")

    if 'RAW' in folder_path.stem:
        raw_images_folder = folder_path
    else:
        children_has_RAW = list(folder_path.glob('*RAW*'))
        if len(children_has_RAW) == 0:
            raise FileNotFoundError(f"No RAW folder found in '{folder_path}'")
        elif len(children_has_RAW) > 1:
            raise FileNotFoundError(f"Ambiguous: Multiple RAW folders found in {folder_path}")
        else:
            raw_images_folder = folder_path / children_has_RAW[0]
            if verbose > 0:
                print(f"Found \"{raw_images_folder.stem}\" folder in {folder_path}")

    if not any(raw_images_folder.glob('*.tif')):
        raise FileNotFoundError(f"No .tif files found in {raw_images_folder}")

    return raw_images_folder

def get_paths(*list_of_paths, verbose=0):
    """ Parses a string, Path object, or iterable thereof, and return the naturally sorted paths to images,
     grouped by stack into a list of lists """

    list_of_paths = to_paths(*list_of_paths)

    all_stacks = []

    # If only 1 path, it is probably a folder. Otherwise, find_RAW_folder() will throw an informative exception.
    if len(list_of_paths) == 1:
        raw_folder = find_RAW_folder(list_of_paths[0], verbose=verbose)

        # Fetch and sort stack names
        uniq_names = set([img_path.stem.split('step')[0] for img_path in raw_folder.glob("*.tif")])
        list_stacks_names = list(uniq_names)

        sorted_stacks = natsorted(list_stacks_names, alg=ns.IGNORECASE)     # Not crucially needed, but why not

        for stack in sorted_stacks:
            images_current_stack = list(raw_folder.glob(f"{stack}*.tif"))
            all_stacks.append(natsorted(images_current_stack, alg=ns.IGNORECASE))

    # If more than 1 path, there are 3 cases:
    elif len(list_of_paths) > 1:

        # - Multiple files, treat them as 1 stack
        if all([f.is_file() if f.exists() else False for f in list_of_paths]):
            all_stacks = [check_files_paths(list_of_paths, verbose=verbose)]

        # - Multiple folders, treat them as separate stacks (and handle wildcards!)
        elif all([f.is_dir() if (f.exists() or '*' in f.name) else False for f in list_of_paths]):
            for dir in list_of_paths:
                if '*' in dir.name:
                    all_stacks.append(list(dir.parent.expanduser().glob(dir.name)))
                else:
                    all_stacks.append(check_files_paths(dir.glob('*'), verbose=verbose))

        # - Mix of both, throw exception
        else:
            raise AssertionError("Please either pass paths to files OR to folders, not a mix of both.")

    if verbose > 0:
        print(f"Found {len(all_stacks)} stack{'s' if len(all_stacks) > 1 else ''}.")
    return all_stacks

def mk_outputdir(all_stacks, verbose=0):
    """ Creates the output directory in the lowest-level parent folder for all stacks, and returns the path. """
    common = Path(commonpath([commonpath(s) for s in all_stacks]))
    if 'RAW' in common.name:
        outputdir_name = common.name.replace('RAW', 'stacked')
    else:
        outputdir_name = common.name + '_stacked'
    stacked_images_folder = common.parent / outputdir_name
    stacked_images_folder.mkdir(parents=True, exist_ok=True)

    if verbose > 0:
        print(f"Created output folder: {stacked_images_folder}")
    return stacked_images_folder


def install_instructions():
    import getpass
    import platform
    sys = platform.system()
    # TODO - add better checks for brew, winget, etc
    print("\nInstallation instructions:\n")
    if "Darwin" in sys:
        print("--- For stacking method 1 ---\n"
              "   1. Install Homebrew (https://brew.sh/) if you do not already have it.\n"
              "   2. Install Hugin:\n      `brew install hugin`\n"
              "   3. Disable Gatekeeper's warnings:\n      `xattr -d com.apple.quarantine /Applications/Hugin/Hugin.app && xattr -d com.apple.quarantine /Applications/Hugin/PTBatcherGUI.app`\n"
              f"   4. Add the executables to your Path:\n      `ln -s /Applications/Hugin/tools_mac/* /Users/{getpass.getuser()}/.local/bin`\n"
              "\n--- For stacking method 2 ---\n"
              "   1. Download focus-stack for macOS (https://github.com/PetteriAimonen/focus-stack/releases) and place focus-stack.app in your /Applications folder.\n"
              "   2. Disable Gatekeeper's warning:\n      `xattr -d com.apple.quarantine /Applications/focus-stack.app`\n"
              f"   3. Add the executable to your Path:\n      `ln -s /Applications/focus-stack.app/Contents/MacOS/focus-stack /Users/{getpass.getuser()}/.local/bin/focus-stack`"
              )
    if "Windows" in sys:
        print("--- For stacking method 1 ---\n"
              "   1. Install Winget (https://apps.microsoft.com/store/detail/app-installer/9NBLGGH4NNS1) if you do not already have it (old Windows 10 verions only).\n"
              "   2. Install Hugin:\n      `winget install --id Hugin.Hugin`\n"
              f'   3. Add the executables to your Path:\n      `setx Path "%Path%;C:\\Program Files\\Hugin\\bin"`\n      (needs to be ran as Admin)\n'
              "\n--- For stacking method 2 ---\n"
              "   1. Download focus-stack for Windows (https://github.com/PetteriAimonen/focus-stack/releases)\n"
              f'   2. Extract the zip to any location (i.e. "C:\\focus-stack") and add this location to your path:\n      `setx Path "%Path%;C:\\focus-stack"`\n      (needs to be ran as Admin)'
              )

def lookup_bin(bin_name, prefer_system=False, verbose=0):

    # Get system bin path
    system_bin = shutil.which(f"{bin_name}")

    if prefer_system:
        if system_bin is not None:
            return Path(system_bin)
        else:
            if verbose:
                print(f"Executable {bin_name} not found in system path.\nTrying in locally-packaged binaries")

    # Lookup locally-packaged binaries folder
    cwd = Path().cwd()
    local_folder = cwd / 'external'
    if not local_folder.exists():
        local_folder = cwd.parent / 'external'

    local_bin = shutil.which(f"{bin_name}", path=local_folder)

    if local_bin is not None:
        return Path(local_bin)
    else:
        if system_bin is None:
            if verbose > 0:
                install_instructions()
            raise FileNotFoundError(f"{bin_name} can't be found")
        else:
            if verbose:
                print(f"Executable {bin_name} not found in locally-packaged binaries\nUsing {bin_name} found at {Path(system_bin).parent}")
            return Path(system_bin)
