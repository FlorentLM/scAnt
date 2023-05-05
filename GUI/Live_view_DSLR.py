# TODO - add support for gphoto on Linux and macOS

import subprocess
from pathlib import Path
from time import sleep
# from scAnt.files_io import lookup_bin

# Update with the path to CameraControlCmd.exe file.
digicam_install_path = Path('C:\Program Files (x86)\digiCamControl')   # TODO use lookup_bin instead

digicam_cmd_bin = 'CameraControlCmd.exe'
digicam_cmd_path = digicam_install_path / digicam_cmd_bin

digicam_app_bin = 'CameraControl.exe'
digicam_app_path = digicam_install_path / digicam_app_bin

digicam_remote_bin = 'CameraControlRemoteCmd.exe'
digicam_remote_path = digicam_install_path / digicam_remote_bin


class customDSLR():

    def __init__(self):

        # get camera info. If none is connected, exit the program
        p = subprocess.Popen(f"{digicam_cmd_path} /list cameras",
                             stdout=subprocess.PIPE,
                             universal_newlines=True,
                             shell=False)
        (output, err) = p.communicate()
        try:
            self.cameras = output.split("New Camera is connected ! Driver :")[1:]
            for i, camera in enumerate(self.cameras):
                self.cameras[i] = camera.split("\n")[0]  # remove line break from each entry

            # by default, set the first found camera as active
            self.camera_model = self.cameras[0]

            if self.camera_model[0:4] == "digi" or not self.camera_model:
                print("No camera detected!")
                self.camera_model = None
                return
            else:
                print(f"Detected DSLR cameras: {' '.join(self.cameras)}")
                print(f"Using: {self.camera_model}")
        except IndexError:
            print("No DSLR detected!")
            return

        # get all values as soon as camera is initialised
        self.all_iso_vals = []
        self.all_aperture_vals = []
        self.all_shutterspeed_vals = []
        self.all_whitebalance_vals = []
        self.all_compression_vals = []

        self.shutterspeed = None
        self.aperture = None
        self.iso = None
        self.whitebalance = None
        self.compression = None

    def initialise_camera(self):
        # launch DigiCamControl
        subprocess.Popen(digicam_app_path)

        # check for instance of CameraControl.exe for 20 seconds until timeout
        for i in range(20):
            sleep(1)
            sp = subprocess.Popen(f"{digicam_remote_path} /c list sessions",
                                  stdout=subprocess.PIPE,
                                  universal_newlines=True,
                                  shell=False)
            (output, err) = sp.communicate()
            message = str(output).split(":")[-1].split("\n")[0]
            if message == "no camera is connected":
                print(f"Waiting for instance of {digicam_app_bin} to launch...")
            else:
                print(message)
                print(f"{digicam_app_bin} launched successfully!")
                sleep(2)
                break

            if i == 19:
                print(f"Timeout! No response from camera or {digicam_app_bin}!")
                return

        # iso
        self.all_iso_vals = self.get_all_settings("iso")
        self.iso = self.get_current_setting("iso")
        # aperture
        self.all_aperture_vals = self.get_all_settings("aperture")
        self.aperture = self.get_current_setting("aperture")
        # shutter speed
        self.all_shutterspeed_vals = self.get_all_settings("shutterspeed")
        self.shutterspeed = self.get_current_setting("shutterspeed")
        # white balance
        self.all_whitebalance_vals = self.get_all_settings("whitebalance")
        self.whitebalance = self.get_current_setting("whitebalance")
        # compression setting
        self.all_compression_vals = self.get_all_settings("compressionsetting")
        self.compression = self.get_current_setting("compressionsetting")

        print("Successfully initialised camera!")

    def get_all_settings(self, key):
        sleep(0.2)  # prevents issuing to many commands at a time
        sp = subprocess.Popen(f"{digicam_remote_path} /c list {key}",
                              stdout=subprocess.PIPE,
                              universal_newlines=True,
                              shell=False)
        (output, err) = sp.communicate()
        raw_vals = (str(output).split("[")[-1].split("]")[0].split(","))
        all_vals = []
        for val in raw_vals:
            all_vals.append(val.split('"')[1])
        return all_vals

    def set_shutterspeed(self, shutterspeed="1/100"):
        subprocess.Popen(f"{digicam_remote_path} /c set shutterspeed {shutterspeed}",
                         stdout=subprocess.DEVNULL,
                         stderr=subprocess.STDOUT)

    def set_iso(self, iso="500"):
        subprocess.Popen(f"{digicam_remote_path} /c set iso {iso}",
                         stdout=subprocess.DEVNULL,
                         stderr=subprocess.STDOUT)

    def set_aperture(self, aperture="5.6"):
        subprocess.Popen(f"{digicam_remote_path} /c set aperture {aperture}",
                         stdout=subprocess.DEVNULL,
                         stderr=subprocess.STDOUT)

    def set_whitebalance(self, whitebalance="Auto"):
        subprocess.Popen(f"{digicam_remote_path} /c set whitebalance {whitebalance}",
                         stdout=subprocess.DEVNULL,
                         stderr=subprocess.STDOUT)

    def set_compression(self, compression):
        subprocess.Popen(f"{digicam_remote_path} /c set compressionsetting {compression}",
                         stdout=subprocess.DEVNULL,
                         stderr=subprocess.STDOUT)

    def start_live_view(self):
        # open live view window
        subprocess.Popen(f"{digicam_remote_path} /c do LiveViewWnd_Show",
                         stdout=subprocess.DEVNULL,
                         stderr=subprocess.STDOUT)

    def stop_live_view(self):
        # close live view window
        subprocess.Popen(f"{digicam_remote_path} /c do LiveViewWnd_Hide",
                         stdout=subprocess.DEVNULL,
                         stderr=subprocess.STDOUT)

    def capture_image(self, img_name=Path("example.jpg")):
        img_name = Path(img_name)   # Just in case
        subprocess.Popen(f"{digicam_remote_path} /c CaptureNoAf {img_name}",
                         stdout=subprocess.DEVNULL,
                         stderr=subprocess.STDOUT)

    def get_current_setting(self, setting):
        sleep(0.2)  # prevents issuing to many commands at a time
        sp = subprocess.Popen(f"{digicam_remote_path} /c get {setting}",
                              stdout=subprocess.PIPE,
                              universal_newlines=True,
                              shell=False)
        (output, err) = sp.communicate()
        val = output.split('"')[1]
        print(f"Current {setting}: {val}")
        return val


if __name__ == '__main__':

    # example settings
    iso = "500"
    aperture = "3.5"
    shutterspeed = "1/50"
    whitebalance = "Kelvin"
    compression = "JPEG (FINE)"

    # where to save images
    current_folder = Path.cwd().parent

    # calling the class for the first time is meant to aid in finding attached cameras.
    # the name of the connected camera is then stored in DSLR.camera_model
    DSLR = customDSLR()
    if DSLR.camera_model is None:
        # exit program when no camera is connected
        exit()

    # if a camera is connected, it needs to be initialised before use
    DSLR.initialise_camera()
    # wait for setting to be applied before sending next capture command
    sleep(1)

    DSLR.set_iso(iso)
    DSLR.set_aperture(aperture)
    DSLR.set_shutterspeed(shutterspeed)
    DSLR.set_whitebalance(whitebalance)
    DSLR.set_compression(compression)

    # wait for setting to be applied before sending next capture command
    sleep(1)

    # TODO get highlight exposure to work
    # system('"' + str(digi_cam_remote_path) + '"' + " /c set liveview.highlightoverexp true")

    # and now capture 3 images with unique names and increase the ISO each time
    iso_vals = ["500", "1000", "2000"]
    for iso_val in iso_vals:
        DSLR.set_iso(iso_val)
        sleep(2)
        # wait for setting to be applied before sending next capture command
        img_name = f"test_image_iso_{iso_val}.jpg"
        DSLR.capture_image(current_folder / img_name)
        sleep(1)
