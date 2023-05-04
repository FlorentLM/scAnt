import PySpin

system = PySpin.System.GetInstance()

# Get current library version
version = system.GetLibraryVersion()
print('Spinnaker library version: %d.%d.%d.%d' % (version.major, version.minor, version.type, version.build))

# Retrieve list of cameras from the system
cam_list = system.GetCameras()

# get all serial numbers of connected and support FLIR cameras
device_names = []

for id, cam in enumerate(cam_list):
    nodemap = cam.GetTLDeviceNodeMap()

    # Retrieve device serial number
    node_device_serial_number = PySpin.CStringPtr(nodemap.GetNode("DeviceSerialNumber"))
    node_device_model = PySpin.CStringPtr(nodemap.GetNode("DeviceModelName"))

    if PySpin.IsAvailable(node_device_serial_number) and PySpin.IsReadable(node_device_serial_number):
        device_names.append([node_device_model.GetValue(), node_device_serial_number.GetValue()])

    print("Detected", device_names[id][0], "with Serial ID", device_names[id][1])

# by default, use the first camera in the retrieved list
cam = cam_list[0]

num_cameras = cam_list.GetSize()

print('Number of cameras detected: %d' % num_cameras)


# cam.Init()

# # Clear camera list before releasing system
# cam_list.Clear()
#
# # Release system instance
# system.ReleaseInstance()
##

