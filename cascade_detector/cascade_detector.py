import argparse
import cv2
import sys
import os

import datetime as dt
import logging as log

from time import sleep

def check_positive(value):
    ivalue = int(value)
    if ivalue < 0:
        raise argparse.ArgumentTypeError("%s is an invalid, should be greater than equal zero" % value)
    return ivalue

# Optional Argument
parser = argparse.ArgumentParser(description='Cascade Detector Options')
parser.add_argument('--file-path', action="store", dest="file_path", type=str,
    default='/usr/share/opencv4/haarcascades/haarcascade_frontalface_default.xml',
    help='specify trained data file path. (default: /usr/share/opencv4/haarcascades/haarcascade_frontalface_default.xml)')
parser.add_argument('--device-id', action="store", dest="device_id", type=check_positive,
    default=0, help='camera device number. (default: 0)')
args_result = parser.parse_args()

# Configuration setting
Cascade = cv2.CascadeClassifier(args_result.file_path)
if Cascade.empty():
    print('failed to open CascadeClassifier, exit')
    sys.exit(1)
log.basicConfig(filename='cascade_detector.log',level=log.INFO)

# Open camera device and get object (default 0, if needed change the device number)
# use `ls -ltrh /dev/video*` command to check the device number.
video_capture = cv2.VideoCapture(args_result.device_id, cv2.CAP_V4L2)

# Check if camera is opened
retry_open_camera = 0
while not video_capture.isOpened():
    if retry_open_camera < 3:
        print('video capturing is not initialized...retry in 5 sec')
        sleep(5)
        retry_open_camera += 1
        continue
    else:
        print('video capturing cannot be initialized...exit')
        sys.exit(1)
else:
    pass

env_val = os.getenv('QT_X11_NO_MITSHM')
if env_val is None:
    os.environ['QT_X11_NO_MITSHM'] = '1'

# Start cascade detection loop until key interuption
anterior = 0 # this is used to keep the detection log.
print('start cascade detection loop, stop with [Ctrl-C]')
try:
    while True:
        # Capture frame-by-frame
        ret, frame = video_capture.read()
        if not ret:
            print('failed to read the image')

        # Convert image into grayscale
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # Detects objects of different sizes in the input image
        objs, _, levelWeights = Cascade.detectMultiScale3(
            gray,
            scaleFactor = 1.1,
            minNeighbors = 5,
            minSize = (30, 30),
            outputRejectLevels = True
        )

        for (obj, levelWeight) in zip(objs, levelWeights):
            cv2.rectangle(frame, (obj[0], obj[1]),
              (obj[0]+obj[2], obj[1]+obj[3]), (0, 255, 0), 2)
            cv2.putText(frame, 'weight: {:.3f}'.format(levelWeight[0]),
              (obj[0] + 10, obj[1] + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 1)

        # keep the detection history in the log
        if anterior != len(objs):
            anterior = len(objs)
            log.info("objs: "+str(len(objs))+" at "+str(dt.datetime.now()))

        # Display the resulting frame
        cv2.imshow('Video', frame)

        # 33msec (30fps) to wait
        cv2.waitKey(33)

except KeyboardInterrupt:
    print('Key Interruption, shutdown')

# When everything is done, release video object and destroy everything.
video_capture.release()
cv2.destroyAllWindows()
del os.environ['QT_X11_NO_MITSHM']
