## opencv_sample

OpenCV related samples. This project includes multiple samples related to OpenCV, each directory has single sample to easy to understand.

### Platform Dependency

- Ubuntu20.04
- Dependent Packages

```
# Install opencv packages.
> apt install libopencv-dev python3-opencv libcanberra-gtk-module

# Check if opencv is installed.
> python3 -c "import cv2; print(cv2.__version__)"
4.2.0
```

### Using Docker Container

It is preferred to use docker container ubuntu:20.04. but using docker container brings us extra setting if the application requires GUI(x11) access. the followings are the guide how to set up the container with x11 application.

```
# pull docker container
> docker pull ubuntu:20.04

# run container with options
docker run -it --privileged --name ubuntu20_work -e DISPLAY=unix$DISPLAY -v /tmp/.X11-unix:/tmp/.X11-unix -v /home/tomoyafujita/DVT/docker_ws:/root/docker_ws ubuntu:20.04

# disable access control for X window (host system)
> xhost +

# install x11-apps in container to confirm access to x11
> apt install x11-apps
> xeyes
# if you do not see the eyes, you are in trouble...
```

## Samples

- [Cascade Detector](./cascade_detector/readme.md)
