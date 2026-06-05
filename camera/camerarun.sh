#!/bin/bash
sudo modprobe -r v4l2loopback
sudo modprobe v4l2loopback exclusive_caps=1 card_label="GalaxyCam"
scrcpy --video-source=camera --no-audio --v4l2-sink=/dev/video0 --camera-size=1920x1080 --camera-fps=30
