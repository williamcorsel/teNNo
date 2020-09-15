# teNNo

The Size Expansion algorithm enables obstacle detection by tracking size changes of objects between frames. This repository contains the code used to perform the experiments performed in the Bachelor's thesis "INSERT_TITLE" at Leiden University. It contains a controller for the Ryze Tello drone that was used, enabling the streaming of video and drone control using the keyboard.

Size Expansion uses object detection to find object in images from the drone's camera to avoid them. Multiple detection methods are included. These can be found in the `detectors` directory

* **SIFT:** Uses SIFT feature matching to identify objects and calculate sizes. Described in the [paper](https://www.mdpi.com/1424-8220/17/5/1061) by Al-Kaff et al.
* **Darknet:** Runs the [YOLOv4](https://github.com/AlexeyAB/darknet) model used in the thesis.
* **TensorFlow:** Can run models found in the Tensorflow object detection [model zoo](https://github.com/tensorflow/models/blob/master/research/object_detection/g3doc/detection_model_zoo.md).
* **EfficientDet:** Test implementation of [EfficientDet]((https://github.com/zylo117/Yet-Another-EfficientDet-Pytorch)). Currently only detects objects

## How to install

This program is written in Python3. Make sure Python3 and pip3 are installed correctly.

### Base + SIFT Detector

Packages needed for the main program:

```bash
pip install -r requirements.txt
```

Install the latest version of TelloPy from source: More information [here](https://github.com/hanyazou/TelloPy). Note: the version on PyPi is out of data and does not work!

```bash
git clone https://github.com/hanyazou/TelloPy
cd TelloPy
sudo apt-get install python3-setuptools
pip install wheel
python setup.py bdist_wheel
pip install dist/tellopy-*.dev*.whl --upgrade
```

OpenCV libraries (v4.3.0 used) can also be installed from source [here](https://github.com/opencv/opencv/releases/tag/4.3.0), together with the extra modules (opencv-contrib) necessary for SIFT [here](https://github.com/opencv/opencv_contrib/releases/tag/4.3.0). For install directions look [here](https://docs.opencv.org/4.3.0/d7/d9f/tutorial_linux_install.html). For installing into a Conda environment see [here](https://jayrobwilliams.com/files/html/OpenCV_Install.html). Installing via [PyPi](https://pypi.org/project/opencv-contrib-python/) might also be possible if the wheel is updated to at least v4.3.0. (Note: the most recent version of OpenCV-Python is now uploaded to PyPi)

Installing these packages will allow you run the main controller functionality and the SIFT obstacle detector.

### Other detector modules

Other detector modules can be found in the `detectors` directory. Each contains a separate README with install instructions

Using the gpu version of the CNN object detectors is highly recommended. For this you need to install [CUDA](https://docs.nvidia.com/cuda/cuda-installation-guide-linux/index.html).

## How to use

Simply run the following command to start the program:

```bash
python tenno.py
```

The following flags are supported:

|Option |Action|Choices|
|---|---|---|
|-d, --detector | Chooses object detector | {sift, yolodark, tf, eff} |
|-z, --zfile | Adds a zone weight file for YOLO detector | Path to file |
|-t, --test | Enables experiment mode | {ratio_size, ratio_kp, avoid} |
|-o, --output | Output file of experiment results | Path to file |
|-r, --record | Enables recording of every flight in test modes | |

To get full functionality using the object detector of choice call:

```bash
python tenno.py -d sift
python tenno.py -d yolodark
```

### Keyboard controls

|Key    | Action    |
|---    |---        |
|w, a, s, d| Move drone horizontally|
|Space, Shift| Move drone vertically|
|q, e | Turn in place|
|o | Toggle detector on/off
|l | Toggle logging on/off
|p, 1 | Resets current position to (0,0,0)|
|h | Sets waypoint to (0,0,0) |
|r | Toggle video recording on/off |
|2 | Toggle waypoint flying on/off |
|3 | Reset position and start flying to waypoint with obstacle detection |
| - | Toggle HUD on/off |
| Esc | Land drone and quit program |

### Test modes

The ratio_size and ratio_kp test modes enable the adjustments of hull size ratio and keypoint size ratio on the fly. Experiments are started using the '3' key. The drone flies forward, flying back when an obstacle is detected, outputting the distance to the obstacle.

|Key    | Action    |
|---    |---        |
| m | Increase test ratio by 0.025 |
| n | Decrease test ratio by 0.025 |

The avoid test enables the default threshold values. The test flies the drone forward, avoiding any detected obstacles.

|Key    | Action    |
|---    |---        |
| Home | Confirm successful avoidance |
| End | Confirm failed avoidance |
