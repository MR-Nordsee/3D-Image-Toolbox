# 3D Image Toolbox

A Python project for converting images and videos into 3D Side-by-Side (SBS) formats.

- It takes Spacial Photos from iPhone in HEIC format and exports Depth map.
- It takes Normal Images and Videos and creates a Depth map using Depth Anything V2.
- It combines Images and Videos to SideBySide files.
- It takes all Images from the Input folder and decides what to do. 

Made and tested for macOS.

**Needs Python 3.12**

## Installation

```bash
git clone https://github.com/yourusername/3d-image-toolbox.git
cd 3d-image-toolbox
sh setup.sh
```

The setup.sh creates python environments, installs the modules and downloads files from Depth Anything.

## Usage

```bash
source "./tools/bin/activate"
python3 convert.py
```

Place files in Input Folder.
Supported are jpg, jpeg, png, mov, mp4

Generating of Videos takes a lot of time and ressources. Have that in mind.

If it gets to long you can switch the Depth Anything model to a smaller model.

## View files

The files have fullsbs in the filename so Skybox VR automatic switch on the FullSBS Mode.

## Dependencies & Referenced Projects

- [OpenCV](https://github.com/opencv/opencv)
- [NumPy](hhttps://github.com/numpy/numpy)
- [Pillow_heif](https://github.com/bigcat88/pillow_heif)
- [Pillow](https://github.com/python-pillow/Pillow)
- [Depth Anything V2](https://github.com/DepthAnything/Depth-Anything-V2)
- [Skybox VR](https://skybox.xyz/)

## License

This project is licensed under the MIT License.