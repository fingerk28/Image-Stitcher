# Image Stitcher

> *Author: Alex Huang*



## Environment Preparation

```shell
conda create --name stitcher python=3.6.9
conda activate stitcher
conda install opencv=3.4.2
conda install opencv-contrib-python=3.4.2
conda install numpy
```



## Usage

```shell
python stitching.py --dir [IMAGE_DIRECTORY] --row [NUMBER_OF_ROWS]

Example:
python stitching.py --dir ./test_images/
python stitching.py --dir ./test_images/ --row 2
```

> `--dir [IMAGE_DIRECTORY]` &rarr; image folder name.
>
> `--row [NUMBER_OF_ROWS]` &rarr; This is optional. If `--row [NUMBER_OF_ROWS]` is given, the different algorithm will be triggered. The stitcher will stitch images for each row first, and then stitch them vertically.
>
> **The stitched result will be save in `[IMAGE_DIRECTORY]/Result/`**



## Guideline for Taking Pictures by iPhone

1. Choose **2x mode** to take picture

    <img src="https://github.com/fingerk28/Image-Stitcher/blob/master/pic.png" alt="Picture1" style="zoom: 20%;" />

2. Keep an appropriate distance between your iPhone and PCB and allow the text on IC can be clearly capture and FOV is as large as possible.
3. If the PCB is too large and can not be capture in one row, take pictures in **Z shape**.
4. The overlapping between photos should be in **20~30%**.
