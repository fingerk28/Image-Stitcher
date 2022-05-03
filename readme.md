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
python stitching.py --dir [IMAGE_DIRECTORY] --row [NUMBER_OF_ROWS] --rec --ratio [WIDTH_HEIGHT_RATIO]
```

> `[IMAGE_DIRECTORY]`: image folder name
>
> `--row [NUMBER_OF_ROWS]`: This is optional. If `--row [NUMBER_OF_ROWS]` is given, the different algorithm will be triggered
>
> `--rec`: This is optional. If `--rec` is given, the image will be rectified
>
> `--ratio [WIDTH_HEIGHT_RATIO]`: This is optional. If `--rec` is given, you can determine to input `--ratio [WIDTH_HEIGHT_RATIO]` or not. If `--ratio [WIDTH_HEIGHT_RATIO]` is given, the image will be rectified with respect to the ratio



