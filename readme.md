# Image Stitcher

> *Author: Alex Huang*



## Environment Preparation

```shell
conda create --name stitcher python=3.6.9

conda install opencv=3.4.2
conda install opencv-contrib-python=3.4.2
conda install numpy
```



## Usage

```shell
python stitching.py --dir [IMAGE_DIRECTORY] --row [NUMBER_OF_ROWS] --rec --ratio [WIDTH_HEIGHT_RATIO]
```

> [IMAGE_DIRECTORY]: image folder name
>
> [NUMBER_OF_ROWS]: This is optional. If this argument is given, the different algorithm will be triggered
>
> --rec: This is optional. If this argument is given, the image will be rectified
>
> --ratio [WIDTH_HEIGHT_RATIO]: This is optional. If the ratio is given, the image will be rectified with respect to the ratio



