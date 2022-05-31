# Image Stitcher

> *Author: Alex Huang*



## Environment Preparation

```shell
conda create --name stitcher python=3.6.9
conda activate stitcher
python -m pip install opencv-python==4.2.0.34 
python -m pip install opencv-contrib-python==3.4.2.17
python -m pip install numpy
```



## Guideline for Taking Pictures by iPhone

1. Choose **2x mode** to take picture

    <img src="C:\Users\Alex Huang\OneDrive - McKinsey & Company\Desktop\pic.png" alt="Picture1" style="zoom: 20%;" />

2. Keep an appropriate distance between your iPhone and PCB and allow the text on IC can be clearly identified and FOV is as large as possible.

3. Please take pictures **from left to right**.

4. If the PCB is too large and can not be capture in one row, please take pictures in **Z shape**. (from left to right in the first row, from right to left in the second row)

4. The overlapping between photos should be in **20~30%**.



## Usage

```shell
python stitching.py --dir [IMAGE_DIRECTORY] --row [NUMBER_OF_ROWS] --result_dir [RESULT_DIR]

Example:
python stitching.py --dir ./test_images/
python stitching.py --dir ./test_images/ --row 2
python stitching.py --dir ./test_images/ --result_dir result_directory
```

> `--dir [IMAGE_DIRECTORY]` &rarr; image folder name.
>
> `--row [NUMBER_OF_ROWS]` &rarr; This is optional. If the PCB is too large such that you need to take pictures in many rows, you can choose whether to give this argument. In default setting, you don't need to input this argument. If `--row [NUMBER_OF_ROWS]` is given, the different algorithm will be triggered. The stitcher will stitch images for each row first (horizontally), and then stitch them vertically.
>
> `--result_dir [RESULT_DIR]` &rarr; This is optional. **The stitched result and log file will be save in `[IMAGE_DIRECTORY]/[RESULT_DIR]/`**. Default value is `Result`
>
> ---
>
> The status return by `stitcher.fit()`
>
> * 0 &rarr; successfully
> * -1 &rarr; fail
>
> The stitching process contains randomness. If the `stitcher.fit()` return -1 in the first trial, you can try again and it may success.
>
> ---
>
> As a reference, the execution time is as follows (image resolution is **3024x4032**)
>
> * 4 images &rarr; **4~5s**
>
> * 6 images &rarr; **5~6s**
>
> * 8 images &rarr; **8~10s**
>
> However, the running time depends on the difficulty of stitching.
>
> 1. Don't use the **2x mode** to take photos
> 2. The distance between iPhone and PCB is too close (FOV is too small)
> 3. The overlapping is too small or too large
>
> If the above conditions happen, the execution time may be as follows
>
> * 8 images &rarr; **35s**
> * 10 images &rarr; **60s** 

