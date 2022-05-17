import cv2
import numpy as np
from glob import glob
import os
import time
import argparse


parser = argparse.ArgumentParser()
parser.add_argument('--dir', type=str)
parser.add_argument('--row', type=int, default=1)
parser.add_argument('--col', type=int)
args = parser.parse_args()


class Stitcher():
    def __init__(self, args,) :
        print('-----------------------------------')
        print('[INITIALIZING]')
        self.args = args
        self.elapsed_time = []
        self.stitcher = cv2.createStitcher(False)
        print('-----------------------------------')

    def prepare_image_paths(self):
        print('[PREPARING IMAGE PATHS]')
        print('------------------')
        source_paths = [path for path in glob(os.path.join(self.args.dir, '*')) if os.path.isfile(path)]
        source_paths = sorted(source_paths, key=lambda s: os.path.splitext(os.path.basename(s))[0])

        if self.args.col is None:
            self.args.col = (len(source_paths) // self.args.row)

        img_paths = []
        for i in range(self.args.row):
            if i % 2 == 1:
                img_paths.append(source_paths[i*args.col: (i+1)*args.col][::-1])
            else:
                img_paths.append(source_paths[i*args.col: (i+1)*args.col])

        print('Number of images: {}'.format(self.args.row * self.args.col))
        print('Number of rows / cols: {} / {}'.format(self.args.row, self.args.col))
        print('------------------')
        print('Image paths:')
        for paths in img_paths:
            print(paths)

        print('-----------------------------------')
        return img_paths

    def read_images(self):
        print('[READING IMAGES]')
        images = []
        for i in range(self.args.row):
            images_one_row = [cv2.imread(self.image_paths[i][j]) for j in range(self.args.col)]
            images.append(images_one_row)
        print('-----------------------------------')

        return images

    def cv2_stitch(self, images):
        self.log_time(flag='start')
        (_, pano) = self.stitcher.stitch(images)
        self.log_time(flag='end')
        self.print_elapsed_time()
        return pano

    def stitch(self, images):
        print('[STARTING TO STITCH IMAGES HORIZONTALLY]')
        stitched_images = []
        for i in range(self.args.row):
            print('------------------')
            print('Stitching row {} images...'.format(i))
            stitched_images.append(self.cv2_stitch(images[i]))
        print('-----------------------------------')

        if self.args.row > 1:
            self.save_per_row_image(stitched_images)
            print('[STARTING TO STITCH IMAGES VERTICALLY]')
            stitched_images = [np.rot90(image) for image in stitched_images]
            stitched_images = self.cv2_stitch(stitched_images)
            stitched_images = np.rot90(stitched_images, k=-1)
        else:
            print('[NOTES] no need to stitch images vertically')
            stitched_images = stitched_images[0]
        print('-----------------------------------')

        return stitched_images

    def log_time(self, flag):
        if flag == 'start':
            self.start_time = time.time()
        else:
            self.end_time = time.time()
    
    def print_elapsed_time(self):
        elapsed_time = self.end_time - self.start_time
        self.elapsed_time.append(elapsed_time)
        print('Elapsed time: {:.3f}'.format(elapsed_time))

        return elapsed_time

    def print_avg_elapsed_time(self):
        print('Average Elapsed time: {:.3f}'.format(np.mean(self.elapsed_time)))
        print('-----------------------------------')

    def save_stitched_image(self, stitched_image):
        save_dir = os.path.join(self.args.dir, 'Result')
        os.makedirs(save_dir, exist_ok=True)
        cv2.imwrite(os.path.join(save_dir, 'result.png'), stitched_image)
        print('Save stitched image...')
        print('-----------------------------------')
    
    def save_rectified_image(self, image):
        save_dir = os.path.join(self.args.dir, 'Result')
        os.makedirs(save_dir, exist_ok=True)
        cv2.imwrite(os.path.join(save_dir, 'rectified_result.png'), image)
        print('Save rectified image...')
        print('-----------------------------------')

    def save_per_row_image(self, images):
        save_dir = os.path.join(self.args.dir, 'Result')
        os.makedirs(save_dir, exist_ok=True)
        for i in range(len(images)):
            cv2.imwrite(os.path.join(save_dir, 'row{}_result.png'.format(i)), images[i])
        print('Save per row image...')
        print('-----------------------------------')

    def rectify(self, image):
        gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        gray_image = cv2.GaussianBlur(gray_image, (5, 5), 0)

        _, thresh = cv2.threshold(gray_image, 127, 255, 0)
        _, contours, _ = cv2.findContours(thresh, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)

        contours = sorted(contours, key=lambda c: cv2.contourArea(c), reverse=True)
        max_contour = contours[1]
        print('Max area: {}'.format(cv2.contourArea(max_contour)))

        x, y, w, h = cv2.boundingRect(max_contour)

        return image[y: y+h, x: x+w]

    def fit(self):
        self.image_paths = self.prepare_image_paths()
        self.images = self.read_images()
        stitched_image = self.stitch(self.images)
        self.save_stitched_image(stitched_image)
        self.print_avg_elapsed_time()
        rectified_image = self.rectify(stitched_image)
        self.save_rectified_image(rectified_image)


def main():
    
    stitcher = Stitcher(args)
    stitcher.fit()


if __name__ == '__main__':
    try: 
        main()

    except:
        print ("Something wrong...")
    
