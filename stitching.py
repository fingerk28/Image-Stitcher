import logging
import cv2
import numpy as np
from glob import glob
import os
import time
import argparse
import logging
import datetime


parser = argparse.ArgumentParser()
parser.add_argument('--dir', type=str)
parser.add_argument('--row', type=int, default=1)
parser.add_argument('--col', type=int)
parser.add_argument('--result_dir', type=str, default='Result')
args = parser.parse_args()


class Stitcher():
    def __init__(self, args,):
        self.save_dir = self.prepare_result_dir(args)
        self.logger = self.prepare_logger()
        self.logger.info('[INITIALIZING]')
        self.args = args
        self.elapsed_time = []
        self.stitcher = cv2.createStitcher(False)
        self.set_divider()
        self.logger.info(self.long_divider)

    def set_divider(self):
        self.long_divider = '--------------------------------------------------'
        self.short_divider = '----------------------'

    def prepare_result_dir(self, args):
        save_dir = os.path.join(args.dir, args.result_dir)
        os.makedirs(save_dir, exist_ok=True)
        
        return save_dir

    def prepare_logger(self):
        logger = logging.getLogger()
        logger.setLevel(logging.DEBUG)
        formatter = logging.Formatter(
            '[%(levelname)1.1s %(asctime)s %(module)s:%(lineno)d] %(message)s',
            datefmt='%Y%m%d %H:%M:%S')

        ch = logging.StreamHandler()
        ch.setLevel(logging.DEBUG)
        ch.setFormatter(formatter)
        
        log_filename = datetime.datetime.now().strftime(os.path.join(self.save_dir, "%Y-%m-%d_%H_%M_%S.log"))
        fh = logging.FileHandler(log_filename)
        fh.setLevel(logging.DEBUG)
        fh.setFormatter(formatter)

        logger.addHandler(ch)
        logger.addHandler(fh)

        return logger

    def prepare_image_paths(self):
        self.logger.info('[PREPARING IMAGE PATHS]')
        self.logger.info(self.short_divider)
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

        self.logger.info('Number of images: {}'.format(self.args.row * self.args.col))
        self.logger.info('Number of rows / cols: {} / {}'.format(self.args.row, self.args.col))
        self.logger.info(self.short_divider)
        self.logger.info('Image paths:')
        for paths in img_paths:
            self.logger.info(paths)

        self.logger.info(self.long_divider)
        return img_paths

    def read_images(self):
        self.logger.info('[READING IMAGES]')
        images = []
        for i in range(self.args.row):
            images_one_row = [cv2.imread(self.image_paths[i][j]) for j in range(self.args.col)]
            images.append(images_one_row)
        self.logger.info(self.long_divider)

        return images

    def cv2_stitch(self, images):
        self.log_time(flag='start')
        (status, pano) = self.stitcher.stitch(images)
        self.log_time(flag='end')
        self.log_elapsed_time()

        if status == 1:
            self.logger.info('ERR_NEED_MORE_IMGS')
        elif status == 2:
            self.logger.info('ERR_HOMOGRAPHY_EST_FAIL')
        elif status == 3:
            self.logger.info('ERR_CAMERA_PARAMS_ADJUST_FAIL')
        
        return status, pano

    def stitch(self, images):
        self.logger.info('[STARTING TO STITCH IMAGES HORIZONTALLY]')
        stitched_images = []
        for i in range(self.args.row):
            self.logger.info(self.short_divider)
            self.logger.info('Stitching row {} images...'.format(i))
            status, result = self.cv2_stitch(images[i])
            if status != 0:
                return -1, None
            stitched_images.append(result)
        self.logger.info(self.long_divider)

        if self.args.row > 1:
            self.save_per_row_image(stitched_images)
            self.logger.info('[STARTING TO STITCH IMAGES VERTICALLY]')
            stitched_images = [np.rot90(image) for image in stitched_images]
            stitched_images = self.cv2_stitch(stitched_images)
            stitched_images = np.rot90(stitched_images, k=-1)
        else:
            self.logger.info('[NOTES] no need to stitch images vertically')
            stitched_images = stitched_images[0]
        self.logger.info(self.long_divider)

        return status, stitched_images

    def log_time(self, flag):
        if flag == 'start':
            self.start_time = time.time()
        else:
            self.end_time = time.time()
    
    def log_elapsed_time(self):
        elapsed_time = self.end_time - self.start_time
        self.elapsed_time.append(elapsed_time)
        self.logger.info('Elapsed time: {:.3f}'.format(elapsed_time))

        return elapsed_time

    def log_avg_elapsed_time(self):
        self.logger.info('Average Elapsed time: {:.3f}'.format(np.mean(self.elapsed_time)))
        self.logger.info(self.long_divider)

    def save_stitched_image(self, stitched_image):
        cv2.imwrite(os.path.join(self.save_dir, 'result.png'), stitched_image)
        self.logger.info('Save stitched image...')
        self.logger.info(self.long_divider)
    
    def save_rectified_image(self, image):
        cv2.imwrite(os.path.join(self.save_dir, 'rectified_result.png'), image)
        self.logger.info('Save rectified image...')
        self.logger.info(self.long_divider)

    def save_per_row_image(self, images):
        for i in range(len(images)):
            cv2.imwrite(os.path.join(self.save_dir, 'row{}_result.png'.format(i)), images[i])
        self.logger.info('Save per row image...')
        self.logger.info(self.long_divider)

    def rectify(self, image):
        gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        gray_image = cv2.GaussianBlur(gray_image, (5, 5), 0)

        _, thresh = cv2.threshold(gray_image, 127, 255, 0)
        _, contours, _ = cv2.findContours(thresh, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)

        contours = sorted(contours, key=lambda c: cv2.contourArea(c), reverse=True)
        max_contour = contours[1]
        self.logger.info('Max area: {}'.format(cv2.contourArea(max_contour)))

        x, y, w, h = cv2.boundingRect(max_contour)

        return image[y: y+h, x: x+w]

    def fit(self):
        self.image_paths = self.prepare_image_paths()
        self.images = self.read_images()
        status, stitched_image = self.stitch(self.images)
        if status == 0:
            self.save_stitched_image(stitched_image)
            self.log_avg_elapsed_time()
            rectified_image = self.rectify(stitched_image)
            self.save_rectified_image(rectified_image)
            self.logger.info('Stitch images successfully')
        else:
            self.logger.info('Stitch images fail')

        return status


def main():
    
    stitcher = Stitcher(args)
    status = stitcher.fit()


if __name__ == '__main__':
    main()

    
