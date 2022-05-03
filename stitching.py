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
parser.add_argument('--ratio', type=float, default=None)
parser.add_argument('--rec', action='store_true', default=False)
args = parser.parse_args()


class Image_Stitching():
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
        img_paths = [path for path in glob(os.path.join(self.args.dir, '*')) if os.path.isfile(path)]
        img_paths = sorted(img_paths, key=lambda s: os.path.splitext(os.path.basename(s))[0])
        self.args.col = len(img_paths) if self.args.row == 1 else (len(img_paths) // self.args.row)
        img_paths = img_paths[:self.args.row * self.args.col]
        for i in range(self.args.row):
            if i % 2 == 1:
                img_paths.insert(0, img_paths[i+i*args.col: i+(i+1)*args.col][::-1])
            else:
                img_paths.insert(0, img_paths[i+i*args.col: i+(i+1)*args.col])
        img_paths = img_paths[:args.row][::-1]

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
            images_one_row = []
            for j in range(self.args.col):
                images_one_row.append(cv2.imread(self.image_paths[i][j]))
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
        if not self.args.rec:
            return None
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

    def get_clicked_corners(self, stitched_image):
        def on_mouse(event, x, y, flags, param):
            if event == cv2.EVENT_LBUTTONDOWN:
                param[0].append((x, y))
        
        points_add = []
        WINDOW_NAME = 'CLICK CORNERS'
        cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL)
        cv2.setMouseCallback(WINDOW_NAME, on_mouse, [points_add])
        while True:
            img_ = stitched_image.copy()
            for p in points_add:
                cv2.circle(img_, tuple(p), 40, (255, 0, 0), -1)
            cv2.imshow(WINDOW_NAME, img_)

            key = cv2.waitKey(20) % 0xFF
            if key == 27 or len(points_add) == 4: 
                time.sleep(0.5)
                break
        cv2.destroyAllWindows()

        return points_add

    def cal_rectified_HW(self, clicked_points):
        H = np.linalg.norm(np.array(clicked_points[3]) - np.array(clicked_points[0]))
        W = np.linalg.norm(np.array(clicked_points[1]) - np.array(clicked_points[0]))

        return H, W

    def rectify(self, stitched_image):
        if not self.args.rec:
            return None

        print('[RECTIFY STITCHED RESULT]')
        clicked_points = self.get_clicked_corners(stitched_image)
        H, W = self.cal_rectified_HW(clicked_points)

        if not self.args.ratio:
            self.args.ratio = W / H

        W = H * self.args.ratio

        rectified_points = np.float32([(0, 0), (W, 0), (W, H), (0, H)])
        H_matrix, status = cv2.findHomography(np.float32(clicked_points), rectified_points, cv2.RANSAC, 0.5)
        rectified_image = cv2.warpPerspective(stitched_image, H_matrix, (int(W), int(H)))
        
        return rectified_image

    def fit(self):
        self.image_paths = self.prepare_image_paths()
        self.images = self.read_images()
        stitched_image = self.stitch(self.images)
        self.save_stitched_image(stitched_image)
        self.print_avg_elapsed_time()
        rectified_image = self.rectify(stitched_image)
        self.save_rectified_image(rectified_image)


def main():
    
    stitcher = Image_Stitching(args)
    stitcher.fit()
    exit()


if __name__ == '__main__':
    try: 
        main()

    except IndexError:
        print ("Something wrong...")
    
