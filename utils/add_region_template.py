import time

from environment import _ocr_preproc, _relative_to_pixel
from utils.custom_ocr import save_templates, match
import cv2 as cv
import numpy as np
import config
import os

from utils.grabscreen import grab_screen


def add_region_template(full_screen: np.ndarray, region, image_path, thresh=(140, 255), erosion=7):
    """
    An util function to add new ocr templates into the dataset. It uses config to define main screen position.
    :param full_screen: Main screen captured as defined in config.main_screen
    :param region: Please insert regions that were defined in config only (or in a similar relative way)
    :param image_path: A full path to the image being saved
    :param thresh: tuple (lower, upper) bounds for pixel value thresholding
    :param erosion: positive int, the kernel size to use when doing erosion
    :return:
    """
    slicer = _relative_to_pixel(region, config.main_screen)
    img = _ocr_preproc(full_screen, slicer, thresh=thresh, erosion=erosion)
    if not os.path.exists(os.path.split(image_path)[0]):
        os.makedirs(os.path.split(image_path)[0])
    cv.imwrite(image_path, img)


if __name__ == '__main__':
    # time.sleep(1)
    screen = grab_screen(config.main_screen)
    add_region_template(screen,
                        config.end_screen_title_region,
                        r"C:\Users\Mykhailo_Tkachuk\PycharmProjects\Brawl-Stars-AI\ocr_data\end_title\VICTORY.png",
                        )
