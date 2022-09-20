from environment import _ocr_preproc, _relative_to_pixel
from custom_ocr import save_templates, match
import cv2 as cv
import numpy as np
import config
import os


def add_region_template(full_screen: np.ndarray, region, image_path, thresh=(150, 255), erosion=7):
    """
    An util function to add new ocr templates into the dataset. It uses config to define main screen position.
    :param full_screen: A screen captured from (0,0) to whatever size you need in absolute coordinates
    :param region: Please insert regions that were defined in config only (or in a similar relative way)
    :param image_path: A full path to the image being saved
    :param thresh: tuple (lower, upper) bounds for pixel value thresholding
    :param erosion: positive int, the kernel size to use when doing erosion
    :return:
    """
    slicer = _relative_to_pixel(region, config.main_screen, absolute=True)
    img = _ocr_preproc(full_screen, slicer, thresh=thresh, erosion=erosion)
    if not os.path.exists(os.path.split(image_path)[0]):
        os.makedirs(os.path.split(image_path)[0])
    cv.imwrite(image_path, img)


if __name__ == '__main__':
    screen = cv.imread(r"C:\Users\Mykhailo_Tkachuk\PycharmProjects\Brawl-Stars-AI\defeated.png")
    screen = cv.cvtColor(screen, cv.COLOR_BGR2RGB)
    add_region_template(screen,
                        config.defeated_region,
                        r"C:\Users\Mykhailo_Tkachuk\PycharmProjects\Brawl-Stars-AI\ocr_data\defeated\Defeated.png",
                        )

    screen = cv.imread(r"C:\Users\Mykhailo_Tkachuk\PycharmProjects\Brawl-Stars-AI\proceed.png")
    screen = cv.cvtColor(screen, cv.COLOR_BGR2RGB)
    add_region_template(screen,
                        config.proceed_region,
                        r"C:\Users\Mykhailo_Tkachuk\PycharmProjects\Brawl-Stars-AI\ocr_data\proceed\proceed.png",
                        )