import warnings
import cv2 as cv
import numpy as np
import os
import json
import pyefd

from pathlib import Path


def _parse_sample(img, crop=(0, 1)):
    img = img[:, int(img.shape[1] * crop[0]):int(img.shape[1] * crop[1])]  # crop
    cnt, hierarchy = cv.findContours(img, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_NONE)  # chain approx simple breaks pyefd
    return img, cnt


def _get_filenames(dir_):
    return [os.path.join(dir_, f) for f in os.walk(dir_).__next__()[-1]]


def _contour_skew(cnt):
    """
    Calculate a vector from bounding box center to a mass center of a contour
    :param cnt: opencv contour
    :return: 2d vector
    """
    M = cv.moments(cnt)
    cx = int(M['m10'] / max(M['m00'], 1))
    cy = int(M['m01'] / max(M['m00'], 1))
    x, y, w, h = cv.boundingRect(cnt)
    return cx - (x + w / 2), cy - (y + h / 2)


def _angle_divergence(alpha, beta, return_angle=True):
    """
    Calculate distance between angles
    :param alpha: angle in radians
    :param beta: andle in radians
    :param return_angle: bool if should return angle. else will leave as cosine
    :return: float either angle or cosine of the angle
    """

    def _mult_func(func):
        return func(alpha) * func(beta)

    out = _mult_func(np.sin) + _mult_func(np.cos)
    if return_angle:
        out = np.arccos(out)
    return out


def sort(cnt):
    def _cnt_xloc(c):
        M = cv.moments(c)
        cx = int(M['m10'] / max(M['m00'], 1))
        return cx

    return sorted(cnt, key=lambda x: _cnt_xloc(x))


def save_templates(data_dir, crop=(0, 1), save=False, filename=None):
    """
    Create a dataset from a directory of images, the names of which contain true labels.
    e.x. a sample img containing the '2+2=4' symbols should be named like 2+2=4.png
    :param data_dir: path to dir with gt images
    :param crop: (float, float) representing what vertical crop should be applied to images. default (0,1)
    :param save: bool. saves as json if True into ../filename
    :param filename: str. if None - Path(data_dir).name + '.json' will be used. default None
    :return: dictionary of characters and their fourier parameters of order 10
    """
    dataset = {}
    for file in _get_filenames(data_dir):
        gt = list(os.path.split(file)[-1].split('.')[0])
        img = cv.imread(file)
        img = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
        img, cnt = _parse_sample(img, crop=crop)
        for label, c in zip(gt, sort(cnt)):
            # save skew direction if flip matters
            vertical_skew = np.sign(_contour_skew(c)[1]) if label in ['2', '5', '6', '9'] else 0

            c_fourier, transforms = pyefd.elliptic_fourier_descriptors(
                c.reshape(-1, 2),
                normalize=True,
                return_transformation=True
            )
            rot = transforms[1] if label in ['1', '-'] else 0
            dataset[label] = {'cnt': c_fourier.tolist(), 'skew': vertical_skew, 'rot': rot}
    if save:
        if filename is None:
            json_path = Path('../') / (Path(data_dir).name + '.json')
        else:
            json_path = Path('../') / filename
        with open(json_path, 'w') as f:
            json.dump(dataset, f)
    return dataset


def match(inp, db, crop=(0, 1), score_thresh=1.2E-4, verbose=0):
    """
    Main ocr func
    The dataset defines all the supported characters.
    Currently, it is [0-9] and -+

    :param inp: a thresholded binary image
    :param db: characters contour database. Can be obtained with save_templates function or loaded from json
    :param score_thresh: float. Score threshold to even consider contours a character. Default value works fine `
    :param crop: (float, float) representing what vertical crop should be applied to images. default (0,1)
    :param verbose: int. the higher, the more logging

    :return: string of characters read
    """
    if not db:
        warnings.warn('Attempted to match with an empty database.', RuntimeWarning)
        return ''
    img, cnt = _parse_sample(inp, crop=crop)
    out = ''
    for c in sort(cnt):  # read left to right
        if c.size < 4:  # if one point
            continue
        chars, scores = [], []
        c_skew = np.sign(_contour_skew(c)[1])
        c_fourier, transforms = pyefd.elliptic_fourier_descriptors(
            c.reshape(-1, 2),
            normalize=True,
            return_transformation=True
        )
        for ch in db.keys():
            if db[ch]['skew'] != 0 and c_skew != db[ch]['skew']:  # skip if skew matters and does not match
                continue

            chars.append(ch)

            score = np.mean(np.square(c_fourier - db[ch]['cnt']))
            angular_score = _angle_divergence(db[ch]['rot'], transforms[1]) ** 2 if db[ch]['rot'] != 0 else 0

            scores.append(score + angular_score)
        min_id = np.argmin(scores)
        if verbose > 0:
            print(dict(zip(chars, scores)))
        if scores[min_id] < score_thresh:
            out += chars[min_id]
    return out
