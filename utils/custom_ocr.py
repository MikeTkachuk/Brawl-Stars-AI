import warnings
import cv2 as cv
import numpy as np
import os
import json
import pyefd

from pathlib import Path

BASE_THRESH = 1.2E-4


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
        gt = os.path.split(file)[-1].split('.')[0]
        gt = gt.split('_')[0]  # ignore comments after _
        gt = gt.replace(' ', '')
        gt = list(gt)
        img = cv.imread(file)
        img = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
        img, cnt = _parse_sample(img, crop=crop)
        for label, c in zip(gt, sort(cnt)):
            # save skew direction if flip matters
            vertical_skew = np.sign(_contour_skew(c)[1]) if label in ['2', '5', '6', '9', 'n', 'u'] else 0

            c_fourier, transforms = pyefd.elliptic_fourier_descriptors(
                c.reshape(-1, 2),
                normalize=True,
                return_transformation=True
            )
            rot = transforms[1] if label in ['1', '-'] else 0

            if label not in dataset:
                dataset[label] = {'cnt': [c_fourier.tolist()], 'skew': [vertical_skew], 'rot': [rot]}
            else:
                similarity = [_match_score(c_fourier, dataset[label]['cnt'][i],
                                           rot if rot != 0 else None, dataset[label]['rot'][i])
                              for i in range(len(dataset[label]['cnt']))]
                if all(np.greater(similarity, BASE_THRESH)):
                    dataset[label]['cnt'].append(c_fourier.tolist())
                    dataset[label]['skew'].append(vertical_skew)
                    dataset[label]['rot'].append(rot)

    if save:
        if filename is None:
            json_path = Path('../') / (Path(data_dir).name + '.json')
        else:
            json_path = Path('../') / filename
        with open(json_path, 'w') as f:
            json.dump(dataset, f)
    return dataset


def _match_score(cf1, cf2, rot1=None, rot2=None):
    score = np.mean(np.square(cf1 - cf2))
    if rot1 is not None and rot2 is not None:
        angular_score = _angle_divergence(rot1, rot2) ** 2
        score += angular_score
    return score


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
            for i in range(len(db[ch]['cnt'])):
                if db[ch]['skew'][i] != 0 and c_skew not in [0.0, db[ch]['skew'][
                    i]]:  # skip if skew matters and does not match
                    continue

                chars.append(ch)
                score = _match_score(c_fourier,
                                     db[ch]['cnt'][i],
                                     db[ch]['rot'][i] if db[ch]['rot'][i] != 0 else None,
                                     transforms[1])

                scores.append(score)
        min_id = np.argmin(scores)
        if verbose > 0:
            print(dict(zip(chars, scores)))
        if scores[min_id] < score_thresh:
            out += chars[min_id]
    return out


def learned_comparison(x1, x2):
    diff = np.sqrt(1000) * (x1 - x2)
    diff = diff.reshape(-1)
    weights = np.array([-1.0000e+00, -1.0000e+00, -1.0000e+00, -4.3289e+00,
                        -5.0549e-01, -5.0598e-01, -4.3815e-01, -4.4509e-01,
                        -4.3521e-01, -4.8219e-01, -5.9286e-01, -7.8976e-01,
                        -3.0901e-02, -1.4133e-01, -5.3326e-02, -1.9868e-01,
                        -5.8596e-02, -7.4078e-02, 4.7613e-02, -6.2624e-02,
                        -1.3891e-01, -1.2226e-01, -3.3368e-03, 4.4394e-02,
                        1.1403e-01, 2.6991e-02, 1.2110e-01, 1.0116e-01,
                        1.5295e-01, 1.0566e-01, 2.8567e-01, 2.7430e-01,
                        2.6891e-01, 3.0749e-01, 5.1849e-01, 5.2175e-01,
                        4.1152e-01, 3.7644e-01, 5.7100e-01, 5.7538e-01])
    bias = 2.1269

    match_ = np.mean(diff * weights) + bias
    return match_ > 0
