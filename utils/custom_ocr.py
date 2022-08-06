import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt
import os
import json


def _parse_sample(img, crop=True):
    # TODO maybe blur for robustness?
    if crop:
        img = img[:, img.shape[1] * 3 // 32:img.shape[1] * 9 // 16]  # crop
    cnt, hierarchy = cv.findContours(img, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
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
    cx = int(M['m10'] / M['m00'])
    cy = int(M['m01'] / M['m00'])
    x, y, w, h = cv.boundingRect(cnt)
    return cx - (x + w / 2), cy - (y + h / 2)


def save_templates(data_dir):
    dataset = {}
    for file in _get_filenames(data_dir):
        gt = list(os.path.split(file)[-1].split('.')[0])
        img = cv.imread(file)
        img = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
        img, cnt = _parse_sample(img)
        for label, c in zip(gt, sort(cnt)):
            # save skew direction if flip matters
            vertical_skew = np.sign(_contour_skew(c)[1]) if label in ['2', '5', '6', '9'] else 0
            dataset[label] = {'cnt': c.reshape(-1, 2).tolist(), 'skew': vertical_skew}
    with open('../dataset.json', 'w') as f:
        json.dump(dataset, f)
    return dataset


def match(inp, db, crop=False, score_thresh=0.15):
    """
    Main ocr func
    :param score_thresh: float. Score threshold to even consider contours a character
    :param crop: True if an inbuilt cropping should be used
    :param inp: a thresholded binary image
    :param db: characters contour database. Can be obtained with save_templates function or loaded from json
    :return: string of characters read
    """
    img, cnt = _parse_sample(inp, crop=crop)
    out = ''
    for c in sort(cnt):  # read left to right
        chars, scores = [], []
        c_skew = np.sign(_contour_skew(c)[1])
        for ch in db.keys():
            if db[ch]['skew'] != 0 and c_skew != db[ch]['skew']:  # skip if skew matters and does not match
                continue
            chars.append(ch)
            scores.append(cv.matchShapes(c, np.array(db[ch]['cnt']).reshape((-1, 1, 2)), cv.CONTOURS_MATCH_I1, 0.0))
        min_id = np.argmin(scores)
        print(dict(zip(chars, scores)))
        if scores[min_id] < score_thresh:
            out += chars[min_id]
    return out


def sort(cnt):
    def _cnt_xloc(c):
        M = cv.moments(c)
        cx = int(M['m10'] / M['m00'])
        return cx

    return sorted(cnt, key=lambda x: _cnt_xloc(x))


database = save_templates('../digits')
for file in _get_filenames('../digits/test'):
    gt = list(os.path.split(file)[-1].split('.')[0])
    img = cv.imread(file)
    img = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    print(''.join(gt), match(img,database, score_thresh=0.15))
