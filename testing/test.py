from pathlib import Path

from tqdm import tqdm

from utils.grabscreen import grab_screen
import matplotlib.pyplot as plt
import cv2 as cv
import time
import numpy as np
from utils.custom_ocr import match, save_templates
from environment import _ocr_preproc, ScreenParser
import torch
import pyefd
from utils.custom_ocr import learned_comparison


def generate_contour():
    cf = pyefd.reconstruct_contour(np.random.normal(0, 1, size=(10, 4)))
    cf -= np.min(cf, axis=0)
    cf *= 64 / np.max(cf, axis=0)
    cf = cf.astype(np.int32)
    img = cv.fillPoly(np.zeros((64, 64)), [cf], 255).astype(np.uint8)
    img = cv.dilate(img, np.ones((3, 3)))

    cnts, _ = cv.findContours(img, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_NONE)
    assert len(cnts) == 1
    return cnts[0].reshape(-1, 2)


def augment_contour(cnt, angle=None, distortion_strength=1.0):
    angle = np.random.uniform(-np.pi, np.pi) if angle is None else angle
    rot_mat = np.array(
        [
            [np.cos(angle), -np.sin(angle)],
            [np.sin(angle), np.cos(angle)]
        ]
    )
    xv, yv = np.meshgrid(np.arange(64), np.arange(64))

    def _random_wave(amplitude=1, frequency=1):
        vec = np.random.normal(0, 1, size=2)
        vec /= np.linalg.norm(vec)
        return amplitude * np.sin(frequency * (vec[0] * xv + vec[1] * yv))

    amplitudes = 1 / (1 + np.arange(16))
    frequencies = (1 + np.arange(16)) / 64
    distortion_map = sum([_random_wave(a, f) for a, f in zip(amplitudes, frequencies)])
    distortion_map -= np.min(distortion_map)
    distortion_map = distortion_map * (1.2 - 0.8) / np.max(distortion_map) + 0.8
    cnt_mean = cnt.mean(axis=0)
    cnt_distorted = (cnt - cnt_mean) * distortion_map[cnt.astype(np.int32)[:, 1], cnt.astype(np.int32)[:, 0]][..., None]
    cnt_distorted += cnt_mean
    cnt_distorted = cnt * (1 - distortion_strength) + distortion_strength * cnt_distorted

    cnt_rotated = cnt_distorted @ rot_mat.T
    # cnt_rotated -= np.min(cnt_rotated, axis=0)
    # cnt_rotated *= 64 / np.max(cnt_rotated, axis=0)
    # cnt_rotated = cnt_rotated.astype(np.int32)

    scale = 0.33 + np.exp(np.random.uniform(-3, 2))
    cnt_scaled = cnt_rotated * scale

    return cnt_scaled, (angle, scale, distortion_map)


def sample_batch(batch_size=16, strength=1.0):
    x1, x2, y = [], [], []
    for i in range(batch_size // 2):
        c = generate_contour()
        x1.append(torch.tensor(pyefd.elliptic_fourier_descriptors(c, normalize=True)))
        x2.append(torch.tensor(pyefd.elliptic_fourier_descriptors(
            augment_contour(c, distortion_strength=strength)[0],
            normalize=True)))
        y.append(1.0)

    for i in range(batch_size // 2):
        x1.append(torch.tensor(pyefd.elliptic_fourier_descriptors(generate_contour(), normalize=True)))
        x2.append(torch.tensor(pyefd.elliptic_fourier_descriptors(generate_contour(), normalize=True)))
        y.append(0.0)
    return torch.stack(x1, dim=0).float(), torch.stack(x2, dim=0).float(), torch.tensor(y)


class Model(torch.nn.Module):
    def __init__(self):
        super(Model, self).__init__()
        self.weights = torch.nn.Parameter(-torch.ones((40,)))

        self.bias = torch.nn.Parameter(torch.tensor(2.4))

    def forward(self, x):
        x = x.view(x.shape[0], -1)
        return (self.weights * x ** 2).mean(dim=-1) + self.bias


if __name__ == "__main__":
    # m = Model()
    # m.train()
    # opt = torch.optim.Adam(m.parameters(), lr=0.001)
    # loss_running = 0.0
    # for i in range(10000):
    #     opt.zero_grad()
    #     x1, x2, y = sample_batch(64, 0.91)
    #     logits = m(np.sqrt(1000)*(x1-x2))
    #     loss = torch.nn.functional.binary_cross_entropy_with_logits(logits[:, None], y[:, None])
    #     loss.backward()
    #     loss_running = loss_running * 0.99 + 0.01 * loss.item()
    #     if i % 50 == 0:
    #         print(loss_running, m.bias)
    #         print(m.weights)
    #         print(sum((torch.sigmoid(logits)>0.5).float() == y))
    #     opt.step()
    #
    # exit()
    train, train_labels = [], []
    for i in tqdm(range(100)):
        c = generate_contour()
        for k in range(8):
            train.append(augment_contour(c, angle=None, distortion_strength=0.91)[0])
            train_labels.append(i)

    train_labels = np.array(train_labels)
    pair_mat = train_labels[None, :] == train_labels[:, None]

    descriptors = [pyefd.elliptic_fourier_descriptors(c, normalize=True) for c in train]
    descriptors = np.array(descriptors).reshape(len(descriptors), -1)

    pair_dst = np.square(descriptors[None, ...] - descriptors[:, None, ...]).mean(axis=-1)
    from sklearn.metrics import roc_curve

    fp, tp, th = roc_curve(pair_mat.flatten(), -pair_dst.flatten())
    print([i for i in np.stack([fp, tp, th], axis=1)[:]])
    plt.plot(*roc_curve(pair_mat.flatten(), -pair_dst.flatten())[:2])
    plt.show()
    exit()
    # scr = ScreenParser()
    # scr.calibrate('./calibr')

    db_dir = Path(r"C:\Users\Mykhailo_Tkachuk\PycharmProjects\Brawl-Stars-AI\ocr_data\end_title")
    db = save_templates(db_dir)
    test_dir = Path(r"C:\Users\Mykhailo_Tkachuk\PycharmProjects\Brawl-Stars-AI\testing\from_run")
    for f in list(test_dir.glob('*.png')):  # + list(db_dir.glob('*.png')):
        print(f.name, match(cv.imread(str(f))[..., [0]], db, verbose=0, score_thresh=3.5E-4))
