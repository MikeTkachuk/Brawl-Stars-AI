import numpy as np

from utils.getkeys import key_check
from controls import act
from environment import ScreenParser, GymEnv
from model import *
from training import *
import config

import time

"""
A TODO list:

- 
- environment class with state reward (if not battle reward present - compute internally) functionality + end/start battle control
- create a mapping from dir vector to a sequence of key controls, experiment with latency
- create mouse control function to change position in radial coordinates
- add action derivatives to act function
- finalize act function
- standardize whole model latency to make the game discrete interval MDP (perhaps make it a continuous transition idk how)
- run dummy model
"""

parser = ScreenParser()
env = GymEnv(parser=parser)


def run_loop():
    model = get_model()
    weights = np.random.normal(0,0.05, size=(768, 2))
    obs = env.reset()

    paused = True
    angle_ = 0
    while True:
        if not paused:
            start_ = time.time()
            features = model.forward_features(img_to_tensor(obs)).cpu().detach().numpy()

            direction = np.dot(features, weights).flatten()
            print(direction)
            obs = env.step({
                'direction': direction,
                'make_move': 1,
            })
            print('elapsed: ', time.time() - start_)

        if 'Z' in key_check():
            paused = not paused
            env.step({
                'direction': [0, 1],
                'make_move': 0,
            })
        if 'Q' in key_check():
            env.__exit__()
            break


if __name__ == '__main__':
    run_loop()
