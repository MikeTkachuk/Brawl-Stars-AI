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
        keys_pressed = key_check()
        if keys_pressed:
            print(keys_pressed)
        if not paused:
            start_ = time.time()
            features = model.forward_features(img_to_tensor(obs)).cpu().detach().numpy()

            direction = np.dot(features, weights).flatten()
            direction = np.array([np.cos(angle_), np.sin(angle_)])
            #direction = (1,0)
            make_shot = 0
            super_ability = 1
            if 'U' in keys_pressed:
                make_shot = 0
                super_ability = 0
            if 'I' in keys_pressed:
                make_shot = 0
                super_ability = 1
            if 'O' in keys_pressed:
                make_shot = 1
                super_ability = 0
            if 'P' in keys_pressed:
                make_shot = 1
                super_ability = 1
            if keys_pressed:
                obs, reward, done, _, info = env.step({
                    'direction': direction,
                    'make_move': 0,
                    'make_shot': make_shot,
                    'shoot_direction': direction,
                    'shoot_strength': 1.0,
                    'super_ability': super_ability
                })
                angle_ += 0.1
                angle_ %= 2*np.pi
                print('elapsed: ', time.time() - start_)
                print(reward, done)
                if done:
                    env.reset()

        if 'Z' in keys_pressed:
            paused = not paused
            env.step({
                'direction': [0, 1],
                'make_move': 0,
            })
        if 'Q' in keys_pressed:
            env.__exit__()
            break
        time.sleep(0.05)
        print(paused)


if __name__ == '__main__':
    run_loop()
