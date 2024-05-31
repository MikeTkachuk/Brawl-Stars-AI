import random

import numpy as np
import cv2 as cv
import torch

from environment import make_env
from utils.getkeys import key_check
import time

if __name__ == "__main__":
    env = make_env(move_shot_anchors=(4, 4))
    observations = [env.reset()]
    actions = []
    count = 0
    anch_map = {
        -1: "no_action",
        0: "right",
        1: "up",
        2: "left",
        3: "down",
    }
    while True:
        if env.done:
            ep = {
                "observations": torch.tensor(observations).byte().permute(0, 3, 1, 2),
                "actions": torch.tensor(actions).long()
            }
            torch.save(ep, r"C:\Users\Michael\Downloads\test_env" + rf"\{count:03}.pt")
            observations = [env.reset()]
            actions = []
        try:
            keys_pressed = key_check()
            move_anchor = -1
            for i in range(4):
                if str(i) in keys_pressed:
                    move_anchor = i
            move_anchor = count % 5 - 1
            token = random.randint(0, 159)
            bins = env.split_into_bins(token, )
            bins[0] = int(move_anchor >= 0)
            bins[-2] = max(0, move_anchor)

            # test shooting
            bins[-1] = count % 4
            bins[2] = count % 2
            # end test shooting

            actions.append(bins)
            token = env.create_action_token(*bins
                                            )
            print(keys_pressed, token)
            cv.imwrite(r"C:\Users\Michael\Downloads\test_env" + rf"\{count:03}_{anch_map[move_anchor]}.png",
                       observations[-1].astype(np.uint8)[..., ::-1])
            obs, *_ = env.step([token, random.random(), random.random(), random.random()])
            observations.append(obs)

            time.sleep(0.5)
            count += 1
        except Exception as e:
            print(e)
            continue
