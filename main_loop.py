from utils.getkeys import key_check
from controls import act
from environment import ScreenEnv
from model import
from training import
import config


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

env = ScreenEnv()
def run_loop():
    paused = False

    while True:
        if not paused:


        if 'Z' in key_check():
            paused = not paused
        if 'Q' in key_check():
            break