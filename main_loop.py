from utils.getkeys import key_check
from controls import act
from environment import get_state, get_reward
from model import
from training import
import config


def run_loop():
    paused = False

    while True:
        if not paused:


        if 'Z' in key_check():
            paused = not paused
        if 'Q' in key_check():
            break