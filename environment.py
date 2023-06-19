from utils.grabscreen import grab_screen
import cv2 as cv
import numpy as np
import config
from config import _relative_to_pixel
import pytesseract
import os
import time
import re
from utils.custom_ocr import save_templates, match
from utils.utils import is_power_of_two
from controls import act, reset_controls, exit_end_screen, start_battle, exit_defeated

import gym
from gym import spaces
from typing import Optional, Union, List, Any
from multiprocessing import Process
from multiprocessing.sharedctypes import Value
from ctypes import Structure, c_double
from utils.getkeys import key_check


def _ocr_preproc(rgb_screen, region, thresh=(150, 255), erosion=7):
    # region in xywh
    cropper = (slice(region[1], region[1] + region[3]),
               slice(region[0], region[0] + region[2]))
    rgb_region = rgb_screen[cropper]

    if min(rgb_region.shape[:2]) < 240:
        scale = 240 / min(rgb_region.shape[:2])
        rgb_region = cv.resize(rgb_region, None, fx=scale, fy=scale)
    rgb_region = cv.cvtColor(rgb_region, cv.COLOR_RGB2GRAY)
    _, rgb_region = cv.threshold(rgb_region, thresh[0], thresh[1], cv.THRESH_BINARY)
    rgb_region = cv.erode(rgb_region, np.ones((erosion, erosion), np.uint8))
    rgb_region = cv.cvtColor(np.expand_dims(rgb_region, -1), cv.COLOR_GRAY2RGB)
    return rgb_region


class ScreenParser:
    """Custom visual parser of a brawl stars UI"""
    def __init__(self):
        # screen region helpers
        self.main_screen = config.main_screen

        self.end_screen_title_region = config.end_screen_title_region
        self.score_region = config.score_region
        self.player_trophies_region = config.player_trophies_region

        self.exit_end_screen_region = config.exit_end_screen_region
        self.start_battle_region = config.start_battle_region
        self.defeated_region = config.defeated_region
        self.proceed_region = config.proceed_region

        # init ocr
        pytesseract.pytesseract.tesseract_cmd = config.tesseract_cmd
        self.digit_database = save_templates(config.digit_database, crop=(3 / 32, 9 / 16))
        self.digit_signed_database = save_templates(config.digit_signed_database, crop=(3 / 32, 9 / 16))
        self.exit_database = save_templates(config.exit_database)
        self.play_database = save_templates(config.play_database)
        self.defeated_database = save_templates(config.defeated_database)
        self.proceed_database = save_templates(config.proceed_database)

    def _parse_screen(self, screen):
        """
        Recognize rewards and screen state
        :param screen: HxWxC RGB array of a screen
        :return: tuple of recognized texts
        """

        exit_end_screen_region = _relative_to_pixel(self.exit_end_screen_region, self.main_screen)
        exit_end_screen = _ocr_preproc(screen, exit_end_screen_region, thresh=(200, 255))
        exit_end_screen_text = match(cv.cvtColor(exit_end_screen, cv.COLOR_RGB2GRAY), self.exit_database)
        if exit_end_screen_text == 'exit':  # run slow tesseract only if inside exit screen
            end_title_region = _relative_to_pixel(self.end_screen_title_region, self.main_screen)
            end_title = _ocr_preproc(screen, end_title_region)

            # tesseract ocr params were found by brute force
            end_title_text = pytesseract.image_to_string(end_title, config='--psm 8')

            # read end screen score
            score_region = _relative_to_pixel(self.score_region, self.main_screen)
            score = _ocr_preproc(screen, score_region)
            score_text = match(cv.cvtColor(score, cv.COLOR_RGB2GRAY), self.digit_signed_database)
        else:
            end_title_text = ''
            score_text = ''

        start_battle_region = _relative_to_pixel(self.start_battle_region, self.main_screen)
        start_battle = _ocr_preproc(screen, start_battle_region, thresh=(200, 255))
        start_battle_text = match(cv.cvtColor(start_battle, cv.COLOR_RGB2GRAY), self.play_database)
        if start_battle_text == 'play':  # read brawler total trophies only in the main menu
            player_trophies_region = _relative_to_pixel(self.player_trophies_region, self.main_screen)
            player_trophies = _ocr_preproc(screen, player_trophies_region)
            player_trophies_text = match(cv.cvtColor(player_trophies, cv.COLOR_RGB2GRAY), self.digit_database)
        else:
            player_trophies_text = ''

        defeated_region = _relative_to_pixel(self.defeated_region, self.main_screen)
        defeated = _ocr_preproc(screen, defeated_region)
        defeated_text = match(cv.cvtColor(defeated, cv.COLOR_RGB2GRAY), self.defeated_database)

        proceed_region = _relative_to_pixel(self.proceed_region, self.main_screen)
        proceed = _ocr_preproc(screen, proceed_region)
        proceed_text = match(cv.cvtColor(proceed, cv.COLOR_RGB2GRAY), self.proceed_database)
        return end_title_text, score_text, player_trophies_text, \
               exit_end_screen_text, start_battle_text, defeated_text, proceed_text

    def get_state(self):
        """
        Captures a predefined screen region, parses screen to produce reward and game state
        :return: HxWxC RGB array, float reward, bool is_training
        """
        screen = grab_screen(self.main_screen)
        parse_results = self._parse_screen(screen)

        (end_title_text, score_text, player_trophies,
         exit_text, play_text, defeated_text, proceed_text) = parse_results

        print('Parsed:')
        print(end_title_text, score_text, player_trophies,
              exit_text, play_text, defeated_text, proceed_text)
        return screen, [s.lower().strip(':!. \t—\n') for s in parse_results]


class Vector(Structure):
    _fields_ = [('x', c_double), ('y', c_double)]


class ActingProcess:
    """
    A class that handles UI controlling (keyboard and mouse)
    and uninterrupted communication with the program
    """
    def __init__(self, proc, shared_data=None):
        """

        :param proc: A raw process that has not been terminated yet
        :param shared_data: an optional dict of multiprocessing shared data objects
        """
        self.proc = proc
        self.shared_data = shared_data or {}
        self._started = self.is_running  # get current process state
        self._exited = False

    @property
    def is_running(self):
        return self.proc.is_alive()

    def __exit__(self, exc_type=None, exc_val=None, exc_tb=None):
        if self._exited or not self._started:
            raise RuntimeError("The process is not alive, can not exit.")
        self.proc.terminate()
        reset_controls()
        self._exited = True
        print('Terminated latest control process.')

    def exit(self):
        """Terminates the processes"""
        self.__exit__()

    def start(self):
        """Starts the process"""
        if self._exited or self._started:
            raise RuntimeError("Repeated initialization is not supported.")
        self.proc.start()
        self._started = True
        print('New control process started.')

    def update_data(self, data):
        """
        Update the shared data
        :param data: dict of data with the keys present in the shared_data param upon object init
        :return:
        """
        for k, v in self.shared_data.items():
            if k not in data:
                continue
            if 'direction' in k:
                v.x = data[k][0]
                v.y = data[k][1]
            else:
                v.value = data[k]
        with self.shared_data['changed'].get_lock():
            self.shared_data['changed'].value = 1


class GymEnv(gym.Env):
    def __init__(self, parser: ScreenParser, move_shot_anchors=(8, 8)):
        """
        Real time Brawl Stars environment
        :param parser: an instance of screen parser to interpret the game frame
        :param move_shot_anchors: int or (int, int) - num anchors for movement dir and shoot dir respectively
                                    used for action tokenization during training. default 8
        """
        print('Creating env instance')
        self.parser = parser
        self.acting_process: ActingProcess = None
        self.done = False
        self.move_shot_anchors = move_shot_anchors if isinstance(move_shot_anchors, tuple) else (move_shot_anchors,) * 2
        if not is_power_of_two(self.move_shot_anchors[0]) or not is_power_of_two(self.move_shot_anchors[1]):
            raise ValueError('Num anchors that is not a power of 2 is not allowed.')

        self.action_space = spaces.Discrete(
            n=2 ** 4 * np.prod(self.move_shot_anchors))  # 2^#binary_actions * prod(#anchors)
        self.continuous_action_space = spaces.Box(0, 1, shape=(3,))

    def _init_control_process(self):
        """
            Creates an instance of ActingProcess,
            connects it to a program with shared variables,
            and runs the acting process
        :return:
        :raises: RuntimeError if called repeatedly
        """

        if self.acting_process is not None and self.acting_process.is_running:
            raise RuntimeError("Repeated env control process initialization encountered")

        direction = Value(Vector, 1, 0)
        make_move = Value('i', 0)
        make_shot = Value('i', 1)  # init mouse released
        shoot_direction = Value(Vector, 1, 0)
        shoot_strength = Value('d', 0.0)
        super_ability = Value('i', 0)
        use_gadget = Value('i', 0)
        changed = Value('i', 0)

        proc = Process(target=act, args=(
            direction,
            make_move,
            make_shot,
            shoot_direction,
            shoot_strength,
            super_ability,
            use_gadget,
            changed
        ))

        action_transmitter = {
            'direction': direction,
            'make_move': make_move,
            'make_shot': make_shot,
            'shoot_direction': shoot_direction,
            'shoot_strength': shoot_strength,
            'super_ability': super_ability,
            'use_gadget': use_gadget,
            'changed': changed
        }

        self.acting_process = ActingProcess(proc=proc, shared_data=action_transmitter)
        self.acting_process.start()

    def _interpret_parsed_screen(self, parsed=None, max_patience=5):
        """
        A func to both interpret on-screen texts
         and operate post-battle (kills controll process and leads to the main menu screen)
        :param parsed: optional parsed screen texts
        :return: parsed numeric reward or None,
                 bool whether entered any of the terminal screens,
                 dict info placeholder
        """
        if parsed is None:
            parsed = self.parser.get_state()[1]
        (end_title_text, score_text, player_trophies,
         exit_text, play_text, defeated_text, proceed_text) = parsed

        reward = 0
        terminated = False
        info = {}

        if any([defeated_text == 'defeated',
                exit_text == 'exit',
                proceed_text == 'proceed',
                play_text == 'play']):
            if self.acting_process is not None:
                self.acting_process.exit()
            terminated = True
            patience = 0
            while True:
                if defeated_text == 'defeated':
                    exit_defeated()
                elif exit_text == 'exit' or proceed_text == 'proceed':
                    if score_text:
                        reward = float(score_text)
                    else:
                        raw_score = 0  # TODO remove
                        if end_title_text in ['defeat', 'victory', 'draw']:
                            if end_title_text == 'defeat':
                                raw_score = -8
                            elif end_title_text == 'victory':
                                raw_score = 8
                            else:
                                raw_score = 0

                        else:
                            if 'rank' in end_title_text:
                                raw_rank = int(''.join(re.findall('[0-9]', end_title_text.replace('rank', ''))))
                                raw_score = np.linspace(-8, 8, 10)[-raw_rank]

                            if 'you are' in end_title_text:
                                raw_rank = 1
                                raw_score = np.linspace(-8, 8, 10)[-raw_rank]
                        reward = raw_score  # TODO add score weighting based on the total trophies
                    exit_end_screen()
                else:
                    patience += 1
                    if patience > max_patience:
                        break
                (end_title_text, score_text, player_trophies,
                 exit_text, play_text, defeated_text, proceed_text) = self.parser.get_state()[1]
                time.sleep(0.2)

        return reward, terminated, info

    def _parse_action_token(self, action):
        """
        Parse 1 multi-binary and 3 continuous action values
        :param action: array-like of action values
        :return: dict of parsed actions
        """
        assert len(action) == 4  # 1 token + 3 continuous [0,1]: (move, shot, strength)
        assert 0 <= action[0] < self.action_space.n
        bins = str(bin(int(action[0])))[2:]
        desired_len = len(str(bin(self.action_space.n - 1))) - 2
        bins = '0'*(desired_len - len(bins)) + bins  # pad with 0

        make_move, make_shot, super_ability, use_gadget = bins[:4]

        move_n_bits = len(str(bin(self.move_shot_anchors[0] - 1))) - 2
        move_anchor = int(bins[4:4 + move_n_bits], 2)
        assert len(bins[4:4 + move_n_bits]) == move_n_bits  # just to check myself

        shot_n_bits = len(str(bin(self.move_shot_anchors[1] - 1))) - 2
        shot_anchor = int(bins[4 + move_n_bits: 4 + move_n_bits + shot_n_bits], 2)
        assert len(bins[4 + move_n_bits: 4 + move_n_bits + shot_n_bits]) == shot_n_bits

        def _get_anchor_dir(anchor_num, total, shift=0.0):
            angle_shift = 1 / total * 2 * np.pi * (shift - 0.5)  # assumes shift is in [0, 1]
            angle = anchor_num / total * 2 * np.pi + angle_shift
            anchor = np.array([np.cos(angle), np.sin(angle)])
            return anchor

        parsed_action = {
            'direction': _get_anchor_dir(move_anchor, self.move_shot_anchors[0], action[1]),
            'make_move': int(make_move),
            'make_shot': int(make_shot),
            'shoot_direction': _get_anchor_dir(shot_anchor, self.move_shot_anchors[1], action[2]),
            'shoot_strength': action[3],
            'super_ability': int(super_ability),
            'use_gadget': int(use_gadget),
        }
        return parsed_action

    def _obs_preproc(self, obs):
        return cv.resize(obs, (256, 256)).astype(np.float32)

    def step(self, action: Union[dict, Any]):
        """
        Update the action params valid until the next step call. Return the screen observed at the same time

        :param action: optional kwargs of form {direction': (x,y), norm != 0,
            'make_move': int bool like,
            'make_shot': int bool like,
            'shoot_direction': (x,y), norm != 0,
            'shoot_strength': float in [0,1],
            'super_ability': int bool like,
            'use_gadget': int bool like}

            If not dict it is parsed separately as an array-like:
                1st place - action token
                nth others - continuous params
        :return: np.ndarray. screen img
        """
        if config.terminate_program in key_check():  # exits env if the user pressed the specified key
            self.__exit__()
            exit()

        if not self.acting_process.is_running:  # if resume is needed
            self._init_control_process()

        if self.done:
            return None

        if not isinstance(action, dict):
            action = self._parse_action_token(action)

        self.acting_process.update_data(action)
        screen, parse_results = self.parser.get_state()
        reward, terminated, info = self._interpret_parsed_screen(parse_results)
        obs = self._obs_preproc(screen)
        self.done = terminated
        return obs, reward, terminated, info

    def reset(
            self,
            timeout=50
    ):
        """
            Kills acting process and waits for battle to end.
            After that enters new battle and starts a new acting process
        :param timeout: num seconds to wait for battle to end
        :return: observation after reset
        """
        print('environment.GymEnv.reset: reset called')
        if not self.done:  # TODO handle idle disconnection
            for attempt in range(timeout // 5):
                terminated = self._interpret_parsed_screen()[1]  # if outside battle
                if terminated:
                    break
                time.sleep(5)
            else:
                raise RuntimeError('Env reset timeout.')

        while True:  # takes some time for _interpret_parsed_screen to enter main menu
            if self.parser.get_state()[1][4] == 'play':  # if in main menu
                start_battle()
            else:
                break
            time.sleep(0.2)

        # TODO skip loading screen (technically it is starts here)

        self._init_control_process()
        observation = self._obs_preproc(self.parser.get_state()[0])
        self.done = False
        return observation

    def __exit__(self, exc_type=None, exc_val=None, exc_tb=None):
        if self.acting_process.is_running:
            self.acting_process.exit()

    def render(self, mode="human"):
        return


def make_env():
    return GymEnv(ScreenParser(), )
