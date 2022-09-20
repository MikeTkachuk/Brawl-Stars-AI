from utils.grabscreen import grab_screen
import cv2 as cv
import numpy as np
import config
from config import _relative_to_pixel
import pytesseract
import os
from utils.custom_ocr import save_templates, match
from controls import act

import gym
from gym import spaces
from typing import Optional, Union, List
from multiprocessing import Process
from multiprocessing.sharedctypes import Value
from ctypes import Structure, c_double


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
        # cv.imwrite('exit.png', exit_end_screen)
        if exit_end_screen_text == 'exit':  # run slow tesseract only if inside exit screen
            end_title_region = _relative_to_pixel(self.end_screen_title_region, self.main_screen)
            end_title = _ocr_preproc(screen, end_title_region)

            # tesseract ocr params were found by brute force
            end_title_text = pytesseract.image_to_string(end_title, config='--psm 8')

            # read end screen score
            score_region = _relative_to_pixel(self.score_region, self.main_screen)
            score = _ocr_preproc(screen, score_region)
            score_text = match(cv.cvtColor(score, cv.COLOR_RGB2GRAY), self.digit_signed_database)
            # cv.imwrite('+8.png', score)
        else:
            end_title_text = ''
            score_text = ''

        start_battle_region = _relative_to_pixel(self.start_battle_region, self.main_screen)
        start_battle = _ocr_preproc(screen, start_battle_region, thresh=(200, 255))
        start_battle_text = match(cv.cvtColor(start_battle, cv.COLOR_RGB2GRAY), self.play_database)
        # cv.imwrite('play.png', start_battle)
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

        print(end_title_text, score_text, player_trophies,
              exit_text, play_text, defeated_text, proceed_text)
        return screen, parse_results


class Vector(Structure):
    _fields_ = [('x', c_double), ('y', c_double)]


class GymEnv(gym.Env):
    def __init__(self, parser: ScreenParser):
        self.parser = parser
        self.acting_process = None
        self.action_transmitter = None

    def _init_control_process(self):
        if self.acting_process is not None:
            self.acting_process.terminate()
            print('Terminated latest control.')

        direction = Value(Vector, 1, 0)
        make_move = Value('i', 0)
        make_shot = Value('i', 0)
        shoot_direction = Value(Vector, 1, 0)
        shoot_strength = Value('d', 0.0)
        super_ability = Value('i', 0)
        use_gadget = Value('i', 0)

        self.acting_process = Process(target=act, args=(
            direction,
            make_move,
            make_shot,
            shoot_direction,
            shoot_strength,
            super_ability,
            use_gadget
        ))
        self.acting_process.start()
        print('New control started.')

        self.action_transmitter = {
            'direction': direction,
            'make_move': make_move,
            'make_shot': make_shot,
            'shoot_direction': shoot_direction,
            'shoot_strength': shoot_strength,
            'super_ability': super_ability,
            'use_gadget': use_gadget
        }

    def _interpret_parsed_screen(self, parsed):
        (end_title_text, score_text, player_trophies,
         exit_text, play_text, defeated_text, proceed_text) = parsed

    def step(self, action):
        """
        Update the action params valid until the next step call. Return the screen observed at the same time

        :param action: optional kwargs of form {direction': (x,y), norm != 0,
            'make_move': int bool like,
            'make_shot': int bool like,
            'shoot_direction': (x,y), norm != 0,
            'shoot_strength': float in [0,1],
            'super_ability': int bool like,
            'use_gadget': int bool like}
        :return: np.ndarray. screen img
        """
        for k, v in self.action_transmitter.items():
            if k not in action:
                continue
            if 'direction' in k:
                v.x = action[k][0]
                v.y = action[k][1]
            else:
                v.value = action[k]
        screen, parse_results = self.parser.get_state()
        return screen

    def reset(
            self,
            *,
            seed=None,
            return_info=False,
            options=None,
    ):
        # TODO add skip end of the showdown battle
        super().reset(seed=seed)
        self._init_control_process()
        observation = grab_screen(self.parser.main_screen)
        info = {}
        return (observation, info) if return_info else observation

    def __exit__(self, exc_type=None, exc_val=None, exc_tb=None):
        self.acting_process.terminate()

    def render(self, mode="human"):
        return
