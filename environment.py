from utils.grabscreen import grab_screen
import cv2 as cv
import numpy as np
import config
import pytesseract
import os
from utils.custom_ocr import save_templates, match


class ScreenEnv:
    def __init__(self):
        # screen region helpers
        self.main_screen = config.main_screen

        self.end_screen_title_region = config.end_screen_title_region
        self.score_region = config.score_region
        self.player_trophies_region = config.player_trophies_region

        self.exit_end_screen_region = config.exit_end_screen_region
        self.start_battle_region = config.start_battle_region

        # Environment state attributes
        self.is_end_screen = False
        self.is_start_screen = True
        self.player_trophies_count = None

        # init ocr
        pytesseract.pytesseract.tesseract_cmd = config.tesseract_cmd
        self.digit_database = save_templates(config.digit_database)
        self.digit_signed_database = save_templates(config.digit_signed_database)
        self.exit_database = save_templates(config.exit_database)
        self.play_database = save_templates(config.play_database)

    def _relative_to_pixel(self, point, absolute=False):
        """
        Convert relative to absolute
        :param point: (x,y) or region (x, y, w, h) relative to main top left corner
        :param absolute: bool, if False calculates pixel locations relative to main screen
        :return: pixel point or region
        """
        mx, my, mw, mh = self.main_screen
        if len(point) == 2:
            out = np.array([point[0] * mw, point[1] * mh], dtype=np.int32)
            if absolute:
                out += np.array([mx, my], dtype=np.int32)
            return out
        if len(point) == 4:
            out = np.array([point[0] * mw, point[1] * mh, point[2] * mw, point[3] * mh], dtype=np.int32)
            if absolute:
                out += np.array([mx, my, 0, 0], dtype=np.int32)
            return out

    def _parse_end_screen(self, screen):
        """
        Recognize rewards and screen state
        :param screen: HxWxC RGB array of a screen
        :return:
        """

        def _ocr_preproc(rgb_screen, region):
            # region in xywh
            cropper = (slice(region[1], region[1] + region[3]),
                       slice(region[0], region[0] + region[2]))
            rgb_region = rgb_screen[cropper]

            if min(rgb_region.shape[:2]) < 240:
                scale = 240 / min(rgb_region.shape[:2])
                rgb_region = cv.resize(rgb_region, None, fx=scale, fy=scale)
            rgb_region = cv.cvtColor(rgb_region, cv.COLOR_RGB2GRAY)
            _, rgb_region = cv.threshold(rgb_region, 150, 255, cv.THRESH_BINARY)
            rgb_region = cv.erode(rgb_region, np.ones((7, 7), np.uint8))
            rgb_region = cv.cvtColor(np.expand_dims(rgb_region, -1), cv.COLOR_GRAY2RGB)
            return rgb_region

        exit_end_screen_region = self._relative_to_pixel(self.exit_end_screen_region)
        exit_end_screen = _ocr_preproc(screen, exit_end_screen_region)
        exit_end_screen_text = match(cv.cvtColor(exit_end_screen, cv.COLOR_RGB2GRAY), self.exit_database)
        if exit_end_screen_text == 'exit':  # run slow tesseract only if inside exit screen
            end_title_region = self._relative_to_pixel(self.end_screen_title_region)
            end_title = _ocr_preproc(screen, end_title_region)

            # tesseract ocr params were found by brute force
            end_title_text = pytesseract.image_to_string(end_title, config='--psm 8')

            # read end screen score
            score_region = self._relative_to_pixel(self.score_region)
            score = _ocr_preproc(screen, score_region)
            score_text = match(cv.cvtColor(score, cv.COLOR_RGB2GRAY), self.digit_signed_database)
        else:
            end_title_text = ''
            score_text = ''

        start_battle_region = self._relative_to_pixel(self.start_battle_region)
        start_battle = _ocr_preproc(screen, start_battle_region)
        start_battle_text = match(cv.cvtColor(start_battle, cv.COLOR_RGB2GRAY), self.play_database)
        if start_battle_text == 'play':  # read brawler total trophies only in the main menu
            player_trophies_region = self._relative_to_pixel(self.player_trophies_region)
            player_trophies = _ocr_preproc(screen, player_trophies_region)
            player_trophies_text = match(cv.cvtColor(player_trophies, cv.COLOR_RGB2GRAY), self.digit_database)
        else:
            player_trophies_text = ''

        return end_title_text, score_text, player_trophies_text

    def get_state(self):
        """
        Captures a predefined screen region, parses screen to produce reward and game state
        :return: HxWxC RGB array, float reward, bool is_training
        """
        screen = grab_screen(self.main_screen)
        end_title_text, score_text, player_trophies = self._parse_end_screen(screen)
        print(f"End:{end_title_text}")
        print(f"Reward:{score_text}")
        print(f"Total:{player_trophies}")


