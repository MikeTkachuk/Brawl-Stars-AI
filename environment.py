from utils.grabscreen import grab_screen
import cv2 as cv
import numpy as np
import config
import pytesseract


class ScreenEnv:
    def __init__(self):
        # screen region helpers
        self.main_screen = config.main_screen

        self.end_screen_title = config.end_screen_title
        self.score_region = config.score_region
        self.player_trophies_region = config.player_trophies_region

        self.exit_end_screen = config.exit_end_screen
        self.start_battle = config.start_battle

        # Environment state attributes
        self.is_end_screen = False
        self.is_start_screen = True

        pytesseract.pytesseract.tesseract_cmd = config.tesseract_cmd

    def _relative_to_pixel(self, point):
        """
        Convert relative to absolute
        :param point: (x,y) or region (x, y, w, h) relative to main top left corner
        :return: absolute point or region
        """
        mx, my, mw, mh = self.main_screen
        if len(point) == 2:
            return (
                int(mx + point[0] * mw),
                int(my + point[1] * mh)
            )
        if len(point) == 4:
            return (
                int(mx + point[0] * mw),
                int(my + point[1] * mh),
                int(mw * point[2]),
                int(mh * point[3])
            )

    def _parse_end_screen(self, screen):
        """
        Recognize rewards and screen state
        :param screen: HxWxC RGB array of a screen
        :return:
        """
        # TODO do gram screen once
        def _ocr_preproc(rgb_region):
            if min(rgb_region.shape[:2]) < 240:
                scale = 240/min(rgb_region.shape[:2])
                rgb_region = cv.resize(rgb_region, None, fx=scale, fy=scale)
            rgb_region = cv.cvtColor(rgb_region, cv.COLOR_RGB2GRAY)
            _, rgb_region = cv.threshold(rgb_region, 150, 255, cv.THRESH_BINARY)
            rgb_region = cv.erode(rgb_region, np.ones((7, 7), np.uint8))
            rgb_region = cv.cvtColor(np.expand_dims(rgb_region, -1), cv.COLOR_GRAY2RGB)
            return rgb_region

        end_title_region = self._relative_to_pixel(self.end_screen_title)
        end_title = grab_screen(end_title_region)
        end_title = _ocr_preproc(end_title)

        score_region = self._relative_to_pixel(self.score_region)
        score = grab_screen(score_region)
        score = _ocr_preproc(score)

        player_trophies_region = self._relative_to_pixel(self.player_trophies_region)
        player_trophies = grab_screen(player_trophies_region)
        player_trophies = _ocr_preproc(player_trophies)

        #remove !!
        cv.imwrite('end.png', end_title)
        cv.imwrite('score.png', score)
        cv.imwrite('trophy.png', player_trophies)

        # tesseract ocr params were found by brute force
        end_title_text = pytesseract.image_to_string(end_title, config='--psm 8')
        score_text = pytesseract.image_to_string(score, config='-c tessedit_char_whitelist=0123456789+- --psm 7')
        player_trophies_text = pytesseract.image_to_string(player_trophies, config='-c tessedit_char_whitelist=0123456789+- --psm 7')
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

