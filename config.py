import numpy as np

# ===============
# paths
# ===============
tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract'
digit_database = r'C:\Users\Mykhailo_Tkachuk\PycharmProjects\Brawl-Stars-AI\ocr_data\digits'
digit_signed_database = r'C:\Users\Mykhailo_Tkachuk\PycharmProjects\Brawl-Stars-AI\ocr_data\digits_signed'
exit_database = r'C:\Users\Mykhailo_Tkachuk\PycharmProjects\Brawl-Stars-AI\ocr_data\exit'
play_database = r'C:\Users\Mykhailo_Tkachuk\PycharmProjects\Brawl-Stars-AI\ocr_data\play'
defeated_database = r"C:\Users\Mykhailo_Tkachuk\PycharmProjects\Brawl-Stars-AI\ocr_data\defeated"
proceed_database = r"C:\Users\Mykhailo_Tkachuk\PycharmProjects\Brawl-Stars-AI\ocr_data\proceed"

# ===============
# controls config
# ===============
forward = 17
backward = 31
left = 30
right = 32

gadget = 33

terminate_program = 'Q'  # press Q to terminate everything with exit()

# ===============
# screen locations config
# regions of form (x, y, width, height)
# locations of form (x, y)
# ===============

# --- main screen absolute size ---
main_screen = (0, 41, 1388, 781)  # absolute. The only thing that should be changed in case of main screen displacement

# do NOT change without full recalibration
ref_main_screen = (0, 41, 1388, 781)    # to fix the relative calculations.
                                        # Should NOT be changed unless calibrating from scratch!
####

screen_x, screen_y = ref_main_screen[0], ref_main_screen[1]
screen_width, screen_height = ref_main_screen[2], ref_main_screen[3]


# --- helper controls relative to main screen ---
def to_relative(region):
    if len(region) == 2:
        return (
            (region[0] - screen_x) / screen_width,
            (region[1] - screen_y) / screen_height,
        )
    if len(region) == 4:
        return (
            (region[0] - screen_x) / screen_width,
            (region[1] - screen_y) / screen_height,
            region[2] / screen_width,
            region[3] / screen_height,
        )


def _relative_to_pixel(point, main_scr, absolute=False):
    """
    Convert relative to absolute
    :param point: (x,y) or region (x, y, w, h) relative to main top left corner
    :param main_scr: tuple xywh of main screen in pixels
    :param absolute: bool, if False calculates pixel locations relative to main screen
    :return: pixel point or region
    """
    mx, my, mw, mh = main_scr
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


end_screen_title_region = to_relative((55, 55, 360, 120))
score_region = to_relative((117, 177, 80, 45))
player_trophies_region = to_relative((629, 148, 64, 30))

exit_end_screen_region = to_relative((1185, 740, 80, 40))
start_battle_region = to_relative((1140, 720, 120, 50))
proceed_region = to_relative((1155, 742, 143, 39))
defeated_region = to_relative((530, 200, 330, 80))

click_defeated = to_relative((696, 764))
click_exit_proceed = to_relative((1225, 760))
click_play = to_relative((1200, 745))

regular_joystick = to_relative((1244,614))
super_joystick = to_relative((1088, 677))
joystick_radius = 78
