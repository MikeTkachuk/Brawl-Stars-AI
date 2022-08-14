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
forward = hex(17)
backward = hex(31)
left = hex(30)
right = hex(32)

gadget = hex(33)

# ===============
# screen locations config
# regions of form (x, y, width, height)
# locations of form (x, y)
# ===============

# --- main screen absolute size ---
main_screen = (0, 41, 1388, 781)  # absolute. Should be changed in case of main screen displacement

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


end_screen_title_region = to_relative((55, 55, 360, 120))
score_region = to_relative((117, 177, 80, 45))
player_trophies_region = to_relative((629, 148, 64, 30))

exit_end_screen_region = to_relative((1185, 740, 80, 40))
start_battle_region = to_relative((1140, 720, 120, 50))
proceed_region = to_relative((1155, 742, 143, 39))
defeated_region = to_relative((530, 200, 330, 80))
