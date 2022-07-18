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
main_screen = (0, 41, 1388, 822)  # absolute
screen_width, screen_height = main_screen[2], main_screen[3]

# --- helper controls relative to main screen ---
end_screen_title = (60 / screen_width, 0 / screen_height,
                    340 / screen_width, 135 / screen_height)
score_region = (120 / screen_width, 170 / screen_height,
                80 / screen_width, 60 / screen_height)

exit_end_screen = (1224 / screen_width, 763 / screen_height)
start_battle = (1192 / screen_width, 742 / screen_height)
