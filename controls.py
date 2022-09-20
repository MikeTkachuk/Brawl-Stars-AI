import time

from utils.directkeys import PressKey, ReleaseKey
import config
from config import _relative_to_pixel
import numpy as np
import mouse
import ctypes

ctypes.windll.shcore.SetProcessDpiAwareness(2)  # https://github.com/boppreh/mouse/issues/122


def straight():
    PressKey(config.forward)
    ReleaseKey(config.left)
    ReleaseKey(config.right)
    ReleaseKey(config.backward)


def left():
    ReleaseKey(config.forward)
    PressKey(config.left)
    ReleaseKey(config.backward)
    ReleaseKey(config.right)


def right():
    ReleaseKey(config.forward)
    PressKey(config.right)
    ReleaseKey(config.left)
    ReleaseKey(config.backward)


def reverse():
    PressKey(config.backward)
    ReleaseKey(config.left)
    ReleaseKey(config.forward)
    ReleaseKey(config.right)


def forward_left():
    PressKey(config.forward)
    PressKey(config.left)
    ReleaseKey(config.right)
    ReleaseKey(config.backward)


def forward_right():
    PressKey(config.forward)
    PressKey(config.right)
    ReleaseKey(config.left)
    ReleaseKey(config.backward)


def reverse_left():
    PressKey(config.backward)
    PressKey(config.left)
    ReleaseKey(config.forward)
    ReleaseKey(config.right)


def reverse_right():
    PressKey(config.backward)
    PressKey(config.right)
    ReleaseKey(config.forward)
    ReleaseKey(config.left)


def stop_movement():
    ReleaseKey(config.backward)
    ReleaseKey(config.right)
    ReleaseKey(config.forward)
    ReleaseKey(config.left)


def start_battle():
    pass


def exit_end_screen():
    pass


# assemble anchor directions and their corresponding controls
angles = np.arange(32) / 32 * 2 * np.pi
anchors = np.stack([np.cos(angles), np.sin(angles)], axis=1)
anchor_controls = []
func_order = (right, forward_right, straight, forward_left, left, reverse_left, reverse, reverse_right, right)
combinations_loop = ((1, 0), (2, 1), (1, 1), (1, 2))  # (0,1) turns into (1,0) upon next iteration
for i_func in range(len(func_order) - 1):
    for comb_ in combinations_loop:
        anchor_controls.append(
            (func_order[i_func],) * comb_[0] + (func_order[i_func + 1],) * comb_[1]
        )


def generate_dir(direction):
    direction = np.array(direction)
    direction /= np.linalg.norm(direction)
    scores = np.dot(anchors, direction.reshape(2, 1))
    dir_control = anchor_controls[np.argmax(scores.flatten())]
    return dir_control


def shooting_routine(old, new):
    movement_duration = 0.05

    def _press_move(center, dir_xy, strength):
        mouse.press()
        time.sleep(movement_duration)
        x_center, y_center = _relative_to_pixel(center, config.main_screen, absolute=True)
        cos, sin = np.array(dir_xy) / np.linalg.norm(dir_xy)
        x = cos * config.joystick_radius * strength + x_center
        y = sin * config.joystick_radius * strength + y_center
        mouse.move(x, y, duration=movement_duration)

    def _reset_joystick(center):
        _press_move(center, (1, 0), 1.0)
        time.sleep(movement_duration)
        _press_move(center, (1, 0), 0.0)
        time.sleep(movement_duration)
        mouse.release()
        time.sleep(movement_duration)

    (make_shot,
     shoot_direction_x,
     shoot_direction_y,
     shoot_strength,
     super_ability,) = list(zip(old, new))

    joystick = config.super_joystick if super_ability[1] else config.regular_joystick
    prev_joystick = config.super_joystick if super_ability[0] else config.regular_joystick

    # if mouse is not clicked or no change in joysticks, do the job
    print(mouse.is_pressed())
    if make_shot[0] or super_ability[0] == super_ability[1]:
        print('case NO CHANGE')
        if make_shot[0]:
            mouse.move(*_relative_to_pixel(joystick, config.main_screen, absolute=True),
                       duration=movement_duration)  # move to init location if released
        _press_move(joystick, (shoot_direction_x[1], shoot_direction_y[1]), shoot_strength[1])
        if make_shot[1]:
            time.sleep(movement_duration)
            mouse.release()
    else:  # if already aiming at different joystick
        print('case RESET')
        _reset_joystick(prev_joystick)
        mouse.move(*_relative_to_pixel(joystick, config.main_screen, absolute=True),
                   duration=movement_duration)  # move to init location
        _press_move(joystick, (shoot_direction_x[1], shoot_direction_y[1]), shoot_strength[1])
        if make_shot[1]:
            time.sleep(movement_duration)
            mouse.release()


def act(
        direction=None,
        make_move=None,
        make_shot=None,
        shoot_direction=None,
        shoot_strength=None,
        super_ability=None,
        use_gadget=None,
        changed=None,
):
    """
    Run an action. Use shared states to change the action during the run. Runs an infinite loop

    :param direction: float (x, y), movement vector, will be normalized
    :param make_move: int bool-like, whether to move
    :param make_shot: int bool-like, whether to release the shoot button
    :param shoot_direction: float (x, y), aiming vector, will be normalized
    :param shoot_strength: float [0,1], how far to throw (for throwing abilities)
    :param super_ability: int bool-like, whether to use the super ability
    :param use_gadget: int bool-like, whether to use a gadget

    :param changed: int bool-like. if params were updated

    :return:
    """

    # TODO implement mouse radial movement (based on difference
    #  in shoot_directions), clone for each of the shooting modes
    # TODO calculate shoot_strength => pixel distance. Add to config?
    # TODO make super <=> not super mode change via proper control release (spot if changed)

    # how to aim inter-frame?
    # make shot inter-frame?
    # use throwing only when specified&
    # multi-shot?

    # init act.vars
    def init():
        store_old = (act.make_shot_local,
                     act.shoot_direction_x_local,
                     act.shoot_direction_y_local,
                     act.shoot_strength_local,
                     act.super_ability_local,)
        (
            act.direction_x_local,
            act.direction_y_local,
            act.make_move_local,

            act.make_shot_local,
            act.shoot_direction_x_local,
            act.shoot_direction_y_local,
            act.shoot_strength_local,
            act.super_ability_local,

            act.use_gadget_local
        ) = _make_args_local_copy()

        if not act.make_move_local:
            stop_movement()
            movements_ = ()
        else:
            movements_ = generate_dir((act.direction_x_local, act.direction_y_local))

        if act.use_gadget_local:
            PressKey(config.gadget)
            ReleaseKey(config.gadget)

        shooting_routine(store_old, (act.make_shot_local,
                                     act.shoot_direction_x_local,
                                     act.shoot_direction_y_local,
                                     act.shoot_strength_local,
                                     act.super_ability_local,))

        return movements_

    def _make_args_local_copy():
        direction_x_local = float(direction.x)
        direction_y_local = float(direction.y)
        make_move_local = int(make_move.value)

        make_shot_local = int(make_shot.value)
        shoot_direction_x_local = float(shoot_direction.x)
        shoot_direction_y_local = float(shoot_direction.y)
        shoot_strength_local = float(shoot_strength.value)
        super_ability_local = int(super_ability.value)

        use_gadget_local = int(use_gadget.value)
        return (
            direction_x_local,
            direction_y_local,
            make_move_local,

            make_shot_local,
            shoot_direction_x_local,
            shoot_direction_y_local,
            shoot_strength_local,
            super_ability_local,

            use_gadget_local
        )

    (
        act.direction_x_local,
        act.direction_y_local,
        act.make_move_local,

        act.make_shot_local,
        act.shoot_direction_x_local,
        act.shoot_direction_y_local,
        act.shoot_strength_local,
        act.super_ability_local,

        act.use_gadget_local
    ) = _make_args_local_copy()
    movement_controls = init()
    while True:
        # inside loop
        if changed.value:
            with changed.get_lock():
                changed.value = 0
            movement_controls = init()

        # if np.random.uniform() > 0.9997: print('im here', len(movement_controls))
        for func in movement_controls:
            func()
            time.sleep(0.1)
