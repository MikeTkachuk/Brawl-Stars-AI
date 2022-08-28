import time

from utils.directkeys import PressKey, ReleaseKey
import config
import numpy as np


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


def act(
        direction=None,
        make_move=None,
        make_shot=None,
        shoot_direction=None,
        super_ability=None,
        use_gadget=None
):
    """
    Run an action until rerun with new
    :param direction:
    :param make_move:
    :param make_shot:
    :param shoot_direction:
    :param super_ability:
    :param use_gadget:
    :return:
    """

    # how to aim inter-frame?
    # make shot inter-frame?

    # init act.vars
    def init():
        pulled = _make_args_local_copy()
        act.direction_x_local = pulled[0]
        act.direction_y_local = pulled[1]
        act.make_move_local = pulled[2]

        if not act.make_move_local:
            stop_movement()
            movements_ = ()
        else:
            movements_ = generate_dir((act.direction_x_local, act.direction_y_local))

        return movements_

    def _make_args_local_copy():
        direction_x_local = float(direction.x)
        direction_y_local = float(direction.y)
        make_move_local = int(make_move.value)
        return (
            direction_x_local,
            direction_y_local,
            make_move_local,
        )

    def _spot_change():
        pulled = _make_args_local_copy()
        changed_ = [a != b for a, b in
                    zip([act.direction_x_local, act.direction_y_local, act.make_move_local],
                        pulled)
                    ]
        if any(changed_):
            act.direction_x_local = pulled[0]
            act.direction_y_local = pulled[1]
            act.make_move_local = pulled[2]
            return True
        else:
            return False

    movement_controls = init()
    while True:
        # inside loop
        changed = _spot_change()
        if changed:
            movement_controls = init()

        print('im here', len(movement_controls))
        for func in movement_controls:
            func()
            time.sleep(0.15)
