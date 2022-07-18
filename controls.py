from utils.directkeys import PressKey, ReleaseKey
import config


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


def act(
        direction=None,
        make_move=True,
        make_shot=True,
        shoot_vector=None,
        super_ability=False,
        use_gadget=False
        ):
    # how to aim inter-frame?
    pass


def start_battle():
    pass


def exit_end_screen():
    pass
