from __future__ import annotations

import ctypes
import json
import time
from pathlib import Path
from typing import Union, List

import mouse
import keyboard

ctypes.windll.shcore.SetProcessDpiAwareness(2)  # https://github.com/boppreh/mouse/issues/122


def parse_event(to_parse) -> dict:
    if isinstance(to_parse, keyboard.KeyboardEvent):
        return {
            'type': 'keyboard',
            'name': to_parse.name,
            'action': to_parse.event_type,
            'time': to_parse.time
        }
    elif isinstance(to_parse, mouse.MoveEvent):
        return {
            'type': 'mouse_move',
            'x': to_parse.x,
            'y': to_parse.y,
            'time': to_parse.time
        }
    elif isinstance(to_parse, mouse.ButtonEvent):
        return {
            'type': 'mouse_press',
            'button': to_parse.button,
            'action': to_parse.event_type,
            'time': to_parse.time
        }
    elif isinstance(to_parse, mouse.WheelEvent):
        return {
            'type': 'mouse_scroll',
            'delta': to_parse.delta,
            'time': to_parse.time
        }
    else:
        raise NotImplementedError


def play_event(event):
    if event['type'] == 'keyboard':
        if event['action'] == 'up':
            keyboard.release(event['name'])
        elif event['action'] == 'down':
            keyboard.press(event['name'])
        else:
            raise NotImplementedError
    elif event['type'] == 'mouse_move':
        mouse.move(event['x'], event['y'], absolute=True)
    elif event['type'] == 'mouse_press':
        if event['action'] == 'double':
            mouse.double_click(event['button'])
        elif event['action'] == 'up':
            mouse.release(event['button'])
        elif event['action'] == 'down':
            mouse.press(event['button'])
        else:
            raise NotImplementedError
    elif event['type'] == 'mouse_scroll':
        mouse.wheel(event['delta'])
    elif event['type'] == 'sleep':
        time.sleep(event['interval'])
    else:
        raise NotImplementedError


class Macro:
    def __init__(self, name: str = None, load_path=None):
        self.events: List[dict] = []
        self.name = name

        if load_path is not None:
            self.load(load_path)

        self._mouse_hook = None
        self._keyboard_hook = None

    def get_event_parser(self, mode='none'):
        """

        :param mode: str in ['none', 'simple']. Simple mode omits all extra movement
        :return:
        """
        def mode_none(event):
            if not len(self.events):
                self.events.append(parse_event(mouse.MoveEvent(*mouse.get_position(), event.time)))
                self.events.append(parse_event(event))
            else:
                self.events.append(parse_event(event))

        if mode == 'none':
            return mode_none
        else:
            raise NotImplementedError

    def add_hooks(self):
        self._mouse_hook = mouse.hook(self.get_event_parser())
        self._keyboard_hook = keyboard.hook(self.get_event_parser())

    def start_recording(self, out_dir=None):
        self.add_hooks()
        print("macro.start_recording: Started recording")
        keyboard.wait('ctrl+q', suppress=True)
        self.stop_recording(out_dir=out_dir)

    def stop_recording(self, out_dir=None):
        mouse.unhook(self._mouse_hook)
        keyboard.unhook(self._keyboard_hook)
        self.events.pop(-1)  # remove alt press

        if out_dir is None:
            out_dir = Path('./')
        out_dir.mkdir(parents=True, exist_ok=True)
        self.dump(out_dir)

    def dump(self, out_dir):
        with open(Path(out_dir) / f"{self.name}.json", 'w') as out_file:
            json.dump(self.events, out_file)

    def load(self, load_path):
        with open(load_path) as l_file:
            self.events = json.load(l_file)
        self.name = Path(load_path).stem

    def play(self):
        print(f"Macro.play: playing {self.name}")
        for i in range(len(self.events) - 1):
            play_event(self.events[i])
            time.sleep(self.events[i + 1]['time'] - self.events[i]['time'])
        play_event(self.events[-1])
        print(f"Macro.play: ended {self.name}")


if __name__ == "__main__":
    time.sleep(2)
    m = Macro(name="reload_brawlstars2")
    m.start_recording()
    m.play()
