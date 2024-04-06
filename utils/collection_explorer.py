import tkinter as tk

import numpy as np
import torch

from utils.misc import create_token


class Main:
    def __init__(self, master: tk.Tk):
        self.master = master
        self.frame_master = tk.Frame(self.master)
        self.frame_master.grid(row=0, column=0)

        self.stats_frame = tk.Frame(self.frame_master)
        self.stats_frame.grid(row=0, column=0)
        self.labels = ["Value: ", "Move: ", "move_dir: ", "Shoot: ", "shot_dir: ", "Strength: ", "Super: ", "Gadget: "]
        self.step_variables = [tk.StringVar() for _ in self.labels]
        for i, (l, v) in enumerate(zip(self.labels, self.step_variables)):
            tk.Label(self.stats_frame, text=l).grid(row=i, column=0)
            tk.Label(self.stats_frame, textvariable=v).grid(row=i, column=1)

    @torch.no_grad()
    def update_stats(self, env, ac_trainer):
        self.step_variables[0].set(f"{ac_trainer.outputs[-1].means_values[0].item():.3f}")
        action = ac_trainer.actions[-1][0].cpu().detach()
        action_logits = ac_trainer.outputs[-1].logits_actions[0].flatten().detach().cpu().numpy()
        action_token = create_token(action[:-3].int().numpy().flatten(),
                                    anchors=env.move_shot_anchors)
        parsed = env._parse_action_token([action_token] + torch.sigmoid(action[-3:]).numpy().tolist())

        def _get_angle(vec):
            angle = np.arccos(vec[0]) * np.sign(vec[1])
            angle *= 180 / np.pi
            if angle < 0:
                angle = 360 + angle
            return float(angle)

        def _sigmoid(x):
            return float(1 / (1 + np.exp(-x)))

        self.step_variables[1].set(f"{parsed['make_move']} | {_sigmoid(action_logits[0]):.2f} | {action_logits[0]:.2f}")
        self.step_variables[2].set(f"{_get_angle(parsed['direction']):.2f}")
        self.step_variables[3].set(f"{parsed['make_shot']} | {_sigmoid(action_logits[1]):.2f} | {action_logits[1]:.2f}")
        self.step_variables[4].set(f"{_get_angle(parsed['shoot_direction']):.2f}")
        self.step_variables[5].set(f'{parsed["shoot_strength"]:.2f}')
        self.step_variables[6].set(f'{parsed["super_ability"]} | {_sigmoid(action_logits[2]):.2f} | {action_logits[2]:.2f}')
        self.step_variables[7].set(f'{parsed["use_gadget"]} | {_sigmoid(action_logits[3]):.2f} | {action_logits[3]:.2f}')

        self.master.update_idletasks()


def run_explorer(hook_callback):
    root = tk.Tk()
    root.geometry("+1500+0")
    app = Main(root)
    hook_callback(app)
    root.mainloop()
