import time
from collections import defaultdict

import numpy as np
import torch
from torch.distributions.categorical import Categorical
from torch.distributions.normal import Normal
import hydra
from omegaconf import DictConfig
import wandb

from src.trainer import Trainer
from src.utils import compute_lambda_returns, LossWithIntermediateLosses
from src.models.actor_critic import ActorCriticOutput, ImagineOutput, ActorCritic
from environment import make_env, RELOAD_MACRO, GymEnv
from utils.misc import create_token

MAX_EPISODES = 1000


def limit_speed(speed_constraint=2):
    def decorator(func):
        def inner(*args, **kwargs):
            start = time.time()
            out = func(*args, **kwargs)
            extra_time = (1 / speed_constraint) - (time.time() - start) if speed_constraint is not None else 0.0
            if extra_time > 0:
                time.sleep(extra_time)
            else:
                print(f"off rate by {-extra_time}")
            return out

        return inner

    return decorator


def ac_loss(actor: ActorCritic, outputs: ImagineOutput, entropy_weight=0.001):
    with torch.no_grad():
        lambda_returns = compute_lambda_returns(
            rewards=outputs.rewards,
            values=outputs.values,
            ends=outputs.ends,
            gamma=0.995,
            lambda_=0.95,
        )[:, :-1]
    values = outputs.values[:, :-1]

    to_print = torch.cat([values, lambda_returns], dim=0).T
    print("values and returns\n", to_print.detach().numpy())

    (log_probs, entropy), (log_probs_continuous, entropy_cont) = actor.get_proba_entropy(outputs)

    loss_actions = -1 * (
            log_probs * (lambda_returns - values.detach()).unsqueeze(-1)).mean()

    loss_continuous_actions = -1 * (
            log_probs_continuous * (lambda_returns - values.detach()).unsqueeze(-1)).mean()

    loss_entropy = - entropy_weight * entropy.mean()
    loss_entropy_continuous = - entropy_weight * entropy_cont.mean()
    loss_values = torch.nn.functional.mse_loss(values, lambda_returns)

    return LossWithIntermediateLosses(loss_actions=loss_actions,
                                      loss_continuous_actions=loss_continuous_actions,
                                      loss_values=loss_values,
                                      loss_entropy=loss_entropy,
                                      loss_entropy_continuous=loss_entropy_continuous)


class ACTrainer:
    def __init__(self, env: GymEnv, actor: ActorCritic, optimizer):
        self.env = env
        self.actor = actor
        self.optimizer = optimizer

        self.actions, self.rewards, self.dones, self.outputs = [], [], [], []
        self.observations = []
        self.metrics = defaultdict(float)

        self._step = 0

    def reset(self):
        self.metrics = defaultdict(float)
        self._step = 0

        self.actions, self.rewards, self.dones, self.outputs = [], [], [], []
        self.observations = [self.env.reset()]
        self.actor.reset(1)
        self.actor.train()

    def _update_metrics(self, metrics: dict):
        for k, v in metrics.items():
            name = f"actor_critic/train/{k}"
            self.metrics[name] = v

    @limit_speed(speed_constraint=2)
    def step(self):
        self.optimizer.zero_grad()

        # sample action from s_t
        output = self.actor.forward(torch.tensor(self.observations[-1].transpose(2, 0, 1)).unsqueeze(0).float() / 255.0)

        action, action_cont = self.actor.sample_actions(output, eps=0.05)
        action_raw = torch.cat([action.flatten(), action_cont.flatten()]).unsqueeze(0)
        action_token = create_token(action.int().numpy().flatten(),
                                    anchors=self.env.move_shot_anchors)
        action_sigmoid = torch.cat([torch.tensor([action_token]),
                                    torch.sigmoid(action_cont).flatten()])
        self.actions.append(action_raw)
        self.outputs.append(output)

        obs, reward, done, _ = self.env.step(action_sigmoid.detach())

        self.observations.append(obs)
        self.rewards.append(torch.tensor([reward]) / 4.0)
        self.dones.append(torch.tensor([done]))

        self._step += 1
        return self.metrics

    def episode_end(self):
        # update value estimation
        self.rewards[-2] = self.rewards[-1]
        episode_output = ImagineOutput(
            observations=None,
            actions=torch.stack(self.actions, dim=1)[..., :-3],
            actions_continuous=torch.stack(self.actions, dim=1)[..., -3:],
            logits_actions=torch.cat([out.logits_actions for out in self.outputs], dim=1),
            continuous_means=torch.cat([out.mean_continuous for out in self.outputs], dim=1),
            continuous_stds=torch.cat([out.std_continuous for out in self.outputs], dim=1),
            values=torch.stack([out.means_values for out in self.outputs], dim=1).reshape(1, -1),
            rewards=torch.stack(self.rewards).reshape(1, -1),
            ends=torch.stack(self.dones).reshape(1, -1)
        )

        self.optimizer.zero_grad()
        loss = ac_loss(self.actor, episode_output)
        loss.loss_total.backward()
        self.optimizer.step()
        self._update_metrics(loss.intermediate_losses)
        self.metrics["actor_critic/train/total_loss"] = sum(self.metrics.values())
        self.metrics["reward"] = self.rewards[-2]
        self.metrics["episode_length"] = len(self.dones)
        return self.metrics


@hydra.main(config_path=r"C:\Users\Mykhailo_Tkachuk\PycharmProjects\Brawl_iris\config", config_name="trainer")
def main(cfg: DictConfig):
    trainer = Trainer(cfg)

    env = trainer.train_collector.env.env
    actor = trainer.agent.actor_critic
    optimizer = trainer.optimizer_actor_critic
    ac_trainer = ACTrainer(env, actor, optimizer)

    for n_episode in range(MAX_EPISODES):
        ac_trainer.reset()
        skip_update = False
        while not env.done:
            step_metrics = ac_trainer.step()
            if ac_trainer._step % 5 == 0:
                print(f"step value {ac_trainer.outputs[-1].means_values.item()}")
            if len(ac_trainer.observations) > 20 and \
                    not np.count_nonzero(ac_trainer.observations[-20] - ac_trainer.observations[-1]):
                print("Train loop: env froze, attempting reload")
                skip_update = True
                break  # if froze inside episode

        if skip_update:
            RELOAD_MACRO.play()
        else:
            episode_metrics = ac_trainer.episode_end()
            wandb.log({"epoch": n_episode, **episode_metrics})
            trainer.save_checkpoint(n_episode, True)


if __name__ == "__main__":
    main()
