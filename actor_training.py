import os
import shutil
import sys
import time
from collections import defaultdict
from pathlib import Path
from typing import Optional
from multiprocessing import Process, Event, Pipe
from threading import Thread

import numpy as np
import torch
import tqdm
import hydra
from omegaconf import DictConfig
import wandb
import matplotlib.pyplot as plt

plt.switch_backend("AGG")

from src.dataset import EpisodesDataset
from src.episode import Episode
from src.trainer import Trainer
from src.utils import compute_lambda_returns, LossWithIntermediateLosses, adaptive_gradient_clipping
from src.models.actor_critic import ActorCriticOutput, ImagineOutput, ActorCritic
from environment import make_env, RELOAD_MACRO, GymEnv
from utils.misc import create_token

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

MAX_EPISODES = 1000
WEIGHT_QUANTILE = 0.5  # value loss quantile to assign episode weights
RANDOM_ACTION = 0.01  # proba
POSITIVE_WEIGHT = 1.0  # weigh positive updates (divides neg ones)

# align weights so that positive episode with loss 0.1 was on par with the
# negative one via loss + neg_ep_w_shift
NEGATIVE_EPISODE_WEIGHT_SHIFT = 0.0  # TODO maybe increase for exploration

# set action_limit_mask to False and the desired actions to lock them during the run.

# make_move, make_shot, super_ability, use_gadget, move_anchor, shot_anchor
ACTION_NAMES = ["make_move", "make_shot", "super_ability", "use_gadget", "move_anchor", "shot_anchor"]
ACTION_LOCK_MASK = torch.tensor([True, True, True, False, True, False]).to(device)
ACTION_LOGIT_LOCK_MASK = torch.repeat_interleave(ACTION_LOCK_MASK, torch.tensor([1, 1, 1, 1, 4, 4]).to(device))
ACTION_LOCK = torch.tensor([0.0, 0.0, 0.0, 0.0, 0.0, 0.0]).to(device)
# move_shift, shot_shift, shot_strength
ACTION_C_NAMES = ["move_shift", "shot_shift", "shot_strength"]
ACTION_C_LOCK_MASK = torch.tensor([True, False, False]).to(device)
ACTION_C_LOCK = torch.tensor([0.0, 0.0, -10.0]).to(device)


def limit_speed(speed_constraint=2):
    def decorator(func):
        def inner(*args, **kwargs):
            start = time.time()
            out = func(*args, **kwargs)
            extra_time = (1 / speed_constraint) - (time.time() - start) if speed_constraint is not None else 0.0
            if extra_time > 0:
                time.sleep(extra_time)
            else:
                if np.random.random() < 0.2:
                    print(f"off rate by {-extra_time}")
                pass
            return out

        return inner

    return decorator


def compute_masked_lambda_returns(rewards,
                                  values,
                                  ends,
                                  mask_paddings,
                                  gamma=0.995,
                                  lambda_=0.95,
                                  ):
    lambda_returns = torch.zeros_like(values)
    for b in range(rewards.size(0)):
        rewards_masked = rewards[b][mask_paddings[b]]
        values_masked = values[b][mask_paddings[b]]
        ends_masked = ends[b][mask_paddings[b]]
        if mask_paddings[b].count_nonzero():
            lambda_returns[b][mask_paddings[b]] = compute_lambda_returns(rewards_masked[None, ...],
                                                                         values_masked[None, ...],
                                                                         ends_masked[None, ...],
                                                                         gamma,
                                                                         lambda_)
    return lambda_returns


def warmup_lr_lambda(epoch, init_lr_scale, num_warmup_steps):
    if epoch < num_warmup_steps:
        return init_lr_scale + (1 - init_lr_scale) * (epoch / num_warmup_steps)
    return 1.0


def lambda_return_schedule(epoch, last_epoch=MAX_EPISODES // 2, start_lambda=0.95, last_lambda=0.99):
    start_pow = np.log(1 - start_lambda)
    last_pow = np.log(1 - last_lambda)
    if epoch < last_epoch:
        return 1 - np.exp(start_pow + (last_pow - start_pow) * epoch / last_epoch)
    return last_lambda


def process_collection(event: Event, config, env: GymEnv,
                       actor_critic, dataset: EpisodesDataset,
                       weights_path, log_connection, verbose=10):
    sys.stdin = open(0)  # otherwise input() will fail
    ac_trainer = ACTrainer(config, env, actor_critic, None, dataset)
    while True:
        try:
            if event.is_set():
                # collection
                actor_weights = torch.load(weights_path, map_location=device)
                field_name = "actor_critic."
                actor_weights = {k[len(field_name):]: v for k, v in actor_weights.items() if field_name in k}
                ac_trainer.actor.load_state_dict(actor_weights)

                for _ in range(config.training.actor_critic.collection_steps):
                    ac_trainer.reset(collection=True)
                    skip_update = False
                    while not env.done:
                        step_metrics = ac_trainer.step(collection=True)
                        if verbose and ac_trainer.episode_step % verbose == 0:
                            print(f"step value {ac_trainer.outputs[-1].means_values[0].item()}")
                            print(f"actions: {ac_trainer.actions[-1][0].cpu().detach().numpy()}")
                            print(f"act_logits: {ac_trainer.outputs[-1].logits_actions[0].cpu().detach().numpy()}")
                            print(f"means_stds: {ac_trainer.outputs[-1].mean_continuous[0].cpu().detach().numpy()}"
                                  f" {ac_trainer.outputs[-1].std_continuous[0].cpu().detach().numpy()}")

                        if len(ac_trainer.observations) > 20 and \
                                not np.count_nonzero(ac_trainer.observations[-20] -
                                                     ac_trainer.observations[-1]) or step_metrics['buggy_episode']:
                            print("Train loop: env froze, attempting reload")
                            skip_update = True
                            break  # if froze inside episode

                    if skip_update:
                        env.__exit__(soft=False)
                        RELOAD_MACRO.play()
                    else:
                        if not ac_trainer.actions:
                            print("Empty episode. Skipping")
                            return
                        ac_trainer.episode_end(collection=True, do_step=False)
                        dataset.update_disk_checkpoint(Path("checkpoints/dataset"))
                    log_connection.send(ac_trainer.metrics)

                # reset event flag
                event.clear()
            else:
                time.sleep(1)
        except Exception as e:
            import traceback
            traceback.print_exc()
            time.sleep(1)


class ACTrainer:
    def __init__(self,
                 cfg,
                 env: Optional[GymEnv],
                 actor: ActorCritic,
                 optimizer,
                 dataset: EpisodesDataset,
                 replay_only=False):
        self.cfg = cfg
        self.env = env
        self.actor = actor.to(device)
        self.optimizer = optimizer
        if self.optimizer is not None:
            self.lr_scheduler = torch.optim.lr_scheduler.LambdaLR(
                self.optimizer,
                lambda epoch: warmup_lr_lambda(epoch,
                                               self.cfg.training.actor_critic.lr_warmup_scale,
                                               self.cfg.training.actor_critic.lr_warmup_steps)
            )
        else:
            self.lr_scheduler = None

        self.dataset = dataset
        self.replay_only = replay_only

        self.actions, self.rewards, self.dones, self.outputs, self.mask_paddings = [], [], [], [], []
        self.observations = []
        self.metrics = defaultdict(float)

        self.batch_size = self.cfg.training.actor_critic.batch_num_samples
        self.accumulation_steps = self.cfg.training.actor_critic.grad_acc_steps

        self.episode_step = 0
        self._replay = None
        self._active_episodes = None
        self._update_weights = None

    def get_episode_proba(self, rank=True, alpha=0.7, beta=0.1, zero_center=True, use_age=True):
        if not use_age:
            episode_weights = np.array([self.dataset.episode_weights[i] for i in self.dataset.disk_episodes])
        else:
            rank = True
            episode_weights = np.array(self.dataset.disk_episodes, dtype=np.float32)  # sort last to first

        if rank:
            criterion_rank = np.argsort(episode_weights, )
            criterion = np.zeros_like(episode_weights)
            criterion[criterion_rank] = 1 / (len(episode_weights) - np.arange(len(episode_weights)))
        else:
            criterion = episode_weights + 1E-3

        assert all(criterion > 0.0)

        probas = criterion ** alpha
        probas /= np.sum(probas)
        if zero_center:
            pass  # removed because episodes are now offloaded
            # associated_rewards = np.array([ep.get_last_reward() for ep in self.dataset.episodes])
            # pos_mask = associated_rewards > 0
            # neg_mask = ~pos_mask
            # if np.any(pos_mask) and np.any(neg_mask):
            #     neg_scale = np.sum(associated_rewards[pos_mask] * probas[pos_mask]) / \
            #         np.abs(np.sum(associated_rewards[neg_mask] * probas[neg_mask]))
            #     probas[neg_mask] *= neg_scale
            #     probas /= np.sum(probas)
            # print("new ev: ", sum(associated_rewards * probas))
        update_weights = 1 / (len(self.dataset) * probas) ** beta
        return probas, update_weights

    @torch.no_grad()
    def _assign_episode_weights(self, loss, mask_padding):
        for i, episode in enumerate(self._active_episodes):
            if episode is None:  # if doing replay it will remain None
                continue
            episode_loss = loss[i][mask_padding[i]]  # for scale with what is logged
            old_weight = self.dataset.episode_weights[episode]
            new_weight = torch.quantile(episode_loss, WEIGHT_QUANTILE).detach().item()
            self.dataset.episode_weights[episode] = new_weight
            print(f"Changed weight: {old_weight:.3f} -> {new_weight:.3f}")

    def reset(self, collection=False, prioritize=True, zero_center=True):
        print("ACTrainer.reset: Started reset")
        self.metrics = defaultdict(float)
        self.episode_step = 0
        self.dataset.load_disk_checkpoint(Path("checkpoints/dataset"))

        # sample episodes via prioritized sampling
        if len(self.dataset) >= self.batch_size - 1 and self.batch_size > 1:
            probas, weights = self.get_episode_proba(alpha=0.7 if prioritize else 0.0, zero_center=zero_center)
            episode_ids = np.random.choice(np.arange(len(self.dataset)),
                                           size=(self.batch_size - 1,),
                                           p=probas,
                                           replace=False)  # todo: fix replacement/caching/accumulation
            disk_ids = [self.dataset.disk_episodes[i] for i in episode_ids]
            self._active_episodes = ([None if self.replay_only else self.dataset.num_seen_episodes] +
                                     disk_ids)
            self._update_weights = np.concatenate([np.ones(1), weights[episode_ids]])
            start_ = time.time()
            self._replay = self.dataset.sample_replay(samples=disk_ids)
            self.metrics["batch_mean_reward"] = sum([
                r[torch.nonzero(r)[-1]].item() if torch.count_nonzero(r) else 0.0 for r in self._replay["rewards"]
            ]) / (self.batch_size - 1)
            print(self.metrics)
            print("replay sample elapsed: ", time.time() - start_)
        else:
            self._update_weights = np.ones(self.batch_size)
            self._replay = self._get_placeholder_replay()

        # self._replay = self._get_placeholder_replay(length=250, mask=False)  # debug memory etc

        self.actions, self.rewards, self.dones, self.outputs, self.mask_paddings = [], [], [], [], []
        self.observations = [self.env.reset()] if not self.replay_only else []
        self.actor.reset(1 if collection else self.batch_size)
        self.actor.train()

    def _update_metrics(self, metrics: dict):
        for k, v in metrics.items():
            name = f"actor_critic/train/{k}"
            self.metrics[name] = v

    @limit_speed(speed_constraint=2)
    def step(self, collection=False):
        if collection:
            return self.collection_step()

        curr_obs = torch.tensor(self.observations[-1].transpose(2, 0, 1)).unsqueeze(0).float() / 255.0
        if self.episode_step < self._replay['ends'].size(1):
            replay_obs = self._replay['observations'][:, self.episode_step]
            replay_mask_padding = self._replay['mask_padding'][:, self.episode_step]
            replay_actions = torch.cat([self._replay['actions'][:, self.episode_step],
                                        self._replay['actions_continuous'][:, self.episode_step]], dim=-1)
            replay_rewards = self._replay['rewards'][:, self.episode_step]
            replay_ends = self._replay['ends'][:, self.episode_step]
        else:
            replay_obs = torch.zeros_like(self._replay['observations'][:, -1])
            replay_mask_padding = torch.zeros_like(self._replay['mask_padding'][:, -1])
            replay_actions = torch.cat([torch.zeros_like(self._replay['actions'][:, -1]),
                                        torch.zeros_like(self._replay['actions_continuous'][:, -1])], dim=-1)
            replay_rewards = torch.zeros_like(self._replay['rewards'][:, -1])
            replay_ends = torch.ones_like(self._replay['ends'][:, -1])

        full_obs = torch.cat([curr_obs, replay_obs], dim=0)
        full_mask_padding = torch.cat([torch.tensor([True]), replay_mask_padding], dim=0)
        self.mask_paddings.append(full_mask_padding)
        output = self.actor.forward(full_obs.to(device),
                                    mask_padding=full_mask_padding.to(device))

        # sample action from s_t
        action, action_cont = self.actor.sample_actions(output, eps=RANDOM_ACTION)
        action = torch.where(ACTION_LOCK_MASK, action[0], ACTION_LOCK)
        action_cont = torch.where(ACTION_C_LOCK_MASK, action_cont[0], ACTION_C_LOCK)
        action_raw = torch.cat([action.flatten(), action_cont.flatten()]).reshape(1, -1)
        action_token = create_token(action.int().cpu().numpy().flatten(),
                                    anchors=self.env.move_shot_anchors)
        action_sigmoid = torch.cat([torch.tensor([action_token]).to(device),
                                    torch.sigmoid(action_cont).flatten()])
        full_action = torch.cat([action_raw, replay_actions.to(device)], dim=0)
        self.actions.append(full_action)
        self.outputs.append(output)

        obs, reward, done, _ = self.env.step(action_sigmoid.cpu().detach())

        self.observations.append(obs)
        full_rewards = torch.cat([torch.tensor([reward / 8.0]), replay_rewards], dim=0)
        self.rewards.append(full_rewards)
        full_ends = torch.cat([torch.tensor([done]), replay_ends], dim=0)
        self.dones.append(full_ends)

        self.episode_step += 1
        if reward == -100 or self.episode_step > 400:
            self.metrics['buggy_episode'] = 1.0
        else:
            self.metrics['buggy_episode'] = 0.0
        return self.metrics

    @torch.no_grad()
    @limit_speed(speed_constraint=2)
    def collection_step(self):
        curr_obs = torch.tensor(self.observations[-1].transpose(2, 0, 1)).unsqueeze(0).float() / 255.0
        full_mask_padding = torch.tensor([True])
        self.mask_paddings.append(full_mask_padding)
        output = self.actor.forward(curr_obs.to(device),
                                    mask_padding=full_mask_padding.to(device))

        # sample action from s_t
        action, action_cont = self.actor.sample_actions(output, eps=RANDOM_ACTION)
        # lock actions
        action = torch.where(ACTION_LOCK_MASK, action[0], ACTION_LOCK)
        action_cont = torch.where(ACTION_C_LOCK_MASK, action_cont[0], ACTION_C_LOCK)
        # compute token and sigmoid actions
        action_raw = torch.cat([action.flatten(), action_cont.flatten()]).reshape(1, -1)
        action_token = create_token(action.int().cpu().numpy().flatten(),
                                    anchors=self.env.move_shot_anchors)
        action_sigmoid = torch.cat([torch.tensor([action_token]).to(device),
                                    torch.sigmoid(action_cont).flatten()])

        self.actions.append(action_raw)
        self.outputs.append(output)

        obs, reward, done, _ = self.env.step(action_sigmoid.cpu().detach())

        self.observations.append(obs)
        full_rewards = torch.tensor([reward / 8.0])
        self.rewards.append(full_rewards)
        full_ends = torch.tensor([done])
        self.dones.append(full_ends)

        self.episode_step += 1
        if reward == -100 or self.episode_step > 400:
            self.metrics['buggy_episode'] = 1.0
        else:
            self.metrics['buggy_episode'] = 0.0
        return self.metrics

    @torch.no_grad()
    def _create_episode(self):
        obs = np.stack(self.observations, axis=0)
        obs = torch.ByteTensor(obs).permute(0, 3, 1, 2).contiguous()
        actions = torch.LongTensor(
            torch.stack(self.actions, dim=1)[0, :self.episode_step, ..., :-3].long().cpu())
        actions_continuous = torch.FloatTensor(
            torch.stack(self.actions, dim=1)[0, :self.episode_step, ..., -3:].cpu())
        rewards = torch.FloatTensor(torch.stack(self.rewards, dim=1)[0, :self.episode_step].float())
        ends = torch.LongTensor(torch.stack(self.dones, dim=1)[0, :self.episode_step].long())
        mask_padding = torch.BoolTensor(torch.ones(self.episode_step, dtype=torch.bool))

        episode = Episode(
            observations=obs,
            actions=actions,
            actions_continuous=actions_continuous,
            rewards=rewards,
            ends=ends,
            mask_padding=mask_padding,
            reward=rewards[-1]
        )
        return episode

    def _get_placeholder_replay(self, length=1, mask=True):
        return {
            'observations': torch.zeros((self.batch_size - 1, length, 3, 192, 192)),
            'actions': torch.zeros((self.batch_size - 1, length, 6)),
            'actions_continuous': torch.zeros((self.batch_size - 1, length, 3)),
            'rewards': torch.zeros((self.batch_size - 1, length)),
            'ends': torch.ones((self.batch_size - 1, length), dtype=torch.bool) if mask else
            torch.zeros((self.batch_size - 1, length), dtype=torch.bool),
            'mask_padding': torch.zeros((self.batch_size - 1, length), dtype=torch.bool) if mask else
            torch.ones((self.batch_size - 1, length), dtype=torch.bool)
        }

    @staticmethod
    def _create_hist(data_, bins=32, names=None, mean_line=False):
        if isinstance(data_, torch.Tensor):
            data_ = data_.detach().cpu().numpy()
        if len(data_.shape) == 1:
            data_ = data_.reshape(-1, 1)
        else:
            data_ = data_.reshape(-1, data_.shape[-1])
        f, ax = plt.subplots()

        ax: plt.Subplot
        for i in range(data_.shape[-1]):
            ax.hist(data_[:, i], bins=bins, alpha=0.7, label=str(i) if names is None else names[i])
            if mean_line:
                ax.axvline(data_[:, i].mean(), color='k', linestyle='dashed', linewidth=1)
        ax.legend()
        img = wandb.Image(f)
        plt.close("all")
        return img

    def episode_end(self, epoch=None, collection=False, do_step=True):
        # finish replaying if needed
        if self._replay['ends'].size(1) > self.episode_step and not collection:
            start_catchup = time.time()
            for i in range(self.episode_step, self._replay['ends'].size(1)):
                full_mask_padding = torch.cat([torch.tensor([False]), self._replay['mask_padding'][:, i]], dim=0)
                self.mask_paddings.append(full_mask_padding)

                obs = torch.cat([torch.zeros_like(self._replay['observations'][[0], 0]),
                                 self._replay['observations'][:, i]], dim=0)
                output = self.actor.forward(obs.to(device),
                                            mask_padding=full_mask_padding.to(device))
                self.outputs.append(output)
                replay_actions = torch.cat([self._replay['actions'][:, i],
                                            self._replay['actions_continuous'][:, i]], dim=-1).to(device)
                full_actions = torch.cat([torch.zeros_like(replay_actions)[[0]], replay_actions])
                self.actions.append(full_actions)
                full_rewards = torch.cat([torch.tensor([0.0]), self._replay['rewards'][:, i]], dim=0)
                self.rewards.append(full_rewards)
                full_ends = torch.cat([torch.tensor([True]), self._replay['ends'][:, i]], dim=0)
                self.dones.append(full_ends)
            print("replay catch-up elapsed: ", time.time() - start_catchup)

        if not self.replay_only:
            # try to learn neg reward
            true_reward = self.rewards[self.episode_step - 1][0].item()
            self.rewards[self.episode_step - 1][0] = true_reward if true_reward > 0.375 else (
                                                                                                     true_reward - 0.375) / 1.375  # positive starts at rank 3
            # update value estimation
            self.rewards[self.episode_step - 3][0] = self.rewards[self.episode_step - 1][
                0]  # move reward closer bc of lambda returns

            self.metrics["reward"] = self.rewards[self.episode_step - 3][0]
            self.metrics["true_reward"] = true_reward
            self.metrics["episode_length"] = self.episode_step

            # pop trailing obs
            self.observations.pop(-1)
            self.dataset.add_episode(
                self._create_episode()
            )
        actual_batch_size = len(self.dones[-1])
        episode_output = ImagineOutput(
            observations=None,
            actions=torch.stack(self.actions, dim=1)[..., :-3].to(device),
            actions_continuous=torch.stack(self.actions, dim=1)[..., -3:].to(device),
            logits_actions=torch.cat([out.logits_actions for out in self.outputs], dim=1).to(device),
            continuous_means=torch.cat([out.mean_continuous for out in self.outputs], dim=1).to(device),
            continuous_stds=torch.cat([out.std_continuous for out in self.outputs], dim=1).to(device),
            values=torch.stack([out.means_values for out in self.outputs], dim=1).reshape(actual_batch_size, -1).to(
                device),
            rewards=torch.stack(self.rewards, dim=1).reshape(actual_batch_size, -1).to(device),
            ends=torch.stack(self.dones, dim=1).reshape(actual_batch_size, -1).to(device)
        )
        mask_paddings = torch.stack(self.mask_paddings, dim=1).to(device)
        if collection:
            # log action and value histograms
            self.metrics["values"] = self._create_hist(episode_output.values[mask_paddings], names=["value"],
                                                       mean_line=True)
            self.metrics["binary_logits"] = self._create_hist(
                episode_output.logits_actions[mask_paddings][..., :4], names=ACTION_NAMES[:4])
            self.metrics["anchor_logits"] = self._create_hist(torch.stack([
                episode_output.logits_actions[mask_paddings][..., 4:8],
                episode_output.logits_actions[mask_paddings][..., 8:],
            ], dim=-1), names=ACTION_NAMES[4:])
            self.metrics["means"] = self._create_hist(episode_output.continuous_means[mask_paddings],
                                                      names=ACTION_C_NAMES)
            self.metrics["stds"] = self._create_hist(episode_output.continuous_stds[mask_paddings],
                                                     names=ACTION_C_NAMES)
            return self.metrics

        self.metrics["actor_critic/train/lr"] = self.lr_scheduler.get_last_lr()[0]

        loss = self.ac_loss(episode_output, mask_paddings, epoch=epoch)
        print("ACTrainer.episode_end: backward started")
        start_backward = time.time()
        loss.loss_total.backward()
        if do_step:
            adaptive_gradient_clipping(self.actor.parameters(), lam=self.cfg.training.actor_critic.agc_lambda)
            torch.nn.utils.clip_grad_norm_(self.actor.parameters(),
                                           self.cfg.training.actor_critic.max_grad_norm,
                                           error_if_nonfinite=True)
            self.optimizer.step()
            self.lr_scheduler.step()
            self.optimizer.zero_grad()
        loss = loss / (1 / self.accumulation_steps)
        print("backward update elapsed: ", time.time() - start_backward)
        self._update_metrics(loss.intermediate_losses)
        self.metrics["actor_critic/train/total_loss"] = loss.loss_total.item()
        return self.metrics

    def ac_loss(self, outputs: ImagineOutput, mask_paddings, epoch=None):
        mask_paddings = torch.logical_and(mask_paddings, outputs.ends.logical_not())  # do not include end into loss
        values = outputs.values
        values_std = torch.masked_select(values, mask_paddings).std()
        values_mean = torch.masked_select(values, mask_paddings).mean()
        self.metrics["value_mean"] = values_mean.item()
        self.metrics["value_std"] = values_std.item()
        values = (values - values_mean) / values_std
        if epoch is None or not self.cfg.training.actor_critic.lambda_warmup_init:
            lambda_ = self.cfg.training.actor_critic.lambda_
        else:
            lambda_ = lambda_return_schedule(epoch, start_lambda=self.cfg.training.actor_critic.lambda_warmup_init,
                                             last_lambda=self.cfg.training.actor_critic.lambda_)
        # self.metrics["actor_critic/train/lambda_"] = lambda_
        with torch.no_grad():
            lambda_returns = compute_masked_lambda_returns(
                rewards=outputs.rewards,
                values=values,
                ends=outputs.ends,
                mask_paddings=mask_paddings,
                gamma=self.cfg.training.actor_critic.gamma,
                lambda_=lambda_,
            )

        to_print = torch.stack([values[0], lambda_returns[0]], dim=0).T[mask_paddings[0]]
        print("values and returns\n", to_print.cpu().detach().numpy())

        (log_probs, entropy), (log_probs_continuous, entropy_cont) = self.actor.get_proba_entropy(outputs)

        update_weight = torch.tensor(self._update_weights, device=device).reshape(self.batch_size, 1)
        advantage_factor = lambda_returns - values.detach()
        # self.metrics["advantages"] = self._create_hist(advantage_factor[mask_paddings], mean_line=True)
        # self.metrics["weighted_advantages"] = self._create_hist((update_weight * advantage_factor)[mask_paddings],
        #                                                         mean_line=True)
        self.metrics["advantage_mean"] = advantage_factor[mask_paddings].mean().item()
        advantage_factor[advantage_factor < 0] /= POSITIVE_WEIGHT

        # compute losses  # todo: log -ln(x) | -ln(1-x) where adv > 0
        loss_actions = -1 * (log_probs * advantage_factor.unsqueeze(-1))[..., ACTION_LOCK_MASK]
        loss_actions_masked = torch.masked_select(update_weight.unsqueeze(-1) * loss_actions,
                                                  mask_paddings.unsqueeze(-1)).mean()

        loss_continuous_actions = -1 * (log_probs_continuous.clamp(-5, 5) * advantage_factor.unsqueeze(-1))[
            ..., ACTION_C_LOCK_MASK]
        loss_continuous_actions_masked = torch.masked_select(update_weight.unsqueeze(-1) * loss_continuous_actions,
                                                             mask_paddings.unsqueeze(-1)).mean()

        loss_entropy = torch.masked_select(
            - self.cfg.training.actor_critic.entropy_weight * torch.log(entropy[..., ACTION_LOCK_MASK]),
            mask_paddings.unsqueeze(-1)).mean() + \
                       torch.masked_select(
                           # logit l2 regularization loss. cat(logits=[10,10,-10,-10]).entropy() is 0.69!!
                           self.cfg.training.actor_critic.entropy_weight * outputs.logits_actions[
                               ..., ACTION_LOGIT_LOCK_MASK] ** 2,
                           mask_paddings.unsqueeze(-1)
                       ).mean()
        loss_entropy_continuous = torch.masked_select(
            - self.cfg.training.actor_critic.entropy_continuous_weight * entropy_cont[..., ACTION_C_LOCK_MASK],
            mask_paddings.unsqueeze(-1)).mean()
        loss_values = torch.square(values - lambda_returns)
        loss_values_masked = torch.masked_select(update_weight * loss_values, mask_paddings).mean()

        full_loss = LossWithIntermediateLosses(loss_actions=loss_actions_masked,
                                               loss_continuous_actions=loss_continuous_actions_masked,
                                               loss_values=loss_values_masked,
                                               loss_entropy=loss_entropy,
                                               loss_entropy_continuous=loss_entropy_continuous)
        full_loss = full_loss / self.accumulation_steps

        # episode weight ~ absolute loss per episode
        loss_actions = loss_actions.detach()
        loss_actions[loss_actions < 0] = loss_actions[loss_actions < 0] + NEGATIVE_EPISODE_WEIGHT_SHIFT
        loss_continuous_actions = loss_continuous_actions.detach()
        loss_continuous_actions[loss_continuous_actions < 0] = loss_continuous_actions[
                                                                   loss_continuous_actions < 0
                                                                   ] + NEGATIVE_EPISODE_WEIGHT_SHIFT

        self._assign_episode_weights(loss_values + loss_actions.mean(dim=-1) + loss_continuous_actions.mean(dim=-1),
                                     mask_paddings)
        return full_loss


@hydra.main(config_path=r"C:\Users\Michael\PycharmProjects\Brawl_iris\config", config_name="trainer")
def main(cfg: DictConfig):
    trainer = Trainer(cfg)

    shutil.copytree(
        Path(
            r"C:\Users\Michael\PycharmProjects\Brawl-Stars-AI\outputs\agc_revisit\2024-03-24_22-09-00\checkpoints\dataset"),
        Path("checkpoints\dataset"), dirs_exist_ok=True, )
    for f in Path("checkpoints\dataset").glob("*"):
        if f.stem.isnumeric() and int(f.stem) > cfg.training.actor_critic.batch_num_samples:
            f.unlink()
    trainer.train_dataset.load_disk_checkpoint(trainer.ckpt_dir / 'dataset')

    # shutil.copytree(
    #     Path(r"C:\Users\Michael\PycharmProjects\Brawl-Stars-AI\outputs\faster_and_fix\2024-03-31_04-47-02\checkpoints"),
    #     Path("checkpoints"), dirs_exist_ok=True, ignore=shutil.ignore_patterns("checkpoint*"))
    #
    # trainer.load_checkpoint()

    start_epoch = trainer.start_epoch
    env: GymEnv = trainer.train_collector.env.env
    actor = trainer.agent.actor_critic
    actor.checkpoint_backbone = trainer.cfg.training.actor_critic.checkpoint_backbone
    actor.checkpoint_lstm = trainer.cfg.training.actor_critic.checkpoint_lstm
    actor.fp16 = trainer.cfg.common.fp16
    optimizer = trainer.optimizer_actor_critic
    ac_trainer = ACTrainer(trainer.cfg, env, actor, optimizer, trainer.train_dataset)
    ac_trainer.lr_scheduler.last_epoch = start_epoch - 1
    ac_trainer.replay_only = True

    def collect_one(do_replay=False, do_log=False, do_step=False, verbose=0):
        ac_trainer.reset(collection=not do_replay)
        skip_update = False
        while not env.done:
            step_metrics = ac_trainer.step(collection=not do_replay)
            if verbose and ac_trainer.episode_step % verbose == 0:
                print(f"step value {ac_trainer.outputs[-1].means_values[0].item()}")
                print(f"actions: {ac_trainer.actions[-1][0].cpu().detach().numpy()}")
                print(f"act_logits: {ac_trainer.outputs[-1].logits_actions[0].cpu().detach().numpy()}")
                print(f"means_stds: {ac_trainer.outputs[-1].mean_continuous[0].cpu().detach().numpy()}"
                      f" {ac_trainer.outputs[-1].std_continuous[0].cpu().detach().numpy()}")

            if len(ac_trainer.observations) > 20 and \
                    not np.count_nonzero(ac_trainer.observations[-20] -
                                         ac_trainer.observations[-1]) or step_metrics['buggy_episode']:
                print("Train loop: env froze, attempting reload")
                skip_update = True
                break  # if froze inside episode

        if skip_update:
            env.__exit__(soft=False)
            RELOAD_MACRO.play()
        else:
            if not ac_trainer.actions:
                print("Empty episode. Skipping")
                return
            ac_trainer.episode_end(collection=not do_replay, do_step=do_step)

        if do_log:
            wandb.log({"epoch": n_episode, **ac_trainer.metrics})

    while len(ac_trainer.dataset) < ac_trainer.batch_size:
        collect_one()
    collection_event = Event()
    log_pipe_here, log_pipe_there = Pipe()

    def _wandb_log_listener():
        while True:
            if log_pipe_here.poll(1):
                wandb.log(log_pipe_here.recv())

    wandb_log_listener = Thread(target=_wandb_log_listener)
    wandb_log_listener.start()

    collection_process = Process(target=process_collection, args=(collection_event, cfg, env,
                                                                  actor.to('cpu'),  # cuda weights pickling issue
                                                                  ac_trainer.dataset,
                                                                  Path("checkpoints/last.pt"),
                                                                  log_pipe_there))
    collection_process.start()
    actor.to(device)
    trainer.save_checkpoint(start_epoch, False, flush=False)
    try:
        for n_episode in range(start_epoch, MAX_EPISODES):
            collection_event.set()
            for _ in range(cfg.training.actor_critic.grad_acc_steps - 1):
                ac_trainer.reset()
                episode_metrics = ac_trainer.episode_end(do_step=False)
                wandb.log({"epoch": n_episode, **episode_metrics})
            ac_trainer.reset()
            episode_metrics = ac_trainer.episode_end(do_step=True)
            wandb.log({"epoch": n_episode, **episode_metrics})

            while collection_event.is_set():  # wait for collection to finish
                time.sleep(1)

            print('Saving checkpoint')
            if (n_episode + 1) % 50 == 0:
                shutil.copytree('checkpoints', f'checkpoints/checkpoint_{n_episode}',
                                ignore=shutil.ignore_patterns("dataset*", "checkpoint*"))
            trainer.save_checkpoint(n_episode, True)
    finally:
        collection_process.terminate()
        wandb.finish()


@hydra.main(config_path=r"C:\Users\Michael\PycharmProjects\Brawl_iris\config", config_name="trainer")
def main_replay(cfg: DictConfig):
    trainer = Trainer(cfg)
    shutil.copytree(
        Path(
            r"C:\Users\Michael\PycharmProjects\Brawl-Stars-AI\outputs\parallel_collection\2024-03-30_11-03-19\checkpoints\dataset"),
        Path("checkpoints\dataset"), dirs_exist_ok=True)
    # for f in Path("checkpoints\dataset").glob("*"):
    #     if f.stem.isnumeric() and int(f.stem) > 48:
    #         f.unlink()
    trainer.train_dataset.load_disk_checkpoint(trainer.ckpt_dir / 'dataset')

    start_epoch = trainer.start_epoch
    env = None
    actor = trainer.agent.actor_critic
    actor.checkpoint_backbone = trainer.cfg.training.actor_critic.checkpoint_backbone
    actor.checkpoint_lstm = trainer.cfg.training.actor_critic.checkpoint_lstm
    actor.fp16 = trainer.cfg.common.fp16

    optimizer = trainer.optimizer_actor_critic
    ac_trainer = ACTrainer(trainer.cfg, env, actor, optimizer, trainer.train_dataset, replay_only=True)
    ac_trainer.batch_size = ac_trainer.batch_size + 1
    ac_trainer.lr_scheduler.last_epoch = start_epoch - 1

    for n_step in tqdm.tqdm(range(start_epoch, MAX_EPISODES), desc="Step: ", total=MAX_EPISODES, ):
        ac_trainer.reset(prioritize=False)
        episode_metrics = ac_trainer.episode_end(epoch=n_step)
        print('Logging')
        wandb.log({"epoch": n_step, **episode_metrics})
        if n_step % 100 == 0:
            continue
            print('Saving checkpoint')
            trainer.save_checkpoint(n_step, False, save_dataset=False)


@hydra.main(config_path=r"C:\Users\Michael\PycharmProjects\Brawl_iris\config", config_name="trainer")
def inspect(cfg: DictConfig):
    try:
        cfg.wandb.mode = "offline"
        trainer = Trainer(cfg)

        shutil.copytree(
            Path(
                r"C:\Users\Michael\PycharmProjects\Brawl-Stars-AI\outputs\parallel_collection\2024-03-29_22-57-56\checkpoints"),
            Path("checkpoints"), dirs_exist_ok=True, ignore=shutil.ignore_patterns("checkpoint*"))
        trainer.load_checkpoint()

        # shutil.copytree(
        #     Path(r"C:\Users\Michael\PycharmProjects\Brawl-Stars-AI\outputs\2023-12-14\05-00-41\checkpoints\dataset"),
        #     Path("checkpoints\dataset"), dirs_exist_ok=True)
        # for f in Path("checkpoints\dataset").glob("*"):
        #     if int(f.stem) > 0:
        #         continue
        #         # f.unlink()
        # trainer.train_dataset.load_disk_checkpoint(trainer.ckpt_dir / 'dataset')

        env: GymEnv = trainer.train_collector.env.env
        actor = trainer.agent.actor_critic
        actor.checkpoint_backbone = trainer.cfg.training.actor_critic.checkpoint_backbone
        actor.checkpoint_lstm = trainer.cfg.training.actor_critic.checkpoint_lstm
        actor.fp16 = trainer.cfg.common.fp16

        optimizer = trainer.optimizer_actor_critic
        ac_trainer = ACTrainer(trainer.cfg, env, actor, optimizer, trainer.train_dataset, replay_only=True)
        ac_trainer.reset()
        ac_trainer.episode_end()
    finally:
        wandb.finish()
        shutil.rmtree(os.getcwd(), ignore_errors=True)


if __name__ == "__main__":
    main()
