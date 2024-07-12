import copy
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
from hydra.utils import instantiate
from omegaconf import DictConfig, OmegaConf
import wandb
import matplotlib.pyplot as plt

plt.switch_backend("AGG")
sys.path.append(r"C:\Users\Michael\PycharmProjects\Brawl_iris")

from src.dataset import EpisodesDataset, _collate_fn as sample_batch
from src.episode import Episode
from src.trainer import Trainer
from src.utils import compute_lambda_returns, LossWithIntermediateLosses, adaptive_gradient_clipping, set_seed
from src.models.actor_critic import ActorCriticOutput, ImagineOutput, ActorCritic
from environment import make_env, RELOAD_MACRO, GymEnv
from utils.collection_explorer import run_explorer

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

MAX_EPOCHS = 1000
WEIGHT_QUANTILE = 0.5  # value loss quantile to assign episode weights
RANDOM_ACTION = 0.01  # proba
POSITIVE_WEIGHT = 1.0  # weigh positive updates (divides neg ones)


# set action_limit_mask to False and the desired actions to lock them during the run.

# move_shift, shot_shift, shot_strength
ACTION_C_NAMES = ["move_shift", "shot_shift", "shot_strength"]
ACTION_C_LOCK_MASK = torch.tensor([False, False, False]).to(device)
ACTION_C_LOCK = torch.tensor([0.0, 0.0, -10.0]).to(device)


class RateLimiter:
    def __init__(self):
        """

        :param calls_per_second: optional int or float, max number of function calls per second
        :param verbosity:
        """
        self._last_call_timestamp = -1

    def limit(self, calls_per_second=2.0, verbosity=0.2, outlier_threshold=4.0):
        """
        Function decorator, limits
        :param calls_per_second: optional int or float, max number of function calls per second.
         overrides instance attribute
        :param verbosity: float, proba of printed delay warning
        :param outlier_threshold: int or float, delay in seconds more than which the warning will not be
         triggered. Useful for expected large delays.
        :return:
        """
        def decorator(func):
            def inner(*args, **kwargs):
                if self._last_call_timestamp < 0:
                    self._last_call_timestamp = time.time()
                out = func(*args, **kwargs)

                extra_time = (1 / calls_per_second) - (time.time() - self._last_call_timestamp)
                if extra_time > 0:
                    time.sleep(extra_time)
                elif extra_time > -outlier_threshold:
                    if np.random.random() < verbosity:
                        print(f"<{func.__name__}> off rate by {-extra_time}")
                    pass
                self._last_call_timestamp = time.time()
                return out

            return inner

        return decorator


collection_limiter = RateLimiter()


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


def lambda_return_schedule(epoch, last_epoch=MAX_EPOCHS // 2, start_lambda=0.95, last_lambda=0.99):
    start_pow = np.log(1 - start_lambda)
    last_pow = np.log(1 - last_lambda)
    if epoch < last_epoch:
        return 1 - np.exp(start_pow + (last_pow - start_pow) * epoch / last_epoch)
    return last_lambda


def process_collection(event: Event, config, log_connection, verbose=1):
    sys.stdin = open(0)  # otherwise input() will fail
    wandb.log = log_connection.send  # try to globalize wandb logging in scope of this process (macros.py e.g.)
    if config.training.actor_critic.join_collection:
        # if not blocking, make sure to always use the latest version of actor
        config.training.actor_critic.collection_steps = 1
    actor_critic = instantiate(config.actor_critic)
    dataset = instantiate(config.datasets.train)
    env: GymEnv = instantiate(config.env.train)
    ac_trainer = ACTrainer(config, env, actor_critic, None, dataset)
    hook_place = [None]

    def _get_hook(hook):
        hook_place[0] = hook

    explorer_thread = Thread(target=run_explorer, args=(_get_hook,), daemon=True)
    explorer_thread.start()
    while True:
        if hook_place[0] is not None:
            print("process_collection: got explorer hook")
            explorer = hook_place[0]
            break

    while True:
        try:
            if event.is_set():
                # collection
                ac_trainer.load_checkpoint(optimizer=False, scheduler=False, dataset=False)
                ac_trainer.actor.eval()

                for _ in range(config.training.actor_critic.collection_steps):
                    ac_trainer.reset(collection=True)
                    skip_update = False
                    while not env.done:
                        step_metrics = ac_trainer.step(collection=True)
                        if verbose and ac_trainer.episode_step % verbose == 0:
                            explorer.update_stats(env, ac_trainer)
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
                        if len(ac_trainer.actions) < 3:
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

    def get_episode_proba(self, rank=True, alpha=0.7, beta=0.1, use_age=True):
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

    def reset(self, collection=False, prioritize=True, replay=None, burn_in=False, segments=False):
        """
        Always reserves actor state at id 0 for real-time env.
        When doing replaying provide sampled replay of size batch_size-1
        """
        print("ACTrainer.reset: Started reset")
        assert not (collection and self.replay_only), "Can't collect when replay_only is set"
        self.metrics = defaultdict(float)
        self.episode_step = 0
        self.dataset.load_disk_checkpoint(Path("checkpoints/dataset"))

        # sample episodes via prioritized sampling
        if replay is not None:
            self._update_weights = np.ones(self.batch_size)
            self._replay = replay
        elif not collection and len(self.dataset) >= self.batch_size - 1 and self.batch_size > 1:
            probas, weights = self.get_episode_proba(alpha=0.3 if prioritize else 0.0)
            episode_ids = np.random.choice(np.arange(len(self.dataset)),
                                           size=(self.batch_size - 1,),
                                           p=probas,
                                           replace=False)  # todo: fix replacement/caching/accumulation
            disk_ids = [self.dataset.disk_episodes[i] for i in episode_ids]
            self._active_episodes = ([None if self.replay_only else self.dataset.num_seen_episodes] +
                                     disk_ids)
            self._update_weights = np.concatenate([np.ones(1), weights[episode_ids]])
            start_ = time.time()
            if not segments:
                self._replay = self.dataset.sample_replay(samples=disk_ids)
            else:
                self._replay = sample_batch([(self.dataset.get_episode(i), None) for i in disk_ids],
                                            self.dataset.resolution,
                                            sample_segments=True,
                                            segment_len=self.cfg.training.actor_critic.sequence_length,
                                            end_proba=0.1)[0]
            self.metrics["actor_critic/train/batch_mean_reward"] = sum([
                r[torch.nonzero(r)[-1]].item() if torch.count_nonzero(r) else 0.0 for r in self._replay["rewards"]
            ]) / (self.batch_size - 1)
            print(self.metrics)
            print("replay sample elapsed: ", time.time() - start_)
        else:
            self._update_weights = np.ones(self.batch_size)
            if not collection:
                self._replay = self._get_placeholder_replay()

        # self._replay = self._get_placeholder_replay(length=250, mask=False)  # debug memory etc

        self.actions, self.rewards, self.dones, self.outputs, self.mask_paddings = [], [], [], [], []
        if not self.replay_only:
            self.env.reset()
        self.observations = []
        self.actor.reset(1 if collection else self.batch_size)
        if self.replay_only and burn_in:
            self.burn_in(self.cfg.training.actor_critic.burn_in)

    @torch.no_grad()
    def burn_in(self, steps: int):
        """
        Uses self.episode_step to track burn-in steps. Will fail if not replay_only
        """
        assert self.replay_only, "Burn-in available only in replay mode"
        assert steps < self._replay["ends"].size(1) - 2, "Can't burn-in more than the length of replay"
        assert self.episode_step == 0, "Called burn-in after step was made"
        while self.episode_step < steps:
            if self._replay["mask_padding"][:, self.episode_step].any():
                full_mask_padding = torch.cat([torch.tensor([False]),
                                               self._replay['mask_padding'][:, self.episode_step]], dim=0)
                obs = torch.cat([torch.zeros_like(self._replay['observations'][[0], 0]),
                                 self._replay['observations'][:, self.episode_step]], dim=0)
                self.actor.forward(obs.to(device), full_mask_padding.to(device))
            self.episode_step += 1

    def _update_metrics(self, metrics: dict, mode="train"):
        for k, v in metrics.items():
            name = f"actor_critic/{mode}/{k}"
            self.metrics[name] = v

    @staticmethod
    def reward_map(r):
        if r >= 8:
            return 1.0
        elif r >= 6:
            return 0.25
        elif r >= 4:
            return -0.3
        elif r >= 2:
            return -0.5
        else:
            return -1.0

    @collection_limiter.limit(calls_per_second=2, verbosity=1.0)
    def step(self, collection=False):
        assert not self.replay_only, "Can't call env step in replay mode. Use episode_end instead"
        if collection:
            return self._collection_step()
        obs, reward, done, _ = self.env.request_state()
        self.observations.append(obs)

        curr_obs = torch.tensor(self.observations[-1].transpose(2, 0, 1)).unsqueeze(0).float() / 255.0
        if self.episode_step < self._replay['ends'].size(1):
            replay_obs = self._replay['observations'][:, self.episode_step]
            replay_mask_padding = self._replay['mask_padding'][:, self.episode_step]
            replay_actions = torch.cat([self._replay['actions'][:, self.episode_step, None],
                                        self._replay['actions_continuous'][:, self.episode_step]], dim=-1)
            replay_rewards = self._replay['rewards'][:, self.episode_step]
            replay_ends = self._replay['ends'][:, self.episode_step]
        else:
            replay_obs = torch.zeros_like(self._replay['observations'][:, -1])
            replay_mask_padding = torch.zeros_like(self._replay['mask_padding'][:, -1])
            replay_actions = torch.cat([torch.zeros_like(self._replay['actions'][:, -1, None]),
                                        torch.zeros_like(self._replay['actions_continuous'][:, -1])], dim=-1)
            replay_rewards = torch.zeros_like(self._replay['rewards'][:, -1])
            replay_ends = torch.ones_like(self._replay['ends'][:, -1])

        full_rewards = torch.cat([torch.tensor([reward]), replay_rewards], dim=0)
        self.rewards.append(full_rewards)
        full_ends = torch.cat([torch.tensor([done]), replay_ends], dim=0)
        self.dones.append(full_ends)

        full_obs = torch.cat([curr_obs, replay_obs], dim=0)
        full_mask_padding = torch.cat([torch.tensor([True]), replay_mask_padding], dim=0)
        self.mask_paddings.append(full_mask_padding)
        output = self.actor.forward(full_obs.to(device),
                                    mask_padding=full_mask_padding.to(device))

        # sample action from s_t
        action_token, action_cont = self.actor.sample_actions(output, eps=RANDOM_ACTION)
        action_raw = torch.cat([action_token.flatten(), action_cont.flatten()]).reshape(1, -1)
        action_sigmoid = torch.cat([torch.tensor([action_token]).to(device),
                                    torch.sigmoid(action_cont).flatten()])
        full_action = torch.cat([action_raw, replay_actions.to(device)], dim=0)
        self.actions.append(full_action)
        self.outputs.append(output)

        self.env.step(action_sigmoid.cpu().detach(), return_state=False)

        self.episode_step += 1
        if reward == -100 or self.episode_step > 400:
            self.metrics['collection/buggy_episode'] = 1.0
        else:
            self.metrics['collection/buggy_episode'] = 0.0
        return self.metrics

    @torch.no_grad()
    def _collection_step(self):
        obs, reward, done, _ = self.env.request_state()
        self.observations.append(obs)
        full_rewards = torch.tensor([reward])
        self.rewards.append(full_rewards)
        full_ends = torch.tensor([done])
        self.dones.append(full_ends)

        curr_obs = torch.tensor(self.observations[-1].transpose(2, 0, 1)).unsqueeze(0).float() / 255.0
        full_mask_padding = torch.tensor([True])
        self.mask_paddings.append(full_mask_padding)
        output = self.actor.forward(curr_obs.to(device),
                                    mask_padding=full_mask_padding.to(device))

        # sample action from s_t
        action_token, action_cont = self.actor.sample_actions(output, eps=RANDOM_ACTION)
        # compute token and sigmoid actions
        action_raw = torch.cat([action_token.flatten(), action_cont.flatten()]).reshape(1, -1)
        action_sigmoid = torch.cat([torch.tensor([action_token]).to(device),
                                    torch.sigmoid(action_cont).flatten()])

        self.actions.append(action_raw)
        self.outputs.append(output)

        self.env.step(action_sigmoid.cpu().detach(), return_state=False)

        self.episode_step += 1
        if reward == -100 or self.episode_step > 400:
            self.metrics['collection/buggy_episode'] = 1.0
        else:
            self.metrics['collection/buggy_episode'] = 0.0
        return self.metrics

    @torch.no_grad()
    def _create_episode(self):
        obs = np.stack(self.observations, axis=0)
        obs = torch.ByteTensor(obs).permute(0, 3, 1, 2).contiguous()
        actions = torch.LongTensor(
            torch.stack(self.actions, dim=1)[0, :self.episode_step, ..., 0].long().cpu())
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
            'actions': torch.zeros((self.batch_size - 1, length, 1)),
            'actions_continuous': torch.zeros((self.batch_size - 1, length, 3)),
            'rewards': torch.zeros((self.batch_size - 1, length)),
            'ends': torch.ones((self.batch_size - 1, length), dtype=torch.bool) if mask else
            torch.zeros((self.batch_size - 1, length), dtype=torch.bool),
            'mask_padding': torch.zeros((self.batch_size - 1, length), dtype=torch.bool) if mask else
            torch.ones((self.batch_size - 1, length), dtype=torch.bool)
        }

    @staticmethod
    def _create_hist(data_, bins=32, names=None, mean_line=False, line_at=None, xlim=None):
        if isinstance(data_, torch.Tensor):
            data_ = data_.detach().cpu().numpy()
        if len(data_.shape) == 1:
            data_ = data_.reshape(-1, 1)
        else:
            data_ = data_.reshape(-1, data_.shape[-1])

        f, ax = plt.subplots()
        ax: plt.Subplot
        if xlim is not None:
            if 'q' in str(xlim):
                q = float(xlim.replace('q', ''))
                if q > 0.5:
                    q = 1 - q
                low = np.quantile(data_.reshape(-1), q)
                high = np.quantile(data_.reshape(-1), 1 - q)
                xlim = (low, high)
            ax.set_xlim(*xlim)
        for i in range(data_.shape[-1]):
            ax.hist(data_[:, i], bins=bins, alpha=0.7, range=xlim, label=str(i) if names is None else names[i])
            if mean_line:
                ax.axvline(data_[:, i].mean(), color='k', linestyle='dashed', linewidth=1)
            if line_at is not None:
                ax.axvline(line_at, color='k', linestyle='dashed', linewidth=1)
        ax.legend()
        img = wandb.Image(f)
        plt.close("all")
        return img

    def _catch_up(self):
        if self._replay['ends'].size(1) > self.episode_step:
            start_catchup = time.time()
            for i in range(self.episode_step, self._replay['ends'].size(1)):
                full_mask_padding = torch.cat([torch.tensor([False]), self._replay['mask_padding'][:, i]], dim=0)
                self.mask_paddings.append(full_mask_padding)

                obs = torch.cat([torch.zeros_like(self._replay['observations'][[0], 0]),
                                 self._replay['observations'][:, i]], dim=0)
                output = self.actor.forward(obs.to(device),
                                            mask_padding=full_mask_padding.to(device))
                self.outputs.append(output)
                replay_actions = torch.cat([self._replay['actions'][:, i, None],
                                            self._replay['actions_continuous'][:, i]], dim=-1).to(device)
                full_actions = torch.cat([torch.zeros_like(replay_actions)[[0]], replay_actions])
                self.actions.append(full_actions)
                full_rewards = torch.cat([torch.tensor([0.0]), self._replay['rewards'][:, i]], dim=0)
                self.rewards.append(full_rewards)
                full_ends = torch.cat([torch.tensor([True]), self._replay['ends'][:, i]], dim=0)
                self.dones.append(full_ends)
            print("replay catch-up elapsed: ", time.time() - start_catchup)

    def episode_end(self, epoch=None, collection=False, do_step=True):
        # finish replaying if needed
        if not collection:
            self._catch_up()

        if not self.replay_only:
            # try to learn neg reward
            true_reward = self.rewards[self.episode_step - 1][0].item()
            self.rewards[self.episode_step - 1][0] = self.reward_map(true_reward)
            self.rewards[self.episode_step - 3][0] = self.rewards[self.episode_step - 1][
                0]  # move reward closer bc of lambda returns

            self.metrics["collection/reward"] = self.rewards[self.episode_step - 3][0]
            self.metrics["collection/true_reward"] = true_reward
            self.metrics["collection/episode_length"] = self.episode_step

            # pop trailing obs
            self.dataset.add_episode(
                self._create_episode()
            )
        actual_batch_size = len(self.dones[-1])
        episode_output = ImagineOutput(
            observations=None,
            actions=torch.stack(self.actions, dim=1)[..., 0].to(device),
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
            # self.metrics["binary_logits"] = self._create_hist(
            #     episode_output.logits_actions[mask_paddings][..., :4], names=ACTION_NAMES[:4])
            # self.metrics["anchor_logits"] = self._create_hist(torch.stack([
            #     episode_output.logits_actions[mask_paddings][..., 4:8],
            #     episode_output.logits_actions[mask_paddings][..., 8:],
            # ], dim=-1), names=ACTION_NAMES[4:])
            self.metrics["means"] = self._create_hist(episode_output.continuous_means[mask_paddings],
                                                      names=ACTION_C_NAMES)
            self.metrics["stds"] = self._create_hist(episode_output.continuous_stds[mask_paddings],
                                                     names=ACTION_C_NAMES)
            loss = self.ac_loss(episode_output, mask_paddings, mode="eval")
            self._update_metrics(loss.intermediate_losses, mode="eval")
            return self.metrics

        self.metrics["actor_critic/train/lr"] = self.lr_scheduler.get_last_lr()[0]

        loss = self.ac_loss(episode_output, mask_paddings, epoch=epoch)
        print("ACTrainer.episode_end: backward started")
        start_backward = time.time()
        loss.loss_total.backward()
        if do_step:
            self.metrics["gradients"] = self._create_hist(torch.cat(
                [p.grad.flatten() for p in self.actor.parameters() if p.requires_grad and p.grad is not None]),
                names=["gradients"], xlim="0.05q")
            ratios = adaptive_gradient_clipping(self.actor.parameters(), lam=self.cfg.training.actor_critic.agc_lambda)
            self.metrics["agc_ratios"] = self._create_hist(torch.cat([r.flatten() for r in ratios]),
                                                           line_at=self.cfg.training.actor_critic.agc_lambda,
                                                           names=["agc_ratios"],
                                                           xlim=[0, 2])
            torch.nn.utils.clip_grad_norm_(self.actor.parameters(),
                                           self.cfg.training.actor_critic.max_grad_norm,
                                           error_if_nonfinite=True)
            self.optimizer.step()
            self.lr_scheduler.step()
            self.optimizer.zero_grad()
        loss = loss / (1 / self.accumulation_steps)  # for logging to match the scale of other runs
        print("backward update elapsed: ", time.time() - start_backward)
        self._update_metrics(loss.intermediate_losses)
        self.metrics["actor_critic/train/total_loss"] = loss.loss_total.item()
        return self.metrics

    def ac_loss(self, outputs: ImagineOutput, mask_paddings, epoch=None, mode="train"):
        mask_paddings = torch.logical_and(mask_paddings, outputs.ends.logical_not())  # do not include end into loss
        # use normalization for value loss and absolute value for advantage
        values_raw = outputs.values
        values_std = torch.masked_select(values_raw, mask_paddings).std()
        values_mean = torch.masked_select(values_raw, mask_paddings).mean()
        self.metrics[f"actor_critic/{mode}/value_mean"] = values_mean.item()
        self.metrics[f"actor_critic/{mode}/value_std"] = values_std.item()
        values = (values_raw - values_mean) / values_std
        if epoch is None or not self.cfg.training.actor_critic.lambda_warmup_init:
            lambda_ = self.cfg.training.actor_critic.lambda_
        else:
            lambda_ = lambda_return_schedule(epoch, start_lambda=self.cfg.training.actor_critic.lambda_warmup_init,
                                             last_lambda=self.cfg.training.actor_critic.lambda_)
        # self.metrics["actor_critic/train/lambda_"] = lambda_
        with torch.no_grad():
            lambda_returns_raw = compute_masked_lambda_returns(
                rewards=outputs.rewards,
                values=values_raw,
                ends=outputs.ends,
                mask_paddings=mask_paddings,
                gamma=self.cfg.training.actor_critic.gamma,
                lambda_=lambda_,
            )[:, :-1]
            lambda_returns = compute_masked_lambda_returns(
                rewards=outputs.rewards,
                values=values,
                ends=outputs.ends,
                mask_paddings=mask_paddings,
                gamma=self.cfg.training.actor_critic.gamma,
                lambda_=lambda_,
            )[:, :-1]
        mask_paddings = mask_paddings[:, :-1]

        (log_probs, entropy), (log_probs_continuous, entropy_cont) = self.actor.get_proba_entropy(outputs)

        update_weight = torch.tensor(self._update_weights, device=device).reshape(self.batch_size, 1)
        advantage_factor = (lambda_returns_raw - values_raw.detach()[:, :-1]) / values_std.detach()  # match loss_values scale
        # print(advantage_factor[1][mask_paddings[1]])
        # self.metrics["advantages"] = self._create_hist(advantage_factor[mask_paddings], mean_line=True)
        # self.metrics["weighted_advantages"] = self._create_hist((update_weight * advantage_factor)[mask_paddings],
        #                                                         mean_line=True)
        self.metrics[f"actor_critic/{mode}/advantage_mean"] = advantage_factor[mask_paddings].mean().item()
        advantage_factor[advantage_factor < 0] /= POSITIVE_WEIGHT

        # compute losses
        # loss_actions = -1 * (log_probs * advantage_factor.unsqueeze(-1))[..., ACTION_LOCK_MASK]
        # loss_actions_masked = torch.masked_select(update_weight.unsqueeze(-1) * loss_actions,
        #                                           mask_paddings.unsqueeze(-1)).mean()
        loss_actions_masked = torch.masked_select(torch.where(
            advantage_factor >= 0,
            -1 * (log_probs * advantage_factor),
            (torch.log(1 - torch.exp(log_probs)) * advantage_factor)
        ), mask_paddings).mean()
        self.metrics[f"actor_critic/{mode}/custom_weighted_actions"] = loss_actions_masked.item()
        self.metrics[f"actor_critic/{mode}/custom_actions"] = torch.masked_select(torch.where(
            advantage_factor >= 0,
            -log_probs,
            -torch.log(1 - torch.exp(log_probs))
        ), mask_paddings).mean().item()

        loss_continuous_actions = -1 * (log_probs_continuous.clamp(-5, 5) * advantage_factor.unsqueeze(-1))
        loss_continuous_actions_masked = torch.masked_select(update_weight.unsqueeze(-1) * loss_continuous_actions,
                                                             mask_paddings.unsqueeze(-1)).mean()
        # self.metrics["actor_critic/train/custom_continuous"] = torch.masked_select(torch.where(
        #     loss_continuous_actions >= 0,
        #     loss_continuous_actions,
        #     (torch.log(1 - torch.exp(log_probs_continuous.clamp(-5, 5))) * advantage_factor.unsqueeze(-1))[..., ACTION_C_LOCK_MASK]
        # ), mask_paddings.unsqueeze(-1)).mean().item()
        # todo
        loss_entropy = torch.masked_select(
            - 0 * self.cfg.training.actor_critic.entropy_weight * torch.log(entropy),
            mask_paddings).mean() + \
                       torch.masked_select(
                           # logit l2 regularization loss. cat(logits=[10,10,-10,-10]).entropy() is 0.69!!
                           self.cfg.training.actor_critic.entropy_weight *
                           torch.where(outputs.logits_actions.abs() > 2, outputs.logits_actions, 0)[:, :-1] ** 2,
                           mask_paddings[..., None]
                       ).mean()
        loss_entropy_continuous = torch.masked_select(
            - self.cfg.training.actor_critic.entropy_continuous_weight * entropy_cont[..., ACTION_C_LOCK_MASK],
            mask_paddings.unsqueeze(-1)).mean()
        loss_values = torch.square(values[:, :-1] - lambda_returns)
        loss_values_masked = torch.masked_select(update_weight * loss_values, mask_paddings).mean()
        # todo
        full_loss = LossWithIntermediateLosses(loss_actions=loss_actions_masked,
                                               loss_continuous_actions=torch.zeros_like(loss_values_masked),
                                               # * loss_continuous_actions_masked,
                                               loss_values=loss_values_masked,
                                               loss_entropy=loss_entropy,
                                               loss_entropy_continuous=torch.zeros_like(
                                                   loss_values_masked), )  # * loss_entropy_continuous)
        full_loss = full_loss / self.accumulation_steps

        # episode weight ~ absolute loss per episode
        # loss_actions = loss_actions.detach()
        # loss_actions[loss_actions < 0] = loss_actions[loss_actions < 0] + NEGATIVE_EPISODE_WEIGHT_SHIFT
        # loss_continuous_actions = loss_continuous_actions.detach()
        # loss_continuous_actions[loss_continuous_actions < 0] = loss_continuous_actions[
        #                                                            loss_continuous_actions < 0
        #                                                            ] + NEGATIVE_EPISODE_WEIGHT_SHIFT
        #
        # self._assign_episode_weights(loss_values + loss_actions.mean(dim=-1) + loss_continuous_actions.mean(dim=-1),
        #                              mask_paddings)
        return full_loss

    def save_checkpoint(self, epoch: int, dataset=True):
        Path("checkpoints/dataset").mkdir(parents=True, exist_ok=True)
        torch.save(epoch, "checkpoints/epoch.pt")
        torch.save(self.optimizer.state_dict(), "checkpoints/optimizer.pt")
        torch.save(self.lr_scheduler.state_dict(), "checkpoints/lr_scheduler.pt")
        torch.save(self.actor.state_dict(), "checkpoints/last.pt")
        if dataset:
            self.dataset.update_disk_checkpoint(Path("checkpoints/dataset"))

    def load_checkpoint(self, path=None, dataset_path="checkpoints/dataset",
                        optimizer=True, scheduler=True, dataset=True) -> int:
        """Loads state with optional components and returns the last saved epoch"""
        if path is None:
            path = "checkpoints"
        path = Path(path)
        if dataset_path is None:
            dataset_path = path / "dataset"
        else:
            dataset_path = Path(dataset_path)

        epoch = torch.load(path / "epoch.pt")
        self.actor.load_state_dict(torch.load(path / "last.pt", map_location=device))
        if optimizer:
            self.optimizer.load_state_dict(torch.load(path / "optimizer.pt"))
        if scheduler:
            self.lr_scheduler.load_state_dict(torch.load(path / "lr_scheduler.pt"))
        if dataset:
            self.dataset.load_disk_checkpoint(dataset_path)
        return epoch


def custom_setup(cfg):
    torch.backends.cudnn.benchmark = True
    if sys.gettrace() is not None:  # if debugging
        cfg.wandb.mode = "offline"
        cfg.training.actor_critic.batch_num_samples = 2
    cfg.wandb.tags = list(set(cfg.wandb.tags or [] + ["collection"]))
    wandb.init(config=OmegaConf.to_container(cfg, resolve=True),
               reinit=True,
               resume=True,
               **cfg.wandb)


@hydra.main(config_path=r"C:\Users\Michael\PycharmProjects\Brawl_iris\config", config_name="trainer")
def main(cfg: DictConfig):
    # todo: add motion attention mask (easier architecture)
    #       predict next frame (more signal density)
    #       remove value normalization (batchnorm is fine)
    #       debug midway actor converge
    #       input average of frames between actions instead of a single screenshot

    custom_setup(cfg)

    start_epoch = 1
    env: GymEnv = instantiate(cfg.env.train)
    actor = instantiate(cfg.actor_critic)

    head_parameters = []
    other_parameters = []
    for n, p in actor.named_parameters():
        if any([part in n for part in ["actor_head"]]):
            head_parameters.append(p)
        else:
            other_parameters.append(p)

    optimizer = torch.optim.Adam([
        {"params": other_parameters},
        {"params": head_parameters, "lr": cfg.training.actor_critic.lr_heads}
    ],
        lr=cfg.training.learning_rate,
        betas=cfg.training.actor_critic.adam_betas)
    dataset = instantiate(cfg.datasets.train)
    Path("checkpoints/dataset").mkdir(parents=True, exist_ok=True)

    ac_trainer = ACTrainer(cfg, env, actor, optimizer, dataset)
    # # TODO ATTENTION REMOVE
    # # used for random collection based on tokens
    # ac_trainer.batch_size = 99999
    #
    # def sample_from_random_token(*args, **kwargs):
    #     t = np.random.randint(0, env.action_space.n)
    #     return (torch.tensor(env.split_into_bins(t)).to(device)[None],
    #             torch.zeros((3,)).float().to(device)[None])
    #
    # ac_trainer.actor.sample_actions = sample_from_random_token
    # # TODO ATTENTION REMOVE (end)
    # shutil.copytree(
    #     Path(
    #         r"C:\Users\Michael\PycharmProjects\Brawl_iris\input_artifacts\dataset"),
    #     Path("checkpoints\dataset"), dirs_exist_ok=True, )
    total_batch_size = cfg.training.actor_critic.batch_num_samples * cfg.training.actor_critic.grad_acc_steps
    # for f in Path("checkpoints\dataset").glob("*"):
    #     if f.stem.isnumeric() and int(f.stem) < 1627:
    #         f.unlink()
    # ac_trainer.dataset.load_disk_checkpoint(Path("checkpoints/dataset"))

    # start_epoch = ac_trainer.load_checkpoint(
    #     r"C:\Users\Michael\PycharmProjects\Brawl-Stars-AI\outputs\smaller_discounts\2024-04-21_20-41-57\checkpoints",
    #     dataset=False)

    ac_trainer.lr_scheduler.last_epoch = start_epoch - 1

    wandb.init(config=OmegaConf.to_container(cfg, resolve=True),
               reinit=True,
               resume=True,
               **cfg.wandb)
    set_seed(cfg.common.seed)

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
            if len(ac_trainer.actions) < 3:
                print("Empty episode. Skipping")
                return
            ac_trainer.episode_end(collection=not do_replay, do_step=do_step)

        if do_log:
            wandb.log({**ac_trainer.metrics})

    while ac_trainer.batch_size > len(ac_trainer.dataset):
        print("collecting", len(ac_trainer.dataset), ac_trainer.batch_size)
        collect_one(do_log=True)
        ac_trainer.save_checkpoint(start_epoch)  # save collected
    ac_trainer.save_checkpoint(start_epoch)  # make checkpoint for collection process

    # init collection and monitoring threads
    collection_event = Event()
    log_pipe_here, log_pipe_there = Pipe()

    def _wandb_log_listener():
        while True:
            if log_pipe_here.poll(1):
                wandb.log(log_pipe_here.recv())

    wandb_log_listener = Thread(target=_wandb_log_listener)
    wandb_log_listener.start()

    collection_process = Process(target=process_collection, args=(collection_event, cfg, log_pipe_there), )
    collection_process.start()
    ac_trainer.replay_only = True
    actor.to(device)
    actor.train()

    converge_actor_head = False
    first_time_converge = False
    try:
        for n_episode in range(start_epoch, 2**20):
            collection_event.set()
            for _ in range(cfg.training.actor_critic.grad_acc_steps - 1):
                ac_trainer.reset(segments=True, burn_in=True)
                episode_metrics = ac_trainer.episode_end(do_step=False)
                wandb.log({"epoch": n_episode, **episode_metrics})
            ac_trainer.reset(segments=True, burn_in=True)
            episode_metrics = ac_trainer.episode_end(do_step=True)
            wandb.log({"epoch": n_episode, **episode_metrics})
            if cfg.training.actor_critic.join_collection:
                while collection_event.is_set():  # wait for collection to finish
                    time.sleep(1)

            print('Saving checkpoint')
            if (n_episode + 1) % 200 == 0:
                shutil.copytree('checkpoints', f'checkpoints/checkpoint_{n_episode}',
                                ignore=shutil.ignore_patterns("dataset*", "checkpoint*"))
            ac_trainer.save_checkpoint(n_episode, dataset=False)

            if converge_actor_head:
                if n_episode % 100 == 0 or first_time_converge:  # do actor head converge every epochs
                    first_time_converge = False
                    print("Running advantages precomputation")
                    dataloader = iter(dataset.torch_dataloader(ac_trainer.batch_size, random_sampling=False))
                    compute_advantages(ac_trainer, dataloader, cfg)
                    print("Running actor head fit")
                    train_actor_head(Path("./"), cfg, head=ac_trainer.actor.actor_head, do_log=True)
                    ac_trainer.actor.train()

    finally:
        collection_process.terminate()
        wandb.finish()


@hydra.main(config_path=r"C:\Users\Michael\PycharmProjects\Brawl_iris\config", config_name="trainer")
def main_replay(cfg: DictConfig):
    # todo: main goal - converge on 480 episodes.
    #       main problems - noisy value objective and general task difficulty
    #  - play with lambda - residuals might be noisy
    #  - add/replace with optical flow to filter out static info

    shutil.copytree(
        Path(
            r"C:\Users\Michael\PycharmProjects\Brawl-Stars-AI\outputs\cleaner_rewards\2024-04-04_10-07-21\checkpoints\dataset"),
        Path("checkpoints\dataset"), dirs_exist_ok=True)
    # for f in Path("checkpoints\dataset").glob("*"):
    #     if f.stem.isnumeric() and int(f.stem) > 48:
    #         f.unlink()

    # shutil.copytree(
    #     Path(r"C:\Users\Michael\PycharmProjects\Brawl-Stars-AI\outputs\actor_mlp_head\2024-03-31_22-44-53\checkpoints"),
    #     Path("checkpoints"), dirs_exist_ok=True, ignore=shutil.ignore_patterns("checkpoint*"))
    # for f in Path("checkpoints\dataset").glob("*"):
    #     if f.stem.isnumeric() and int(f.stem) > 257 + 48:
    #         f.unlink()

    start_epoch = 1
    env = None
    actor: torch.nn.Module = instantiate(cfg.actor_critic)
    # actor.load_state_dict(torch.load(
    #     r"C:\Users\Michael\PycharmProjects\Brawl-Stars-AI\outputs\param_groups_converge\2024-04-08_09-12-01\checkpoints\last.pt"))
    # actor_head = torch.load(
    #     r"C:\Users\Michael\PycharmProjects\Brawl-Stars-AI\outputs\step_head_converge\2024-04-08_23-30-22\head.pt")  # todo:
    # new_weight = torch.cat([actor_head[-1].weight, torch.zeros(6, actor_head[-1].weight.shape[-1])])
    # actor_head[-1].weight = torch.nn.Parameter(new_weight)
    # actor.actor_head = actor_head

    head_parameters = []
    other_parameters = []
    for n, p in actor.named_parameters():
        if any([part in n for part in ["actor_head"]]):
            head_parameters.append(p)
        else:
            other_parameters.append(p)

    optimizer = torch.optim.Adam([
        {"params": other_parameters},
        {"params": head_parameters, "lr": cfg.training.actor_critic.lr_heads}
    ],
        lr=cfg.training.learning_rate,
        betas=cfg.training.actor_critic.adam_betas)
    dataset: EpisodesDataset = instantiate(cfg.datasets.train)
    ac_trainer = ACTrainer(cfg, env, actor, optimizer, dataset, replay_only=True)
    ac_trainer.dataset.load_disk_checkpoint(Path("checkpoints/dataset"))

    total_steps = MAX_EPOCHS * cfg.training.actor_critic.grad_acc_steps
    dataloader = iter(dataset.torch_dataloader(ac_trainer.batch_size, total_steps, sample_segments=True,
                                               segment_len=cfg.common.sequence_length))

    def sample_next():
        start_replay = time.time()
        replay, ids = next(dataloader)
        print(f"Dataloader elapsed: {time.time() - start_replay}")
        return replay, ids

    ac_trainer.lr_scheduler.last_epoch = start_epoch - 1
    ac_trainer.batch_size = ac_trainer.batch_size + 1

    # r, i = sample_next()
    # print(i)
    # # ac_trainer.actor.eval()
    # ac_trainer.reset(replay=r)
    # ac_trainer.episode_end(do_step=False)
    # ac_trainer.reset(replay=r)
    # ac_trainer.episode_end(do_step=False)
    # exit()
    cfg.wandb.tags.append("converge")
    wandb.init(config=OmegaConf.to_container(cfg, resolve=True),
               reinit=True,
               resume=True,
               **cfg.wandb)
    set_seed(cfg.common.seed)
    for n_step in tqdm.tqdm(range(start_epoch, MAX_EPOCHS), desc="Step: ", total=MAX_EPOCHS, ):
        for _ in range(cfg.training.actor_critic.grad_acc_steps - 1):
            ac_trainer.reset(replay=sample_next()[0], burn_in=True)
            episode_metrics = ac_trainer.episode_end(epoch=n_step, do_step=False)
            wandb.log({"epoch": n_step, **episode_metrics})

        ac_trainer.reset(replay=sample_next()[0], burn_in=True)
        episode_metrics = ac_trainer.episode_end(epoch=n_step, do_step=True)
        wandb.log({"epoch": n_step, **episode_metrics})
        if (n_step + 1) % 50 == 0:
            print('Saving checkpoint')
            ac_trainer.save_checkpoint(n_step)

        # if n_step % 100 == 0:  # do actor head converge every epochs
        #     print("Running advantages precomputation")
        #     advantage_dataloader = iter(dataset.torch_dataloader(ac_trainer.batch_size,
        #                                                          random_sampling=False, parallelize=False))
        #     compute_advantages(ac_trainer, advantage_dataloader, cfg)
        #     print("Running actor head fit")
        #     train_actor_head(Path("./"), cfg, head=ac_trainer.actor.actor_head, do_log=True)
        #     ac_trainer.actor.train()


@torch.no_grad()
def compute_advantages(ac_trainer, dataloader, cfg, ):
    embeddings_dir = Path("embeddings")
    if embeddings_dir.exists():
        shutil.rmtree(embeddings_dir)
    embeddings_dir.mkdir()
    out = {}
    actions = {}
    ac_trainer.actor.eval()
    for (replay, ids) in tqdm.tqdm(dataloader, desc="Batch: "):
        ac_trainer.batch_size = len(replay["ends"])
        ac_trainer.reset(replay=replay)
        outputs, embeddings = [], []
        for i in range(replay["ends"].size(1)):
            output, embedding = ac_trainer.actor.forward(replay["observations"][:, i].to(device),
                                                         replay["mask_padding"][:, i].to(device),
                                                         return_embedding=True)
            outputs.append(output)
            embeddings.append(embedding)

        actual_batch_size = replay["ends"].size(0)
        ends = replay["ends"].to(device)
        mask_paddings = replay["mask_padding"].to(device)
        mask_paddings = torch.logical_and(mask_paddings, ends.logical_not())  # do not include end into loss
        values = torch.stack([out.means_values for out in outputs], dim=1).reshape(actual_batch_size, -1).to(
            device)
        lambda_returns_raw = compute_masked_lambda_returns(
            rewards=replay["rewards"].to(device),
            values=values,
            ends=ends,
            mask_paddings=mask_paddings,
            gamma=cfg.training.actor_critic.gamma,
            lambda_=cfg.training.actor_critic.lambda_
        )
        advantages = (lambda_returns_raw - values)
        assert len(advantages) == len(ids)
        out.update(zip(ids, [a[m] for a, m in zip(advantages, mask_paddings)]))
        actions.update(zip(ids, [ac[m] for ac, m in zip(replay["actions"], mask_paddings)]))
        for i in range(actual_batch_size):
            emb = torch.stack([e[i] for e in embeddings])[mask_paddings[i]]
            torch.save(emb, embeddings_dir / f"{ids[i]}.pt")
    torch.save(out, "advantages.pt")
    torch.save(actions, "actions.pt")


@hydra.main(config_path=r"C:\Users\Michael\PycharmProjects\Brawl_iris\config", config_name="trainer")
def main_compute_advantages(cfg: DictConfig):
    # todo:
    shutil.copytree(
        Path(
            r"C:\Users\Michael\PycharmProjects\Brawl-Stars-AI\outputs\param_groups_converge\2024-04-08_09-12-01\checkpoints\dataset"),
        Path("checkpoints\dataset"), dirs_exist_ok=True)
    # for f in Path("checkpoints\dataset").glob("*"):
    #     if f.stem.isnumeric() and int(f.stem) > 48:
    #         f.unlink()

    # shutil.copytree(
    #     Path(r"C:\Users\Michael\PycharmProjects\Brawl-Stars-AI\outputs\actor_mlp_head\2024-03-31_22-44-53\checkpoints"),
    #     Path("checkpoints"), dirs_exist_ok=True, ignore=shutil.ignore_patterns("checkpoint*"))
    # for f in Path("checkpoints\dataset").glob("*"):
    #     if f.stem.isnumeric() and int(f.stem) > 257 + 48:
    #         f.unlink()

    start_epoch = 1
    env = None
    actor = instantiate(cfg.actor_critic)

    optimizer = torch.optim.Adam(actor.parameters(),
                                 lr=cfg.training.learning_rate,
                                 betas=cfg.training.actor_critic.adam_betas)
    dataset: EpisodesDataset = instantiate(cfg.datasets.train)
    ac_trainer = ACTrainer(cfg, env, actor, optimizer, dataset, replay_only=True)
    ac_trainer.dataset.load_disk_checkpoint(Path("checkpoints/dataset"))
    ac_trainer.load_checkpoint(
        r"C:\Users\Michael\PycharmProjects\Brawl-Stars-AI\outputs\param_groups_converge\2024-04-08_09-12-01\checkpoints",
        dataset=False, optimizer=False)

    dataloader = iter(dataset.torch_dataloader(ac_trainer.batch_size, random_sampling=False))

    ac_trainer.lr_scheduler.last_epoch = start_epoch - 1
    ac_trainer.batch_size = ac_trainer.batch_size + 1
    compute_advantages(ac_trainer, dataloader, cfg)


def train_actor_head(dataset_path, cfg, epochs=100, do_log=False, head=None):
    from torch.utils.data import DataLoader, Dataset
    from torch.distributions import Bernoulli, Categorical

    class EmbDataset(Dataset):
        def __init__(self, dataset_path):
            dataset_path = Path(dataset_path)
            self.paths = [p for p in (dataset_path / "embeddings").glob("*")]
            self.episodes = [torch.load(p) for p in self.paths]
            self.advantages_grouped = torch.load(dataset_path / "advantages.pt")
            self.actions_grouped = torch.load(dataset_path / "actions.pt")
            self.id_to_ep_id = {}

            for i, ep in enumerate(self.episodes):  # sample_id -> (ep_id, frame_id)
                cur_len = len(self.id_to_ep_id)
                self.id_to_ep_id.update(zip(range(cur_len, cur_len + len(ep)),
                                            zip([i] * len(ep), range(len(ep)))))

        def __len__(self):
            return len(self.id_to_ep_id)

        def __getitem__(self, idx):
            ep_id, frame_id = self.id_to_ep_id[idx]
            ep_name = int(self.paths[ep_id].stem)
            emb = self.episodes[ep_id][frame_id]
            advantage = self.advantages_grouped[ep_name][frame_id]
            action = self.actions_grouped[ep_name][frame_id]
            return emb, action.float(), advantage

    if head is None:
        head = torch.nn.Sequential(
            torch.nn.LazyConv2d(256, 3),
            torch.nn.BatchNorm2d(256), torch.nn.LeakyReLU(0.1),

            torch.nn.LazyConv2d(128, 3),
            torch.nn.BatchNorm2d(128), torch.nn.LeakyReLU(0.1),
            torch.nn.Flatten(),
            torch.nn.LazyLinear(18, bias=False)
        ).to(device)
    head.train()

    dataset = EmbDataset(dataset_path)
    dataloader = DataLoader(dataset, batch_size=512)
    optim = torch.optim.Adam(head.parameters(), lr=0.001)

    def compute_loss(logits, actions, advantages):
        logits_per_action_type = logits.split([1, 1, 1, 1, 4, 4], dim=-1)
        action_per_type = actions.split(1, dim=-1)
        log_probs, entropies = [], []
        for logit, action in zip(logits_per_action_type, action_per_type):
            if logit.size(-1) == 1:
                d = Bernoulli(logits=logit)
            else:
                d = Categorical(logits=logit[:, None, :])
            log_probs.append(d.log_prob(action[:, :]))
            entropies.append(d.entropy())
        log_probs = torch.cat(log_probs, dim=-1)
        metrics = {}
        metrics["actor_critic/train/custom_actions"] = torch.where(
            advantages.unsqueeze(-1) >= 0,
            -log_probs[..., ACTION_LOCK_MASK],
            -torch.log(1 - torch.exp(log_probs))[..., ACTION_LOCK_MASK]
        ).mean().item()
        loss = torch.where(
            advantages.unsqueeze(-1) >= 0,
            -1 * (log_probs * advantages.unsqueeze(-1))[..., ACTION_LOCK_MASK],
            (torch.log(1 - torch.exp(log_probs)) * advantages.unsqueeze(-1))[..., ACTION_LOCK_MASK]
        ).mean()
        metrics["actor_critic/train/custom_weighted_actions"] = loss.item()
        entropy_loss = (cfg.training.actor_critic.entropy_weight *
                        torch.where(logits.abs() > 2, logits, 0)[
                            ..., ACTION_LOGIT_LOCK_MASK] ** 2).mean()
        metrics["actor_critic/train/loss_entropy"] = entropy_loss.item()
        if do_log:
            wandb.log(metrics)
        return loss + entropy_loss

    for n_epoch in tqdm.tqdm(range(epochs), desc="Epoch: "):
        for inp, actions, advantages in dataloader:
            logits = head(inp.to(device))[..., :12]
            loss = compute_loss(logits, actions.to(device), advantages.to(device))
            loss.backward()
            adaptive_gradient_clipping(head.parameters(), lam=cfg.training.actor_critic.agc_lambda)
            optim.step()
            optim.zero_grad()
        torch.save(head, "head.pt")
    return head


@hydra.main(config_path=r"C:\Users\Michael\PycharmProjects\Brawl_iris\config", config_name="trainer")
def main_train_actor_head(cfg: DictConfig):
    wandb.init(config=OmegaConf.to_container(cfg, resolve=True),
               reinit=True,
               resume=True,
               **cfg.wandb)
    train_actor_head("", cfg, epochs=400, do_log=True)


@hydra.main(config_path=r"C:\Users\Michael\PycharmProjects\Brawl_iris\config", config_name="trainer")
def inspect(cfg: DictConfig):
    try:
        cfg.wandb.mode = "offline"
        trainer = Trainer(cfg)

        shutil.copytree(
            Path(
                r"C:\Users\Michael\PycharmProjects\Brawl-Stars-AI\outputs\actor_mlp_head\2024-03-31_22-44-53\checkpoints"),
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
