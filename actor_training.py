import os
import shutil
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
import tqdm
import hydra
from omegaconf import DictConfig
import wandb

from src.dataset import EpisodesDataset
from src.episode import Episode
from src.trainer import Trainer
from src.utils import compute_lambda_returns, LossWithIntermediateLosses
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
ACTION_LOCK_MASK = torch.tensor([True, True, True, False, True, True]).to(device)
ACTION_LOCK = torch.tensor([0.0, 0.0, 0.0, 0.0, 0.0, 0.0]).to(device)
# move_shift, shot_shift, shot_strength
ACTION_C_LOCK_MASK = torch.tensor([True, True, True]).to(device)
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


class ACTrainer:
    def __init__(self,
                 cfg,
                 env: GymEnv,
                 actor: ActorCritic,
                 optimizer,
                 dataset: EpisodesDataset,
                 replay_only=False):
        self.cfg = cfg
        self.env = env
        self.actor = actor.to(device)
        self.optimizer = optimizer
        self.dataset = dataset
        self.replay_only = replay_only

        self.actions, self.rewards, self.dones, self.outputs, self.mask_paddings = [], [], [], [], []
        self.observations = []
        self.metrics = defaultdict(float)

        self.batch_size = self.cfg.training.actor_critic.batch_num_samples

        self.collection_num_step = 0
        self._replay = None
        self._active_episodes = None
        self._update_weights = None

    def get_episode_proba(self, rank=True, alpha=0.7, beta=0.5):
        episode_weights = np.array([getattr(ep, 'weight', 0.0) for ep in self.dataset.episodes])
        if rank:
            criterion_rank = np.argsort(episode_weights, )
            criterion = np.zeros_like(episode_weights)
            criterion[criterion_rank] = 1 / (len(episode_weights) - np.arange(len(episode_weights)))
        else:
            criterion = episode_weights + 1E-3

        assert all(criterion > 0.0)

        probas = criterion ** alpha
        probas = probas / np.sum(probas)
        update_weights = 1 / (len(self.dataset) * probas) ** beta
        return probas, update_weights

    @torch.no_grad()
    def _assign_episode_weights(self, loss, mask_padding):
        for i, episode in enumerate(self._active_episodes):
            if episode is None:
                continue
            episode_loss = loss[i][mask_padding[i]]  # for scale with what is logged
            old_weight = episode.weight
            new_weight = torch.quantile(episode_loss, WEIGHT_QUANTILE).detach().item()
            episode.weight = new_weight
            print(f"Changed weight: {old_weight:.3f} -> {new_weight:.3f}")

    def reset(self):
        print("ACTrainer.reset: Started reset")
        self.metrics = defaultdict(float)
        self.collection_num_step = 0

        # sample episodes via prioritized sampling
        if len(self.dataset) >= self.batch_size - 1 and self.batch_size > 1:
            probas, weights = self.get_episode_proba()
            episode_ids = np.random.choice(np.arange(len(self.dataset)),
                                           size=(self.batch_size - 1,),
                                           p=probas,
                                           replace=False)
            self._active_episodes = [None] + [self.dataset.episodes[i] for i in episode_ids]
            self._update_weights = np.concatenate([np.ones(1), weights[episode_ids]])
            self._replay = self.dataset.sample_replay(samples=episode_ids)
        else:
            self._update_weights = np.ones(self.batch_size)
            self._replay = self._get_placeholder_replay()
            self._active_episodes = [None]

        self.actions, self.rewards, self.dones, self.outputs, self.mask_paddings = [], [], [], [], []
        self.observations = [self.env.reset()] if not self.replay_only else []
        self.actor.reset(self.batch_size)
        self.actor.train()

    def _update_metrics(self, metrics: dict):
        for k, v in metrics.items():
            name = f"actor_critic/train/{k}"
            self.metrics[name] = v

    @limit_speed(speed_constraint=2)
    def step(self):
        curr_obs = torch.tensor(self.observations[-1].transpose(2, 0, 1)).unsqueeze(0).float() / 255.0
        if self.collection_num_step < self._replay['ends'].size(1):
            replay_obs = self._replay['observations'][:, self.collection_num_step]
            replay_mask_padding = self._replay['mask_padding'][:, self.collection_num_step]
            replay_actions = torch.cat([self._replay['actions'][:, self.collection_num_step],
                                        self._replay['actions_continuous'][:, self.collection_num_step]], dim=-1)
            replay_rewards = self._replay['rewards'][:, self.collection_num_step]
            replay_ends = self._replay['ends'][:, self.collection_num_step]
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

        self.collection_num_step += 1
        if reward == -100 or self.collection_num_step > 400:
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

        self.collection_num_step += 1
        if reward == -100 or self.collection_num_step > 400:
            self.metrics['buggy_episode'] = 1.0
        else:
            self.metrics['buggy_episode'] = 0.0
        return self.metrics

    @torch.no_grad()
    def _create_episode(self):
        obs = np.stack(self.observations, axis=0)
        obs = torch.ByteTensor(obs).permute(0, 3, 1, 2).contiguous()
        actions = torch.LongTensor(torch.stack(self.actions, dim=1)[0, :self.collection_num_step, ..., :-3].long().cpu())
        actions_continuous = torch.FloatTensor(torch.stack(self.actions, dim=1)[0, :self.collection_num_step, ..., -3:].cpu())
        rewards = torch.FloatTensor(torch.stack(self.rewards, dim=1)[0, :self.collection_num_step].float())
        ends = torch.LongTensor(torch.stack(self.dones, dim=1)[0, :self.collection_num_step].long())
        mask_padding = torch.BoolTensor(torch.ones(self.collection_num_step, dtype=torch.bool))

        episode = Episode(
            observations=obs,
            actions=actions,
            actions_continuous=actions_continuous,
            rewards=rewards,
            ends=ends,
            mask_padding=mask_padding
        )
        # set for convenient weight assignment after loss compute
        self._active_episodes[0] = episode
        return episode

    def _get_placeholder_replay(self):
        return {
            'observations': torch.zeros((self.batch_size - 1, 1, 3, 192, 192)),
            'actions': torch.zeros((self.batch_size - 1, 1, 6)),
            'actions_continuous': torch.zeros((self.batch_size - 1, 1, 3)),
            'rewards': torch.zeros((self.batch_size - 1, 1)),
            'ends': torch.ones((self.batch_size - 1, 1), dtype=torch.bool),
            'mask_padding': torch.zeros((self.batch_size - 1, 1), dtype=torch.bool)
        }

    def episode_end(self):
        # finish replaying if needed
        if self._replay['ends'].size(1) > self.collection_num_step:
            for i in range(self.collection_num_step, self._replay['ends'].size(1)):
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

        if not self.replay_only:
            # try to learn neg reward
            true_reward = self.rewards[self.collection_num_step - 1][0].item()
            self.rewards[self.collection_num_step - 1][0] = true_reward if true_reward > 0.375 else (
                        true_reward - 0.375) / 1.375  # positive starts at rank 3
            # update value estimation
            self.rewards[self.collection_num_step - 2][0] = self.rewards[self.collection_num_step - 1][
                0]  # move reward closer bc of lambda returns

            self.metrics["reward"] = self.rewards[self.collection_num_step - 2][0]
            self.metrics["true_reward"] = true_reward
            self.metrics["episode_length"] = self.collection_num_step

            # pop trailing obs
            self.observations.pop(-1)
            self.dataset.add_episode(
                self._create_episode()
            )

        episode_output = ImagineOutput(
            observations=None,
            actions=torch.stack(self.actions, dim=1)[..., :-3].to(device),
            actions_continuous=torch.stack(self.actions, dim=1)[..., -3:].to(device),
            logits_actions=torch.cat([out.logits_actions for out in self.outputs], dim=1).to(device),
            continuous_means=torch.cat([out.mean_continuous for out in self.outputs], dim=1).to(device),
            continuous_stds=torch.cat([out.std_continuous for out in self.outputs], dim=1).to(device),
            values=torch.stack([out.means_values for out in self.outputs], dim=1).reshape(self.batch_size, -1).to(
                device),
            rewards=torch.stack(self.rewards, dim=1).reshape(self.batch_size, -1).to(device),
            ends=torch.stack(self.dones, dim=1).reshape(self.batch_size, -1).to(device)
        )

        self.optimizer.zero_grad()
        loss = self.ac_loss(episode_output, torch.stack(self.mask_paddings, dim=1).to(device))
        print("ACTrainer.episode_end: backward started")
        loss.loss_total.backward()
        torch.nn.utils.clip_grad_norm_(self.actor.parameters(), self.cfg.training.actor_critic.max_grad_norm)
        self.optimizer.step()
        self._update_metrics(loss.intermediate_losses)
        self.metrics["actor_critic/train/total_loss"] = loss.loss_total.item()
        return self.metrics

    def ac_loss(self, outputs: ImagineOutput, mask_paddings):
        with torch.no_grad():
            lambda_returns = compute_masked_lambda_returns(
                rewards=outputs.rewards,
                values=outputs.values,
                ends=outputs.ends,
                mask_paddings=mask_paddings,
                gamma=self.cfg.training.actor_critic.gamma,
                lambda_=self.cfg.training.actor_critic.lambda_,
            )
        mask_paddings = torch.logical_and(mask_paddings, outputs.ends.logical_not())  # do not include end into loss
        values = outputs.values

        to_print = torch.stack([values[0], lambda_returns[0]], dim=0).T[mask_paddings[0]]
        print("values and returns\n", to_print.cpu().detach().numpy())

        (log_probs, entropy), (log_probs_continuous, entropy_cont) = self.actor.get_proba_entropy(outputs)

        batch_lengths = torch.count_nonzero(mask_paddings, dim=1).unsqueeze(-1)  # TODO questionable
        update_weight = torch.tensor(self._update_weights, device=device).reshape(self.batch_size, 1) * \
                        torch.min(batch_lengths[batch_lengths > 0].flatten()) / batch_lengths
        advantage_factor = lambda_returns - values.detach()
        advantage_factor[advantage_factor < 0] /= POSITIVE_WEIGHT

        # compute losses
        loss_actions = -1 * (log_probs * advantage_factor.unsqueeze(-1))[..., ACTION_LOCK_MASK]
        loss_actions_masked = torch.masked_select(update_weight.unsqueeze(-1) * loss_actions,
                                                  mask_paddings.unsqueeze(-1)).mean()

        loss_continuous_actions = -1 * (log_probs_continuous.clamp(-5, 5) * advantage_factor.unsqueeze(-1))[
            ..., ACTION_C_LOCK_MASK]
        loss_continuous_actions_masked = torch.masked_select(update_weight.unsqueeze(-1) * loss_continuous_actions,
                                                             mask_paddings.unsqueeze(-1)).mean()

        loss_entropy = torch.masked_select(
            - self.cfg.training.actor_critic.entropy_weight * entropy[..., ACTION_LOCK_MASK],
            mask_paddings.unsqueeze(-1)).mean()
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
        print('update weights:', self._update_weights)
        # episode weight ~ absolute loss per episode
        loss_actions = loss_actions.detach()
        loss_actions[loss_actions < 0] = loss_actions[loss_actions < 0] + NEGATIVE_EPISODE_WEIGHT_SHIFT
        loss_continuous_actions = loss_entropy_continuous.detach()
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
        Path(r"C:\Users\Michael\PycharmProjects\Brawl-Stars-AI\outputs\2023-12-08\init_data\checkpoints\dataset"),
        Path("checkpoints\dataset"), dirs_exist_ok=True)
    for f in Path("checkpoints\dataset").glob("*"):
        if int(f.stem) > 32:
            f.unlink()
    trainer.train_dataset.load_disk_checkpoint(trainer.ckpt_dir / 'dataset')
    # trainer.load_checkpoint()
    env: GymEnv = trainer.train_collector.env.env
    actor = trainer.agent.actor_critic
    actor.checkpoint_backbone = True
    optimizer = trainer.optimizer_actor_critic
    ac_trainer = ACTrainer(trainer.cfg, env, actor, optimizer, trainer.train_dataset)

    for n_episode in range(MAX_EPISODES):
        ac_trainer.reset()
        skip_update = False
        while not env.done:
            step_metrics = ac_trainer.step()
            if ac_trainer.collection_num_step % 10 == 0:
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
                continue
            episode_metrics = ac_trainer.episode_end()
            print('Logging')
            wandb.log({"epoch": n_episode, **episode_metrics})
            print('Saving checkpoint')
            if (n_episode + 1) % 50 == 0:
                shutil.copytree('checkpoints', f'checkpoints/checkpoint_{n_episode}',
                                ignore=shutil.ignore_patterns("dataset*", "checkpoint*"))
            trainer.save_checkpoint(n_episode, False, flush=False)


@hydra.main(config_path=r"C:\Users\Michael\PycharmProjects\Brawl_iris\config", config_name="trainer")
def main_replay(cfg: DictConfig):
    trainer = Trainer(cfg)
    shutil.copytree(r"C:\Users\Mykhailo_Tkachuk\PycharmProjects\Brawl-Stars-AI\outputs\2023-08-18\22-50-03\checkpoints",
                    "checkpoints", dirs_exist_ok=True)
    trainer.load_checkpoint()

    env = None
    actor = trainer.agent.actor_critic
    optimizer = trainer.optimizer_actor_critic
    ac_trainer = ACTrainer(trainer.cfg, env, actor, optimizer, trainer.train_dataset, replay_only=True)
    ac_trainer.batch_size = 4 + 1

    for n_step in tqdm.tqdm(range(MAX_EPISODES), desc="Step: ", total=MAX_EPISODES):
        ac_trainer.reset()
        episode_metrics = ac_trainer.episode_end()
        print('Logging')
        wandb.log({"epoch": n_step, **episode_metrics})
        if n_step % 5 == 0:
            print('Saving checkpoint')
            trainer.save_checkpoint(n_step, True)


@hydra.main(config_path=r"C:\Users\Michael\PycharmProjects\Brawl_iris\config", config_name="trainer")
def inspect(cfg: DictConfig):
    trainer = Trainer(cfg)

    shutil.copytree(
        Path(r"C:\Users\Michael\PycharmProjects\Brawl-Stars-AI\outputs\2023-12-10\00-32-46\checkpoints"),
        Path("checkpoints"), dirs_exist_ok=True)
    trainer.load_checkpoint()
    env: GymEnv = trainer.train_collector.env.env
    actor = trainer.agent.actor_critic
    actor.checkpoint_backbone = True
    optimizer = trainer.optimizer_actor_critic
    ac_trainer = ACTrainer(trainer.cfg, env, actor, optimizer, trainer.train_dataset, replay_only=True)
    ac_trainer.reset()
    ac_trainer.episode_end()
    shutil.rmtree(os.getcwd())


if __name__ == "__main__":
    main()
