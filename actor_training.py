import time
from collections import defaultdict

import numpy as np
import torch
from torch.distributions.categorical import Categorical
from torch.distributions.normal import Normal
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

MAX_EPISODES = 1000
BATCH_SIZE = 4
WEIGHT_QUANTILE = 0.5  # value loss quantile to assign episode weights
RANDOM_ACTION = 0.05  # proba
POSITIVE_WEIGHT = 1.0  # weigh positive updates (divides neg ones)

# align weights so that positive episode with loss 0.1 was on par with the
# negative one via loss + neg_ep_w_shift
NEGATIVE_EPISODE_WEIGHT_SHIFT = 0.10

# set action_limit_mask and the desired actions to lock them during the run

# make_move, make_shot, super_ability, use_gadget, move_anchor, shot_anchor
ACTION_LOCK_MASK = torch.tensor([False, True, True, True, False, True])
ACTION_LOCK = torch.tensor([0.0, 0.0, 0.0, 0.0])
#
ACTION_C_LOCK_MASK = torch.tensor([False, True, True])
ACTION_C_LOCK = torch.tensor([0.0, 0.0])


def limit_speed(speed_constraint=2):
    def decorator(func):
        def inner(*args, **kwargs):
            start = time.time()
            out = func(*args, **kwargs)
            extra_time = (1 / speed_constraint) - (time.time() - start) if speed_constraint is not None else 0.0
            if extra_time > 0:
                time.sleep(extra_time)
            else:
                # print(f"off rate by {-extra_time}")
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
    def __init__(self, env: GymEnv, actor: ActorCritic, optimizer, dataset: EpisodesDataset):
        self.env = env
        self.actor = actor
        self.optimizer = optimizer
        self.dataset = dataset

        self.actions, self.rewards, self.dones, self.outputs, self.mask_paddings = [], [], [], [], []
        self.observations = []
        self.metrics = defaultdict(float)

        self.batch_size = BATCH_SIZE

        self._step = 0
        self._mean_lambda_return = 0.0
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
            episode_loss = loss[i][mask_padding[i]]
            episode_loss[episode_loss < 0] = episode_loss[episode_loss < 0] + NEGATIVE_EPISODE_WEIGHT_SHIFT
            episode.weight = torch.quantile(episode_loss,
                                            WEIGHT_QUANTILE).detach().item()

    def reset(self):
        print("ACTrainer.reset: Started reset")
        self.metrics = defaultdict(float)
        self._step = 0

        # sample episodes via prioritized sampling
        if len(self.dataset) >= self.batch_size - 1:
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
        self.observations = [self.env.reset()]
        self.actor.reset(self.batch_size)
        self.actor.train()

    def _update_metrics(self, metrics: dict):
        for k, v in metrics.items():
            name = f"actor_critic/train/{k}"
            self.metrics[name] = v

    @limit_speed(speed_constraint=2)
    def step(self):
        curr_obs = torch.tensor(self.observations[-1].transpose(2, 0, 1)).unsqueeze(0).float() / 255.0
        if self._step < self._replay['ends'].size(1):
            replay_obs = self._replay['observations'][:, self._step]
            replay_mask_padding = self._replay['mask_padding'][:, self._step]
            replay_actions = torch.cat([self._replay['actions'][:, self._step],
                                        self._replay['actions_continuous'][:, self._step]], dim=-1)
            replay_rewards = self._replay['rewards'][:, self._step]
            replay_ends = self._replay['ends'][:, self._step]
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
        output = self.actor.forward(full_obs, mask_padding=full_mask_padding)

        # sample action from s_t
        action, action_cont = self.actor.sample_actions(output, eps=RANDOM_ACTION)
        action = torch.where(ACTION_LOCK_MASK, action[0], ACTION_LOCK)
        action_cont = torch.where(ACTION_C_LOCK_MASK, action_cont[0], ACTION_C_LOCK)
        action_raw = torch.cat([action.flatten(), action_cont.flatten()]).reshape(1, -1)
        action_token = create_token(action.int().numpy().flatten(),
                                    anchors=self.env.move_shot_anchors)
        action_sigmoid = torch.cat([torch.tensor([action_token]),
                                    torch.sigmoid(action_cont).flatten()])
        full_action = torch.cat([action_raw, replay_actions], dim=0)
        self.actions.append(full_action)
        self.outputs.append(output)

        obs, reward, done, _ = self.env.step(action_sigmoid.detach())

        self.observations.append(obs)
        full_rewards = torch.cat([torch.tensor([reward / 8.0]), replay_rewards], dim=0)
        self.rewards.append(full_rewards)
        full_ends = torch.cat([torch.tensor([done]), replay_ends], dim=0)
        self.dones.append(full_ends)

        self._step += 1
        if reward == -100 or self._step > 400:
            self.metrics['buggy_episode'] = 1.0
        else:
            self.metrics['buggy_episode'] = 0.0
        return self.metrics

    @torch.no_grad()
    def _create_episode(self):
        obs = np.stack(self.observations, axis=0)
        obs = torch.ByteTensor(obs).permute(0, 3, 1, 2).contiguous()
        actions = torch.LongTensor(torch.stack(self.actions, dim=1)[0, :self._step, ..., :-3].long())
        actions_continuous = torch.FloatTensor(torch.stack(self.actions, dim=1)[0, :self._step, ..., -3:])
        rewards = torch.FloatTensor(torch.stack(self.rewards, dim=1)[0, :self._step].float())
        ends = torch.LongTensor(torch.stack(self.dones, dim=1)[0, :self._step].long())
        mask_padding = torch.BoolTensor(torch.ones(self._step, dtype=torch.bool))

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
        if self._replay['ends'].size(1) > self._step:
            for i in range(self._step, self._replay['ends'].size(1)):
                full_mask_padding = torch.cat([torch.tensor([False]), self._replay['mask_padding'][:, i]], dim=0)
                self.mask_paddings.append(full_mask_padding)

                obs = torch.cat([torch.zeros_like(self._replay['observations'][[0], 0]),
                                 self._replay['observations'][:, i]], dim=0)
                output = self.actor.forward(obs, mask_padding=full_mask_padding)
                self.outputs.append(output)
                replay_actions = torch.cat([self._replay['actions'][:, self._step],
                                            self._replay['actions_continuous'][:, self._step]], dim=-1)
                full_actions = torch.cat([torch.zeros_like(self.actions[-1][[0]]), replay_actions])
                self.actions.append(full_actions)
                full_rewards = torch.cat([torch.tensor([0.0]), self._replay['rewards'][:, i]], dim=0)
                self.rewards.append(full_rewards)
                full_ends = torch.cat([torch.tensor([True]), self._replay['ends'][:, i]], dim=0)
                self.dones.append(full_ends)

        # update value estimation
        self.rewards[self._step - 2][0] = self.rewards[self._step - 1][0]  # move reward closer bc of lambda returns

        # pop trailing obs
        self.observations.pop(-1)
        self.dataset.add_episode(
            self._create_episode()
        )
        episode_output = ImagineOutput(
            observations=None,
            actions=torch.stack(self.actions, dim=1)[..., :-3],
            actions_continuous=torch.stack(self.actions, dim=1)[..., -3:],
            logits_actions=torch.cat([out.logits_actions for out in self.outputs], dim=1),
            continuous_means=torch.cat([out.mean_continuous for out in self.outputs], dim=1),
            continuous_stds=torch.cat([out.std_continuous for out in self.outputs], dim=1),
            values=torch.stack([out.means_values for out in self.outputs], dim=1).reshape(self.batch_size, -1),
            rewards=torch.stack(self.rewards, dim=1).reshape(self.batch_size, -1),
            ends=torch.stack(self.dones, dim=1).reshape(self.batch_size, -1)
        )

        self.optimizer.zero_grad()
        loss = self.ac_loss(episode_output, torch.stack(self.mask_paddings, dim=1))
        print("ACTrainer.episode_end: backward started")
        loss.loss_total.backward()
        self.optimizer.step()
        self._update_metrics(loss.intermediate_losses)
        self.metrics["actor_critic/train/total_loss"] = loss.loss_total.item()
        self.metrics["reward"] = self.rewards[self._step - 2][0]
        self.metrics["episode_length"] = self._step
        self.metrics["mean_lambda_return"] = self._mean_lambda_return
        return self.metrics

    def ac_loss(self, outputs: ImagineOutput, mask_paddings, entropy_weight=0.001):
        with torch.no_grad():
            lambda_returns = compute_masked_lambda_returns(
                rewards=outputs.rewards,
                values=outputs.values,
                ends=outputs.ends,
                mask_paddings=mask_paddings,
                gamma=0.995,
                lambda_=0.95,
            )
        mask_paddings = torch.logical_and(mask_paddings, outputs.ends.logical_not())  # do not include end into loss
        values = outputs.values

        to_print = torch.stack([values[0], lambda_returns[0]], dim=0).T[mask_paddings[0]]
        print("values and returns\n", to_print.detach().numpy())

        (log_probs, entropy), (log_probs_continuous, entropy_cont) = self.actor.get_proba_entropy(outputs)

        batch_lengths = torch.count_nonzero(mask_paddings, dim=1).clip(1, ).unsqueeze(-1)  # TODO questionable
        update_weight = torch.tensor(self._update_weights).reshape(self.batch_size, 1) * \
                        torch.min(batch_lengths) / batch_lengths
        advantage_factor = lambda_returns - values.detach()
        advantage_factor[advantage_factor < 0] /= POSITIVE_WEIGHT

        # compute losses
        loss_actions = -1 * (log_probs * advantage_factor.unsqueeze(-1))[..., ACTION_LOCK_MASK]
        loss_actions_masked = torch.masked_select(update_weight * loss_actions, mask_paddings.unsqueeze(-1)).mean()

        loss_continuous_actions = -1 * (log_probs_continuous * advantage_factor.unsqueeze(-1))[..., ACTION_C_LOCK_MASK]
        loss_continuous_actions_masked = torch.masked_select(update_weight * loss_continuous_actions,
                                                             mask_paddings.unsqueeze(-1)).mean()

        loss_entropy = torch.masked_select(- entropy_weight * entropy, mask_paddings.unsqueeze(-1)).mean()
        loss_entropy_continuous = torch.masked_select(- entropy_weight * entropy_cont,
                                                      mask_paddings.unsqueeze(-1)).mean()
        loss_values = torch.square(values - lambda_returns)
        loss_values_masked = torch.masked_select(update_weight * loss_values, mask_paddings).mean()

        full_loss = LossWithIntermediateLosses(loss_actions=loss_actions_masked,
                                               loss_continuous_actions=loss_continuous_actions_masked,
                                               loss_values=loss_values_masked,
                                               loss_entropy=loss_entropy,
                                               loss_entropy_continuous=loss_entropy_continuous)

        # episode weight ~ loss per episode
        self._assign_episode_weights(loss_values + loss_actions + loss_continuous_actions, mask_paddings)
        return full_loss


@hydra.main(config_path=r"C:\Users\Mykhailo_Tkachuk\PycharmProjects\Brawl_iris\config", config_name="trainer")
def main(cfg: DictConfig):
    trainer = Trainer(cfg)

    env: GymEnv = trainer.train_collector.env.env
    actor = trainer.agent.actor_critic
    optimizer = trainer.optimizer_actor_critic
    ac_trainer = ACTrainer(env, actor, optimizer, trainer.train_dataset)

    for n_episode in range(MAX_EPISODES):
        ac_trainer.reset()
        skip_update = False
        while not env.done:
            step_metrics = ac_trainer.step()
            if ac_trainer._step % 5 == 0:
                print(f"step value {ac_trainer.outputs[-1].means_values[0].item()}")
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
            episode_metrics = ac_trainer.episode_end()
            print('Logging')
            wandb.log({"epoch": n_episode, **episode_metrics})
            print('Saving checkpoint')
            trainer.save_checkpoint(n_episode, False, flush=False)


if __name__ == "__main__":
    main()
