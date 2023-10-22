from dataclasses import dataclass

import torch


@dataclass
class TrainingState:
    # Level Configuration
    max_level: int = 4
    level: int = 1
    proportion_successful_to_increase_level: float = 0.9
    # Training Configuration
    geodesic_loss_epsilon: float = 1e-3
    zero_entropy_threshold: float = 0.1
    geodesic_max_rollouts: int = 3
    episode_batch_size: int = 1024
    max_std_dev = 0.002
    save_freq: int = 1000
    lr_actor: float = 1e-4
    actor_geodesic_weight: float = 1.0
    actor_entropy_weight: float = 1e-10
    actor_q_learning_weight: float = 1.0
    lr_critic: float = 1e-4
    gamma: float = 0.99
    policy_freq: int = 4
    target_update_freq_cte: int = 16
    tau: float = 0.05
    qlearning_batch_size: int = 32
    max_action_clip: float = torch.pi
    max_time_steps: float = 1e8
    max_episodes_in_buffer: int = 10
    data_generation_without_actor_iterations: int = 20
    t: int = 0
    eval_freq: int = 200
    # Error Configuration
    weights = torch.Tensor([1.0, 1.0, 1.0, 1.0, 1.0, 1.0])
    max_steps_done: int = 20

    def training_is_finished(self):
        return self.level > self.max_level or self.t > self.max_time_steps

    def actor_geodesic_optimization_rollouts(self):
        return min(
            self.level,
            self.max_steps_done,
            self.geodesic_max_rollouts,
        )

    def target_update_freq(self):
        return self.target_update_freq_cte

    def initial_theta_std_dev(self):
        return torch.pi

    def pose_error_successful_threshold(self):
        return max(1.0 / (100.0 * (2**self.level)), 1e-4)

    def replay_buffer_max_size(self):
        return (
            self.max_episodes_in_buffer
            * self.max_steps_done
            * self.episode_batch_size
            * self.qlearning_batch_size
        )

    def can_train_buffer(self):
        return self.t >= self.data_generation_without_actor_iterations

    def use_actor_for_data_generation(self):
        return self.t >= self.data_generation_without_actor_iterations

    def can_save(self):
        return ((self.t + 1) % self.save_freq == 0) and self.can_train_buffer()

    def can_evaluate_policy(self):
        return ((self.t + 1) % self.eval_freq == 0) and self.can_train_buffer()

    def batch_size(self):
        return self.qlearning_batch_size


@dataclass
class EpisodeState:
    gamma = 0
    discounted_reward = None
    discounted_gamma = 0
    initial_reward = None
    done = None

    def __init__(self, label, initial_reward, gamma):
        self.label = label
        self.gamma = gamma
        self.initial_reward = initial_reward.detach().cpu()
        self.discounted_gamma = gamma

    def step(self, reward, done, time_step, summary):
        self.done = done.detach().cpu()
        if self.discounted_reward is None:
            self.discounted_reward = reward.detach().cpu()
        else:
            self.discounted_reward[self.done == 0] += (
                self.discounted_gamma * reward[self.done == 0].detach().cpu()
            )
        self.discounted_gamma *= self.gamma
        everything_is_done = self.done[self.done == 1].shape[0] == self.done.shape[0]
        if everything_is_done:
            final_reward = reward.detach().cpu()
            reward_times_worse = (
                (final_reward - self.initial_reward) / (self.initial_reward + 1e-10)
            ).mean()
            """
            How to interpret this:
              => +2 : 2 times worse than initially (200% worsening)
              => +1 : 1 time worse than initially  (100% worsening)
              =>  0  : No improvement, final reward is equal to initial reward
              => -1  : 100% improvement, final reward is zero
            """
            summary.add_scalar(
                f"{self.label} / Done Reward Times Worse",
                reward_times_worse,
                time_step,
            )
            summary.add_scalar(
                f"{self.label} / Done Acc. Disc. Reward",
                self.discounted_reward.mean(),
                time_step,
            )
        return everything_is_done
