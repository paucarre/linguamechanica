from dataclasses import dataclass
import torch


@dataclass
class TrainingState:
    level: int = 1
    geodesic_rollouts: int = 3
    geodesic_threshold_done: float = 1e-6
    proportion_successful_to_increase_level: float = 0.7
    episode_batch_size: int = 1024
    max_std_dev = 0.002
    save_freq: int = 10000
    lr_actor: float = 1e-4  # 1e-6
    lr_actor_geodesic: float = 1e-4
    lr_actor_entropy: float = 1e-6
    lr_critic: float = 1e-4
    gamma: float = 0.99
    policy_freq: int = 16
    tau: float = 0.05
    eval_freq: int = 200
    max_time_steps: float = 1e6
    data_generation_without_actor_iterations: int = 20
    qlearning_batch_size: int = 32
    max_action_clip: float = torch.pi
    t: int = 0
    weights = torch.Tensor([1.0, 1.0, 1.0, 1.0, 1.0, 1.0])
    max_steps_done: int = 20
    max_episodes_in_buffer: int = 10

    def initial_theta_std_dev(self):
        return 0.1 * self.level

    def pose_error_successful_threshold(self):
        return max(1.0 / (10 ** max(self.level, 0)), 0.0001)

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

    def step(self, reward, done, step, summary):
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
            """
            How to interpret this:
              => +2 : 2 times worse than initially (200% worsening)
              => +1 : 1 time worse than initially  (100% worsening)
              =>  0  : No improvement, final reward is equal to initial reward
              => -1  : 100% improvement, final reward is zero
            """
            summary.add_scalar(
                f"{self.label} / Reward Times Worse",
                (
                    (final_reward - self.initial_reward) / (self.initial_reward + 1e-10)
                ).mean(),
                step,
            )
            summary.add_scalar(
                f"{self.label} / Acc. Disc. Reward",
                self.discounted_reward.mean(),
                step,
            )
        return everything_is_done
