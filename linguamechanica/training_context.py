from dataclasses import dataclass
import torch


@dataclass
class TrainingState:
    episode_batch_size: int = 64
    save_freq: int = 10000
    lr_actor: float = 1e-5
    lr_actor_geodesic: float = 1e-3
    lr_critic: float = 1e-5
    gamma: float = 0.99
    policy_freq: int = 16
    tau: float = 0.05
    eval_freq: int = 200
    max_time_steps: float = 1e6
    data_generation_without_actor_iterations: int = 20
    qlearning_batch_size: int = 32
    """
    The higher the noise, the more the episodes will explore.
    As the episodes will explore more, the Quality Network Q(a, s)
    will be able to learn from the distribution of the environment,
    ( P(a, s, a', s') and thus be more accurate and less brittle.
    This will stabilize the learning of the policy network 
    as the action network will be as good as the quality network is.
    """
    initial_action_variance: int = 1e-3
    max_variance: int = 1e-3
    max_noise_clip: int = 1e-4
    max_action: int = 0.2
    t: int = 0
    weights = torch.Tensor([1.0, 0.0, 0.0, 0.0, 0.0, 0.0])
    max_steps_done: int = 20
    max_episodes_in_buffer: int = 50

    def replay_buffer_max_size(self):
        return self.max_episodes_in_buffer * self.max_steps_done

    def can_train_buffer(self):
        return self.t >= self.data_generation_without_actor_iterations

    def use_actor_for_data_generation(self):
        return self.t >= self.data_generation_without_actor_iterations

    def can_save(self):
        return (self.t + 1) % self.save_freq == 0 and self.can_train_buffer()

    def can_evaluate_policy(self):
        return (self.t + 1) % self.eval_freq == 0 and self.can_train_buffer()

    def batch_size(self):
        return self.qlearning_batch_size


@dataclass
class EpisodeState:
    gamma = 0
    discounted_reward = None
    discounted_gamma = 0
    initial_reward = None
    done = None

    def __init__(self, label, gamma):
        self.label = label
        self.gamma = gamma
        self.discounted_gamma = gamma

    def step(self, reward, done, step, summary):
        self.done = done.detach().cpu()
        if self.initial_reward is None:
            self.initial_reward = reward.detach().cpu()
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
            summary.add_scalar(
                f"Loss / {self.label} / Final Reward to Initial Reward Ratio in {self.label}",
                (final_reward / (self.initial_reward + 1e-10)).mean(),
                step,
            )
            summary.add_scalar(
                f"Loss / {self.label} / Accumulated Discontinued Reward in {self.label}",
                self.discounted_reward.mean(),
                step,
            )
            self.create_new()
        return everything_is_done

    def create_new(self):
        self.gamma = 0
        self.discounted_reward = None
        self.discounted_gamma = 0
        self.initial_reward = None
        self.done = None
