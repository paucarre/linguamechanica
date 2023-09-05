from dataclasses import dataclass
import torch


@dataclass
class TrainingState:
    episode_batch_size: int = 64
    save_freq: int = 1000
    lr_actor: float = 1e-3
    lr_critic: float = 1e-3
    gamma: float = 0.99
    policy_freq: int = 16
    tau: float = 0.05
    eval_freq: int = 200
    max_timesteps: float = 1e6
    data_generation_without_actor_iterations: int = 20
    qlearning_batch_size: int = 1024  # 32
    '''
    The higher the noise, the more the episodes will explore.
    As the episodes will explore more, the Quality Network Q(a, s)
    will be able to learn from the distribution of the environment,
    ( P(a, s, a', s') and thus be more accurate and less brittle.
    This will estabilize the learning of the policy network 
    as the action network will be as good as the quality network is.
    '''
    initial_action_variance: int = 1e-5
    max_variance: int = 2e-3
    max_noise_clip: int = 2e-5
    max_action: int = 0.2
    t: int = 0
    weights = torch.Tensor([1.0, 0.0, 0.0, 0.0, 0.0, 0.0])
    max_steps_done: int = 20
    max_episodes_in_buffer:int = 50

    def replay_buffer_max_size(self):
        return self.max_episodes_in_buffer * self.max_steps_done

    def can_train_buffer(self):
        return self.t >= self.data_generation_without_actor_iterations

    def use_actor_for_data_generation(self):
        return self.t >= self.data_generation_without_actor_iterations

    def can_save(self):
        return (
            self.t + 1
        ) % self.save_freq == 0 and self.can_train_buffer()

    def can_eval_policy(self):
        return (
            self.t + 1
        ) % self.eval_freq == 0 and self.can_train_buffer()

    def batch_size(self):
        return self.qlearning_batch_size


@dataclass
class EpisodeState:
    reward = 0
    timesteps = 0
    num = 0
    gamma = 0
    discounted_gamma = 0

    def __init__(self, gamma):
        self.gamma = gamma
        self.discounted_gamma = gamma

    def step(self, reward):
        self.reward = self.reward + (self.discounted_gamma * reward)
        self.discounted_gamma *= self.gamma

    def create_new(self):
        self.reward = 0
        self.timesteps = 0
        self.num += 1
        self.discounted_gamma = self.gamma


class EpisodeBatch:

    def __init__(self, open_chain, batch_size):
        self.create_new()
        self.open_chain = open_chain
        self.batch_size = batch_size

    def step(self):
        next_state, reward, done = env.step(action)

    def create_new(self):
        self.actions = []
        self.states = []
        self.next_states = []
        self.rewards = []
        self.dones = []
        episode_batch_index += 1