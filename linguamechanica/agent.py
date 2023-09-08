import torch
import torch.nn.functional as F
import math
from linguamechanica.kinematics import DifferentiableOpenChainMechanism
from linguamechanica.models import InverseKinematicsActor, InverseKinematicsCritic
from torchrl.data import ReplayBuffer
from dataclasses import asdict
import torch.optim as optim
from torchrl.data.replay_buffers import ListStorage


def add_actions_to_thetas(actions, thetas):
    deltas_sin = actions[:, :, 0]
    deltas_cos = actions[:, :, 1]
    deltas = torch.atan2(deltas_sin, deltas_cos)
    return thetas + deltas


def compute_geodesic_loss(thetas, target_pose, open_chain, weights):
    error_pose = open_chain.compute_error_pose(thetas, target_pose)
    error = DifferentiableOpenChainMechanism.compute_weighted_error(
        error_pose, weights.to(thetas.device)
    )
    return error.mean()


def thetas_target_pose_from_state(state):
    return state[:, 6:], state[:, :6]


class IKAgent:
    def __init__(
        self,
        open_chain,
        summary,
        lr_actor,
        lr_actor_geodesic,
        lr_critic,
        state_dims,
        action_dims,
        gamma=0.99,
        policy_freq=8,
        tau=0.005,
        max_action=1.0,
        max_noise_clip=1e-3,
        initial_action_variance=1e-3,
        max_variance=1e-3,
        policy_noise=1e-2,
        replay_buffer_max_size=10000,
    ):
        self.summary = summary
        self.max_noise_clip = max_noise_clip
        self.gamma = gamma
        self.lr_actor = lr_actor
        self.lr_critic = lr_critic
        self.lr_actor_geodesic = lr_actor_geodesic
        self.max_action = max_action
        self.initial_action_variance = initial_action_variance
        self.max_variance = max_variance
        self.policy_noise = policy_noise
        self.open_chain = open_chain
        self.actor = InverseKinematicsActor(
            open_chain=self.open_chain, max_variance=self.max_variance
        ).to(open_chain.device)
        self.actor_target = InverseKinematicsActor(
            open_chain=self.open_chain, max_variance=self.max_variance
        ).to(open_chain.device)
        self.critic_q1 = InverseKinematicsCritic(open_chain=open_chain).to(
            open_chain.device
        )
        self.critic_q2 = InverseKinematicsCritic(open_chain=open_chain).to(
            open_chain.device
        )
        self.critic_target_q1 = InverseKinematicsCritic(open_chain=open_chain).to(
            open_chain.device
        )
        self.critic_target_q2 = InverseKinematicsCritic(open_chain=open_chain).to(
            open_chain.device
        )
        self.replay_buffer = ReplayBuffer(
            storage=ListStorage(max_size=replay_buffer_max_size)
        )
        self.total_it = 0
        self.policy_freq = policy_freq
        self.tau = tau
        self.create_optimizers()

    def create_optimizers(self):
        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=self.lr_actor)
        self.actor_geodesic_optimizer = optim.Adam(
            self.actor.parameters(), lr=self.lr_actor_geodesic
        )
        self.critic_q1_optimizer = optim.Adam(
            self.critic_q1.parameters(), lr=self.lr_critic
        )
        self.critic_q2_optimizer = optim.Adam(
            self.critic_q2.parameters(), lr=self.lr_critic
        )

    def save(self, training_state):
        training_state_dict = asdict(training_state)
        torch.save(
            training_state_dict,
            f"checkpoints/state_{training_state.t + 1}.pt",
        )
        model_dictionary = {
            "critic_target_q1": self.critic_target_q1.state_dict(),
            "critic_target_q2": self.critic_target_q2.state_dict(),
            "critic_q1": self.critic_q1.state_dict(),
            "critic_q2": self.critic_q2.state_dict(),
            "actor_target": self.actor_target.state_dict(),
            "actor": self.actor.state_dict(),
        }
        torch.save(
            model_dictionary,
            f"checkpoints/model_{training_state.t + 1}.pt",
        )

    def load(self, name):
        model_dictionary = torch.load(f"checkpoints/model_{name}.pt")
        self.critic_target_q1.load_state_dict(model_dictionary["critic_target_q1"])
        self.critic_target_q2.load_state_dict(model_dictionary["critic_target_q2"])
        self.critic_q1.load_state_dict(model_dictionary["critic_q1"])
        self.critic_q2.load_state_dict(model_dictionary["critic_q2"])
        self.actor.load_state_dict(model_dictionary["actor"])
        self.actor_target.load_state_dict(model_dictionary["actor_target"])
        training_state_dict = torch.load(f"checkpoints/state_{name}.pt")
        return training_state_dict

    def store_transition(self, state, action, reward, next_state, done):
        """
        Note that the `replay_buffer` is using a `RoundRobinWriter` and
        thus it will get updated with new data despite the storege
        being full.
        """
        # print("store_transition", state.shape, action.shape, reward.shape, next_state.shape, done.shape)
        self.replay_buffer.add([state, action, reward, next_state, done])

    def compute_log_prob(self, mu_v, var_v, actions_v):
        log_prob_part_1 = ((mu_v - actions_v) ** 2) / (2 * var_v.clamp(min=1e-3))
        log_prob_part_2 = torch.log(torch.sqrt(2 * math.pi * var_v))
        # NOTE: It is addition as it is a multiplication in the non-log domain,
        # but in the log space it is a sum. There is a single probability.
        log_prob = -(log_prob_part_1 + log_prob_part_2)
        return log_prob

    def choose_action(self, state, training_state, summary=None):
        mu_v, var_v = None, None
        state = state.to(self.open_chain.device)
        current_thetas, target_pose = thetas_target_pose_from_state(state)
        # TODO: this shouldn't be here
        self.actor = self.actor.cuda()
        mu_v, var_v = self.actor(current_thetas, target_pose)
        # else:
        #    mu_v, var_v = self.jacobian_actor(state)
        # TODO: make (1, 6) constant parametrizable
        """
        mu_v = torch.zeros(1, 6).to(self.jacobian_actor.device)
        var_v = (
            torch.normal(torch.zeros(1, 6), torch.ones(6) * 0.0001)
            .to(self.jacobian_actor.device)
            .abs()
        )
        """
        actions_v, noise = self.sample(mu_v, var_v)
        log_prob = self.compute_log_prob(mu_v, var_v, actions_v)
        if summary is not None:
            summary.add_scalar(
                "Data Distributions / Data Generation Action Mean",
                actions_v.mean(),
                training_state.t,
            )
            summary.add_scalar(
                "Data Distributions / Data Generation Action Std",
                actions_v.std(),
                training_state.t,
            )
            summary.add_scalar(
                "Data Distributions / Noise w.r.t Action Ratio Mean",
                (noise / (actions_v + 1e-20)).mean(),
                training_state.t,
            )
            summary.add_scalar(
                "Data Distributions / Data Generation Noise Std",
                noise.std(),
                training_state.t,
            )
        return actions_v, log_prob, mu_v, noise

    def sample(self, mu, var):
        std = torch.sqrt(var)
        noise = torch.randn_like(mu) * std
        """
            TODO: this might work for angular actuators, but not for
            prismatic actuators. It is necessary a max_noise_clip
            that is congruent with the type of actuator.
        """
        noise = torch.clip(noise, min=-self.max_noise_clip, max=self.max_noise_clip)
        actions = mu + noise
        return actions, noise

    def train_buffer(self, training_state):
        self.total_it += 1
        state, action, reward, next_state, done = self.replay_buffer.sample(
            training_state.batch_size()
        )
        # TODO: do this properly !
        state = state.detach().clone().view(-1, state.shape[2])
        action = action.detach().clone().view(-1, action.shape[2])
        reward = reward.detach().clone().view(-1, reward.shape[2])
        next_state = next_state.detach().clone().view(-1, next_state.shape[2])
        done = done.detach().clone().view(-1, done.shape[2])
        # print("replay buffer sample", training_state.batch_size(), state.shape, action.shape, reward.shape, next_state.shape, done.shape)
        self.summary.add_scalar(
            "Data Distributions / Train Buffer Action Mean",
            action.mean(),
            training_state.t,
        )
        self.summary.add_scalar(
            "Data Distributions / Train Buffer Action Std",
            action.std(),
            training_state.t,
        )

        action = action.to(self.actor.device())
        state = state.to(self.actor.device())
        next_state = next_state.to(self.actor.device())
        reward = reward.to(self.actor.device())
        done = done.to(self.actor.device())

        with torch.no_grad():
            # Select action according to policy and add clipped noise
            # print("if self.state == AgentState.QLEARNING_TRANING", next_state.shape)
            # TODO: do this properly
            next_thetas, next_target_pose = thetas_target_pose_from_state(next_state)
            next_mu, next_var = self.actor_target(next_thetas, next_target_pose)
            # next_actions, noise = self.sample(next_mu, next_var)
            # Compute the target Q value
            # TODO: do this in a nicer way
            next_next_thetas = add_actions_to_thetas(next_mu, next_thetas)
            target_Q1 = self.critic_target_q1(next_next_thetas, next_target_pose)
            target_Q2 = self.critic_target_q2(next_next_thetas, next_target_pose)
            target_Q = reward + (
                (1.0 - done.float()) * self.gamma * torch.min(target_Q1, target_Q2)
            )
        self.critic_q1_optimizer.zero_grad()
        self.critic_q2_optimizer.zero_grad()

        current_thetas, target_pose = thetas_target_pose_from_state(state)
        current_Q1 = self.critic_q1(current_thetas, target_pose)
        current_Q2 = self.critic_q2(current_thetas, target_pose)

        quality_loss = F.mse_loss(current_Q1, target_Q) + F.mse_loss(
            current_Q2, target_Q
        )
        quality_loss.backward()
        # torch.nn.utils.clip_grad_norm_(self.critic.parameters(), 1.0)
        self.critic_q1_optimizer.step()
        self.critic_q2_optimizer.step()
        self.summary.add_scalar(
            "Loss / Train / Quality Loss (Q1 + Q2)",
            quality_loss,
            training_state.t,
        )

        mu, var = self.actor(current_thetas, target_pose)
        next_thetas = add_actions_to_thetas(mu, current_thetas)

        self.actor_geodesic_optimizer.zero_grad()
        actor_geodesic_loss = compute_geodesic_loss(
            next_thetas, target_pose, self.open_chain, training_state.weights
        )
        self.summary.add_scalar(
            "Loss / Train / Actor Geodesic Loss",
            actor_geodesic_loss,
            training_state.t,
        )
        actor_geodesic_loss.backward()
        self.actor_geodesic_optimizer.step()

        # Delayed policy updates
        if self.total_it % self.policy_freq == 0:
            mu, var = self.actor(current_thetas, target_pose)
            next_thetas = add_actions_to_thetas(mu, current_thetas)
            self.actor_optimizer.zero_grad()
            actor_q_learning_loss = -self.critic_q1(next_thetas, target_pose).mean()
            self.summary.add_scalar(
                "Loss / Train / Actor Q Learning Loss",
                actor_q_learning_loss,
                training_state.t,
            )
            actor_q_learning_loss.backward()
            self.actor_optimizer.step()

        if self.total_it % self.policy_freq == 0:
            # Update the frozen target models
            for param, target_param in zip(
                self.critic_q1.parameters(), self.critic_target_q1.parameters()
            ):
                target_param.data.copy_(
                    self.tau * param.data + (1 - self.tau) * target_param.data
                )

            for param, target_param in zip(
                self.critic_q2.parameters(), self.critic_target_q2.parameters()
            ):
                target_param.data.copy_(
                    self.tau * param.data + (1 - self.tau) * target_param.data
                )
            for param, target_param in zip(
                self.actor.parameters(), self.actor_target.parameters()
            ):
                target_param.data.copy_(
                    self.tau * param.data + (1 - self.tau) * target_param.data
                )
