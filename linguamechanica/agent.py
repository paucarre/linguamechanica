import os
from dataclasses import asdict

import torch
import torch.nn.functional as F
import torch.optim as optim
from dacite import from_dict
from torchrl.data import ReplayBuffer
from torchrl.data.replay_buffers import ListStorage

from linguamechanica.environment import Environment
from linguamechanica.kinematics import DifferentiableOpenChainMechanism
from linguamechanica.models import InverseKinematicsActor, InverseKinematicsCritic
from linguamechanica.training_context import TrainingState


def compute_geodesic_loss(thetas, target_pose, open_chain, weights):
    error_pose = open_chain.compute_error_pose(thetas, target_pose)
    error = DifferentiableOpenChainMechanism.compute_weighted_error(
        error_pose, weights.to(thetas.device)
    )
    return error.mean()


class IKAgent:
    def __init__(
        self,
        open_chain,
        summary,
        training_state,
    ):
        self.training_state = training_state
        self.summary = summary
        self.open_chain = open_chain
        self.actor = InverseKinematicsActor(
            open_chain=self.open_chain,
            max_action=self.training_state.max_action_clip,
            max_std_dev=self.training_state.max_std_dev,
        ).to(open_chain.device)
        self.actor_target = InverseKinematicsActor(
            open_chain=self.open_chain,
            max_action=self.training_state.max_action_clip,
            max_std_dev=self.training_state.max_std_dev,
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
            storage=ListStorage(max_size=self.training_state.replay_buffer_max_size())
        )
        self.total_it = 0
        self.create_optimizers()
        self.max_pose = None

    def create_optimizers(self):
        self.actor_optimizer = optim.Adam(
            self.actor.parameters(), lr=self.training_state.lr_actor
        )
        self.critic_q1_optimizer = optim.Adam(
            self.critic_q1.parameters(), lr=self.training_state.lr_critic
        )
        self.critic_q2_optimizer = optim.Adam(
            self.critic_q2.parameters(), lr=self.training_state.lr_critic
        )

    def save(self, training_state):
        checkpoint_path = f"checkpoints/model_{self.training_state.t + 1}.pt"
        if not os.path.exists(checkpoint_path):
            training_state_dict = asdict(self.training_state)
            torch.save(
                training_state_dict,
                f"checkpoints/state_{self.training_state.t + 1}.pt",
            )
            model_dictionary = {
                # Models
                "critic_target_q1": self.critic_target_q1.state_dict(),
                "critic_target_q2": self.critic_target_q2.state_dict(),
                "critic_q1": self.critic_q1.state_dict(),
                "critic_q2": self.critic_q2.state_dict(),
                "actor_target": self.actor_target.state_dict(),
                "actor": self.actor.state_dict(),
                # Optimizers
                "actor_optimizer": self.actor_optimizer.state_dict(),
                "critic_q1_optimizer": self.critic_q1_optimizer.state_dict(),
                "critic_q2_optimizer": self.critic_q2_optimizer.state_dict(),
            }
            torch.save(
                model_dictionary,
                checkpoint_path,
            )

    @staticmethod
    def from_checkpoint(open_chain, checkpoint_id, summary=None):
        model_dictionary = torch.load(f"checkpoints/model_{checkpoint_id}.pt")
        training_state_dict = torch.load(f"checkpoints/state_{checkpoint_id}.pt")
        training_state = from_dict(data_class=TrainingState, data=training_state_dict)
        training_state.t += 1
        agent = IKAgent(
            open_chain,
            summary,
            training_state,
        )
        # Models
        agent.critic_target_q1.load_state_dict(model_dictionary["critic_target_q1"])
        agent.critic_target_q2.load_state_dict(model_dictionary["critic_target_q2"])
        agent.critic_q1.load_state_dict(model_dictionary["critic_q1"])
        agent.critic_q2.load_state_dict(model_dictionary["critic_q2"])
        agent.actor.load_state_dict(model_dictionary["actor"])
        agent.actor_target.load_state_dict(model_dictionary["actor_target"])
        # Optimizers
        agent.actor_optimizer.load_state_dict(model_dictionary["actor_optimizer"])
        agent.critic_q1_optimizer.load_state_dict(
            model_dictionary["critic_q1_optimizer"]
        )
        agent.critic_q2_optimizer.load_state_dict(
            model_dictionary["critic_q2_optimizer"]
        )
        return agent

    def store_transition(self, state, action, reward, next_state, done):
        state = state.detach().cpu()
        action = action.detach().cpu()
        reward = reward.detach().cpu()
        next_state = next_state.detach().cpu()
        done = done.detach().cpu()
        self.replay_buffer.add([state, action, reward, next_state, done])

    def cuda(self):
        self.actor = self.actor.cuda()
        self.actor_target = self.actor_target.cuda()
        self.critic_q1 = self.critic_q1.cuda()
        self.critic_q2 = self.critic_q2.cuda()
        self.critic_target_q1 = self.critic_target_q1.cuda()
        self.critic_target_q2 = self.critic_target_q2.cuda()
        return self

    def inference(self, iterations, state, environment, top_n):
        iteration = 0
        reward = None
        while iteration < iterations:
            thetas, target_pose = Environment.thetas_target_pose_from_state(state)
            action_mean, actions, log_probabilities, entropy = self.choose_action(
                state, self.training_state
            )
            actions, next_state, reward, done, level_increased = environment.step(
                actions
            )
            state = next_state
            iteration += 1
        reward_sorted, indices = torch.sort(reward[:, 0], descending=True)
        thetas_sorted = thetas[indices.to(thetas.device), :][:top_n, :]
        reward_sorted = reward_sorted.to(thetas.device).unsqueeze(1)[:top_n, :]
        return thetas_sorted, reward_sorted

    def choose_action(self, state, training_state):
        mu_v, var_v = None, None
        state = state.to(self.open_chain.device)
        current_thetas, target_pose = Environment.thetas_target_pose_from_state(state)
        # TODO: this shouldn't be here
        self.actor = self.actor.cuda()
        actions_mean, actions, log_probabilities, entropy = self.actor(
            current_thetas, target_pose
        )
        """
        For low pose error, std dev is ignored.
        """
        current_thetas = current_thetas + actions_mean.data
        pose_error = -Environment.compute_reward(
            self.open_chain,
            current_thetas,
            target_pose,
            self.training_state.weights,
        )
        low_pose_error = (
            pose_error.squeeze(1) < self.training_state.zero_entropy_threshold
        )
        actions[low_pose_error, :] = actions_mean[low_pose_error, :]
        if self.summary is not None:
            self.summary.add_scalar(
                "Data / Action Mean",
                actions.mean(),
                training_state.t,
            )
            self.summary.add_scalar(
                "Data / Action Std",
                actions.std(),
                training_state.t,
            )
            self.summary.add_scalar(
                "Data / Log Prob. Mean",
                log_probabilities.mean(),
                training_state.t,
            )
            self.summary.add_scalar(
                "Data / Entropy Mean",
                entropy.std(),
                training_state.t,
            )
        return actions_mean, actions, log_probabilities, entropy

    def update_target_models(self):
        if self.total_it % self.training_state.target_update_freq() == 0:
            for param, target_param in zip(
                self.critic_q1.parameters(), self.critic_target_q1.parameters()
            ):
                target_param.data.copy_(
                    self.training_state.tau * param.data
                    + (1 - self.training_state.tau) * target_param.data
                )

            for param, target_param in zip(
                self.critic_q2.parameters(), self.critic_target_q2.parameters()
            ):
                target_param.data.copy_(
                    self.training_state.tau * param.data
                    + (1 - self.training_state.tau) * target_param.data
                )
            for param, target_param in zip(
                self.actor.parameters(), self.actor_target.parameters()
            ):
                target_param.data.copy_(
                    self.training_state.tau * param.data
                    + (1 - self.training_state.tau) * target_param.data
                )

    def compute_delayed_actor_q_learning_loss(self, state, actions):
        if self.total_it % self.training_state.policy_freq == 0:
            current_thetas, target_pose = Environment.thetas_target_pose_from_state(
                state
            )
            next_thetas = current_thetas + actions
            critic_prediction = self.critic_q1(next_thetas, target_pose)
            actor_q_learning_loss = -critic_prediction.mean()
            if self.summary is not None:
                self.summary.add_scalar(
                    "Train / Actor Q Learning Loss",
                    actor_q_learning_loss,
                    self.training_state.t,
                )
            return actor_q_learning_loss
        return 0.0

    def critic_update(self, state, reward, next_state, done):
        with torch.no_grad():
            next_thetas, next_target_pose = Environment.thetas_target_pose_from_state(
                next_state
            )
            next_actions, _, _, _ = self.actor_target(next_thetas, next_target_pose)
            # Compute the target Q value
            next_next_thetas = next_thetas + next_actions
            target_Q1 = self.critic_target_q1(next_next_thetas, next_target_pose)
            target_Q2 = self.critic_target_q2(next_next_thetas, next_target_pose)
            target_Q = reward + (
                (1.0 - done.float())
                * self.training_state.gamma
                * torch.min(target_Q1, target_Q2)
            )
        self.critic_q1_optimizer.zero_grad()
        self.critic_q2_optimizer.zero_grad()

        current_thetas, target_pose = Environment.thetas_target_pose_from_state(state)
        current_Q1 = self.critic_q1(current_thetas, target_pose)
        current_Q2 = self.critic_q2(current_thetas, target_pose)

        quality_loss = F.mse_loss(current_Q1, target_Q) + F.mse_loss(
            current_Q2, target_Q
        )
        quality_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.critic_q1.parameters(), 1.0)
        torch.nn.utils.clip_grad_norm_(self.critic_q2.parameters(), 1.0)
        self.critic_q1_optimizer.step()
        self.critic_q2_optimizer.step()
        if self.summary is not None:
            self.summary.add_scalar(
                "Train / Quality Loss (Q1 + Q2)",
                quality_loss,
                self.training_state.t,
            )

    def is_corrupted(self, tensor):
        return torch.logical_or(torch.isinf(tensor), torch.isnan(tensor))

    def corrupted_rows(self, tensor):
        corr = self.is_corrupted(tensor).sum(1) > 0
        return tensor[corr, :]

    def compute_actor_geodesic_loss(self, state):
        thetas, target_pose = Environment.thetas_target_pose_from_state(state)
        initial_reward = Environment.compute_reward(
            self.open_chain, thetas, target_pose, self.training_state.weights
        )
        angle_delta = None
        entropy = None
        for rollout in range(
            self.training_state.actor_geodesic_optimization_rollouts()
        ):
            angle_delta, _, _, entropy = self.actor(thetas, target_pose)
            thetas = thetas + angle_delta
        final_reward = Environment.compute_reward(
            self.open_chain,
            thetas,
            target_pose,
            self.training_state.weights,
            self.summary,
            self.training_state.t,
        )
        geodesic_loss = (
            (final_reward - initial_reward.data)
            / (initial_reward.data + self.training_state.geodesic_loss_epsilon)
        ).mean()
        if self.summary is not None:
            self.summary.add_scalar(
                "Train / Final Reward",
                final_reward.mean(),
                self.training_state.t,
            )
        if self.summary is not None:
            self.summary.add_scalar(
                "Train / Actor Reward Times Worse Loss",
                geodesic_loss,
                self.training_state.t,
            )
        return geodesic_loss, entropy, final_reward, angle_delta

    def actor_optimization(self, state):
        (
            geodesic_loss,
            entropy,
            final_reward,
            angle_delta,
        ) = self.compute_actor_geodesic_loss(state)
        actor_entropy_loss = self.compute_actor_entropy_loss(
            entropy, pose_error=-final_reward
        )
        q_learning_actor_loss = self.compute_delayed_actor_q_learning_loss(
            state, angle_delta
        )
        loss = (
            (geodesic_loss * self.training_state.actor_geodesic_weight)
            + (actor_entropy_loss * self.training_state.actor_entropy_weight)
            + (q_learning_actor_loss * self.training_state.actor_q_learning_weight)
        )
        if self.summary is not None:
            self.summary.add_scalar(
                "Train / Actor Loss",
                loss,
                self.training_state.t,
            )

        self.actor_optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.actor.parameters(), 1.0)
        self.actor_optimizer.step()
        return loss

    def sample_from_buffer(self):
        state, action, reward, next_state, done = self.replay_buffer.sample(
            self.training_state.batch_size()
        )
        # TODO: this is adding batches to batches which is not the way it should be done...
        state = state.detach().clone().view(-1, state.shape[2])
        action = action.detach().clone().view(-1, action.shape[2])
        reward = reward.detach().clone().view(-1, reward.shape[2])
        next_state = next_state.detach().clone().view(-1, next_state.shape[2])
        done = done.detach().clone().view(-1, done.shape[2])
        if self.summary is not None:
            self.summary.add_scalar(
                "Data / Train Action Mean",
                action.mean(),
                self.training_state.t,
            )
            self.summary.add_scalar(
                "Data / Train Action Std",
                action.std(),
                self.training_state.t,
            )

        action = action.to(self.actor.device())
        state = state.to(self.actor.device())
        next_state = next_state.to(self.actor.device())
        reward = reward.to(self.actor.device())
        done = done.to(self.actor.device())
        return state, action, reward, next_state, done

    def compute_actor_entropy_loss(self, entropy, pose_error):
        """
        The entropy loss is the negative mean
        because we want to maximize entropy to
        make maximum exploration thus maximizing
        discovery of action space.
        """
        """
        We want the pose error to be at most 1.0
        because we don't want to make entropy modulation
        to amplify it. This is archived by dynamically 
        estimate the `max_pose`.
        """
        current_max_pose, idx = pose_error.data.squeeze(1).max(0)
        current_max_pose = current_max_pose.item()
        if self.max_pose is None or current_max_pose > self.max_pose:
            self.max_pose = current_max_pose
        entropy_weight = (pose_error.data / self.max_pose) ** 2
        """
        If the pose error is small entropy is ignored as it's
        not used.
        """
        entropy_weight[pose_error < self.training_state.zero_entropy_threshold] = 0.0
        actor_entropy_loss = -(entropy.unsqueeze(1) * entropy_weight).mean()
        if self.summary is not None:
            self.summary.add_scalar(
                "Train / Actor Entropy Loss",
                actor_entropy_loss,
                self.training_state.t,
            )
        return actor_entropy_loss

    def train_buffer(self):
        self.total_it += 1
        state, action, reward, next_state, done = self.sample_from_buffer()
        self.actor_optimization(state)
        self.critic_update(state, reward, next_state, done)
        self.update_target_models()
