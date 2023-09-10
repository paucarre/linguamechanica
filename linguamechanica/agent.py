import torch
import torch.nn.functional as F
from linguamechanica.kinematics import DifferentiableOpenChainMechanism
from linguamechanica.models import InverseKinematicsActor, InverseKinematicsCritic
from torchrl.data import ReplayBuffer
from dataclasses import asdict
import torch.optim as optim
from torchrl.data.replay_buffers import ListStorage
from dacite import from_dict
from linguamechanica.training_context import TrainingState
import os
from linguamechanica.environment import Environment


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

    def create_optimizers(self):
        self.actor_optimizer = optim.Adam(
            self.actor.parameters(), lr=self.training_state.lr_actor
        )
        self.actor_geodesic_optimizer = optim.Adam(
            self.actor.parameters(), lr=self.training_state.lr_actor_geodesic
        )
        self.actor_entropy_optimizer = optim.Adam(
            self.actor.parameters(), lr=self.training_state.lr_actor_entropy
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
                "actor_geodesic_optimizer": self.actor_geodesic_optimizer.state_dict(),
                "actor_entropy_optimizer": self.actor_entropy_optimizer.state_dict(),
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
        agent.actor_geodesic_optimizer.load_state_dict(model_dictionary["actor_geodesic_optimizer"])
        agent.actor_entropy_optimizer.load_state_dict(model_dictionary["actor_entropy_optimizer"])
        agent.critic_q1_optimizer.load_state_dict(model_dictionary["critic_q1_optimizer"])
        agent.critic_q2_optimizer.load_state_dict(model_dictionary["critic_q2_optimizer"])
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

    def choose_action(self, state, training_state):
        mu_v, var_v = None, None
        state = state.to(self.open_chain.device)
        current_thetas, target_pose = Environment.thetas_target_pose_from_state(state)
        # TODO: this shouldn't be here
        self.actor = self.actor.cuda()
        actions_mean, actions, log_probabilities, entropy = self.actor(
            current_thetas, target_pose
        )
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
        if self.total_it % self.training_state.policy_freq == 0:
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

    def delayed_actor_update(self, state):
        if self.total_it % self.training_state.policy_freq == 0:
            current_thetas, target_pose = Environment.thetas_target_pose_from_state(
                state
            )
            actions, _, _, _ = self.actor(current_thetas, target_pose)
            next_thetas = current_thetas + actions
            self.actor_optimizer.zero_grad()
            actor_q_learning_loss = -self.critic_q1(next_thetas, target_pose).mean()
            if self.summary is not None:
                self.summary.add_scalar(
                    "Train / Actor Q Learning Loss",
                    actor_q_learning_loss,
                    self.training_state.t,
                )
            actor_q_learning_loss.backward()
            torch.nn.utils.clip_grad_norm_(self.actor.parameters(), 1.0)
            self.actor_optimizer.step()

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

    def compute_reward(self, thetas, target_pose):
        error_pose = self.open_chain.compute_error_pose(thetas, target_pose)
        pose_error = DifferentiableOpenChainMechanism.compute_weighted_error(
            error_pose, self.training_state.weights
        )
        return -pose_error.unsqueeze(1)

    def actor_geodesic_optimization(self, state, epsilon=1e-10):
        thetas, target_pose = Environment.thetas_target_pose_from_state(state)
        for rollout in range(self.training_state.geodesic_rollouts):
            angle_delta_mean, _, _, _ = self.actor(thetas, target_pose)
            thetas = thetas + angle_delta_mean
        loss = -self.compute_reward(thetas=thetas, target_pose=target_pose).mean()
        if self.summary is not None:
            self.summary.add_scalar(
                "Train / Actor Reward Loss",
                loss,
                self.training_state.t,
            )
        self.actor_geodesic_optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.actor.parameters(), 1.0)
        self.actor_geodesic_optimizer.step()
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

    def actor_entropy_update(self, state):
        current_thetas, target_pose = Environment.thetas_target_pose_from_state(state)
        _, _, _, entropy = self.actor(current_thetas, target_pose)
        self.actor_entropy_optimizer.zero_grad()
        actor_entropy_loss = entropy.mean()
        if self.summary is not None:
            self.summary.add_scalar(
                "Train / Actor Entropy Loss",
                actor_entropy_loss,
                self.training_state.t,
            )
        actor_entropy_loss.backward()
        self.actor_entropy_optimizer.step()

    def train_buffer(self):
        self.total_it += 1
        state, action, reward, next_state, done = self.sample_from_buffer()
        self.critic_update(state, reward, next_state, done)
        self.actor_geodesic_optimization(state)
        self.delayed_actor_update(state)
        self.actor_entropy_update(state)
        self.update_target_models()
