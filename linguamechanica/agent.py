import os
from dataclasses import asdict

import torch
import torch.nn.functional as F
import torch.optim as optim
from dacite import from_dict
from torchmetrics.aggregation import RunningMean
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
        self.actor_q_loss_running_mean = RunningMean(window=500).to(open_chain.device)
        self.q_loss_running_mean = RunningMean(window=500).to(open_chain.device)

    def create_optimizers(self):
        self.actor_geodesic_optimizer = optim.Adam(
            self.actor.parameters(), lr=self.training_state.lr_actor_geodesic()
        )
        self.actor_optimizer = optim.Adam(
            self.actor.parameters(), lr=self.training_state.lr_actor
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
        agent.actor_geodesic_optimizer.load_state_dict(
            model_dictionary["actor_geodesic_optimizer"]
        )
        agent.actor_entropy_optimizer.load_state_dict(
            model_dictionary["actor_entropy_optimizer"]
        )
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
        self.actor_q_loss_running_mean = self.actor_q_loss_running_mean.cuda()
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

    def delayed_actor_update(self, state):
        if self.total_it % self.training_state.policy_freq == 0:
            current_thetas, target_pose = Environment.thetas_target_pose_from_state(
                state
            )
            actions, _, _, _ = self.actor(current_thetas, target_pose)
            next_thetas = current_thetas + actions
            self.actor_optimizer.zero_grad()
            critic_prediction = self.critic_q1(next_thetas, target_pose)
            # current_actor_q_loss_running_mean = self.actor_q_loss_running_mean.compute()
            actor_q_learning_loss = -critic_prediction.mean()
            if self.summary is not None:
                self.summary.add_scalar(
                    "Train / Actor Q Learning Loss",
                    actor_q_learning_loss,
                    self.training_state.t,
                )
            # actor_q_learning_loss_derivative_error = (
            #    current_actor_q_loss_running_mean - actor_q_learning_loss
            # ).abs() / (current_actor_q_loss_running_mean.abs() + 1e-6)
            # if self.summary is not None:
            #    self.summary.add_scalar(
            #        "Train / Actor Q Learning Derivative Error",
            #        actor_q_learning_loss_derivative_error,
            #        self.training_state.t,
            #    )
            # actor_q_learning_derivative_correction = 1.0 / max(
            #    1.0, 100.0 * actor_q_learning_loss_derivative_error
            # )
            # if self.summary is not None:
            #    self.summary.add_scalar(
            #        "Train / Actor Q Learning Derivative Correction",
            #        actor_q_learning_derivative_correction,
            #        self.training_state.t,
            #    )
            # self.actor_q_loss_running_mean(actor_q_learning_loss)
            # Only use derivative loss regulation if there is a down-regulation (dampening)
            # if (
            #    not math.isnan(actor_q_learning_derivative_correction)
            #    and actor_q_learning_derivative_correction < 1.0
            # ):
            # actor_q_learning_loss = (
            #    actor_q_learning_loss #* actor_q_learning_derivative_correction
            # )
            # if self.summary is not None:
            #   self.summary.add_scalar(
            #        "Train / Actor Q Learning Loss Corrected",
            #        actor_q_learning_loss,
            #        self.training_state.t,
            #    )
            actor_q_learning_loss.backward()
            torch.nn.utils.clip_grad_norm_(
                self.actor.parameters(), self.training_state.delayed_actor_grad_clip()
            )
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
        self.q_loss_running_mean(quality_loss)
        quality_loss.backward()
        torch.nn.utils.clip_grad_norm_(
            self.critic_q1.parameters(), self.training_state.critic_clip()
        )
        torch.nn.utils.clip_grad_norm_(
            self.critic_q2.parameters(), self.training_state.critic_clip()
        )
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


    def actor_geodesic_optimization(self, state):
        thetas, target_pose = Environment.thetas_target_pose_from_state(state)
        thetas.requires_grad=True
        target_pose.requires_grad=True
        initial_reward = Environment.compute_reward(
            self.open_chain, thetas, target_pose, self.training_state.weights
        ).mean()
        for rollout in range(
            self.training_state.actor_geodesic_optimization_rollouts()
        ):
            angle_delta_mean, _, _, _ = self.actor(thetas, target_pose)
            thetas = thetas + angle_delta_mean
        final_reward = Environment.compute_reward(
            self.open_chain,
            thetas,
            target_pose,
            self.training_state.weights,
            self.summary,
            self.training_state.t,
        ).mean()
        if self.summary is not None:
            self.summary.add_scalar(
                "Train / Final Reward",
                final_reward,
                self.training_state.t,
            )
        loss = (
            (final_reward - initial_reward.data) / (initial_reward.data + 1e-3)
        ).mean()
        #self.actor.retain_grad()
        loss.retain_grad()
        if self.summary is not None:
            self.summary.add_scalar(
                "Train / Actor Reward Times Worse Loss",
                loss,
                self.training_state.t,
            )
        self.actor_geodesic_optimizer.zero_grad()
        thetas.retain_grad()
        target_pose.retain_grad()
        loss.backward()
        '''

        '''
        for idx, param in enumerate(self.actor.parameters()):            
            if param.grad is not None:
                if self.is_corrupted(param).sum() > 0 or self.is_corrupted(param.grad).sum() > 0 or self.is_corrupted(loss).sum() > 0:
                    print("------------- Actor Grad -------------")
                    torch.set_printoptions(precision=20)
                    print(self.is_corrupted(param).sum() > 0, self.is_corrupted(param.grad).sum() >0 ,  self.is_corrupted(loss).sum() > 0)
                    print(f"Time Step {self.training_state.t}")
                    print(f"Current Gradient: {idx}")
                    print("loss", loss)
                    print("self.corrupted_rows(thetas.grad)", self.corrupted_rows(thetas.grad))
                    print("self.corrupted_rows(target_pose.grad)", self.corrupted_rows(target_pose.grad))                    
                    print("param", param)
                    print("param.grad.shape", param.grad.shape)
                    print("param.grad", self.corrupted_rows(param.grad))
                    print("thetas", thetas)
                    print("target_pose", target_pose)
                    rows_with_corrupted_values = self.is_corrupted(target_pose.grad).sum(1) > 0
                    print("param.grad with rows_with_corrupted_values", param.grad[rows_with_corrupted_values, :])
                    print("--------------------------------------")
        torch.nn.utils.clip_grad_norm_(
            self.actor.parameters(), self.training_state.gradient_clip_actor_geodesic()
        )
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
        self.actor_geodesic_optimization(state)
        #self.critic_update(state, reward, next_state, done)
        #self.delayed_actor_update(state)
        #self.actor_entropy_update(state)
        self.update_target_models()


if __name__ == "__main__":
    from linguamechanica.kinematics import UrdfRobotLibrary
    from linguamechanica.se3 import ImplicitDualQuaternion
    urdf_robot = UrdfRobotLibrary.dobot_cr5()
    se3 = ImplicitDualQuaternion()    
    open_chain = urdf_robot.extract_open_chains(se3, 0.1)[-1].cuda()
    target_pose = torch.tensor([[ 0.31304278969764709473, -0.00631798803806304932,
        1.22056627273559570312, -1.25425827503204345703,
        0.11632148176431655884,  0.80687266588211059570]], device='cuda:0',
        requires_grad=True)
    thetas = torch.tensor([[-3.51662302017211914062,  0.27089971303939819336,
        3.34850096702575683594, -2.87017297744750976562,
        0.16795067489147186279,  0.00000000000000000000000000001]], device='cuda:0',
        requires_grad=True)
    thetas.retain_grad()
    target_pose.retain_grad()
    #current_pose = 
    loss = Environment.compute_reward(
            open_chain, thetas, target_pose, torch.ones(6).cuda()).mean()
    print(loss)
    #loss = se3.exp(target_pose).mean()
    print(target_pose.grad)
    #open_chain = urdf_robot.extract_open_chains(se3, 0.1)[-2].cuda()
    #loss = open_chain.forward_kinematics(thetas[:, :5]).mean()
    #loss.backward()
    #print(thetas.grad)
    #open_chain = urdf_robot.extract_open_chains(se3, 0.1)[-1].cuda()
    #loss = open_chain.forward_kinematics(thetas[:, :]).mean()
    #loss.backward()
    #print(thetas.grad)
