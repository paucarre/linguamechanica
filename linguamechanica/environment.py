import math
import random

import torch

from linguamechanica.kinematics import DifferentiableOpenChainMechanism


class Environment:
    def __init__(self, open_chain, training_state):
        self.open_chain = open_chain
        self.training_state = training_state
        """
        State dims should be for now:
            - Target pose, 6 
            - Current thetas DoF(open_chain)
        """
        self.state_dimensions = 6 + self.open_chain.dof()
        self.current_step = None
        # TODO: make this nicer
        self.device = "cuda:0"
        self.open_chain = self.open_chain.to(self.device)
        self.training_state.weights = self.training_state.weights.to(self.device)

    def to(self, device):
        self.device = device
        return self

    def cuda(self):
        # TODO: this is not elegant
        return self.to("cuda:0")

    def uniformly_sample_parameters_within_constraints(self):
        samples = []
        for sample_idx in range(self.training_state.episode_batch_size):
            coordinates = []
            for i in range(len(self.open_chain.joint_limits)):
                """
                TODO: use constraints once they are properly tested...
                self.open_chain.joint_limits[i][0],
                self.open_chain.joint_limits[i][1])
                """
                coordinates.append(
                    random.uniform(
                        -torch.pi,
                        torch.pi,
                    )
                )
            samples.append(torch.Tensor(coordinates).unsqueeze(0))
        return torch.cat(samples, 0).to(self.device)

    def generate_observation(self):
        state = torch.zeros(
            self.training_state.episode_batch_size, self.state_dimensions
        )
        state[:, :6] = self.target_pose.detach()
        state[:, 6:] = self.current_thetas.detach()
        return state

    @staticmethod
    def thetas_target_pose_from_state(state):
        return state[:, 6:], state[:, :6]

    def reset_to_target_pose(self, target_pose, summary=None):
        samples = self.training_state.episode_batch_size
        self.target_pose = target_pose.unsqueeze(0).repeat(samples, 1).to(self.device)
        self.current_thetas = self.uniformly_sample_parameters_within_constraints()
        return self._reset(summary)

    def current_pose(self):
        current_transformation = self.open_chain.forward_kinematics(self.current_thetas)
        return self.open_chain.se3.log(current_transformation)

    def reset_to_random_targets(self, summary=None):
        target_thetas_batch = self.uniformly_sample_parameters_within_constraints()
        return self._reset_to_target_thetas_batch(
            target_thetas_batch=target_thetas_batch, summary=summary
        )

    def reset_to_target_thetas(self, target_thetas, summary=None):
        samples = self.training_state.episode_batch_size
        target_thetas_batch = (
            target_thetas.unsqueeze(0).repeat(samples, 1).to(self.device)
        )
        return self._reset_to_target_thetas_batch(
            target_thetas_batch=target_thetas_batch
        )

    def _reset_to_target_thetas_batch(self, target_thetas_batch, summary=None):
        self.target_thetas = target_thetas_batch
        target_transformation = self.open_chain.forward_kinematics(self.target_thetas)
        self.target_pose = self.open_chain.se3.log(target_transformation)
        noise = (
            torch.randn_like(self.target_thetas)
            * self.training_state.initial_theta_std_dev()
        )
        self.current_thetas = (self.target_thetas.detach().clone() + noise).to(
            self.device
        )
        return self._reset(summary)

    def _reset(self, summary=None):
        observation = self.generate_observation().to(self.device)
        self.current_step = torch.zeros(self.training_state.episode_batch_size, 1).to(
            self.device
        )
        self.initial_reward = Environment.compute_reward(
            self.open_chain,
            self.current_thetas,
            self.target_pose,
            self.training_state.weights,
        ).to(self.device)
        if summary is not None:
            summary.add_scalar(
                f"Env / Level",
                self.training_state.level,
                self.training_state.t,
            )
            summary.add_scalar(
                f"Env / Noise Constant",
                self.training_state.initial_theta_std_dev(),
                self.training_state.t,
            )
            summary.add_scalar(
                f"Env / Success Threshold",
                self.training_state.pose_error_successful_threshold(),
                self.training_state.t,
            )
        return observation, self.initial_reward

    @staticmethod
    def compute_reward(open_chain, thetas, target_pose, weights, summary=None, t=None):
        error_pose = open_chain.compute_error_pose(
            thetas, target_pose, summary=summary, t=t
        )
        pose_error = DifferentiableOpenChainMechanism.compute_weighted_error(
            error_pose,
            weights,
        )
        reward = -pose_error.unsqueeze(1)
        return reward

    def within_error_success(self, reward):
        error_pose = -reward
        return error_pose < self.training_state.pose_error_successful_threshold()

    def step(self, action, summary=None):
        level_increased = False
        within_steps = self.current_step < self.training_state.max_steps_done
        self.current_step[within_steps] += 1
        self.current_thetas[:, :] += action[:, :]
        self.current_thetas[self.current_thetas > math.pi] = self.current_thetas[
            self.current_thetas > math.pi
        ] - (2.0 * math.pi)
        self.current_thetas[self.current_thetas < -math.pi] = self.current_thetas[
            self.current_thetas < -math.pi
        ] + (2.0 * math.pi)

        reward = Environment.compute_reward(
            self.open_chain,
            self.current_thetas,
            self.target_pose,
            self.training_state.weights,
        ).to(self.device)
        is_success_reward = self.within_error_success(reward)
        proportion_successful = float(
            is_success_reward[is_success_reward == 1].shape[0]
        ) / float(is_success_reward.shape[0])
        if (
            proportion_successful
            >= self.training_state.proportion_successful_to_increase_level
        ):
            self.training_state.level += 1
            level_increased = True
        if summary is not None:
            summary.add_scalar(
                f"Env / Proportion Success",
                proportion_successful,
                self.training_state.t,
            )
        observation = self.generate_observation()
        done = torch.logical_or(
            is_success_reward, self.current_step >= self.training_state.max_steps_done
        )
        if summary is not None and done[done == 1].shape[0] > 0:
            summary.add_scalar(
                f"Env / Final Reward Done",
                reward[done == 1].mean(),
                self.training_state.t,
            )
            summary.add_scalar(
                f"Env / Initial Reward Done",
                self.initial_reward[done == 1].mean(),
                self.training_state.t,
            )
        return action, observation, reward, done, level_increased
