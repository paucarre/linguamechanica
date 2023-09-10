import numpy as np
import random
import torch
from pytorch3d import transforms
from linguamechanica.kinematics import DifferentiableOpenChainMechanism


class Environment:
    def __init__(self, open_chain, training_state):
        self.open_chain = open_chain
        self.training_state = training_state
        """
        State dims should be for now:
            - Target pose, 6 
            - Current thetas 6 
        """
        self.state_dimensions = 12
        self.action_dims = np.zeros(6).shape
        self.current_steps = None
        # TODO: make this nicer
        self.device = "cuda:0"
        self.open_chain = self.open_chain.to(self.device)
        self.training_state.weights = self.training_state.weights.to(self.device)
        self.noise_cte = 0.0

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
                coordinates.append(
                    random.uniform(
                        self.open_chain.joint_limits[i][0],
                        self.open_chain.joint_limits[i][1],
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

    def reset(self, summary=None):
        self.target_thetas = self.uniformly_sample_parameters_within_constraints()
        target_transformation = self.open_chain.forward_transformation(
            self.target_thetas
        )
        self.target_pose = transforms.se3_log_map(target_transformation.get_matrix())
        self.noise_cte = 0.1 * self.training_state.level
        noise = torch.randn_like(self.target_thetas) * self.noise_cte
        self.current_thetas = (self.target_thetas.detach().clone() + noise).to(
            self.device
        )
        self.pose_error_successful_threshold = 1.0 / (
            10 ** max(self.training_state.level, 0)
        )

        observation = self.generate_observation().to(self.device)
        self.current_step = torch.zeros(self.training_state.episode_batch_size, 1).to(
            self.device
        )
        self.initial_reward = self.compute_reward()[0]
        if summary is not None:
            summary.add_scalar(
                f"Env / Level",
                self.training_state.level,
                self.training_state.t,
            )
            summary.add_scalar(
                f"Env / Noise Constant",
                self.noise_cte,
                self.training_state.t,
            )
            summary.add_scalar(
                f"Env / Success Threshold",
                self.pose_error_successful_threshold,
                self.training_state.t,
            )
        return observation, self.initial_reward

    def compute_reward(self):
        error_pose = self.open_chain.compute_error_pose(
            self.current_thetas, self.target_pose
        )
        pose_error = DifferentiableOpenChainMechanism.compute_weighted_error(
            error_pose, self.training_state.weights
        )
        done = pose_error < self.pose_error_successful_threshold
        return -pose_error.unsqueeze(1).to(self.device), done.unsqueeze(1).to(
            self.device
        )

    def step(self, action, summary=None):
        within_steps = self.current_step < self.training_state.max_steps_done
        self.current_step[within_steps] += 1
        self.current_thetas[:, :] += action[:, :]
        reward, done = self.compute_reward()
        all_solutions_within_error_threshold = done[done == 1].shape[0] == done.shape[0]
        if all_solutions_within_error_threshold:
            self.training_state.level += 1
        if summary is not None:
            summary.add_scalar(
                f"Env / Proportion Success",
                float(done[done == 1].shape[0]) / float(done.shape[0]),
                self.training_state.t,
            )
        observation = self.generate_observation()
        done = torch.logical_or(
            done, self.current_step >= self.training_state.max_steps_done
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
        return action, observation, reward, done
