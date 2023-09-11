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

    def reset_to_target_pose(self, target_pose, samples, summary=None):
        target_thetas = target_theta.unsqueeze(1).repeat(samples, 1)
        return self._reset(target_thetas, summary)

    def reset_to_random_targets(self, summary=None):
        target_thetas = self.uniformly_sample_parameters_within_constraints()
        target_transformation = self.open_chain.forward_transformation(target_thetas)
        self.target_pose = transforms.se3_log_map(target_transformation.get_matrix())
        noise = (
            torch.randn_like(target_thetas)
            * self.training_state.initial_theta_std_dev()
        )
        self.current_thetas = (target_thetas.detach().clone() + noise).to(self.device)
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
    def compute_reward(open_chain, thetas, target_pose, weights):
        error_pose = open_chain.compute_error_pose(thetas, target_pose)
        pose_error = DifferentiableOpenChainMechanism.compute_weighted_error(
            error_pose, weights
        )
        reward = -pose_error.unsqueeze(1)
        return reward

    def within_error_success(self, reward):
        error_pose = -reward
        return error_pose < self.training_state.pose_error_successful_threshold()

    def step(self, action, summary=None):
        within_steps = self.current_step < self.training_state.max_steps_done
        self.current_step[within_steps] += 1
        self.current_thetas[:, :] += action[:, :]
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
        return action, observation, reward, done
