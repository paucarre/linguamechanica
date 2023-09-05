import numpy as np
import random
import torch
from pytorch3d import transforms
import math
from linguamechanica.kinematics import DifferentiableOpenChainMechanism


def force_parameters_within_bounds_to_delete(params):
    bigger_than_pi = params[:, :] > math.pi
    params[bigger_than_pi] = params[bigger_than_pi] - (2.0 * math.pi)
    less_than_minus_pi = params[:, :] < -math.pi
    params[less_than_minus_pi] = params[less_than_minus_pi] + (2.0 * math.pi)


class Environment:
    def __init__(self, batch_size, open_chain, training_state):
        self.batch_size = batch_size
        self.open_chain = open_chain
        """
        State dims should be for now:
            - Target pose, 6 
            - Current pose, 6
            - Current parameters, 6 
              Note that the current pose might
              not be informative enough as to know
              the current parameters one would need to solve the
              inverse kinematics for the current pose,
              which might be an even more difficult task.
        Action size should be:
            - Angle: sigmoid(x) - 0.5 or something similar
        """
        self.weights = training_state.weights
        self.state_dimensions = 12
        self.action_dims = np.zeros(6).shape
        self.current_steps = None
        self.max_steps_done = training_state.max_steps_done
        # TODO: make this nicer
        self.device = "cuda:0"
        self.open_chain = self.open_chain.to(self.device)
        self.weights = self.weights.to(self.device)
        self.max_noise_clip = training_state.max_noise_clip
        self.max_action = training_state.max_action

    def to(self, device):
        self.device = device
        return self

    def uniformly_sample_parameters_within_constraints(self):
        samples = []
        for sample_idx in range(self.batch_size):
            coordinates = []
            for i in range(len(self.open_chain.joint_limits)):
                # TODO: check if unconstrained works
                coordinates.append(
                    random.uniform(
                        self.open_chain.joint_limits[i][0],
                        self.open_chain.joint_limits[i][1],
                    )
                )
            samples.append(torch.Tensor(coordinates).unsqueeze(0))
        return torch.cat(samples, 0).to(self.device)

    #def sample_random_action(self):
    #    # TODO: this is a bit silly for now
    #    return self.uniformly_sample_parameters_within_constraints()# / math.pi

    def generate_observation(self):
        state = torch.zeros(self.batch_size, self.state_dimensions)
        state[:, :6] = self.target_pose.detach()
        state[:, 6:] = self.current_thetas.detach()
        return state

    def reset(self):
        self.target_parameters = self.uniformly_sample_parameters_within_constraints()
        target_transformation = self.open_chain.forward_transformation(
            self.target_parameters
        )
        self.target_pose = transforms.se3_log_map(
            target_transformation.get_matrix()
        )
        # TODO:
        # - Add a level which modulates upwards the noise
        # - Constraint values to the actuator constraints
        # The higher the level is, the higher the noise
        # so that the network learns to solve harder problems
        max_episode_cumulative_action = self.max_action * self.max_steps_done
        noise = torch.randn_like(self.target_parameters) * max_episode_cumulative_action
        noise = noise.clamp(-max_episode_cumulative_action * 0.5, max_episode_cumulative_action * 0.5)
        self.current_thetas = (self.target_parameters.detach().clone() + noise).to(self.device)
        # self.uniformly_sample_parameters_within_constraints()
        observation = self.generate_observation().to(self.device)
        self.current_step = torch.zeros(self.batch_size, 1).to(self.device)
        return observation

    def compute_reward(self):
        error_pose = self.open_chain.compute_error_pose(
            self.current_thetas, self.target_pose
        )
        pose_error = DifferentiableOpenChainMechanism.compute_weighted_error(
            error_pose, self.weights
        )
        # TODO: use a better threshold
        done = pose_error < 1e-3
        return -pose_error.unsqueeze(1), done.unsqueeze(1)

    def step(self, action):
        not_done = self.current_step < self.max_steps_done
        self.current_step[not_done] += 1
        # print(f"Action {action}, {self.current_parameter_index}")
        """
        TODO:
        Clip the current parameters to the max and min values.
        Even if there are no constraints this is necessary. For
        instance, the revolute joints will go from (-pi, pi)
        or (0, 2 * pi).
        """
        #print(action.shape)
        #TODO: do this niceely
        theta_deltas_sin = action[:, :, 0]
        theta_deltas_cos = action[:, :, 1]
        theta_deltas = torch.atan2(theta_deltas_sin, theta_deltas_cos)
        self.current_thetas[:, :] += theta_deltas[:, :]
        #force_parameters_within_bounds(self.current_thetas)
        # self.current_parameter_index = (self.current_parameter_index + 1) % len(
        #    self.open_chain
        # )
        # print(f"{self.target_parameters} | {self.current_thetas}")
        reward, done = self.compute_reward()
        observation = self.generate_observation()
        done = torch.logical_or(done, self.current_step >= self.max_steps_done)
        return theta_deltas, observation, reward, done
