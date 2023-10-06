import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions.normal import Normal


def compute_manifold_error(open_chain, thetas, target_pose):
    pose, error_pose = open_chain.compute_pose_and_error_pose(thetas, target_pose)
    # pose decomposition
    pose_linear = pose[:, :3]
    pose_angular_cos = pose[:, 3:].cos()
    pose_angular_sin = pose[:, 3:].sin()
    # error pose decomposition
    error_pose_linear = error_pose[:, :3]
    error_pose_angular_cos = error_pose[:, 3:].cos()
    error_pose_angular_sin = error_pose[:, 3:].sin()
    # manifold error with all information
    manifold_error = torch.cat(
        [
            pose_linear,
            pose_angular_cos,
            pose_angular_sin,
            error_pose_linear,
            error_pose_angular_cos,
            error_pose_angular_sin,
            thetas.cos(),
            thetas.sin(),
        ],
        1,
    )
    return manifold_error


def fc_init(layer, weight=1e-10, bias_const=0.0):
    torch.nn.init.uniform_(layer.weight, a=-weight, b=weight)
    torch.nn.init.constant_(layer.bias, bias_const)
    return layer


class InverseKinematicsActor(nn.Module):
    def __init__(self, open_chain, max_action, max_std_dev, weights=128):
        super(InverseKinematicsActor, self).__init__()
        self.open_chain = open_chain
        self.max_action = max_action
        self.max_std_dev = max_std_dev
        on_manifold_count = 18 + (2 * self.open_chain.dof())
        thetas_count = self.open_chain.screws.shape[0]
        self.fc1 = fc_init(
            nn.Linear(in_features=on_manifold_count, out_features=weights, bias=True)
        )
        self.fc2 = fc_init(
            nn.Linear(
                in_features=weights + on_manifold_count, out_features=weights, bias=True
            )
        )
        self.fc3 = fc_init(
            nn.Linear(
                in_features=weights + on_manifold_count, out_features=weights, bias=True
            )
        )
        self.fc4 = fc_init(
            nn.Linear(
                in_features=weights + on_manifold_count, out_features=weights, bias=True
            )
        )
        self.fc5 = fc_init(
            nn.Linear(
                in_features=weights + on_manifold_count, out_features=weights, bias=True
            )
        )
        self.fc6 = fc_init(
            nn.Linear(
                in_features=weights + on_manifold_count, out_features=weights, bias=True
            )
        )
        self.fc_angle_mean = nn.Linear(
            in_features=weights + on_manifold_count,
            out_features=thetas_count,
            bias=False,
        )
        torch.nn.init.uniform_(self.fc_angle_mean.weight, a=-1e-5, b=1e-5)

        self.max_log_std_dev = math.log(self.max_std_dev)
        # TODO: make initial std dev configurable, for mow it is set to 1e-8
        initial_log_std = self.max_log_std_dev - math.log(1e-8)
        """
        NOTE:
        We use a network for the log std dev instead of direct parameters
        because the standard deviation should depend on the pose error
        between the current pose and the target pose.
        
        An example of direct parameters is: 
        https://github.com/vwxyzjn/ppo-implementation-details/blob/main/ppo_continuous_action.py#L124
        """
        self.fc_angle_log_std_dev_1 = fc_init(
            nn.Linear(
                in_features=weights + on_manifold_count,
                out_features=weights,
                bias=True,
            )
        )
        self.fc_angle_log_std_dev_2 = fc_init(
            nn.Linear(
                in_features=weights,
                out_features=thetas_count,
                bias=True,
            ),
            weight=1e-8,
            bias_const=initial_log_std,
        )

    def device(self):
        return self.fc1.weight.device

    def forward(self, thetas, target_pose):
        manifold_error = compute_manifold_error(self.open_chain, thetas, target_pose)
        x = torch.cat([F.tanh(self.fc1(manifold_error)), manifold_error], 1)
        x = torch.cat([F.tanh(self.fc2(x)), manifold_error], 1)
        x = torch.cat([F.tanh(self.fc3(x)), manifold_error], 1)
        x = torch.cat([F.tanh(self.fc4(x)), manifold_error], 1)
        x = torch.cat([F.tanh(self.fc5(x)), manifold_error], 1)
        x = torch.cat([F.tanh(self.fc6(x)), manifold_error], 1)
        angle_delta_mean = F.tanh(self.fc_angle_mean(x)) * self.max_action
        x = F.tanh(self.fc_angle_log_std_dev_1(x))
        x = self.fc_angle_log_std_dev_2(x)
        """
        We want the std dev to be limited to `self.max_std_dev` radiants.
        This means the log of the std dev can't be higher than `self.max_log_std_dev`
        By using the negative softplus and adding `self.max_log_std_dev` we archive this.
        """
        action_log_std_dev = self.max_log_std_dev - F.softplus(x)
        action_std_dev = torch.exp(action_log_std_dev)
        angle_delta_probabilities = Normal(angle_delta_mean, action_std_dev)
        angle_delta_action = angle_delta_probabilities.sample()
        # Here the clip is done on a sample which has no gradient, thus the clipping won't affect
        # any gradient
        angle_delta_action = torch.clip(
            angle_delta_action, -self.max_action, self.max_action
        )
        # See https://github.com/vwxyzjn/ppo-implementation-details/blob/main/ppo_continuous_action.py#L130
        return (
            angle_delta_mean,
            angle_delta_action,
            # TODO: check if this makes any sense and use it in case it does
            angle_delta_probabilities.log_prob(angle_delta_action).sum(1),
            angle_delta_probabilities.entropy().sum(1),
        )


class PseudoinvJacobianIKActor(nn.Module):
    def __init__(
        self,
        open_chain,
    ):
        super(PseudoinvJacobianIKActor, self).__init__()
        self.open_chain = open_chain
        # TODO: do this in a more elegant way
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.to(self.device)

    def forward(self, state):
        current_thetas = state[:, 6:]
        target_pose = state[:, :6]
        self.open_chain = self.open_chain.to(state.device)
        error_pose = self.open_chain.compute_error_pose(current_thetas, target_pose)
        # TODO: the constant factor should be a parameter
        mu = -0.01 * self.open_chain.inverse_kinematics_step(current_thetas, error_pose)
        var = torch.zeros(mu.shape).to(mu.device)
        return mu, var


class InverseKinematicsCritic(nn.Module):
    def __init__(self, open_chain, weights=128):
        super(InverseKinematicsCritic, self).__init__()
        self.open_chain = open_chain
        on_manifold_count = 18 + (2 * self.open_chain.dof())
        self.fc1 = fc_init(
            nn.Linear(in_features=on_manifold_count, out_features=weights, bias=True)
        )
        self.fc2 = fc_init(
            nn.Linear(
                in_features=weights + on_manifold_count, out_features=weights, bias=True
            )
        )
        self.fc3 = fc_init(
            nn.Linear(
                in_features=weights + on_manifold_count, out_features=weights, bias=True
            )
        )
        self.fc4 = fc_init(
            nn.Linear(
                in_features=weights + on_manifold_count, out_features=weights, bias=True
            )
        )
        self.fc5 = fc_init(
            nn.Linear(
                in_features=weights + on_manifold_count, out_features=weights, bias=True
            )
        )
        self.fc6 = fc_init(
            nn.Linear(
                in_features=weights + on_manifold_count, out_features=weights, bias=True
            )
        )
        self.fc_critic = fc_init(
            nn.Linear(
                in_features=weights + on_manifold_count, out_features=1, bias=True
            )
        )

    def device(self):
        return self.fc1.weight.device

    def forward(self, thetas, target_pose):
        manifold_error = compute_manifold_error(self.open_chain, thetas, target_pose)
        x = torch.cat([F.tanh(self.fc1(manifold_error)), manifold_error], 1)
        x = torch.cat([F.tanh(self.fc2(x)), manifold_error], 1)
        x = torch.cat([F.tanh(self.fc3(x)), manifold_error], 1)
        x = torch.cat([F.tanh(self.fc4(x)), manifold_error], 1)
        x = torch.cat([F.tanh(self.fc5(x)), manifold_error], 1)
        x = torch.cat([F.tanh(self.fc6(x)), manifold_error], 1)
        quality = -F.relu(self.fc_critic(x))
        return quality
