import torch
import torch.nn as nn
import torch.nn.functional as F
from pytorch3d import transforms

def compute_manifold_error(open_chain, thetas, target_pose):
    pose, error_pose = open_chain.compute_pose_and_error_pose(thetas, target_pose)
    # pose decomposition
    pose_linear =  pose[:, :3]
    pose_angular_cos =  pose[:, 3:].cos()
    pose_angular_sin =  pose[:, 3:].sin()
    # error pose decomposition 
    error_pose_linear =  error_pose[:, :3]
    error_pose_angular_cos =  error_pose[:, 3:].cos()
    error_pose_angular_sin =  error_pose[:, 3:].sin()
    # manifold error with all information
    manifold_error = torch.cat([pose_linear, pose_angular_cos , pose_angular_sin, error_pose_linear, error_pose_angular_cos, error_pose_angular_sin, thetas.cos(), thetas.sin()], 1)      
    return manifold_error
    
class InverseKinematicsActor(nn.Module):
    def __init__(
        self, 
        open_chain,
        max_action,
        max_variance
    ):
        super(InverseKinematicsActor, self).__init__()        
        weights = 128
        self.max_action = max_action
        self.max_variance = max_variance        
        self.open_chain = open_chain
        #self.max_action = max_action
        #self.max_variance = max_variance
        on_manifold_count = 9 + 9 + 6 + 6
        thetas_count = self.open_chain.screws.shape[0]
        self.fc1 = nn.Linear(in_features=on_manifold_count, out_features=weights, bias=True)
        self.fc2 = nn.Linear(in_features=weights+on_manifold_count, out_features=weights, bias=True)
        self.fc3 = nn.Linear(in_features=weights+on_manifold_count, out_features=weights, bias=True)
        self.fc4 = nn.Linear(in_features=weights+on_manifold_count, out_features=weights, bias=True)
        self.fc5 = nn.Linear(in_features=weights+on_manifold_count, out_features=weights, bias=True)
        self.fc6 = nn.Linear(in_features=weights+on_manifold_count, out_features=weights, bias=True)
        self.fc_cos = nn.Linear(in_features=weights+on_manifold_count, out_features=thetas_count, bias=True)
        self.fc_sin = nn.Linear(in_features=weights+on_manifold_count, out_features=thetas_count, bias=True)
        self.fc_cos_var = nn.Linear(in_features=weights+on_manifold_count, out_features=thetas_count, bias=True)
        self.fc_sin_var = nn.Linear(in_features=weights+on_manifold_count, out_features=thetas_count, bias=True)
    
    def device(self):
        return self.fc1.weight.device

    def forward(self, thetas, target_pose):        
        manifold_error = compute_manifold_error(self.open_chain, thetas, target_pose)
        x = torch.cat([F.tanh(self.fc1(manifold_error)), manifold_error],1)
        x = torch.cat([F.tanh(self.fc2(x)), manifold_error],1)
        x = torch.cat([F.tanh(self.fc3(x)), manifold_error],1)
        x = torch.cat([F.tanh(self.fc4(x)), manifold_error],1)
        x = torch.cat([F.tanh(self.fc5(x)), manifold_error],1)
        x = torch.cat([F.tanh(self.fc6(x)), manifold_error],1)
        # cos, sin output
        cos = self.fc_cos(x).cos()
        sin = self.fc_sin(x).sin()
        # cos var, sin var output
        cos_var = ((F.tanh(self.fc_cos_var(x)) + 1.0) / 2.0) * self.max_variance
        sin_var = ((F.tanh(self.fc_sin_var(x)) + 1.0) / 2.0) * self.max_variance
        return torch.cat([sin.unsqueeze(2), cos.unsqueeze(2)], 2), torch.cat([sin_var.unsqueeze(2), cos_var.unsqueeze(2)], 2)



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
        """
        We ignore the current pose from the
        state as we only care about the current parameters
        """
        current_thetas = state[:, 6:]
        target_pose = state[:, :6]
        self.open_chain = self.open_chain.to(state.device)
        error_pose = self.open_chain.compute_error_pose(current_thetas, target_pose)
        # TODO: the constant factor should be something else
        mu = -0.01 * self.open_chain.inverse_kinematics_step(current_thetas, error_pose)
        var = torch.zeros(mu.shape).to(mu.device)
        return mu, var


class InverseKinematicsCritic(nn.Module):
    def __init__(
        self, 
        open_chain,
    ):
        super(InverseKinematicsCritic, self).__init__()        
        weights = 128
        self.open_chain = open_chain
        on_manifold_count = 9 + 9 + 6 + 6        
        self.fc1 = nn.Linear(in_features=on_manifold_count, out_features=weights, bias=True)
        self.fc2 = nn.Linear(in_features=weights+on_manifold_count, out_features=weights, bias=True)
        self.fc3 = nn.Linear(in_features=weights+on_manifold_count, out_features=weights, bias=True)
        self.fc4 = nn.Linear(in_features=weights+on_manifold_count, out_features=weights, bias=True)
        self.fc5 = nn.Linear(in_features=weights+on_manifold_count, out_features=weights, bias=True)
        self.fc6 = nn.Linear(in_features=weights+on_manifold_count, out_features=weights, bias=True)
        self.fc_critic = nn.Linear(in_features=weights+on_manifold_count, out_features=1, bias=True)
    
    def device(self):
        return self.fc1.weight.device

    def forward(self, thetas, target_pose):
        manifold_error = compute_manifold_error(self.open_chain, thetas, target_pose)
        x = torch.cat([F.tanh(self.fc1(manifold_error)), manifold_error],1)
        x = torch.cat([F.tanh(self.fc2(x)), manifold_error],1)
        x = torch.cat([F.tanh(self.fc3(x)), manifold_error],1)
        x = torch.cat([F.tanh(self.fc4(x)), manifold_error],1)
        x = torch.cat([F.tanh(self.fc5(x)), manifold_error],1)
        x = torch.cat([F.tanh(self.fc6(x)), manifold_error],1)
        quality = -F.relu(self.fc_critic(x))
        return quality