import numpy as np
from pytransform3d.transformations import (
    exponential_coordinates_from_transform,
    invert_transform,
)
from pytransform3d.urdf import (
    UrdfTransformManager,
    parse_urdf,
    initialize_urdf_transform_manager,
)
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from pytorch3d import transforms
import torch.optim as optim


def to_left_multiplied(right_multiplied):
    """
     Converts matrix from right multiplied ( most common notation for SE3 )
     to left multiplied, which is the representation used in Pytorch 3D:
     M = [
        [Rxx, Ryx, Rzx, 0],
        [Rxy, Ryy, Rzy, 0],
        [Rxz, Ryz, Rzz, 0],
        [Tx,  Ty,  Tz,  1],
    ]
    This is equivalent to
     M = [
        [             ,  0],
        [ transpose(R),  0],
        [             ,  0],
        [      T      ,  1],
    ]
    """
    shape = right_multiplied.shape
    left_multiplied = right_multiplied.detach().clone()
    if len(shape) == 3:
        left_multiplied = left_multiplied.transpose(1, 2)
        left_multiplied[:, 0:3, 0:3] = right_multiplied[:, 0:3, 0:3].transpose(1, 2)
    elif len(shape) == 2:
        left_multiplied = left_multiplied.transpose(0, 1)
        left_multiplied[0:3, 0:3] = right_multiplied[0:3, 0:3].transpose(0, 1)
    return left_multiplied


class KinematicsNetwork(nn.Module):
    def __init__(self, initial_thetas, chain, error_weights, lr=0.001):
        super(KinematicsNetwork, self).__init__()
        self.chain = chain
        self.thetas = nn.Parameter(initial_thetas.detach().clone())
        self.error_weights = error_weights
        self.optimizer = optim.Adam(self.parameters(), lr=lr)
        self.loss = None

    def forward(self, target_pose):
        error_pose = self.chain.compute_error_pose(self.thetas, target_pose)
        error = DifferentiableOpenChainMechanism.compute_weighted_error(
            error_pose, self.error_weights
        )
        return error

    def train_step(self, target_pose):
        self.optimizer.zero_grad()
        self.loss = self(target_pose)
        self.loss.backward()
        self.optimizer.step()


class DifferentiableOpenChainMechanism:
    def __init__(self, screws, initial_matrix, joint_limits):
        self.screws = screws
        self.initial_matrix = to_left_multiplied(initial_matrix)
        self.joint_limits = joint_limits
        self.device = "cuda:0"
        self.screws = self.screws.to(self.device)
        self.initial_matrix = self.initial_matrix.to(self.screws.device)

    def to(self, device):
        # TODO: this is broken, fix it
        device = "cuda:0"
        self.screws = self.screws.to(device)
        self.initial_matrix = self.initial_matrix.to(device)
        self.device = device
        return self

    def _jacobian_computation_forward(self, thetas):
        transformation = self.forward_transformation(thetas)
        twist = transforms.se3_log_map(transformation.get_matrix())
        return twist
    
    def compute_pose_and_error_pose(self, thetas, target_pose):
        #print(self.device, thetas.device, target_pose.device)
        #print("compute_error_pose", thetas.shape, target_pose.shape)
        current_transformation = self.forward_transformation(thetas)
        target_transformation = transforms.se3_exp_map(target_pose)
        current_trans_to_target = current_transformation.compose(
            transforms.Transform3d(matrix=target_transformation).inverse()
        )
        current_trans_to_target = current_trans_to_target.to(thetas.device).get_matrix()
        error_pose = transforms.se3_log_map(current_trans_to_target)
        pose = transforms.se3_log_map(current_transformation.get_matrix())        
        return pose, error_pose

    def compute_error_pose(self, thetas, target_pose):
        #print("compute_error_pose", thetas.shape, target_pose.shape)
        current_transformation = self.forward_transformation(thetas)
        target_transformation = transforms.se3_exp_map(target_pose)
        current_trans_to_target = current_transformation.compose(
            transforms.Transform3d(matrix=target_transformation).inverse()
        )
        current_trans_to_target = current_trans_to_target.to(thetas.device).get_matrix()
        error_pose = transforms.se3_log_map(current_trans_to_target)
        return error_pose

    def compute_weighted_error(error_pose, weights):
        return (error_pose.abs() * weights.unsqueeze(0)).sum(1)

    def inverse_kinematics_step(self, parameters, error_pose):
        jacobian = self.jacobian(parameters)
        jacobian_pseudoinverse = torch.linalg.pinv(jacobian)
        parameter_delta = torch.bmm(jacobian_pseudoinverse, error_pose.unsqueeze(2))
        return parameter_delta.squeeze(2)

    def inverse_kinematics(
        self,
        initial_thetas,
        target_pose,
        min_error,
        error_weights,
        parameter_update_rate,
        max_steps=1000,
    ):
        current_thetas = initial_thetas.detach().clone()
        error_pose = self.compute_error_pose(current_thetas, target_pose)
        error = DifferentiableOpenChainMechanism.compute_weighted_error(
            error_pose, error_weights
        )
        parameter_update_rate = parameter_update_rate.unsqueeze(0)
        current_step = 0
        while error >= min_error and current_step < max_steps:
            parameter_delta = self.inverse_kinematics_step(current_thetas, error_pose)
            current_thetas -= parameter_delta * parameter_update_rate
            error_pose = self.compute_error_pose(current_thetas, target_pose)
            error = DifferentiableOpenChainMechanism.compute_weighted_error(
                error_pose, error_weights
            )
            current_step += 1
        return current_thetas

    def inverse_kinematics_backprop(
        self,
        initial_thetas,
        target_pose,
        min_error,
        error_weights,
        lr=0.001,
        max_steps=1000,
    ):
        kinematics_network = KinematicsNetwork(
            initial_thetas, self, error_weights, lr=lr
        )
        current_step = 0
        while (
            kinematics_network.loss is None or kinematics_network.loss >= min_error
        ) and current_step < max_steps:
            kinematics_network.train_step(target_pose=target_pose)
            current_step += 1
        return kinematics_network.thetas

    def jacobian(self, coordinates):
        """
        From coordinates of shape:
            [ Batch, Coordinates ]
        Returns Jacobian of shape:
            [ Batch, Velocities, Coordinates]
        Velocities is always 6 with the
        first 3 components being translation
        and the last 3 rotation
        """
        jacobian = torch.autograd.functional.jacobian(
            self._jacobian_computation_forward, coordinates
        )
        """
        Dimensions:
            [batch, screw_coordinates, batch, thetas]
        Need to be reduced to:
            [batch, screw_coordinates, thetas]
        By using `take_along_dim`
        Conceptually this means that coordinates that are
        used in a kinematic chain are not used for other
        kinematic chains and thus the jacobian shall be zero.
        """
        selector = (
            torch.range(0, jacobian.shape[0] - 1)
            .long()
            .unsqueeze(1)
            .unsqueeze(1)
            .unsqueeze(1)
            .to(jacobian.device)
        )
        jacobian = torch.take_along_dim(jacobian, selector, dim=2).squeeze(2)
        return jacobian

    def forward_transformation(self, coordinates):
        self.screws = self.screws.to(coordinates.device)
        #print("forward_transformation", self.screws.shape, coordinates.unsqueeze(2).shape)
        twist = self.screws * coordinates.unsqueeze(2)
        original_shape = twist.shape
        twist = twist.view(-1, original_shape[2])
        transformations = transforms.se3_exp_map(twist)
        """
        Transformations will have indices of this type:
        [
            => The i-th index of the chain
            => The j-th chain ( it can be the same robot 
              with another pose or different robots 
              so long they have the same number of degres of freedom)
            => 4 rows of the left-transformation matrix
            => 4 columns of the left-transformation matrix
        ]
        """
        transformations = transformations.view(
            original_shape[0],
            original_shape[1],
            transformations.shape[1],
            transformations.shape[2],
        )
        chains_length = transformations.shape[1]
        num_chains = transformations.shape[0]
        computed_transforms = transforms.Transform3d(
            matrix=torch.eye(4)
            .unsqueeze(0)
            .repeat(num_chains, 1, 1)
            .to(coordinates.device)
        )
        for chain_idx in range(chains_length):
            current_transformations = transforms.Transform3d(
                matrix=transformations[:, chain_idx, :, :]
            )
            computed_transforms = current_transformations.compose(computed_transforms)
        initial_matrix = transforms.Transform3d(
            matrix=self.initial_matrix.unsqueeze(0).repeat(num_chains, 1, 1)
        )
        return initial_matrix.compose(computed_transforms)

    def __len__(self):
        return len(self.screws)

    def __getitem__(self, i):
        return self.screws[i]


class UrdfRobot:
    def __init__(self, name, links, joints):
        self.name = name
        self.links = links
        self.joints = joints
        self.link_names = [link.name for link in self.links]
        self.joint_names = [joint.joint_name for joint in self.joints]
        self.joint_axis = [joint.joint_axis for joint in self.joints]
        self.joint_types = [joint.joint_type for joint in self.joints]
        self.joint_transformation = [joint.child2parent for joint in self.joints]
        self.joint_limits = [joint.limits for joint in self.joints]
        self.transform_manager = UrdfTransformManager()
        initialize_urdf_transform_manager(self.transform_manager, name, links, joints)
        transform_indices = self.transform_manager.to_dict()["transform_to_ij_index"]
        self.transform_indices = {
            joint_pair: index for joint_pair, index in transform_indices
        }

    def get_transform(self, i):
        link_source = self.link_names[i]
        link_destination = self.link_names[i + 1]
        transform_index = self.transform_indices[(link_destination, link_source)]
        transform = self.transform_manager.to_dict()["transforms"][transform_index][1]
        transform = np.array(
            [transform[:4], transform[4:8], transform[8:12], transform[12:]],
            dtype=np.float32,
        )
        return transform

    def extract_open_chains(self, epsillon):
        open_chains = []
        screws = []
        transform_zero = np.eye(4)
        for i in range(len(self.link_names) - 1):
            for joint_name in self.joint_names:
                self.transform_manager.set_joint(joint_name, 0.0)
            previous_transform_zero = transform_zero
            transform_zero = transform_zero @ self.get_transform(i)
            self.transform_manager.set_joint(self.joint_names[i], epsillon)
            transform_epsillon = self.get_transform(i)
            """
                Mad = Tab(0)Tbc(0)Tcd(0)
                Tab(0)Tbc(0)Tcd(e) = Iexp(screw*e)Mad
                exp(screw*e) = Tab(0)Tbc(0)Tcd(e)inv(Mad)
            """
            exponential = (
                previous_transform_zero
                @ transform_epsillon
                @ invert_transform(transform_zero)
            )
            # coordinates = vee ( log ( exponential ) )
            coordinates = exponential_coordinates_from_transform(exponential)
            screw = coordinates / epsillon
            screws.append(
                np.expand_dims(np.concatenate([screw[3:], screw[:3]]), axis=0)
            )
            screw_torch = torch.Tensor(np.concatenate(screws.copy()))
            initial_torch = torch.Tensor(transform_zero)
            open_chain = DifferentiableOpenChainMechanism(
                screw_torch, initial_torch, self.joint_limits[: i + 1]
            )
            open_chains.append(open_chain)
        return open_chains

    def transformations(self, values):
        """
        This method assumes values is a batch of one element
        """
        assert values.shape[0] == 1
        for i, joint_name in enumerate(self.joint_names):
            self.transform_manager.set_joint(joint_name, values[0][i])
        transform = np.eye(4)
        transformations = []
        for i in range(len(self.link_names) - 1):
            transform = transform @ self.get_transform(i)
            transformations.append(transform)
        return transformations

    def display(self):
        ax = self.transform_manager.plot_frames_in(
            self.link_names[-1], s=0.1, whitelist=self.link_names, show_name=True
        )
        ax = self.transform_manager.plot_connections_in(self.link_names[-1], ax=ax)
        self.transform_manager.plot_visuals(self.link_names[-1], ax=ax)
        ax.set_xlim3d((-0.0, 0.25))
        ax.set_ylim3d((-0.1, 0.25))
        ax.set_zlim3d((0.1, 0.30))
        plt.show()

    def __repr__(self):
        return f"{self}"

    def __str__(self):
        return f"""Robot '{self.name}':
\t- Links: {" | ".join([f"{link.name}" for link in self.links])}
\t- Joints: {" | ".join([f"{joint.joint_name} {joint.joint_type}" for joint in self.joints])}"""


class UrdfRobotLibrary:
    def dobot_cr5():
        urdf_data = None
        with open("./urdf/cr5.urdf", "r") as urdf_file:
            urdf_data = urdf_file.read()
        name, links, joints = parse_urdf(
            urdf_data, mesh_path="./urdf/", package_dir="./urdf/", strict_check=True
        )
        return UrdfRobot(name, links, joints)


if __name__ == "__main__":
    urdf_robot = UrdfRobotLibrary.dobot_cr5()
    open_chains = urdf_robot.extract_open_chains(0.1)

'''
Working:
get_pose_and_pose_error torch.Size([1024, 6]) torch.Size([1024, 6])
compute_error_pose torch.Size([1024, 6]) torch.Size([1024, 6])
forward_transformation torch.Size([6, 6]) torch.Size([1024, 6, 1])                                                      forward_transformation torch.Size([6, 6]) torch.Size([1024, 6, 1]) 
Error:
get_pose_and_pose_error torch.Size([16, 1018, 12]) torch.Size([16, 6, 12])
compute_error_pose torch.Size([16, 1018, 12]) torch.Size([16, 6, 12])
forward_transformation torch.Size([6, 6]) torch.Size([16, 1018, 1, 12])
'''
