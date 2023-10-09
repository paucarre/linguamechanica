import logging
from dataclasses import dataclass

import click
import pybullet as p
import torch

from linguamechanica.agent import IKAgent
from linguamechanica.environment import Environment
from linguamechanica.inference import target_pose_reset, target_thetas_reset
from linguamechanica.kinematics import UrdfRobotLibrary
from linguamechanica.se3 import ImplicitDualQuaternion


@dataclass
class PyBulletRobotIds:
    robot_id: int
    target_robot_id: int
    initial_robot_id: int


def get_logger():
    logging.basicConfig(format="%(asctime)s %(message)s")
    logger = logging.getLogger("ik_test")
    logger.setLevel(logging.INFO)
    return logger


class VisualTester:
    def __init__(
        self,
        se3,
        iterations,
        target_thetas,
        urdf,
        checkpoint,
        samples,
        level,
        target_pose,
    ):
        self.se3 = se3
        self.iterations = iterations
        self.urdf = urdf
        self.checkpoint = checkpoint
        self.samples = samples
        self.level = level
        self.target_thetas = target_thetas
        self.target_pose = target_pose

    def test(self):
        self.robot_ids = self.setup_pybullet()
        environment, agent, state, initial_reward = self.setup_inference()
        self.solve_ik(state, agent, environment, initial_reward)

    def setup_pybullet(self):
        with_target_robot = self.target_thetas is not None
        p.connect(p.GUI)
        p.configureDebugVisualizer(p.COV_ENABLE_GUI, 0)
        robot_id = p.loadURDF(self.urdf, [0, 0, 0])
        target_robot_id = -1
        initial_robot_id = -1
        if with_target_robot:
            target_robot_id = p.loadURDF(self.urdf, [0, 0, 0])
            initial_robot_id = p.loadURDF(self.urdf, [0, 0, 0])
        for link in range(-1, 20):
            if with_target_robot:
                p.changeVisualShape(
                    target_robot_id, link, rgbaColor=[0.1, 0.8, 0.4, 0.5]
                )
                p.changeVisualShape(
                    initial_robot_id, link, rgbaColor=[0.5, 0.5, 0.5, 0.5]
                )
            p.changeVisualShape(robot_id, link, rgbaColor=[0.1, 0.1, 0.8, 0.5])
        p.resetBasePositionAndOrientation(robot_id, [0, 0, 0], [0, 0, 0, 1])
        if with_target_robot:
            p.resetBasePositionAndOrientation(target_robot_id, [0, 0, 0], [0, 0, 0, 1])
            p.resetBasePositionAndOrientation(initial_robot_id, [0, 0, 0], [0, 0, 0, 1])
        p.setGravity(0, 0, 0)
        return PyBulletRobotIds(robot_id, target_robot_id, initial_robot_id)

    def setup_inference(self):
        urdf_robot = UrdfRobotLibrary.from_urdf_path(urdf_path=self.urdf)
        # TODO: make this generic
        open_chain = urdf_robot.extract_open_chains(self.se3, 0.3)[-1].cuda()
        agent = IKAgent.from_checkpoint(
            open_chain=open_chain, checkpoint_id=self.checkpoint
        ).cuda()
        agent.training_state.episode_batch_size = self.samples
        agent.training_state.level = self.level
        environment = Environment(
            open_chain=open_chain, training_state=agent.training_state
        ).cuda()
        state, initial_reward = None, None
        if self.target_thetas is not None:
            state, initial_reward = target_thetas_reset(environment, self.target_thetas)
            for i in range(p.getNumJoints(self.robot_ids.robot_id)):
                p.resetJointState(
                    self.robot_ids.target_robot_id,
                    i,
                    environment.target_thetas[0, i].item(),
                )
        elif target_pose is not None:
            state, initial_reward = target_pose_reset(environment, target_pose)
        thetas, target_pose = Environment.thetas_target_pose_from_state(state)
        self.draw_pose_as_axis(target_pose[0, :], self.robot_ids.robot_id)
        for i in range(p.getNumJoints(self.robot_ids.robot_id)):
            p.resetJointState(self.robot_ids.robot_id, i, thetas[0, i].item())
            # if robot_ids.initial_robot_id != -1:
            #    p.resetJointState(robot_ids.initial_robot_id, i, thetas[0, i].item())
        return environment, agent, state, initial_reward

    def draw_pose_as_axis(
        self, pose, parent_uid=0, life_time=0, len_damp=0.1, lineWidth=3.0, cd=1.0
    ):
        se3_rep = self.se3.exp(pose.unsqueeze(0))
        axis_origin = torch.tensor(
            [
                [0.0, 0.0, 0.0],
            ]
        ).cuda()
        axis_origin = self.se3.act_point(se3_rep, axis_origin)[0]
        axis_orient = torch.tensor(
            [
                [1.0, 0.0, 0.0],
                [0.0, 1.0, 0.0],
                [0.0, 0.0, 1.0],
            ]
        ).cuda()
        axis_orient = self.se3.act_vector(se3_rep, axis_orient)[0]
        po = axis_origin[0, :].tolist()
        px = (axis_origin[0, :] + (axis_orient[0, :] * len_damp)).tolist()
        py = (axis_origin[0, :] + (axis_orient[1, :] * len_damp)).tolist()
        pz = (axis_origin[0, :] + (axis_orient[2, :] * len_damp)).tolist()
        p.addUserDebugLine(
            po,
            px,
            lineColorRGB=[1 * cd, 0, 0],
            lineWidth=lineWidth,
            lifeTime=life_time,
            parentObjectUniqueId=parent_uid,
            parentLinkIndex=-1,
        )
        p.addUserDebugLine(
            po,
            py,
            lineColorRGB=[0, 1 * cd, 0],
            lineWidth=lineWidth,
            lifeTime=life_time,
            parentObjectUniqueId=parent_uid,
            parentLinkIndex=-1,
        )
        p.addUserDebugLine(
            po,
            pz,
            lineColorRGB=[0, 0, 1 * cd],
            lineWidth=lineWidth,
            lifeTime=life_time,
            parentObjectUniqueId=parent_uid,
            parentLinkIndex=-1,
        )

    def solve_ik(self, state, agent, environment, initial_reward):
        logger = get_logger()
        max_initial_reward = initial_reward.max()
        iteration = 0
        max_reward = None
        max_pose = None
        while iteration < self.iterations:
            thetas, target_pose = Environment.thetas_target_pose_from_state(state)
            action_mean, actions, log_probabilities, entropy = agent.choose_action(
                state, agent.training_state
            )
            actions, next_state, reward, done, _ = environment.step(actions)
            current_best_reward_idx = torch.argmax(reward, dim=0)
            current_best_reward = reward[current_best_reward_idx, 0].item()
            if max_reward is None or max_reward < current_best_reward:
                max_reward = current_best_reward
                max_thetas = thetas[current_best_reward_idx.to(thetas.device), :]
                logger.info(
                    f"Current max reward: {max_reward:0.3} from best initial reward of {max_initial_reward:0.3}"
                )
                thetas_str = ", ".join(
                    [f"{theta:1.3}" for theta in max_thetas[0, :].tolist()]
                )
                logger.info(f"Current thetas: {thetas_str}")
                pose = environment.current_pose()
                max_pose = pose[current_best_reward_idx.to(thetas.device), :]
                self.draw_pose_as_axis(
                    max_pose[0, :], self.robot_ids.robot_id, 0.2, 0.05, 1.0
                )
                pose_str = ", ".join(
                    [f"{theta:1.3}" for theta in max_pose[0, :].tolist()]
                )
                logger.info(f"Current pose: {pose_str}")
                pose_str = ", ".join(
                    [f"{theta:1.3}" for theta in target_pose[0, :].tolist()]
                )
                logger.info(f"Target  pose: {pose_str}")
                for i in range(p.getNumJoints(self.robot_ids.robot_id)):
                    p.resetJointState(
                        self.robot_ids.robot_id, i, max_thetas[0, i].item()
                    )
            state = next_state
            iteration += 1
        self.draw_pose_as_axis(
            max_pose[0, :], self.robot_ids.robot_id, 0.0, 0.05, 3.0, 0.5
        )
        while True:
            pass


@click.command(context_settings=dict(help_option_names=["-h", "--help"]))
@click.option(
    "--checkpoint", help="Model checkpoint identifier.", type=int, required=True
)
@click.option(
    "--level",
    help="IK Game Level (theta noise is '0.1 * level').",
    type=int,
    default=100,
    required=True,
)
@click.option(
    "--urdf",
    default="./urdf/cr5.urdf",
    help="URDF of the robot.",
    type=str,
    required=False,
)
@click.option(
    "--samples",
    default=1024,
    help="Number of initial poses to solve the IK problem.",
    type=int,
    required=False,
)
@click.option(
    "--iterations",
    default=100,
    help="Number of IK iterations.",
    type=int,
    required=False,
)
@click.option("--target_thetas", type=str, required=False)
@click.option("--target_pose", type=str, required=False)
def test(checkpoint, urdf, level, samples, iterations, target_thetas, target_pose):
    # TODO: make this generic
    se3 = ImplicitDualQuaternion()
    tester = VisualTester(
        se3, iterations, target_thetas, urdf, checkpoint, samples, level, target_pose
    )
    tester.test()


if __name__ == "__main__":
    test()
