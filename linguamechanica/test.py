import pybullet as p
from dataclasses import dataclass

from linguamechanica.kinematics import UrdfRobotLibrary
from linguamechanica.environment import Environment
from linguamechanica.agent import IKAgent
from linguamechanica.environment import Environment
from linguamechanica.kinematics import UrdfRobotLibrary
import click
import logging
import torch


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


def setup_pybullet(urdf, with_target_robot):
    p.connect(p.GUI)
    p.configureDebugVisualizer(p.COV_ENABLE_GUI, 0)
    robot_id = p.loadURDF(urdf, [0, 0, 0])
    target_robot_id = -1
    if with_target_robot:
        target_robot_id = p.loadURDF(urdf, [0, 0, 0])
    initial_robot_id = p.loadURDF(urdf, [0, 0, 0])
    for link in range(-1, 20):
        if with_target_robot:
            p.changeVisualShape(target_robot_id, link, rgbaColor=[0.1, 0.8, 0.4, 0.5])
        p.changeVisualShape(robot_id, link, rgbaColor=[0.1, 0.1, 0.8, 0.5])
        p.changeVisualShape(initial_robot_id, link, rgbaColor=[0.5, 0.5, 0.5, 0.5])
    p.resetBasePositionAndOrientation(robot_id, [0, 0, 0], [0, 0, 0, 1])
    if with_target_robot:
        p.resetBasePositionAndOrientation(target_robot_id, [0, 0, 0], [0, 0, 0, 1])
    p.resetBasePositionAndOrientation(initial_robot_id, [0, 0, 0], [0, 0, 0, 1])
    p.setGravity(0, 0, 0)
    return PyBulletRobotIds(robot_id, target_robot_id, initial_robot_id)


def target_thetas_reset(environment, target_thetas):
    target_thetas = [float(theta) for theta in target_thetas.split(",")]
    target_thetas = torch.tensor(target_thetas)
    return environment.reset_to_target_thetas(target_thetas)


def target_pose_reset(environment, target_pose):
    target_pose = [float(element) for element in target_pose.split(",")]
    target_pose = torch.tensor(target_pose)
    return environment.reset_to_target_pose(target_pose)


def setup_inference(
    robot_ids, urdf, checkpoint, samples, level, target_thetas, target_pose
):
    urdf_robot = UrdfRobotLibrary.from_urdf_path(urdf_path=urdf)
    open_chain = urdf_robot.extract_open_chains(0.3)[-1].cuda()
    agent = IKAgent.from_checkpoint(
        open_chain=open_chain, checkpoint_id=checkpoint
    ).cuda()
    agent.training_state.episode_batch_size = samples
    agent.training_state.level = level
    environment = Environment(
        open_chain=open_chain, training_state=agent.training_state
    ).cuda()
    state, initial_reward = None, None
    if target_thetas is not None:
        state, initial_reward = target_thetas_reset(environment, target_thetas)
        for i in range(p.getNumJoints(robot_ids.robot_id)):
            p.resetJointState(
                robot_ids.target_robot_id, i, environment.target_thetas[0, i].item()
            )
    elif target_pose is not None:
        state, initial_reward = target_pose_reset(environment, target_pose)
    thetas, target_pose = Environment.thetas_target_pose_from_state(state)
    for i in range(p.getNumJoints(robot_ids.robot_id)):
        p.resetJointState(robot_ids.robot_id, i, thetas[0, i].item())
        p.resetJointState(robot_ids.initial_robot_id, i, thetas[0, i].item())
    return environment, agent, state, initial_reward


def solve_ik(robot_ids, iterations, state, agent, environment, initial_reward):
    logger = get_logger()
    max_initial_reward = initial_reward.max()
    iteration = 0
    max_reward = None
    while iteration < iterations:
        thetas, target_pose = Environment.thetas_target_pose_from_state(state)
        action_mean, actions, log_probabilities, entropy = agent.choose_action(
            state, agent.training_state
        )
        actions, next_state, reward, done = environment.step(actions)
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
            pose_str = ", ".join([f"{theta:1.3}" for theta in max_pose[0, :].tolist()])
            logger.info(f"Current pose: {pose_str}")
            pose_str = ", ".join(
                [f"{theta:1.3}" for theta in target_pose[0, :].tolist()]
            )
            logger.info(f"Target  pose: {pose_str}")
            for i in range(p.getNumJoints(robot_ids.robot_id)):
                p.resetJointState(robot_ids.robot_id, i, max_thetas[0, i].item())
        state = next_state
        iteration += 1
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
    default=3,
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
    with_target_robot = target_thetas is not None
    robot_ids = setup_pybullet(urdf, with_target_robot)
    environment, agent, state, initial_reward = setup_inference(
        robot_ids, urdf, checkpoint, samples, level, target_thetas, target_pose
    )
    solve_ik(robot_ids, iterations, state, agent, environment, initial_reward)


if __name__ == "__main__":
    test()
