
from linguamechanica.kinematics import UrdfRobotLibrary
from linguamechanica.environment import Environment
from linguamechanica.agent import IKAgent
from linguamechanica.environment import Environment
from linguamechanica.kinematics import UrdfRobotLibrary
import click
import torch


def target_thetas_reset(environment, target_thetas):
    target_thetas = [float(theta) for theta in target_thetas.split(",")]
    target_thetas = torch.tensor(target_thetas)
    return environment.reset_to_target_thetas(target_thetas)


def target_pose_reset(environment, target_pose):
    target_pose = [float(element) for element in target_pose.split(",")]
    target_pose = torch.tensor(target_pose)
    return environment.reset_to_target_pose(target_pose)


def setup_inference(robot_ids, urdf, checkpoint, samples, target_thetas, target_pose):
    urdf_robot = UrdfRobotLibrary.from_urdf_path(urdf_path=urdf)
    open_chain = urdf_robot.extract_open_chains(0.3)[-1].cuda()
    agent = IKAgent.from_checkpoint(
        open_chain=open_chain, checkpoint_id=checkpoint
    ).cuda()
    agent.training_state.episode_batch_size = samples
    environment = Environment(
        open_chain=open_chain, training_state=agent.training_state
    ).cuda()
    state, initial_reward = None, None
    if target_thetas is not None:
        state, initial_reward = target_thetas_reset(environment, target_thetas)
    elif target_pose is not None:
        state, initial_reward = target_pose_reset(environment, target_pose)
    thetas, target_pose = Environment.thetas_target_pose_from_state(state)
    return environment, agent, state, initial_reward


def solve_ik(robot_ids, iterations, state, agent, environment, initial_reward):
    initial_reward.max()
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
        state = next_state
        iteration += 1
    while True:
        pass


@click.command(context_settings=dict(help_option_names=["-h", "--help"]))
@click.option(
    "--checkpoint", help="Model checkpoint identifier.", type=int, required=True
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
def inference(checkpoint, urdf, samples, iterations, target_thetas, target_pose):
    with_target_robot = target_thetas is not None
    robot_ids = setup_pybullet(urdf, with_target_robot)
    environment, agent, state, initial_reward = setup_inference(
        robot_ids, urdf, checkpoint, samples, target_thetas, target_pose
    )
    solve_ik(robot_ids, iterations, state, agent, environment, initial_reward)


if __name__ == "__main__":
    inference()
