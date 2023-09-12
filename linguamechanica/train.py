from linguamechanica.kinematics import UrdfRobotLibrary
from linguamechanica.environment import Environment
from linguamechanica.agent import IKAgent
from torch.utils.tensorboard import SummaryWriter
from linguamechanica.training_context import EpisodeState
import click


def evaluate_policy(open_chain, agent, training_state, summary):
    environment = Environment(
        open_chain=open_chain, training_state=training_state
    ).cuda()
    state, initial_reward = environment.reset_to_random_targets()
    episode = EpisodeState("Test", initial_reward, training_state.gamma)
    finished = False
    while not finished:
        action_mean, actions, log_probabilities, entropy = agent.choose_action(
            state, training_state
        )
        actions, next_state, reward, done, level_increased = environment.step(
            actions, summary=summary
        )
        finished = episode.step(reward, done, training_state.t, summary)
        state = next_state


@click.command()
@click.option(
    "--checkpoint",
    help="Model checkpoint identifier.",
    required=False,
    default=None,
    type=int,
)
@click.option(
    "--urdf",
    default="./urdf/cr5.urdf",
    help="URDF of the robot.",
    type=str,
    required=False,
)
@click.option(
    "--level",
    help="IK Game Level (theta noise is '0.1 * level'). By default starts at level in configuration (default to 1) or the one stored in the checkpoint",
    type=int,
    required=False,
)
def train(checkpoint, urdf, level):
    urdf_robot = UrdfRobotLibrary.from_urdf_path(urdf_path=urdf)
    open_chain = urdf_robot.extract_open_chains(0.3)[-1].cuda()
    summary = SummaryWriter()
    agent, training_state = None, None
    if checkpoint is None:
        training_state = TrainingState()
        agent = IKAgent(
            open_chain=open_chain,
            summary=summary,
            training_state=training_state,
        ).cuda()
    else:
        agent = IKAgent.from_checkpoint(
            open_chain=open_chain, checkpoint_id=checkpoint, summary=summary
        ).cuda()
        training_state = agent.training_state
    if level is not None:
        agent.training_state.level = level
    env = Environment(open_chain=open_chain, training_state=training_state).cuda()
    state, initial_reward = env.reset_to_random_targets(summary)
    episode = EpisodeState("Train", initial_reward, training_state.gamma)
    for training_state.t in range(training_state.t, int(training_state.max_time_steps)):
        _, actions, _, _ = agent.choose_action(state, training_state)
        actions, next_state, reward, done, level_increased = env.step(
            actions, summary=summary
        )
        summary.add_scalar("Data / Step Batch Reward", reward.mean(), training_state.t)
        agent.store_transition(state, actions, reward, next_state, done)
        if episode.step(reward, done, training_state.t, summary):
            state, initial_reward = env.reset_to_random_targets(summary)
            episode = EpisodeState("Train", initial_reward, training_state.gamma)
        else:
            state = next_state
        if training_state.can_train_buffer():
            agent.train_buffer(level_increased)
        if training_state.can_evaluate_policy():
            evaluate_policy(open_chain, agent, training_state, summary)
        if training_state.can_save():
            agent.save(training_state)

    summary.close()


if __name__ == "__main__":
    train()
