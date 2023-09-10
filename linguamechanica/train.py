from linguamechanica.kinematics import UrdfRobotLibrary
from linguamechanica.environment import Environment
from linguamechanica.agent import IKAgent
from torch.utils.tensorboard import SummaryWriter
from linguamechanica.training_context import EpisodeState, TrainingState


def evaluate_policy(open_chain, agent, training_state, summary):
    environment = Environment(
        open_chain=open_chain, training_state=training_state
    ).cuda()
    state, initial_reward = environment.reset()
    episode = EpisodeState("Test", initial_reward, training_state.gamma)
    finished = False
    while not finished:
        action_mean, actions, log_probabilities, entropy = agent.choose_action(
            state, training_state
        )
        actions, next_state, reward, done = environment.step(actions, summary=summary)
        finished = episode.step(reward, done, training_state.t, summary)


def train():
    urdf_robot = UrdfRobotLibrary.dobot_cr5()
    open_chain = urdf_robot.extract_open_chains(0.3)[-1].cuda()
    training_state = TrainingState()
    env = Environment(
        open_chain=open_chain, training_state=training_state
    ).cuda()
    summary = SummaryWriter()
    agent = IKAgent(
        open_chain=open_chain,
        summary=summary,
        training_state=training_state,
    ).cuda()
    state, initial_reward = env.reset(summary)
    episode = EpisodeState("Train", initial_reward, training_state.gamma)
    # agent.load(f"checkpoint_46999")

    for training_state.t in range(training_state.t, int(training_state.max_time_steps)):
        _, actions, _, _ = agent.choose_action(state, training_state, summary)
        actions, next_state, reward, done = env.step(actions, summary=summary)
        summary.add_scalar("Data / Step Batch Reward", reward.mean(), training_state.t)
        agent.store_transition(state, actions, reward, next_state, done)
        if episode.step(reward, done, training_state.t, summary):
            state, initial_reward = env.reset(summary)
            episode = EpisodeState("Train", initial_reward, training_state.gamma)
        else:
            state = next_state
        if training_state.can_train_buffer():
            agent.train_buffer()
        if training_state.can_evaluate_policy():
            evaluate_policy(open_chain, agent, training_state, summary)
        if training_state.can_save():
            agent.save(training_state)

    summary.close()


if __name__ == "__main__":
    train()
