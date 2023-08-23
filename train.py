import pong

from tqdm.auto import tqdm

def train(agent1, agent2, episode, dt):
    env = pong.Env()

    for e in tqdm(range(episode), desc="training"): # for the progress bar.
        state1, state2 = env.reset()
        #### write yout code here for task 8
        done = False
        while not done:

            action1 = agent1.act(state1, True)
            action2 = agent2.act(state2, True)

            next_state1, next_state2, reward1, reward2, done = env.step(action1, action2, dt=dt)

            agent1.learn(state1, action1, reward1, next_state1, done)
            agent2.learn(state2, action2, reward2, next_state2, done)

            state1 = next_state1
            state2 = next_state2
        ####

    return env.get_results()