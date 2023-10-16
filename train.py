import pong

from tqdm.auto import tqdm

def train(agent1, agent2, episode, dt):
    env = pong.Env()

    for e in tqdm(range(episode), desc="training"): # for the progress bar.
        state1, state2 = env.reset()
        #### write yout code here for task 8
        pass
        ####

    return env.get_results()