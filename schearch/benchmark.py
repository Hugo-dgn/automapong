from tqdm.auto import tqdm

import pong

def benchmark(agent1, agent2, episode):
    env = pong.Env()
    state1, state2 = env.reset()

    step = 0

    for e in tqdm(range(episode), desc="playing"):
        done = False
        while not done:
            step += 1

            action1 = agent1.act(state1, training=False)  # action est dans {-1, 0, 1}
            action2 = agent2.act(state2, training=False)  # action est dans {-1, 0, 1}

            # Effectuer une étape du jeu
            state1, state2, _, _, done = env.step(action1, action2) # on obtient les nouveaux états
            
            if done:
                state1, state2 = env.reset()

    return env.reward1/step