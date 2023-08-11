import pong

from tqdm.auto import tqdm

def train(agent1, agent2, episode):
    ####write yout code here for task 3
    env = pong.Env()
    state1, state2 = env.reset()

    step = 0

    for e in tqdm(range(episode), desc="training"):
        done = False
        while not done:
            step += 1

            # Obtenir les actions des deux joueurs
            action1 = agent1.act(state1, training=True)  # action est dans {-1, 0, 1}
            action2 = agent2.act(state2, training=True)  # action est dans {-1, 0, 1}

            # Effectuer une étape du jeu
            next_state1, next_state2, reward1, reward2, done = env.step(action1, action2) # on obtient les nouveaux états

            agent1.learn(state1, action1, reward1, next_state1, done)
            agent2.learn(state2, action2, reward2, next_state2, done)

            state1, state2 = next_state1, next_state2
            
            if done:
                state1, state2 = env.reset()
    ####

    return env.reward1/step, env.reward2/step, env.get_results()