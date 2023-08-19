from tqdm.auto import tqdm

import pong

def play(agent1, agent2):
    env = pong.Env()

    ####write yout code here for task 3
    state1, state2 = env.reset()
    run = True
    while run:

        dt, run = pong.render(env)  # affiche le jeu et donne l'intervalle de temps qui s'est écoulé depuis la dernière frame. C'est important pour la physique du jeu (cf delta time sur internet)

        # Obtenir les actions des deux joueurs
        action1 = agent1.act(state1, training=False)  # action est dans {-1, 0, 1}
        action2 = agent2.act(state2, training=False)  # action est dans {-1, 0, 1}

        # Effectuer une étape du jeu
        state1, state2, _, _, done = env.step(action1, action2, dt)  # on obtient les nouveaux états
        
        if done:
            state1, state2 = env.reset()
    ####

    return env.get_results() # return (wins for agent1, wins for agent2, draws)


def benchmark(agent1, agent2, episode, verbose=True):
    env = pong.Env()
    state1, state2 = env.reset()

    step = 0

    for e in tqdm(range(episode), desc="playing") if verbose else range(episode):
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
