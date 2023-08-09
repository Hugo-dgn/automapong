import pong
import time

def play(agent1, agent2):
    pass
    ####write yout code here for task 3
    env = pong.Env()
    state1, state2 = env.reset()

    run = True
    while run:

        dt, run = pong.render(env) # affiche le jeux et donne l'interval de temps qui c'est écoulé depuis la dernière frame. C'est important pour la physique du jeux (cf delta time sur internet)
        # Obtenir les actions des deux joueurs
        action1 = agent1.act(state1, training=False)  # action est dans {-1, 0, 1}
        action2 = agent2.act(state2, training=False)  # action est dans {-1, 0, 1}

        # Effectuer une étape du jeu
        state1, state2, _, _, done = env.step(action1, action2, dt) # on obtient les nouveaux états
        
        if done:
            state1, state2 = env.reset()
    ####

    return env.get_results()


def benchmark(agent1, agent2, episode):
    pass
    ####write yout code here for task 3
    env = pong.Env()
    state1, state2 = env.reset()

    for e in range(episode):
        action1 = agent1.act(state1, training=False)  # action est dans {-1, 0, 1}
        action2 = agent2.act(state2, training=False)  # action est dans {-1, 0, 1}

        # Effectuer une étape du jeu
        state1, state2, _, _, done = env.step(action1, action2) # on obtient les nouveaux états
        
        if done:
            state1, state2 = env.reset()
    ####

    return env.get_results()
