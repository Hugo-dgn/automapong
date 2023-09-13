import pong
import agents

env = pong.Env() #créer l'environnement

agent1 = agents.HumanAgent(player=1)  # Créer un agent controlé par les touches a et q à gauche du jeux (si player=2 alors l'agent est contrôlé par les touches p et m et se trouve à droite du jeux).
agent2 = agents.SimpleAgent(name="simple") # Créer l'agent simple codé précédement.

state1, state2 = env.reset()  # l'environnement fournit les états initiaux

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