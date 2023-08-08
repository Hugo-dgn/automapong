from scipy import stats

from agents import QLearningAgent

n = 1000
alpha = 0.005

def test_task11():
    agent = QLearningAgent("noname", alpha=0.1, gamma=0.9, eps=0.1, end_eps=0, d_step=0.1, eps_decay=0.0001)

    state = (0, 0, (0, 0), (0, 0))

    up = 0
    down = 0
    nothing = 0

    for _ in range(n):

        action = agent.act(state, training = False)

        if action == -1:
            down += 1
        elif action == 0:
            nothing += 1
        else:
            up += 1
    
    observed_frequencies = [up, down, nothing]
    expected_frequencies = [n/3, n/3, n/3]

    chi2_statistic, p_value = stats.chisquare(observed_frequencies, f_exp=expected_frequencies)

    if p_value < alpha:
        message = f"If agent.Q[state][-1] = agent.Q[state][0] = agent.Q[state][1], agent.act must return a random action. Instead got this distribution :\nup : {up/n}\ndown : {down/n}\nnothing : {nothing/n}"
        raise AssertionError(message)