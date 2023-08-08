import numpy as np
import scipy.stats as stats

from agents import QLearningAgent


n = 100
alpha = 0.01

def test_task9_eps():

    for i, eps in enumerate([0.1, 0.4, 0.7]):

        special_action = i%3 - 1

        agent = QLearningAgent("noname", lr=0.1, gamma=0.9, eps=eps, end_eps=0.1, d_step=0.1, eps_decay=0.0001)

        down = 0
        up = 0
        nothing = 0

        state = (0, 0, (0, 0), (0, 0))

        agent.check_q_value(agent.transform_state(state))
        for key in agent.Q:
            agent.Q[key] = {-1 : 0, 0 : 0, 1 : 0}
            agent.Q[key][special_action] = 1

        for _ in range(n):
            agent.step = -1
            action = agent.act(state, training=True)

            if action == -1:
                down += 1
            elif action == 0:
                nothing += 1
            else:
                up += 1

        theo_up = eps/3
        theo_down = eps/3
        theo_nothing = eps/3

        if special_action == -1:
            theo_down = 1 - 2*eps/3
        elif special_action == 0:
            theo_nothing = 1 - 2*eps/3
        elif special_action == 1:
            theo_up = 1 - 2*eps/3

        observed_frequencies = [up, down, nothing]
        expected_frequencies = [theo_up*n, theo_down*n, theo_nothing*n]

        chi2_statistic, p_value = stats.chisquare(observed_frequencies, f_exp=expected_frequencies)

        if p_value < alpha:
            message = f"The Chi-squared test did not pass for the epsilon-greedy algorithm (p-value obtained = {p_value}). When `agent.step = 0`, the agent must use `agent.act` to choose randomly between (-1, 0, 1) with a probability of `eps`, following a uniform distribution."
            message += f"\n\nHere is the distribution of choices recived with eps={eps} and self.Q[state] = {agent.Q[agent.transform_state(state)]}:\nup = {up/n} (should be {theo_up})\ndown = {down/n} (should be {theo_down})\nnothing={nothing/n} (should be {theo_nothing})"
            raise AssertionError(message)

def test_task9_eeps():

    for i, eeps in enumerate(np.linspace(0.1, 1, 10)):

        special_action = i%3 - 1

        agent = QLearningAgent("noname", lr=0.1, gamma=0.9, eps=0.5, end_eps=eeps, d_step=0.1, eps_decay=0.0001)

        down = 0
        up = 0
        nothing = 0

        state = (0, 0, (0, 0), (0, 0))

        agent.check_q_value(agent.transform_state(state))
        for key in agent.Q:
            agent.Q[key] = {-1 : 0, 0 : 0, 1 : 0}
            agent.Q[key][special_action] = 1

        for _ in range(n):
            agent.step = float("inf")
            action = agent.act(state, training=True)

            if action == -1:
                down += 1
            elif action == 0:
                nothing += 1
            else:
                up += 1
        
        theo_up = eeps/3
        theo_down = eeps/3
        theo_nothing = eeps/3

        if special_action == -1:
            theo_down = 1 - 2*eeps/3
        elif special_action == 0:
            theo_nothing = 1 - 2*eeps/3
        elif special_action == 1:
            theo_up = 1 - 2*eeps/3

        observed_frequencies = [up, down, nothing]
        expected_frequencies = [theo_up*n, theo_down*n, theo_nothing*n]

        chi2_statistic, p_value = stats.chisquare(observed_frequencies, f_exp=expected_frequencies)

        if p_value < alpha:
            message = f"The Chi-squared test did not pass for the epsilon-greedy algorithm (p-value obtained = {p_value}). When `agent.step = inf`, the agent must use `agent.act` to choose randomly between (-1, 0, 1) with a probability of `eeps`, following a uniform distribution."
            message += f"\n\nHere is the distribution of choices recived with eps={eeps} and self.Q[state] = {agent.Q[agent.transform_state(state)]}:\nup = {up/n} (should be {theo_up})\ndown = {down/n} (should be {theo_down})\nnothing={nothing/n} (should be {theo_nothing})"
            raise AssertionError(message)

def test_task10_general():

    eps = 0.5
    end_eps = 0.1
    eps_decay = 0.1

    for i, step in enumerate(range(1, 100, 10)):

        special_action = i%3 - 1

        agent = QLearningAgent("noname", lr=0.1, gamma=0.9, eps=eps, end_eps=end_eps, d_step=0.1, eps_decay=eps_decay)

        down = 0
        up = 0
        nothing = 0

        state = (0, 0, (0, 0), (0, 0))

        agent.check_q_value(agent.transform_state(state))
        for key in agent.Q:
            agent.Q[key] = {-1 : 0, 0 : 0, 1 : 0}
            agent.Q[key][special_action] = 1

        for _ in range(n):
            agent.step = step - 1
            action = agent.act(state, training=True)

            if action == -1:
                down += 1
            elif action == 0:
                nothing += 1
            else:
                up += 1
        
        true_eps = (eps - end_eps) * np.exp(-eps_decay*step) + end_eps

        theo_up = true_eps/3
        theo_down = true_eps/3
        theo_nothing = true_eps/3

        if special_action == -1:
            theo_down = 1 - 2*true_eps/3
        elif special_action == 0:
            theo_nothing = 1 - 2*true_eps/3
        elif special_action == 1:
            theo_up = 1 - 2*true_eps/3

        observed_frequencies = [up, down, nothing]
        expected_frequencies = [theo_up*n, theo_down*n, theo_nothing*n]

        chi2_statistic, p_value = stats.chisquare(observed_frequencies, f_exp=expected_frequencies)

        if p_value < alpha:
            message = f"The Chi-squared test did not pass for the epsilon-greedy algorithm (p-value obtained = {p_value}). When `agent.step = step`, the agent must use `agent.act` to choose randomly between (-1, 0, 1) with a probability of `(eps - end_eps)*exp(-eps_decay*step) + end_eps`, following a uniform distribution."
            message += f"\n\nHere is the distribution of choices recived with eps={eps} and self.Q[state] = {agent.Q[agent.transform_state(state)]}:\nup = {up/n} (should be {theo_up})\ndown = {down/n} (should be {theo_down})\nnothing={nothing/n} (should be {theo_nothing})"
            raise AssertionError(message)