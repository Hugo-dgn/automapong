import agents
import network

def test_task16():
    dqn = network.get_topology(1)
    agent = agents.DeepQLearningAgent("noname", dqn=dqn, lr=0.1, gamma=0.9, eps=0.1, eeps=0, edecay=1, capacity=1000, batch=32, tau=0.1, skip=1)

    answer = {}

    target_net_state_dict = agent._target_dqn.state_dict()
    dqn_net_state_dict = agent.dqn.state_dict()

    for key in dqn_net_state_dict:
        answer[key] = (dqn_net_state_dict[key]*agent.tau + target_net_state_dict[key]*(1-agent.tau))

    agent._target_dqn.load_state_dict(target_net_state_dict)

    agent.soft_update_target()

    for key in dqn_net_state_dict:
        if not all(answer[key].reshape(-1) == target_net_state_dict[key].reshape(-1)):
            message = "Soft update is not right."
            raise AssertionError(message)