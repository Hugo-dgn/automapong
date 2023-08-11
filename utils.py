import agents

def get_agent(name_agent):
    if name_agent == "human1":
        return agents.HumanAgent(player=1)
    elif name_agent == "human2":
        return agents.HumanAgent(player=2)
    elif name_agent == "simple":
        agent = agents.SimpleAgent("simple")
        def _save():
            pass
        agent.save = _save
        return agent
    elif name_agent == "strong":
        agent = agents.StrongAgent("strong")
        def _save():
            pass
        agent.save = _save
        return agent
    else:
        return agents.load(name_agent)