import contextlib
with contextlib.redirect_stdout(None):
    import pygame
    pygame.init()

import sys
import os

import humanize
import torch

import argparse
import matplotlib.pyplot as plt
import numpy as np

import agents
import network

from play import play
from train import train
from schearch import grid_schearch


#### Default parameters for agent creation

gamma = 0.9
eps = 1
eeps = 0.1
edecay = 0.01
dqn = 1
lr = 1e-3
capacity = 2_000
batch = 256
tau = 1/batch
skip = 10
d = 0.01

####

do_not_train = ["simple", "strong"]

def _play(args):
    if args.agents[0] == "human":
        args.agents[0] = "human1"
    if args.agents[1] == "human":
        args.agents[1] = "human2"

    agent1 = agents.get_agent(args.agents[0])
    agent2 = agents.get_agent(args.agents[1])
    play(agent1, agent2)

def _train(args):

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Trainig using {device}")

    train_agents = []

    for name in args.agents:
        if name == "human":
            message = "Agent can't be trained by human. Meaning 'human' is not a valid argument for train."
            raise AssertionError(message)
        else:
            train_agents.append(agents.get_agent(name))

    if len(train_agents) == 1:
        train_agents.append(train_agents[0])

    for r in range(1, args.round+1):
        print(f"--- Start round {r} out of {args.round} ---")
        for i, agent1 in enumerate(train_agents):
            for j, agent2 in enumerate(train_agents):
                
                if i <= j or (agent1.name in do_not_train and agent2.name in do_not_train):
                    continue
                print(f"\n{agent1.name} vs {agent2.name}")

                results = train(agent1, agent2, args.episode, args.dt)

                for agent, win in results.items():
                    if agent == 1:
                        print(f"win for agent {agent1.name} : {win}")
                    elif agent == 2:
                        print(f"win for agent {agent2.name} : {win}")
                    elif agent == 0:
                        print(f"draw : {win}")

                if agent1.name in do_not_train:
                    print(f"Saving agent {agent2.name}.")
                elif agent2.name in do_not_train:
                    print(f"Saving agent {agent1.name}.")
                else:
                    print(f"Saving agents {agent1.name} and {agent2.name}.")

                agent1.save()
                agent2.save()

        print(f"\nDone with round {r}.\n")
    
    print("Done.")

def create(args):
    if not os.path.exists("agents/save"):
        os.makedirs("agents/save")
    if args.agent in os.listdir("agents/save"):
        decision = input(f"Agent with name {args.agent} already exist. Do you want to overwrite this agent ? (y/n) :")
        if decision.lower().strip() != "y":
            print("Agent was not overwriten.")
            sys.exit(1)
        else:
            print(f"Overwriting agent {args.agent}.")
    else:
        print(f"Creating agent {args.agent}.")

    if args.type == "ql":
        agent = agents.QLearningAgent(args.agent, lr=args.lr, gamma=args.gamma, eps=args.eps, eeps=args.eeps, d=args.d, edecay=args.edecay)
        agent.save()
    elif args.type == "dql":
        DQN = network.get_topology(args.dqn)
        agent = agents.DeepQLearningAgent(args.agent, dqn=DQN, lr=args.lr, gamma=args.gamma, eps=args.eps, eeps=args.eeps, edecay=args.edecay, capacity=args.capacity, batch=args.batch, tau=args.tau, skip=args.skip)
        agent.save()

def reward(args):

    n_agents = len(args.agents)

    if len(set(args.agents)) < n_agents:
        print("Warning : you are ploting the reward of the same agent at least twice.")

    if n_agents > 4 and not args.split:
        print("Warning : more than 3 agent's reward history are ploted on the same plot. Try using flag '-s' to split the plot.")

    ncols = int(np.sqrt(n_agents)) + 1
    nrows = int(n_agents/ncols)

    if ncols*nrows < n_agents:
        nrows += 1
    
    if n_agents == 1:
        args.split = True
        ncols = 1
        nrows = 1


    if not args.split:
        plt.figure()
        plt.title(f"reward")
        plt.xlabel("episode")
        plt.ylabel("reward")
    
    for i, name_agent in enumerate(args.agents):

        agent = agents.get_agent(name_agent)

        if args.split:
            plt.subplot(nrows, ncols, i+1)
            plt.title(f"reward for agent {agent.name}")
            plt.xlabel("episode")
            plt.ylabel("reward")

        y = agent._reward_history

        if not len(y) > 1:
            message = "Agent must have played more than 1 game for its rewards to be plot."
            raise AssertionError(message)

        x = np.linspace(1, agent.train_games, len(y))

        n1 = int(len(y)/100) + 1
        n2 = int(len(y)/10) + 1

        pad_y1 = np.pad(y, (int(n1/2), n1 - int(n1/2) - 1), 'constant', constant_values=(np.min(y), y[-1]))
        pad_y2 = np.pad(y, (int(n2/2), n2 - int(n2/2) - 1), 'constant', constant_values=(np.min(y), y[-1]))

        window1 = get_gauss_kernel(n1)
        y1 = np.convolve(pad_y1, window1, mode='valid')

        window2 = get_gauss_kernel(n2)
        y2 = np.convolve(pad_y2, window2, mode='valid')

        plt.plot(x, y2, label=agent.name if not args.split else None)
        color = plt.gca().lines[-1].get_color()
        if not args.clean:
            plt.plot(x, y1, color=color, alpha=0.3)
    
    plt.subplots_adjust(wspace=0.7, hspace=0.7)
    if not args.split:
        plt.legend()
    plt.show()

def get_gauss_kernel(n):
    x_win = np.linspace(-n / 2, n / 2, n)
    sigma = n/3
    kernel = np.exp(-x_win**2 / (2 * sigma**2)) / (sigma * np.sqrt(2 * np.pi))
    return kernel / np.sum(kernel)

def info(args):
    agent = agents.get_agent(args.agent)

    size = os.path.getsize(f"agents/save/{agent.name}")
    human_readable_size = humanize.naturalsize(size)

    print(f"size = {human_readable_size}")

    for attr, value in vars(agent).items():
        if attr [0] != "_" and not isinstance(value, (list, tuple, dict)):
            print(f"{attr} = {value}")

def grid(args):

    if args.GridAgent == "dql":
        GridAgent = agents.DeepQLearningAgent
    elif args.GridAgent == "ql":
        GridAgent = agents.QLearningAgent
    
    trainagent = agents.get_agent(args.trainAgent)
    benchmarkagent = agents.get_agent(args.benchmarkAgent)

    args.dqn = tuple([network.get_topology(i) for i in args.dqn])

    kwards = {key: value for key, value in vars(args).items() if key not in ["GridAgent", "trainAgent", "benchmarkAgent", "train_episode", "benchmark_episode", "dt"]}

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Trainig using {device}")

    grid_schearch(GridAgent, trainagent, benchmarkagent, args.train_episode, args.benchmark_episode, args.dt, **kwards)

def copy(args):
    agent = agents.get_agent(args.agent)
    agent.name = args.new_agent
    agent.save()

def main():
    parser = argparse.ArgumentParser(description="Agent Game")
    
    subparsers = parser.add_subparsers(title="subcommands", dest="subcommand")
    
    play_parser = subparsers.add_parser("play", help="Play with agents")
    play_parser.add_argument("agents", nargs=2, help="List of agents to play with")
    play_parser.set_defaults(func=_play)
    
    copy_parser = subparsers.add_parser("copy", help="Copy an agent")
    copy_parser.add_argument("agent", help="name of the agent to copy")
    copy_parser.add_argument("new_agent", help="name of the new agent")
    copy_parser.set_defaults(func=copy)

    train_parser = subparsers.add_parser("train", help="Train agents")
    train_parser.add_argument("agents", nargs="+", help="List of agents to train")
    train_parser.add_argument("-e", "--episode", type=int, help="Number of episode for training.", default=10_000)
    train_parser.add_argument("-r", "--round", type=int, help="Number of round for training.", default=1)
    train_parser.add_argument("-dt", type=float, help="Time stamp for Euler integration.", default=0.1)
    train_parser.set_defaults(func=_train)
    
    create_parser = subparsers.add_parser("create", help="Create an agent")
    create_parser.add_argument("type", help="name of the agent to create", choices=["dql", "ql"])
    create_parser.add_argument("agent", help="name of the agent to create")

    create_parser.add_argument("-gamma", type=float, help="The Bellman's equation gamma.", default=gamma)
    create_parser.add_argument("-eps", type=float, help="Epsilon parameter for epsilon greedy algorithm.", default=eps)
    create_parser.add_argument("-eeps", type=float, help="End epsilon parameter for epsilon greedy algorithm.", default=eeps)
    create_parser.add_argument("-edecay", type=float, help="Exponential decay parameter for epsilon greedy algorithm.", default=edecay)
    create_parser.add_argument("-dqn", type=int, help="Topology of the neural network.", default=dqn)
    create_parser.add_argument("-lr", type=float, help="Learning rate.", default=lr)
    create_parser.add_argument("-capacity", type=int, help="Capacity of the replay memory.", default=capacity)
    create_parser.add_argument("-batch", type=int, help="Batch size.", default=batch)
    create_parser.add_argument("-tau", type=float, help="Inverse of the learn step needed between two update of the target network.", default=tau)
    create_parser.add_argument("-skip", type=int, help="Step skip between two learning step.", default=skip)
    create_parser.add_argument("-d", type=float, help="Discretisation step.", default=d)

    create_parser.set_defaults(func=create)

    reward_parser = subparsers.add_parser("reward", help="plot the reward of an agent")
    reward_parser.add_argument("agents", nargs="+", help="name of the agent")
    reward_parser.add_argument("-s", "--split", action="store_true")
    reward_parser.add_argument("-c", "--clean", action="store_true")
    reward_parser.set_defaults(func=reward)

    info_parser = subparsers.add_parser("info", help="print some info on the agent")
    info_parser.add_argument("agent", help="name of the agent")
    info_parser.set_defaults(func=info)
    
    grid_parser = subparsers.add_parser("grid", help="Does a grid schearch for hyperparameters.")
    grid_parser.add_argument("GridAgent", help="Name of the agent to do the grid schearch on.")
    grid_parser.add_argument("trainAgent", help="Name of the agent to do the training.")
    grid_parser.add_argument("benchmarkAgent", help="Name of the agent to do the benchmark.")
    grid_parser.add_argument("train_episode", type=int, help="Number of train_episode.")
    grid_parser.add_argument("benchmark_episode", type=int, help="Number of benchmark_episode.")
    grid_parser.add_argument("-dt", type=float, help="Euler integration time.", default=0.1)

    grid_parser.add_argument("-name", nargs="*", type=str, help="name of the agent to create", default=("noname",))
    grid_parser.add_argument("-gamma", nargs="*", type=float, help="The Bellman's equation gamma.", default=[gamma])
    grid_parser.add_argument("-eps", nargs="*", type=float, help="Epsilon parameter for epsilon greedy algorithm.", default=[eps])
    grid_parser.add_argument("-eeps", nargs="*", type=float, help="End epsilon parameter for epsilon greedy algorithm.", default=[eeps])
    grid_parser.add_argument("-edecay", nargs="*", type=float, help="Exponential decay parameter for epsilon greedy algorithm.", default=[edecay])
    grid_parser.add_argument("-dqn", nargs="*", type=int, help="Topology of the neural network.", default=[dqn])
    grid_parser.add_argument("-lr", nargs="*",type=float, help="Learning rate.", default=[lr])
    grid_parser.add_argument("-capacity", nargs="*", type=int, help="Capacity of the replay memory.", default=[capacity])
    grid_parser.add_argument("-batch", nargs="*", type=int, help="Batch size.", default=[batch])
    grid_parser.add_argument("-tau", nargs="*", type=float, help="Inverse of the learn step needed between two update of the target network.", default=[tau])
    grid_parser.add_argument("-skip", nargs="*", type=int, help="Step skip between two learning step.", default=[skip])
    grid_parser.add_argument("-d", nargs="*", type=float, help="Discretisation step.", default=[d])

    grid_parser.set_defaults(func=grid)

    args = parser.parse_args()
    
    if hasattr(args, 'func'):
        args.func(args)
    else:
        parser.print_help()

if __name__ == "__main__":
    main()