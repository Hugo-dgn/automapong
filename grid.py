import numpy as np

from itertools import product
import inspect

from train import train
from play import benchmark

def grid_schearch(agentConstructor, benchmarkAgent, train_episode, benchmark_episode , **kwards):

    constructor_signature = list(inspect.signature(agentConstructor.__init__).parameters.keys())

    parameters = {key : value for (key, value) in kwards.items() if key in constructor_signature}


    values = list(parameters.values())
    combinations = list(product(*values))

    keys_combination = [{key : x for (key, x) in zip(list(parameters.keys()), combination)} for combination in combinations]

    fitness = []

    for i, hyperparams in enumerate(keys_combination):

        print(f"\ntrying hyperparameters : {' | '.join([f'{key} ; {value} ' for (key, value) in list(hyperparams.items())])}")

        agent = agentConstructor(**hyperparams)

        train(agent, benchmarkAgent, train_episode)

        results = benchmark(agent, benchmarkAgent, benchmark_episode)

        agent_fitness = 100*(results[1] - results[2])/benchmark_episode
        print(f"fitness : {agent_fitness}")

        fitness.append(agent_fitness)

    
    print(f"\nBest parameters found :\n{' '.join([f'-{key} {value}' for (key, value) in keys_combination[np.argmax(fitness)].items()])}\nwith fitness {max(fitness)}.")