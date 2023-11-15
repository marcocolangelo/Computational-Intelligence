from nim import Nim,Nimply,play_game
from pop_funct import select_parents,mutate,fitness
from nim_agents import pure_random,gabriele,optimal,AgentsVec,NimAgent
import random
import numpy as np
import logging
from matplotlib import pyplot as plt
from matplotlib import cm
from copy import deepcopy
from rules import rule_is_even,rule_is_odd


def main():
    λ = 20
    σ = 0.001
    μ = 5 
    STEPS = 10000
    # Specifica la lista di regole.

    # qui ancora devi mettere le regole. trovi un elenco su rules.py
    rules = [
        rule_is_even,
        rule_is_odd
    ]


    # Inizializza la popolazione di agenti con pesi casauli per ogni regola
    population = np.array([
        NimAgent(f"Agent {i}",np.array(rules), np.random.random(len(rules)),σ)
        for i in range(μ)
    ])

    agents = AgentsVec([optimal])
    opposer = agents.first()

    best_fitness = None
    history = list()

    # Esegui l'algoritmo evolutivo per 1_000_000 // λ generazioni.
    for step in range(STEPS // λ):
        fit_vec = []

        #after each 1/3 of the 
        if step % ((STEPS // λ) // 3) == 0:
            opposer = agents.next()

        # Seleziona gli agenti a caso e creane una copia (figli)
        offspring = select_parents(population,μ,λ)

        # Applica la mutazione agli agenti copiati.
        mutate(offspring)
        # print(f"Offspring dopo mutate: {offspring}")

        for player in offspring:
            #make the order of player randomic
            my_strategy = player.choose_move
            strategy = [my_strategy,opposer]
            random.shuffle(strategy)
            index = strategy.index(my_strategy)      #take the index of our ES actor
            strategy = tuple(strategy)

            #from the match return some parameters as result
            params = play_game(strategy)
            fit = fitness(index,params)
            #save the fitness inside the player class
            player.setFitness(fit)
            #and in a fitness vector
            fit_vec.append(fit)

        #order the offspring vector according to their fitness value    
        #print(f"Fit vec: {fit_vec}")
        
        fit_vec_np = np.array(fit_vec, dtype=[('bool', bool), ('num', int)])
        indices = np.argsort(fit_vec_np, order=['bool', 'num'])

        offspring = offspring[indices]

            # save best (just for the plot)
        if best_fitness is None or best_fitness < max(fit_vec):
            best_fitness = max(fit_vec)
            #history.append((step, best_fitness))

        population = deepcopy(offspring[-μ:])

        #print(f"Agente con il fitness più alto: {population[-1].getName()}")

    logging.info(
        f"Best solution: {population[-1].getFitness()} (with σ={population[-1].σ:0.3g} from actor {population[-1].getName()})"
    )

    logging.info(
        f"Weights: {population[-1].weights} (with σ={population[-1].σ:0.3g})"
    )

    # history = np.array(history)
    # plt.figure(figsize=(14, 4))
    # plt.plot(history[:, 0], history[:, 1], marker=".")

if __name__ == "__main__":
  main()