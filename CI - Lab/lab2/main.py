from nim import Nim,Nimply,play_game
from pop_funct import fitness2, select_parents,mutate,fitness
from nim_agents import pure_random,gabriele,optimal,AgentsVec,NimAgent
import random
import numpy as np
import logging
from matplotlib import pyplot as plt
from matplotlib import cm
from copy import deepcopy
import rules as rs


def main():
    λ = 200
    σ = 0.001
    μ = 50
    STEPS = 10
    # Specifica la lista di regole.

    tr_rules = [
            rs.rule_is_even,
            rs.rule_is_odd,
            rs.rule_is_multiple_of_3_and_5,
            rs.rule_is_perfect_square,
            rs.rule_is_prime,
            rs.rule_is_multiple_of_golden_ratio
        ]

    tr_weights = [-158.37343213 ,-177.89434133,   15.06265569   ,83.63611385  ,-87.40610713,
   49.67343213] 

    # qui ancora devi mettere le regole. trovi un elenco su rules.py
    rules = [
        rs.rule_is_even,
        rs.rule_is_even_max,
        rs.rule_is_even_middle,
        rs.rule_is_odd,
        rs.rule_is_odd_max,
        rs.rule_is_odd_middle,
        rs.rule_is_perfect_square,
        rs.rule_is_perfect_square_max,
        rs.rule_is_perfect_square_middle,
        rs.rule_is_prime,
        rs.rule_is_prime_max,
        rs.rule_is_prime_middle,
        rs.rule_highest_ratio,
        rs.rule_highest_remaining_matches,
        rs.rule_lowest_remaining_matches,
    ]


    # Inizializza la popolazione di agenti con pesi casauli per ogni regola
    population = np.array([
        NimAgent(f"Agent {i}",np.array(rules), np.random.random(len(rules)),σ)
        for i in range(μ)
    ])


    player2 =  NimAgent(f"Agent 1",np.array(tr_rules), tr_weights,σ)
    agents = [pure_random,optimal,pure_random,optimal]
    #opposer = agents.first()

    best_fitness = None
    history = list()

    # Esegui l'algoritmo evolutivo per 1_000_000 // λ generazioni.
    for step in range(STEPS):
        fit_vec = []

        #after each 1/3 of the 
        # if step % ((STEPS // λ) // 3) == 0:
        #     opposer = agents.next()

        # Seleziona gli agenti a caso e creane una copia (figli)
        print(f"Generazione numero: {step}")
        print(f"Popolazione: {[p.getName() for p in population]}")
        offspring = select_parents(population,μ,λ,step)

        # Applica la mutazione agli agenti copiati.
        mutate(offspring,σ)
        # print(f"Offspring dopo mutate: {offspring}")
        
        for player in offspring:
            #make the order of player randomic
            # my_strategy = player.choose_move
            # strategy = [my_strategy,opposer]
            # random.shuffle(strategy)
            # index = strategy.index(my_strategy)      #take the index of our ES actor
            # strategy = tuple(strategy)

            #from the match return some parameters as result
            #params = play_game(strategy)

            fit = fitness2(player,agents)
            #save the fitness inside the player class
            player.setFitness(fit)
            #and in a fitness vector
            fit_vec.append(fit)

        #order the offspring vector according to their fitness value    
        #print(f"Fit vec: {fit_vec}")
        
        #fit_vec_np = np.array(fit_vec, dtype=[('bool', bool), ('num', int)])
        #indices = np.argsort(fit_vec_np, order=['bool', 'num'])
        fit_vec_np = np.array(fit_vec)
        indices = np.argsort(fit_vec_np)

        offspring = offspring[indices]

            # save best (just for the plot)
        if best_fitness is None or best_fitness < max(fit_vec):
            best_fitness = max(fit_vec)
            history.append((step, best_fitness))

        population = deepcopy(offspring[-μ:])

        print(f"Agente con il fitness più alto: {population[-1].getName()} con fitness: {population[-1].getFitness()}")

    print(
        f"Best solution: {population[-1].getFitness()} (with σ={population[-1].σ:0.3g} from actor {population[-1].getName()})"
    )

    print(
        f"Weights: {list(population[-1].weights)} (with σ={population[-1].σ:0.3g})"
    )

    history = np.array(history)
    plt.figure(figsize=(14, 4))
    plt.plot(history[:, 0], history[:, 1], marker=".")

if __name__ == "__main__":
  main()