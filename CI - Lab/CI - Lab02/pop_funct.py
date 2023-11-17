from nim import Nim, Nimply,play_game, play_game2
from nim_agents import  NimAgent, pure_random, optimal, analize
import numpy as np
from copy import deepcopy
import random



# Per la fitness:  
# - vittoria, sconfitta 
# lunghezza del match 
# numero di bastoncini tolti dall'avversario -> + è azzardata la moda dell'oppo più siamo stati bravi noi (?) 
 
def fitness(index, results): 
    won = results.winner == index
    # winner has more prority in the fitness vector
    if won:
        value = -results.turns   #if you win less turns is better
    else:
        value = results.turns    #if you loose, more turns is better
    return won,value

def fitness2(my_agent: NimAgent,AGENTS):
    N = 100
    wins = 0
    next_one = N/len(AGENTS)
    for game in range(N):
        index = int(game // next_one)
        oppo_strategy = AGENTS[index]
        strategy = random.choice([(my_agent.choose_move, oppo_strategy), (oppo_strategy, my_agent.choose_move)])
        #print(strategy)
        #print(f"mio agente idnice: {strategy.index(my_agent.choose_move)}")
        winner = play_game2(strategy)
        #print(f"Vincitore è: {winner}")
        if winner == strategy.index(my_agent.choose_move):
            #print("mio agente vince")
            #print()
            wins+=1
    victory_perc = wins/N * 100
    print(f"L'agente {my_agent.getName()} vince con perc: {victory_perc}")
    return victory_perc

def select_parents(population,μ,λ):
  # Seleziona λ agenti randomici dalla popolazione.
    indices = np.random.randint(0, μ, size=(λ,))
    offspring = [deepcopy(population[i]) for i in indices]  # Restituisci i genitori.
    # print(f"Offspring in select_parents prima di np.array: {offspring}")
    # print(f"Offspring in select_parents dopo di np.array: {np.array(offspring)}")
    return np.array(offspring)

def mutate(population,sigma):         #population must be a list of NimAgent
  # Seleziona λ genitori dalla popolazione.
  #parents = select_parents(population)
    σ = np.random.normal(loc=sigma, scale=0.2)
    if σ < 1e-5:
      σ = 1e-5
  # Muta i genitori.
    for parent in population:
        # Muta il valore di sigma.
        parent.σ = σ 

        # Se il valore di sigma è troppo piccolo, sostituirlo con un piccolo numero.
       
        # Muta i valori delle altre regole in base al valore di sigma corrente.
        parent.weights = np.random.normal(loc=parent.weights, scale=parent.σ)
    return σ
    
def is_prime(n):
    if n <= 1:
        return False
    if n <= 3:
        return True
    if n % 2 == 0 or n % 3 == 0:
        return False
    i = 5
    while i * i <= n:
        if n % i == 0 or n % (i + 2) == 0:
            return False
        i += 6
    return True

def is_perfect_square(n):
    sqrt_n = np.sqrt(n)
    return sqrt_n % 1 == 0
