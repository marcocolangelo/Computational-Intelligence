from nim import Nim, Nimply,play_game
from nim_agents import  pure_random, optimal, analize
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

def select_parents(population,μ,λ):
  # Seleziona λ agenti randomici dalla popolazione.
    indices = np.random.randint(0, μ, size=(λ,))
    offspring = [deepcopy(population[i]) for i in indices]  # Restituisci i genitori.
    # print(f"Offspring in select_parents prima di np.array: {offspring}")
    # print(f"Offspring in select_parents dopo di np.array: {np.array(offspring)}")
    return np.array(offspring)

def mutate(population):         #population must be a list of NimAgent
  # Seleziona λ genitori dalla popolazione.
  #parents = select_parents(population)

  # Muta i genitori.
  for parent in population:
    # Muta il valore di sigma.
    parent.σ = np.random.normal(loc=parent.σ, scale=0.2)

    # Se il valore di sigma è troppo piccolo, sostituirlo con un piccolo numero.
    if parent.σ < 1e-5:
      parent.σ = 1e-5

    # Muta i valori delle altre regole in base al valore di sigma corrente.
    parent.weights = np.random.normal(loc=parent.weights, scale=parent.σ)
    
