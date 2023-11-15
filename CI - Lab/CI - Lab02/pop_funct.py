from nim import Nim, Nimply,play_game
from nim_agents import  pure_random, optimal, analize
import numpy as np
import random

class NimAgent:
    def __init__(self, rules, weights):
        self.rules = rules
        self.weights = weights

    def nim_sum(state: Nim) -> int:
        tmp = np.array([tuple(int(x) for x in f"{c:032b}") for c in state.rows])
        xor = tmp.sum(axis=0) % 2
        return int("".join(str(_) for _ in xor), base=2)
    
    def probability(self, state: Nim):
        # Calcola il valore di nim-sum del gioco dopo l'applicazione della regola.
        nim_sum = nim_sum(state)

        # Se il valore di nim-sum è 0, la regola è vincente e quindi ha probabilità 1.
        if nim_sum == 0:
            return 1.0

        # Altrimenti, la probabilità è inversamente proporzionale al valore di nim-sum.
        return 1.0 / (nim_sum + 1)

    def choose_move(self, state):
        # Calcola la probabilità di ogni regola.
        probabilities = []
        for rule, weight in zip(self.rules, self.weights):
            probabilities.append(weight * self.probability(state))

        # Scegli una regola in base alle probabilità.
        move = random.choices(self.rules, weights=probabilities)[0]

        return move

def fitness(agent, state):
  # Gioca una partita tra l'agente e l'agente ottimale.
  winner = play_game(agent, optimal(state))

  # Se l'agente vince, il suo fitness è 1. Altrimenti, il suo fitness è 0.
  if winner == agent:
    return 1
  else:
    return 0

def select_parents(population,μ,λ):
  # Seleziona λ agenti randomici dalla popolazione.
  parents = population[np.random.randint(0, μ, size=(λ,))]

  # Restituisci i genitori.
  return parents

def mutate(population):
  # Seleziona λ genitori dalla popolazione.
  parents = select_parents(population)

  # Muta i genitori.
  offspring = []
  for parent in parents:
    # Muta il valore di sigma.
    parent.weights[-1] = np.random.normal(loc=parent.weights[-1], scale=0.2)

    # Se il valore di sigma è negativo, sostituirlo con un piccolo numero.
    if parent.weights[-1] < 1e-5:
      parent.weights[-1] = 1e-5

    # Muta i valori delle altre regole in base al valore di sigma corrente.
    parent.weights[0:-1] = np.random.normal(loc=parent.weights[0:-1], scale=parent.weights[-1].reshape(-1, 1))
    offspring.append(parent)

  # Restituisci la popolazione mutata.
  return offspring