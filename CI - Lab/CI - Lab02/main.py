from nim import Nim,Nimply
from pop_funct import select_parents,mutate,NimAgent
import numpy as np

def main():
  λ = 20
  σ = 0.001
  μ = 5 
  # Specifica la lista di regole.

  # qui ancora devi mettere le regole. trovi un elenco su rules.py
  rules = [
    Nimply(0, 1),
    Nimply(1, 1),
    Nimply(2, 1),
    ...
  ]

  # Inizializza i pesi delle regole in modo casuale.
  weights = np.random.random(len(rules))

  # Inizializza la popolazione di agenti.
  population = [
    NimAgent(rules, weights)
    for _ in range(μ)
  ]

  # Esegui l'algoritmo evolutivo per 100 generazioni.
  for i in range(100):
    # Seleziona gli agenti con il fitness più alto.
    population = select_parents(population,μ,λ)

    # Applica la mutazione agli agenti selezionati.
    population = mutate(population)

    # Stampa l'agente con il fitness più alto.
    print(f"Agente con il fitness più alto: {population[0].weights}")

if __name__ == "__main__":
  main()