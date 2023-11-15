from nim import Nimply,Nim
import random
import numpy as np
from copy import deepcopy
import logging
from pprint import pprint, pformat

class NimAgent:
    name = ""
    rules = []
    weights = []
    fitness = None
    σ = None

    def __init__(self, name,rules, weights,σ):
        self.name = name
        self.rules = rules
        self.weights = weights
        self.σ = σ


    
    # def nim_sum(state: Nim) -> int:
    #     tmp = np.array([tuple(int(x) for x in f"{c:032b}") for c in state.rows])
    #     xor = tmp.sum(axis=0) % 2
    #     return int("".join(str(_) for _ in xor), base=2)
    
    # def probability(self, state: Nim):
    #     # Calcola il valore di nim-sum del gioco dopo l'applicazione della regola.
    #     nim_sum = nim_sum(state)

    #     # Se il valore di nim-sum è 0, la regola è vincente e quindi ha probabilità 1.
    #     if nim_sum == 0:
    #         return 1.0

    #     # Altrimenti, la probabilità è inversamente proporzionale al valore di nim-sum.
    #     return 1.0 / (nim_sum + 1)
    
    def analize(raw: Nim) -> list:
        possibles = list()
        for ply in (Nimply(r, o) for r, c in enumerate(raw.rows) for o in range(1, c + 1)):     #so verifies the number of remaining objects for each row and creates a Nimply(row_id,match_id) object
           possibles.append(ply)
        return possibles

    def choose_move(self, state:Nim) -> Nimply:
        # Calcola la probabilità di ogni regola.
        # probabilities = []
        # for rule, weight in zip(self.rules, self.weights):
        #     probabilities.append(weight * self.probability(state))

        # Scegli una regola in base alle probabilità.
        
        np_weights = np.array(self.weights)
        indices = np.argsort(np_weights)

        w_rules = self.rules[indices]

        move = None
        for rule in w_rules:
            ply = rule(state)
            if ply != None:
                move = ply
                break
                
        if move == None:
            # possible_moves = self.analize(state)        #verify which move is feasible according to the current Nim state
            # if len(possible_moves) == 0:
            #     print(f"No possible moves for player {self.name}, something went wrong")
            move = pure_random(state)

        return move
    
    def setFitness(self,value):
        self.fitness = value

    def getFitness(self):
        if self.fitness == None:
            print("Error, you're acceding a None fitness value")
        return self.fitness
    
    def getName(self):
        return self.name


class AgentsVec():
    agents = []
    counter = 0
    lung =  0
    def __init__(self,agents):
        self.agents = agents
        self.counter = 0
        self.lung = len(agents)

    
    def first(self):
        return self.agents[0]
    
    def next(self):
        agent =  self.agents[self.counter%self.lung]
        self.counter+=1
        self.counter=self.counter%self.lung
        return agent

# RANDOM AGENT
def pure_random(state: Nim) -> Nimply:
    """A completely random move"""
    row = random.choice([r for r, c in enumerate(state.rows) if c > 0])
    num_objects = random.randint(1, state.rows[row])
    return Nimply(row, num_objects)


# GABRIELE AGENT
def gabriele(state: Nim) -> Nimply:
    """Pick always the maximum possible number of the lowest row"""
    possible_moves = [(r, o) for r, c in enumerate(state.rows) for o in range(1, c + 1)]
    return Nimply(*max(possible_moves, key=lambda m: (-m[0], m[1])))

# def adaptive(state: Nim) -> Nimply:
#     """A strategy that can adapt its parameters"""
#     genome = {"love_small": 0.5}


# OPTIMAL AGENT
def nim_sum(state: Nim) -> int:
    tmp = np.array([tuple(int(x) for x in f"{c:032b}") for c in state.rows])
    xor = tmp.sum(axis=0) % 2
    return int("".join(str(_) for _ in xor), base=2)


def analize(raw: Nim) -> dict:
    cooked = dict()
    cooked["possible_moves"] = dict()
    for ply in (Nimply(r, o) for r, c in enumerate(raw.rows) for o in range(1, c + 1)):
        tmp = deepcopy(raw)
        tmp.nimming(ply)
        cooked["possible_moves"][ply] = nim_sum(tmp)
    return cooked


def optimal(state: Nim) -> Nimply:
    analysis = analize(state)
    logging.debug(f"analysis:\n{pformat(analysis)}")
    spicy_moves = [ply for ply, ns in analysis["possible_moves"].items() if ns != 0]
    if not spicy_moves:
        spicy_moves = list(analysis["possible_moves"].keys())
    ply = random.choice(spicy_moves)
    return ply