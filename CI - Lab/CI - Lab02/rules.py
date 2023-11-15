from nim import Nim,Nimply
from pop_funct import is_prime,is_perfect_square
import numpy as np
# Regole basate sul numero di oggetti rimasti nel gioco

#take the first row you find with even number of remaining matches
def rule_is_even(state: Nim):
    for index_r,row in enumerate(state.rows):
        if row % 2 == 0 and row != 0:
            return Nimply(index_r,1)
    return None    

def rule_is_even_2(state: Nim):
    for index_r,row in enumerate(state.rows):
        if row % 2 == 0 and row > 1:
            return Nimply(index_r,2)
    return None   

def rule_is_even_3(state: Nim):
    for index_r,row in enumerate(state.rows):
        if row % 2 == 0 and row > 2:
            return Nimply(index_r,3)
    return None   

def rule_is_even_4(state: Nim):
    for index_r,row in enumerate(state.rows):
        if row % 2 == 0 and row > 3:
            return Nimply(index_r,4)
    return None   

def rule_is_even_5(state: Nim):
    for index_r,row in enumerate(state.rows):
        if row % 2 == 0 and row > 4:
            return Nimply(index_r,5)
    return None   

#take the first row you find with odd number of remaining matches
def rule_is_odd(state: Nim):
    for index_r,row in enumerate(state.rows):
        if row % 2 != 0 and row != 0:
            return Nimply(index_r,1)

    return None 

def rule_is_odd_2(state: Nim):
    for index_r,row in enumerate(state.rows):
        if row % 2 != 0 and row > 1 :
            return Nimply(index_r,2)

    return None 

def rule_is_odd_3(state: Nim):
    for index_r,row in enumerate(state.rows):
        if row % 2 != 0 and row > 2 :
            return Nimply(index_r,3)

    return None 

def rule_is_odd_4(state: Nim):
    for index_r,row in enumerate(state.rows):
        if row % 2 != 0 and row > 3:
            return Nimply(index_r,4)

    return None 

def rule_is_odd_5(state: Nim):
    for index_r,row in enumerate(state.rows):
        if row % 2 != 0 and row > 4:
            return Nimply(index_r,5)

    return None 
#take the first row you find with a prime number of remaining matches
def rule_is_prime(state: Nim):
    for index_r,row in enumerate(state.rows):
        if row > 1 and is_prime(row) and row != 0:
            return Nimply(index_r,1)
    return None  

def rule_is_prime_2(state: Nim):
    for index_r,row in enumerate(state.rows):
        if row > 1 and is_prime(row) and row > 1:
            return Nimply(index_r,2)
    return None  

def rule_is_prime_3(state: Nim):
    for index_r,row in enumerate(state.rows):
        if row > 1 and is_prime(row) and row > 2:
            return Nimply(index_r,3)
    return None  

def rule_is_prime_4(state: Nim):
    for index_r,row in enumerate(state.rows):
        if row > 1 and is_prime(row) and row > 3:
            return Nimply(index_r,4)
    return None  

def rule_is_prime_5(state: Nim):
    for index_r,row in enumerate(state.rows):
        if row > 1 and is_prime(row) and row > 4:
            return Nimply(index_r,5)
    return None  

#Take the first row you find with a perfect square number of remaining matches
def rule_is_perfect_square(state: Nim):
    for index_r,row in enumerate(state.rows):
        if is_perfect_square(row) and row != 0:
            return Nimply(index_r,1)
    return None

def rule_is_perfect_square_2(state: Nim):
    for index_r,row in enumerate(state.rows):
        if is_perfect_square(row) and row > 1:
            return Nimply(index_r,2)
    return None

def rule_is_perfect_square_3(state: Nim):
    for index_r,row in enumerate(state.rows):
        if is_perfect_square(row) and row > 2:
            return Nimply(index_r,3)
    return None

def rule_is_perfect_square_4(state: Nim):
    for index_r,row in enumerate(state.rows):
        if is_perfect_square(row) and row > 3:
            return Nimply(index_r,4)
    return None

def rule_is_perfect_square_5(state: Nim):
    for index_r,row in enumerate(state.rows):
        if is_perfect_square(row) and row > 4:
            return Nimply(index_r,5)
    return None

#take the first row you find with a multiple of 3 and multiple of 5 remaining matches
def rule_is_multiple_of_3_and_5(state: Nim):
    for index_r,row in enumerate(state.rows):
        if row % 3 == 0 and row % 5 == 0 and row != 0:
            return Nimply(index_r,1)
    return None

#take the first row with a multiple_of_golden_ratio
def rule_is_multiple_of_golden_ratio(state: Nim):
    golden_ratio = (1 + np.sqrt(5)) / 2
    for index_r,row in enumerate(state.rows):
        if row != 0 and row % golden_ratio == 0:
            return Nimply(index_r,1)
    return None

# #Take the first row you find with the highest ratio of remaining matches to total matches
# def rule_highest_ratio(state: Nim):
#     for index_r,row in enumerate(state.rows):
#         if row != 0:
#             ratio = row / sum(state.rows)
#             return Nimply(index_r,1)
#     return None

# #Take the first row you find with the lowest difference between remaining matches and the number of matches that the opponent can remove in a turn
# def rule_lowest_difference(state: Nim):
#     for index_r,row in enumerate(state.rows):
#         if row != 0:
#             difference = row - state.opponent_moves
#             return Nimply(index_r,1)
#     return None



