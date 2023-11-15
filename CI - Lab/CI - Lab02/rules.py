from nim import Nim,Nimply

# Regole basate sul numero di oggetti rimasti nel gioco

#take the first row you find with even number of remaining matches
def rule_is_even(state: Nim):
    for index_r,row in enumerate(state.rows):
        if row % 2 == 0 and row != 0:
            return Nimply(index_r,1)
    return None    

#take the first row you find with odd number of remaining matches
def rule_is_odd(state: Nim):
    for index_r,row in enumerate(state.rows):
        if row % 2 != 0 and row != 0:
            return Nimply(index_r,1)

    return None 




# def rule_is_even(state: Nim):
#   return Nimply(state.remaining_objects % 2 == 0, 2)

# def rule_is_even(state: Nim):
#   return Nimply(state.remaining_objects % 2 == 0, 3)

# def rule_is_even(state: Nim):
#   return Nimply(state.remaining_objects % 2 == 0, 4)

# def rule_is_odd(state: Nim):
#   return Nimply(state.remaining_objects % 2 == 1, 1)

# def rule_is_multiple_of_3(state: Nim):
#   return Nimply(state.remaining_objects % 3 == 0, 1)

# def rule_is_multiple_of_5(state: Nim):
#   return Nimply(state.remaining_objects % 5 == 0, 1)

# # Regole basate sul numero di oggetti che l'avversario può rimuovere in un turno

# def rule_opponent_can_remove_one(state: Nim):
#   return Nimply(state.opponent_moves == 1, 1)

# def rule_opponent_can_remove_two(state: Nim):
#   return Nimply(state.opponent_moves == 2, 1)

# def rule_opponent_can_remove_three(state: Nim):
#   return Nimply(state.opponent_moves == 3, 1)

# # Regole basate sul fitness degli agenti

# def rule_my_fitness_is_higher(state: Nim):
#   return Nimply(state.my_fitness > state.opponent_fitness, 1)

# def rule_my_fitness_is_lower(state: Nim):
#   return Nimply(state.my_fitness < state.opponent_fitness, 1)

# def rule_my_fitness_is_equal(state: Nim):
#   return Nimply(state.my_fitness == state.opponent_fitness, 1)

# # Regole basate sul comportamento degli agenti

# def rule_opponent_removed_one(state: Nim):
#   return Nimply(state.opponent_move == 1, 1)

# def rule_opponent_removed_two(state: Nim):
#   return Nimply(state.opponent_move == 2, 1)

# def rule_opponent_removed_three(state: Nim):
#   return Nimply(state.opponent_move == 3, 1)