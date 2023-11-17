from nim import Nim,Nimply,play_game, play_game2
from pop_funct import select_parents,mutate,fitness
from nim_agents import pure_random,gabriele,optimal,AgentsVec,NimAgent
import random
import numpy as np
import logging
from matplotlib import pyplot as plt
from matplotlib import cm
from copy import deepcopy
import rules as rs


def test():
    λ = 1
    σ = 0.001
    μ = 1
    STEPS = 1000
    # Specifica la lista di regole.

    #test1
#     rules = [
#         rs.rule_is_even,
#         rs.rule_is_odd,
#         rs.rule_is_multiple_of_3_and_5,
#         rs.rule_is_perfect_square,
#         rs.rule_is_prime,
#         rs.rule_is_multiple_of_golden_ratio
#     ]

#     weights = [-158.37343213 ,-177.89434133,   15.06265569   ,83.63611385  ,-87.40610713,
#    49.67343213] 

#     #test2
#     rules2 = [
#         rs.rule_is_even,
#         rs.rule_is_odd,
#         rs.rule_is_multiple_of_3_and_5,
#         rs.rule_is_perfect_square,
#         rs.rule_is_prime
#     ]

#     weights2 = [ 3.2378269 ,-54.52005794,  -7.36840669, -20.70913926,   4.13040257] 

#     #test3
#     rules3 = [
#         rs.rule_is_even,
#         rs.rule_is_odd
#     ]

#     weights3 = [22.78843219 -7.3754364 ] 

#     #test 4
#     rules4 = [
#         rs.rule_is_even,
#         rs.rule_is_even_2,
#         rs.rule_is_even_3,
#         rs.rule_is_even_4,
#         rs.rule_is_even_5,
#         rs.rule_is_odd,
#         rs.rule_is_odd_2,
#         rs.rule_is_odd_3,
#         rs.rule_is_odd_4,
#         rs.rule_is_odd_5,
#         rs.rule_is_multiple_of_3_and_5,
#         rs.rule_is_perfect_square,
#         rs.rule_is_perfect_square_2,
#         rs.rule_is_perfect_square_3,
#         rs.rule_is_perfect_square_4,
#         rs.rule_is_perfect_square_5,
#         rs.rule_is_prime,
#         rs.rule_is_prime_2,
#         rs.rule_is_prime_3,
#         rs.rule_is_prime_4,
#         rs.rule_is_prime_5,
#         rs.rule_is_multiple_of_golden_ratio,
#     ]
#     weights4 = [ 1405.38354931 , -599.88567261,  1485.95077607,  -989.29607573,
#   1703.96921587  , 596.11566999 , -570.13211763  ,1061.54889004,
#   -375.22587833   ,735.77499086 ,-1218.10752806 ,  770.29893759,
#    503.87139718  , 562.14781402, -1598.10036717  ,1193.26584437,
#   -561.03763855 , 1191.86149589, -1332.80970351 , 2929.44003768,
#  -2502.21567783 ,-2062.53532537]
    
#     #test5
#     rules5 = [
#         rs.rule_is_even,
#         rs.rule_is_even_2,
#         rs.rule_is_even_3,
#         rs.rule_is_even_4,
#         rs.rule_is_even_5,
#         rs.rule_is_odd,
#         rs.rule_is_odd_2,
#         rs.rule_is_odd_3,
#         rs.rule_is_odd_4,
#         rs.rule_is_odd_5,
#         rs.rule_is_perfect_square,
#         rs.rule_is_perfect_square_2,
#         rs.rule_is_perfect_square_3,
#         rs.rule_is_perfect_square_4,
#         rs.rule_is_perfect_square_5,
#         rs.rule_is_prime,
#         rs.rule_is_prime_2,
#         rs.rule_is_prime_3,
#         rs.rule_is_prime_4,
#         rs.rule_is_prime_5,
#     ]

#     weights5 = [ 570.84791841  , 51.31022865,  489.42044838, -583.57579797, -363.96191203,
#    35.25711666 ,-147.17143199 ,-215.26319207  , 55.56792015, -429.00433769,
#   -16.19099264 , 271.50510657, -212.48927882  ,194.46105837,  -27.2401491,
#   281.6172184  ,-110.03494002 ,-156.54319386  ,-46.73099971,  335.43166307]

#     #test con addestramento multiplo
#     rules6 = [
#     rs.rule_is_even,
#     rs.rule_is_even_2,
#     rs.rule_is_even_3,
#     rs.rule_is_even_4,
#     rs.rule_is_even_5,
#     rs.rule_is_odd,
#     rs.rule_is_odd_2,
#     rs.rule_is_odd_3,
#     rs.rule_is_odd_4,
#     rs.rule_is_odd_5,
#     rs.rule_is_perfect_square,
#     rs.rule_is_perfect_square_2,
#     rs.rule_is_perfect_square_3,
#     rs.rule_is_perfect_square_4,
#     rs.rule_is_perfect_square_5,
#     rs.rule_is_prime,
#     rs.rule_is_prime_2,
#     rs.rule_is_prime_3,
#     rs.rule_is_prime_4,
#     rs.rule_is_prime_5,
#     ]

#     weights6 = [ -696.57804172 ,-1229.39762782  ,-535.58885242 , -694.38865035,
#    767.49705436  ,1484.88818565 ,-1195.5382981   ,-477.46897306,
#   -485.95477708 ,-3040.81599153  , 645.10349188 ,-1896.42548403,
#  -2147.28095252 ,-2316.94792066 ,-1269.82789094 ,-2892.48429526,
#   3293.79638384  ,-985.41534646 ,-1143.46266409  , 832.99350167]
    
#     #test 7 training misto solo con optimal e random
#     rules7 = [
#         rs.rule_is_even,
#         rs.rule_is_even_2,
#         rs.rule_is_even_3,
#         rs.rule_is_even_4,
#         rs.rule_is_even_5,
#         rs.rule_is_odd,
#         rs.rule_is_odd_2,
#         rs.rule_is_odd_3,
#         rs.rule_is_odd_4,
#         rs.rule_is_odd_5,
#         rs.rule_is_perfect_square,
#         rs.rule_is_perfect_square_2,
#         rs.rule_is_perfect_square_3,
#         rs.rule_is_perfect_square_4,
#         rs.rule_is_perfect_square_5,
#         rs.rule_is_prime,
#         rs.rule_is_prime_2,
#         rs.rule_is_prime_3,
#         rs.rule_is_prime_4,
#         rs.rule_is_prime_5,
#     ]
#     weights7 = [  358.54028622  ,-547.86700939,   -11.75869638 ,-1780.59527521,
#   -829.35200112  , 400.21696293  ,-492.04440701  ,1776.63392025,
#    859.08385298 ,-1254.63234356   ,-50.947125    ,2191.86451241,
#    711.58923928  , 360.09672995  ,1109.01621819  , 256.51493463,
#     98.21926753 ,-1170.57632218 ,  -86.73682563   ,845.50142203]

#     #test 8
#     weights8 =  [-1571.06866311,  1905.80581769,  1852.86215552,  9659.14357989,
# -2290.27546087 ,  647.65798449 ,-1237.86585609 , 8498.98151122,
# -5538.63327091 , 8135.73468676  ,5131.44691438  ,-587.19810707,
# 5757.34758171 ,-3913.51457201 , 8185.19499578 , 3273.464214,
# 4365.53130607 ,-4478.36170294, -4987.96506237 ,-6045.83430714]
   
    
#     rules8 = [
#     rs.rule_is_even,
#     rs.rule_is_even_2,
#     rs.rule_is_even_3,
#     rs.rule_is_even_4,
#     rs.rule_is_even_5,
#     rs.rule_is_odd,
#     rs.rule_is_odd_2,
#     rs.rule_is_odd_3,
#     rs.rule_is_odd_4,
#     rs.rule_is_odd_5,
#     rs.rule_is_perfect_square,
#     rs.rule_is_perfect_square_2,
#     rs.rule_is_perfect_square_3,
#     rs.rule_is_perfect_square_4,
#     rs.rule_is_perfect_square_5,
#     rs.rule_is_prime,
#     rs.rule_is_prime_2,
#     rs.rule_is_prime_3,
#     rs.rule_is_prime_4,
#     rs.rule_is_prime_5,
# ]
    
     
#     weights9 =  [92.95735525729914, 16.80060384360386, -1.8301953355352594, -12.866204984664119, 82.69336513645119, -48.236898305848634, -4.8281297177635825, 80.3629980555464, 52.89743460774294, -91.90504449014058, -58.0090646982755]
#     rules9 = [
#         rs.rule_is_even,
#         rs.rule_is_even_max,
#         rs.rule_is_odd,
#         rs.rule_is_odd_max,
#         rs.rule_is_perfect_square,
#         rs.rule_is_perfect_square_max,
#         rs.rule_is_prime,
#         rs.rule_is_prime_max,
#         rs.rule_highest_ratio,
#         rs.rule_highest_remaining_matches,
#         rs.rule_lowest_remaining_matches
#     ]

    weights10 = [198.29064288540619, 79.5460039954491, -58.45404274241085, -94.3423753600089, 38.4512122260916, -59.83565390892052, -62.96102210606701, -115.55778595199456, 16.355087547483567, -120.66257440522672, -109.44218338231046, 43.02472726987027]
    rules10 = [
        rs.rule_is_even,
        rs.rule_is_even_max,
        rs.rule_is_odd,
        rs.rule_is_odd_max,
        rs.rule_is_perfect_square,
        rs.rule_is_perfect_square_max,
        rs.rule_is_prime,
        rs.rule_is_prime_max,
        rs.rule_highest_ratio,
        rs.rule_highest_remaining_matches,
        rs.rule_lowest_remaining_matches,
        optimal
    ]

    #best config with 45-47 % of success rate against optimal on 10.000 matches
    weigths = [1.1266635846403643, -0.9359140726131243, 0.6049152248796337, 1.7571856684391496, 1.1100107506223738, 2.7019260928867013, 1.1446534658255665, -0.31356745747898396, 0.8492125889529214, 1.8705204674716744, -0.16193153998494336]
    rules = [
            rs.rule_is_even,
            rs.rule_is_even_max,
            rs.rule_is_odd,
            rs.rule_is_odd_max,
            rs.rule_is_perfect_square,
            rs.rule_is_perfect_square_max,
            rs.rule_is_prime,
            rs.rule_is_prime_max,
            rs.rule_highest_ratio,
            rs.rule_highest_remaining_matches,
            rs.rule_lowest_remaining_matches,
        ]
    # Inizializza la popolazione di agenti con pesi casauli per ogni regola

    #actually test2 config is the best one
    player1 = NimAgent(f"Agent 11",np.array(rules), weigths,σ)
    player2 =  NimAgent(f"Agent 10",np.array(rules10), weights10,σ)
   
    #agents = AgentsVec([optimal])
    #opposer = agents.first()

    best_fitness = None
    #history = list()
    win_counter = 0

    oppo_strategy = optimal
    wins = 0  

    for _ in range(STEPS): 
        strategy = random.choice([(player1.choose_move, oppo_strategy), (oppo_strategy, player1.choose_move)]) 
        winner = play_game2(strategy) 
        if winner == strategy.index(player1.choose_move): 
            wins+=1 
    victory_perc = wins/STEPS * 100 

    print(victory_perc)

    print(
    f"Best solution: {victory_perc} (with σ={player1.σ:0.3g} from actor {player1.getName()})"
    )

    print(
        f"Weights: {player1.weights} (with σ={player1.σ:0.3g})"
    )

    # print(
    #     f"Win counter: {wins} -> success_rate = {victory_perc})"
    # )

    print(
        f"Best move: {rules[np.argmax(player1.weights)]} -> success_rate = {victory_perc})"
    )


if __name__ == "__main__":
  print("Test")
  test()