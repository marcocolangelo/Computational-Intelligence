## NIM Game using  Adaptive (μ+λ)-ES

# In collaboration with Roberto Pulvirenti S317704

The aim of this lab was coding an agent able of competing against the OPTIMAL agent in the Nim game.

We used a set of fixed rules that we bound to the respective weights firstly set as random values, calibrated though the ES strategy then. The weights have been used as priority values during the choice of the rule to apply for the agent in each single turn.

We tried multiple configurations and different sets of rules, but the most efficient one which brought us a performance of 45-47 % winning ratio against the optimal agent is the following one:

    λ = 200
    σ = 0.001
    μ = 50
    STEPS = 100 -> but for each STEP each agent has played 100 times

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

You can find the meaning and the associate code of each rule in the *rules.py* file

You can find the code to train the agents in the *main.py* file

You can simulate several configuration behaviours in the *test.py* file and in the *test2.py* file (in which the most efficient fitness function, called fitness2, has been used)

# POSSIBLE FUTURE IMPLEMENTATIONS:
Instead of a set of prefixed rules where each rule has a priority value and for each turn the actor choices its rule according to these weights, we would create agents composed by a small neural networks, where each neuron can be an ACTION one (representing the effective MOVE) or a SENSOR one (representing an IF condition). ACTION neurons and SENSOR neurons would be linked via weights each others and these weights would be updated though an ES based algorithm, in order to make the program more flexible and able adaptable to different opponents. 
* Look at this project to have an idea about it : https://www.youtube.com/watch?v=N3tRFayqVtk&ab_channel=davidrandallmiller