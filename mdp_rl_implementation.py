from copy import deepcopy
import random
import numpy as np

actions = ['UP', 'DOWN', 'RIGHT', 'LEFT']


def get_cur_states(mdp):
    cur_states = []
    # state = Tuple(Tuple(int, int), str)
    row_number = range(len(mdp.board))
    col_number = range(len(mdp.board[0]))
    for i in row_number:
        for j in col_number:
            if mdp.board[i][j] != "WALL":   #there's actual value
                cur_states.append((i, j, float(mdp.board[i][j])))
    return cur_states

def get_state_expect(mdp, U, row, col):
    values = []
    max_action = None
    for a in actions: #check for each legal action its E value
        s = []
        for index in range(4):
            p = mdp.transition_function[a][index]
            if p == 0:
                continue
            next_state = mdp.step((row, col), actions[index])
            s.append(p * U[next_state[0]][next_state[1]])   # prob * state.val
        values.append([sum(s), a])
        values_list = [i[0] for i in values]
        max_value = max(values_list)
        max_index = values_list.index(max_value)
        max_action = values[max_index][1]

    return (max_value,max_action)               # returns max E value

def value_iteration(mdp, U_init, epsilon=10 ** (-3)):
    #
    # Given the mdp, the initial utility of each state - U_init,
    #   and the upper limit - epsilon.
    # run the value iteration algorithm and
    # return: the U obtained at the end of the algorithms' run.
    #
    new_U = deepcopy(U_init)
    delta = 1
    gamma = mdp.gamma
    counter = 0
    while delta >= epsilon * ((1 - gamma) / gamma) and not (delta == 0 and gamma == 1):
        counter += 1
        for state in mdp.terminal_states:   #terminal state have the same values every iter
            row_idx, col_idx = state[0], state[1]
            new_U[row_idx][col_idx] = float(mdp.board[row_idx][col_idx])
        old_U = deepcopy(new_U)
        delta = 0
        current_states = get_cur_states(mdp)
        for row, col, v in current_states:
            if (row, col) not in mdp.terminal_states: #if not a final state than update it's value
                new_U[row][col] = float(v + gamma * get_state_expect(mdp, old_U, row, col)[0])
                delta = max(delta, abs(new_U[row][col] - old_U[row][col]))
    return old_U


def get_policy(mdp, U):
    #
    # Given the mdp and the utility of each state - U (which satisfies the Belman equation)
    # return: the policy
    #
    policy = deepcopy(U)
    for row in range(len(U)):
        for col in range(len(U[0])):
            val, action = get_state_expect(mdp, U, row, col)
            policy[row][col] = action
    return policy


def out_of_boundries(mdp, s):
    return s[0] < 0 or s[1] < 0 or s[0] >= mdp.num_row or s[1] >= mdp.num_col or mdp.board[s[0]][s[1]] == 'WALL'

def next_step(mdp, s, action):
    real_action = random.choices(population=actions, weights=mdp.transition_function[action], k=1)[0]

    a = mdp.actions[real_action]
    next_state =  (s[0]+a[0], s[1]+a[1])
    # next_state = tuple(map(sum, zip(s, mdp.actions[real_action])))
    if out_of_boundries(mdp, next_state):
        next_state = s
    return next_state


def final_state_val(mdp, learning_rate, q_table, s, next_state, index):
    return learning_rate * (float(mdp.board[next_state[0]][next_state[1]]) - q_table[mdp.num_col * s[0] + s[1]][index] \
                            + mdp.gamma * float(mdp.board[s[0]][s[1]]))


def state_val(mdp, learning_rate, q_table, s, next_state, index):
    max_val = float('-inf')
    for a in range(4):
        max_val = max(q_table[mdp.num_col * next_state[0] + next_state[1]][a], max_val)
    return learning_rate * (float(mdp.board[next_state[0]][next_state[1]]) - q_table[mdp.num_col * s[0] + s[1]][index] \
                            + mdp.gamma * max_val)


def q_learning(mdp, init_state, total_episodes=10000, max_steps=999, learning_rate=0.7, epsilon=1.0,
                      max_epsilon=1.0, min_epsilon=0.01, decay_rate=0.8):
    #
    # Given the mdp and the Qlearning parameters:
    # total_episodes - number of episodes to run for the learning algorithm
    # max_steps - for each episode, the limit for the number of steps
    # learning_rate - the "learning rate" (alpha) for updating the table values
    # epsilon - the starting value of epsilon for the exploration-exploitation choosing of an action
    # max_epsilon - max value of the epsilon parameter
    # min_epsilon - min value of the epsilon parameter
    # decay_rate - exponential decay rate for exploration prob
    # init_state - the initial state to start each episode from
    # return: the Qtable learned by the algorithm
    #

    # ====== YOUR CODE: ======
    q_table = np.zeros((mdp.num_col * mdp.num_row, len(mdp.actions)))
    for episode in range(total_episodes):
        s = init_state
        for step in range(max_steps):
            rand = np.random.uniform()
            if rand > epsilon:
                action_index = np.argmax(q_table[mdp.num_col * s[0] + s[1]], axis=-1)
            else:
                action_index = np.random.choice(range(4))
            action = actions[action_index]
            next_s = next_step(mdp, s, action)

            if next_s in mdp.terminal_states:
                q_table[mdp.num_col * s[0] + s[1]][action_index] +=\
                    final_state_val(mdp, learning_rate, q_table, s, next_s, action_index)
                break
            q_table[mdp.num_col * s[0] + s[1]][action_index] += \
                state_val(mdp, learning_rate, q_table, s, next_s, action_index)
            s = next_s
        epsilon = min_epsilon + (max_epsilon - min_epsilon) * np.exp(-decay_rate * episode)
    return q_table
    # ========================


def q_table_policy_extraction(mdp, qtable):
    #
    # Given the mdp and the Qtable:
    # return: the policy corresponding to the Qtable
    #

    # ====== YOUR CODE: ======
    phi = deepcopy(mdp.board)
    for i in range(len(mdp.board)):
        for j in range(len(mdp.board[0])):
            action_index = np.argmax(qtable[mdp.num_col * i + j], -1)
            phi[i][j] = actions[action_index]
    return phi
    # ========================


# BONUS

def policy_evaluation(mdp, policy):
    # TODO:
    # Given the mdp, and a policy
    # return: the utility U(s) of each state s
    #

    # ====== YOUR CODE: ======
    raise NotImplementedError
    # ========================


def policy_iteration(mdp, policy_init):
    # TODO:
    # Given the mdp, and the initial policy - policy_init
    # run the policy iteration algorithm
    # return: the optimal policy
    #

    # ====== YOUR CODE: ======
    raise NotImplementedError
    # ========================
