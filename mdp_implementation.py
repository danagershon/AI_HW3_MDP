from copy import deepcopy
import numpy as np
import random
from termcolor import colored

"""
def policy_evaluation_matrices(mdp, policy):
    # TODO:
    # Given the mdp, and a policy
    # return: the utility U(s) of each state s
    #

    states = []
    R_vector = []

    for row in range(mdp.num_row):
        for col in range(mdp.num_col):
            if mdp.board[row][col] != 'WALL':
                states.append((row, col))
                state_reward = float(mdp.board[row][col])
                R_vector.append(state_reward)

    n = len(states)
    P_matrix = []  # n x n matrix

    for state in states:
        state_P_row = [0] * n
        if state not in mdp.terminal_states:
            neigh_states = [mdp.step(state, action) for action in mdp.actions]
            policy_action = policy[state[0]][state[1]]
            neigh_probs = mdp.transition_function[policy_action]
            for neigh_state, neigh_prob in zip(neigh_states, neigh_probs):
                neigh_state_idx = states.index(neigh_state)
                state_P_row[neigh_state_idx] = neigh_prob
        P_matrix.append(state_P_row)

    R_vector = np.array(R_vector, dtype=np.double)
    P_matrix = np.array(P_matrix, dtype=np.double)

    U_vector = np.linalg.solve(np.subtract(np.identity(n, dtype=np.double), mdp.gamma * P_matrix),  R_vector)
    U = [[None] * mdp.num_col for _ in range(mdp.num_row)]
    for state, state_util in zip(states, U_vector):
        U[state[0]][state[1]] = state_util

    return U
"""


def get_states(mdp):
    for row in range(mdp.num_row):
        for col in range(mdp.num_col):
            if mdp.board[row][col] != 'WALL':
                yield (row, col)


def get_state_reward(mdp, state):
    row, col = state
    if mdp.board[row][col] == 'WALL':
        return None
    return float(mdp.board[row][col])


def get_initialized_utility(mdp):
    u_init = [[0.0] * mdp.num_col for _ in range(mdp.num_row)]
    for terminal_state in mdp.terminal_states:
        row, col = terminal_state
        u_init[row][col] = get_state_reward(mdp, terminal_state)  # terminal state utility is its reward value only

    return u_init


def get_state_util_under_policy(mdp, policy, util, state):
    row, col = state
    neigh_states = [mdp.step(state, action) for action in mdp.actions]
    policy_action = policy[row][col]
    neigh_probs = mdp.transition_function[policy_action]
    state_reward = get_state_reward(mdp, state)

    # R(s) + gamma * sum(P(s'|s,policy(s)) * U(s') for s' in a successors)
    avg_action_util = sum(prob * util[neigh[0]][neigh[1]] for prob, neigh in zip(neigh_probs, neigh_states))
    return state_reward + mdp.gamma * avg_action_util, avg_action_util


def bellman_eq(mdp, util, state):
    state_reward = get_state_reward(mdp, state)
    max_avg_util = float("-inf")
    best_action = None
    next_states = [mdp.step(state, action) for action in mdp.actions]
    avg_utils = []

    if state in mdp.terminal_states:
        return state_reward, None, None, []

    for action in mdp.actions:
        avg_util_from_action = 0
        for next_state, prob in zip(next_states, mdp.transition_function[action]):
            avg_util_from_action += prob * util[next_state[0]][next_state[1]]
        if avg_util_from_action > max_avg_util:
            best_action = action
            max_avg_util = avg_util_from_action
        avg_utils.append(avg_util_from_action)

    epsilon = 10 ** (-3)
    all_best_actions = [action for action, avg_util in zip(mdp.actions, avg_utils) if abs(avg_util - max_avg_util) < epsilon]

    return state_reward + mdp.gamma * max_avg_util, best_action, max_avg_util, all_best_actions


def value_iteration(mdp, U_init, epsilon=10 ** (-3)):
    # TODO:
    # Given the mdp, the initial utility of each state - U_init, and the upper limit - epsilon.
    # run the value iteration algorithm and
    # return: the U obtained at the end of the algorithms' run.

    next_util = deepcopy(U_init)

    while True:
        util = deepcopy(next_util)
        delta = 0
        states = get_states(mdp)
        for state in states:
            row, col = state
            next_util[row][col], _, _, _ = bellman_eq(mdp, util, state)
            delta = max(delta, abs(next_util[row][col] - util[row][col]))
        if delta < epsilon * (1 - mdp.gamma) / mdp.gamma:
            break

    return util


def get_policy(mdp, U):
    # TODO:
    # Given the mdp and the utility of each state - U (which satisfies the Bellman equation)
    # return: the policy

    policy = [[None] * mdp.num_col for _ in range(mdp.num_row)]

    states = get_states(mdp)
    for state in states:
        if state in mdp.terminal_states:
            continue  # policy must give terminal states None as the action
        row, col = state
        _, best_action, _, _ = bellman_eq(mdp, U, state)
        policy[row][col] = best_action

    return policy


def policy_evaluation(mdp, policy):
    # TODO:
    # Given the mdp, and a policy
    # return: the utility U(s) of each state s

    u_init = get_initialized_utility(mdp)

    epsilon = 10 ** (-3)  # error tolerance, like in value iteration
    util = deepcopy(u_init)

    # the utility derived from the policy will be improved iteratively

    while True:
        next_util = deepcopy(u_init)
        delta = 0
        states = get_states(mdp)
        for state in states:
            if state in mdp.terminal_states:
                continue  # utility was already initialized as state reward
            row, col = state
            next_util[row][col], _ = get_state_util_under_policy(mdp, policy, util, state)
            delta = max(delta, abs(next_util[row][col] - util[row][col]))
        util = next_util
        if delta < epsilon * (1 - mdp.gamma) / mdp.gamma:
            break

    return util


def policy_iteration(mdp, policy_init):
    # TODO:
    # Given the mdp, and the initial policy - policy_init
    # run the policy iteration algorithm
    # return: the optimal policy

    policy = policy_init
    changed = True

    while changed:
        util = policy_evaluation(mdp, policy)
        changed = False
        states = get_states(mdp)
        for state in states:
            row, col = state
            if state in mdp.terminal_states:
                policy[row][col] = None
                continue
            _, best_action, max_action_util, _ = bellman_eq(mdp, util, state)
            _, curr_action_util = get_state_util_under_policy(mdp, policy, util, state)
            if max_action_util > curr_action_util:
                policy[row][col] = best_action
                changed = True

    return policy


"""For this functions, you can import what ever you want """


def arrows_formatted_string(actions):
    res = [[u" ", u" ", u"↑", u" ", u" "],
           [u" ", u"←", u" ", u"→", u" "],
           [u" ", u" ", u"↓", u" ", u" "]]

    if 'UP' not in actions:
        res[0][2] = u" "
    if 'LEFT' not in actions:
        res[1][1] = u" "
    if 'RIGHT' not in actions:
        res[1][3] = u" "
    if 'DOWN' not in actions:
        res[2][2] = u" "

    return ["".join(res_row) for res_row in res]


def print_all_policies(mdp, policy_all_actions):
    cell_width = 5
    cell_height = 3
    empty_row = " " * cell_width
    rows_separator = " " + "-" * mdp.num_col * (cell_width+2) + "\n"

    state_to_str = {}

    for r in range(mdp.num_row):
        for c in range(mdp.num_col):
            state_str = [empty_row] * cell_height
            if mdp.board[r][c] == 'WALL' or (r, c) in mdp.terminal_states:
                color = 'blue' if mdp.board[r][c] == 'WALL' else 'red'
                state_str[1] = colored(mdp.board[r][c].center(cell_width, " "), color)
            else:
                state_str = arrows_formatted_string(policy_all_actions[r][c])
            state_to_str[(r, c)] = state_str

    res = rows_separator
    for r in range(mdp.num_row):
        for i in range(cell_height):
            res += "| "
            for c in range(mdp.num_col):
                res += state_to_str[(r, c)][i] + " |"
            res += '\n'
        res += rows_separator
    print(res)


def get_policy_all_actions(mdp, U):
    policy_all_actions = [[None] * mdp.num_col for _ in range(mdp.num_row)]
    num_policies = 1

    states = get_states(mdp)
    for state in states:
        if state in mdp.terminal_states:
            continue  # policy must give terminal states None as the action
        row, col = state
        _, _, _, all_best_actions = bellman_eq(mdp, U, state)
        policy_all_actions[row][col] = all_best_actions
        num_policies *= len(all_best_actions)

    return policy_all_actions, num_policies


def get_all_policies(mdp, U):  # You can add more input parameters as needed
    # TODO:
    # Given the mdp, and the utility value U (which satisfies the Bellman equation)
    # print / display all the policies that maintain this value
    # (a visualization must be performed to display all the policies)
    #
    # return: the number of different policies

    policy_all_actions, num_policies = get_policy_all_actions(mdp, U)

    print_all_policies(mdp, policy_all_actions)

    return num_policies


def did_policy_change(mdp, prev_policy_all_actions, curr_policy_all_actions):
    for state in get_states(mdp):
        if state not in mdp.terminal_states:
            row, col = state
            prev_policy_actions = set(prev_policy_all_actions[row][col])
            curr_policy_actions = set(curr_policy_all_actions[row][col])
            if curr_policy_actions != prev_policy_actions:
                return True

    return False


def get_policy_for_different_rewards(mdp):  # You can add more input parameters as needed
    # TODO:
    # Given the mdp
    # print / displays the optimal policy as a function of r
    # (reward values for any non-finite state)

    from decimal import Decimal

    r_min = Decimal('-5.0')
    r_max = Decimal('5.0')
    reward_jump = Decimal('0.01')
    zero = Decimal('0.00')
    rewards_in_which_policy_changed = []
    mdp_copy = deepcopy(mdp)

    def update_mdp_reward(new_reward):
        for state in get_states(mdp_copy):
            if state not in mdp_copy.terminal_states:
                row, col = state
                mdp_copy.board[row][col] = str(new_reward)

    reward_to_check = r_min
    update_mdp_reward(reward_to_check)
    u_optimal = value_iteration(mdp_copy, get_initialized_utility(mdp_copy))
    prev_policy_all_actions, _ = get_policy_all_actions(mdp, u_optimal)
    reward_to_check += reward_jump

    while reward_to_check < r_max:
        update_mdp_reward(reward_to_check)
        u_optimal = value_iteration(mdp_copy, get_initialized_utility(mdp_copy))
        curr_policy_all_actions, _ = get_policy_all_actions(mdp_copy, u_optimal)
        policy_changed = did_policy_change(mdp_copy, prev_policy_all_actions, curr_policy_all_actions)

        if policy_changed:
            prev_reward = rewards_in_which_policy_changed[-1] if rewards_in_which_policy_changed else None
            if prev_reward:
                print(f"\n{prev_reward} <= R(s) < {reward_to_check}:")
            else:
                print(f"\nR(s) < {reward_to_check}:")
            print_all_policies(mdp, prev_policy_all_actions)

            prev_policy_all_actions = curr_policy_all_actions
            rewards_in_which_policy_changed.append(float(reward_to_check))

        if reward_to_check < zero and (reward_to_check + reward_jump) >= zero:
            reward_to_check = zero
        else:
            reward_to_check += reward_jump

    prev_reward = rewards_in_which_policy_changed[-1] if rewards_in_which_policy_changed else float("-inf")
    print(f"R(s) >= {prev_reward}:")
    print_all_policies(mdp_copy, prev_policy_all_actions)

    return rewards_in_which_policy_changed

