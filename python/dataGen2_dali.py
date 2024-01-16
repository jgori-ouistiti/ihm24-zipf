##### ======= Code by Mohamed Ali Ben Amara and Julien Gori
# contact: gori@isir.upmc.fr
import random
import numpy as np
import pandas as pd
import numpy
import copy


def get_goal_measure(num_commands=12, zipf_param=1):
    """
    Returns the stationary probability of a zipf distribution that we want to eventually achieve by a markov chain
    """
    goal = np.zeros(num_commands)
    for i in range(num_commands):
        goal[i] = 1 / (i + 1) ** zipf_param
    goal = goal / goal.sum()
    return goal


def initialize_matrix(num_commands, sparsity):
    """
    initializes a transition matrix
    """

    P = np.zeros((num_commands, num_commands))
    irreducible = False

    sparsity = int(sparsity * num_commands**2)

    i = 0
    while not irreducible:
        for i in range(num_commands):
            P[i] = np.random.uniform(0.01, 0.99, num_commands)
            P[i] = P[i] / P[i].sum()
        temp = P.copy()

        sparse_ind_per_line = random.sample(range(num_commands), k=num_commands)
        sparse_ind_per_line = [
            (r, c) for r, c in zip(range(num_commands), sparse_ind_per_line)
        ]
        indexes = [(r, c) for r in range(num_commands) for c in range(num_commands)]

        for item in sparse_ind_per_line:
            indexes.remove(item)

        sparse_inds = random.sample(population=indexes, k=sparsity)
        for i, j in sparse_inds:
            P[i, j] = 0
        P = revert_to_one(P)
        irreducible = verify_irreducibility(P)
        i += 1
        if i == 1000:
            print("Could not find a proper matrix. Consider reducing sparsity")
            return temp
    return P


def revert_to_one(matrix):
    """
    returns any matrix as a matrix in which every line sums up to 1
    """
    for i in range(matrix.shape[0]):
        matrix[i] = matrix[i] / matrix[i].sum()
    return matrix


def zipfianize(matrix, column, s):
    matrix = copy.copy(x=matrix)
    _sum = 0
    for ni, i in enumerate(matrix[:, column]):
        _sum += i * (ni + 1) ** (-s)

    matrix[:, column] = matrix[:, column] / _sum * (column + 1) ** (-s)
    matrix = revert_to_one(matrix)
    return matrix


def get_trial_pi(matrix):
    """
    Stationary probability of the matrix
    """
    matrix = numpy.nan_to_num(matrix)  # handle values
    eigs = np.linalg.eig(matrix.T)
    for index, eig_val in enumerate(eigs[0]):
        if np.linalg.norm(eig_val - 1) < 0.001:
            break
    trial_pi = eigs[1][:, index]
    trial_pi = trial_pi.real
    trial_pi = trial_pi / trial_pi.sum()
    return trial_pi


def verify_irreducibility(matrix):
    """
    Verifies if a matrix is irreductible
    """
    dim = matrix.shape[0]
    carrier = matrix.copy()
    for i in range(dim):
        carrier += carrier @ matrix
    if 0 in carrier.reshape(-1):
        return False
    return True


# Zipf with Dependencies
def get_matrix(num_commands=12, zipf_param=1, sparsity=1, verbose=False, max_iter=None):
    """
    gets a random transition matrix for which the stationary distribution follows a zipf
    """

    goal = get_goal_measure(num_commands, zipf_param)
    P = initialize_matrix(num_commands, sparsity)
    conv = False
    if max_iter is None:
        max_iter = num_commands * 1000
    eps = 0.02
    i = 0
    exp = 0.5
    while not (conv):
        trial = get_trial_pi(P)
        if verbose:
            print(f"Current error is {np.linalg.norm(trial-goal)}")
        # to_change = np.argmax(np.abs(trial-goal))
        to_change = np.random.randint(0, num_commands)

        choice = np.random.randint(0, num_commands)
        P[choice, to_change] = P[choice, to_change] * 1.5 ** ((goal - trial)[to_change])

        P = revert_to_one(P)
        i += 1
        conv = np.linalg.norm(trial - goal) < eps
        if i > max_iter:
            if verbose:
                print("max iterations reached")
            return P
    if verbose:
        print(f"Finished in {i} iterations")
    return P


# Zipf with Dependencies
def get_matrix_bis(
    num_commands=12,
    zipf_param=1,
    verbose=False,
    max_iter=None,
    init_matrix=None,
    sparsity=None,
    error_stop=0.01,
    seed=None,
):
    """
    gets a random transition matrix for which the stationary distribution follows a zipf
    """

    rng = numpy.random.default_rng(seed=seed)

    cost_out = numpy.inf

    def cost(trial, goal):
        # return np.linalg.norm((trial-goal).reshape(1,-1)@trial.reshape(-1,1))
        return np.linalg.norm((trial - goal)) / num_commands

    if init_matrix is None:
        if sparsity is None:
            sparsity = 0.5
        P = initialize_matrix(num_commands, sparsity)
    else:
        P = init_matrix

    goal = get_goal_measure(num_commands, zipf_param)

    trial = get_trial_pi(P)
    for i in range(num_commands):
        P = zipfianize(P, i, zipf_param)
    trial = get_trial_pi(P)
    conv = False

    if max_iter is None:
        max_iter = num_commands * 1000
    eps = error_stop
    add_deps_factor = 100
    i = 0
    while not (conv):
        trial = get_trial_pi(P)
        _cost = cost(trial, goal)
        if _cost < cost_out:
            Pout = P
        if verbose:
            print(f"Current error is {_cost}")
        # to_change = np.argmax(np.abs(trial-goal))
        to_change = np.random.randint(0, num_commands)

        choice = np.random.randint(0, num_commands)
        if P[choice, to_change] == 0:
            continue
        P[choice, to_change] = P[choice, to_change] * 1.5 ** ((goal - trial)[to_change])
        P = numpy.nan_to_num(P)  # deal with nans and infs

        # test =====
        to_change = np.random.randint(0, num_commands)
        choice = np.random.randint(0, num_commands)
        if not i % int(max(add_deps_factor / num_commands, 1)):
            P[choice, to_change] = 1
        # ==== end test

        # P = revert_to_one(P)
        indx = numpy.argsort(trial)[::-1]
        P = zipfianize(P, to_change, zipf_param)
        P = P[indx, :]

        i += 1
        conv = cost(trial, goal) < eps
        if i > max_iter:
            if verbose:
                print("max iterations reached")
            Pout = revert_to_one(Pout)
            return Pout, get_trial_pi(Pout), cost(get_trial_pi(Pout), goal)
    if verbose:
        print(f"Finished in {i} iterations")
    return P, get_trial_pi(P), cost(trial, goal)


def transition_matrix(command_sequence):
    """
    gets a transition matrix from a sequence
    """
    dim = len(set(command_sequence))
    Matrix = np.zeros((dim, dim))
    counts = {i: 0 for i in range(dim)}

    for e, actual in enumerate(command_sequence[1:]):
        prev = command_sequence[e]
        Matrix[prev, actual] += 1
        counts[prev] += 1
    for i in range(dim):
        Matrix[i] = Matrix[i] / counts[i]
    return Matrix


def generate_from_matrix(matrix, size):
    dim = matrix.shape[0]
    commands = [i for i in range(dim)]
    prev = np.random.choice(commands)
    sequence = [prev]
    for i in range(size - 1):
        actual = np.random.choice(commands, p=matrix[prev])
        prev = actual
        sequence.append(prev)
    return sequence


def zipfDep(size, num_commands=12, zipf_param=1, seed=None, sparsity=5):
    if seed is not None:
        np.random.seed(seed)
    matrix = get_matrix(num_commands, zipf_param, sparsity)
    sequence = generate_from_matrix(matrix, size)
    return sequence
