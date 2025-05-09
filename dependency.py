import os
import itertools
import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from mpl_toolkits.mplot3d import Axes3D
from math import ceil
from numba import njit
from timeit import default_timer as timer
from datetime import datetime
from itertools import combinations
from scipy.stats import mannwhitneyu
from math import erf
from pymoo.indicators.hv import HV
from scipy.stats import ttest_ind
from math import comb


@njit
def random_choice(x, size=1, replace=True, p=None):
    n = len(x)
    selected = np.zeros(size, dtype=np.int64)
    possible_indices = np.arange(n)

    if p is None:
        p = np.ones(n, dtype=np.float64) / n

    for i in range(size):
        cdf = np.cumsum(p)
        r = np.random.rand()
        for index, threshold in enumerate(cdf):
            if r <= threshold:
                selected[i] = x[possible_indices[index]]
                break

        if replace == False and size > 1:
            possible_indices = np.delete(possible_indices, index)
            p = np.array([a for k, a in enumerate(p) if k != index])
            p = p / p.sum()

    return selected


@njit
def is_dominate(x, y):
    """
    Check if x dominates y (minimization objectives).

    Parameters:
        x (np.ndarray): First point.
        y (np.ndarray): Second point.

    Returns:
        bool: True if x dominates y, False otherwise.
    """
    num_objectives = len(x)
    strictly_better = False  # Tracks if x is strictly better in at least one objective

    for i in range(num_objectives):
        if x[i] > y[i]:
            return False  # x does not dominate y
        if x[i] < y[i]:
            strictly_better = True  # x is strictly better in at least one objective

    return strictly_better
    
@njit
def fast_non_dominated_sort(objective_values):
    """
    Perform fast non-dominated sorting on the given objective values.

    Parameters:
        objective_values (np.ndarray): 2D array of shape (N, M), where N is the number of solutions
                                       and M is the number of objectives.

    Returns:
        fronts (list of np.ndarray): List of fronts, where each front is an array of solution indices.
        ranks (np.ndarray): Array of ranks for each solution.
    """
    N = len(objective_values)
    S = np.zeros((N, N), dtype=np.int64)  # Dominance matrix
    ranks = np.zeros(N, dtype=np.int64)   # Rank of each solution
    fronts = []                           # List of fronts
    n = np.zeros(N, dtype=np.int64)       # Number of solutions dominating each solution

    # Step 1: Compute dominance matrix and count dominators
    for p in range(N):
        for q in range(N):
            if is_dominate(objective_values[p], objective_values[q]):
                S[p, q] = 1
                n[q] += 1

    # Step 2: Identify the first front (solutions with no dominators)
    front = np.where(n == 0)[0]
    fronts.append(front)
    ranks[front] = 0

    # Step 3: Build subsequent fronts
    i = 0
    while len(fronts[i]) > 0:
        next_front = []
        for p in fronts[i]:
            for q in range(N):
                if S[p, q] == 1:
                    n[q] -= 1
                    if n[q] == 0:
                        next_front.append(q)
                        ranks[q] = i + 1
        fronts.append(np.array(next_front))
        i += 1

    # Remove the last empty front
    if len(fronts[-1]) == 0:
        fronts.pop()

    return fronts, ranks


@njit
def find_pareto_front_indexes(objective_values):
    N = len(objective_values)
    n = np.zeros(N, dtype=np.int64)
    for p in range(N):
        for q in range(N):
            if is_dominate(objective_values[p], objective_values[q]):
                n[q] += 1
    front = np.where(n == 0)[0]
    return front

@njit
def find_pareto_front(population, ranks, objective_values):
    pf_indexes = np.where(ranks == 0)[0]
    pareto_front = population[pf_indexes]
    pov = objective_values[pf_indexes]
    return pareto_front, pov

@njit
def calculate_crowding_distance(fronts, objective_values):
    pop_size, num_objectives = objective_values.shape
    crowding_distances = np.zeros(pop_size, dtype=np.float64)

    for front in fronts:
        lf = len(front)
        if lf == 0:
            continue

        # Initialize distances for the current front
        distances = np.zeros(lf, dtype=np.float64)
        distances[0] = 123456789  # Boundary solutions have infinite distance
        distances[-1] = 123456789

        for k in range(num_objectives):
            # Sort the front based on the current objective
            obj_values = objective_values[front, k]
            sorted_indices = np.argsort(obj_values)
            sorted_front = front[sorted_indices]

            # Normalize the objective values
            min_obj = obj_values[sorted_indices[0]]
            max_obj = obj_values[sorted_indices[-1]]
            if max_obj == min_obj:
                continue  # Avoid division by zero

            # Calculate crowding distance for intermediate solutions
            for j in range(1, lf - 1):
                distances[j] += (
                    objective_values[sorted_front[j + 1], k] - objective_values[sorted_front[j - 1], k]
                ) / (max_obj - min_obj)

        # Assign crowding distances to the solutions in the front
        for idx in range(lf):
            crowding_distances[sorted_front[idx]] = distances[idx]

    return crowding_distances


@njit
def locate_point_in_the_grid(x, mins, maxs, num_points_per_dim):
    if not check_if_point_is_covered(x, mins, maxs):
        raise ValueError("Point is outside the objective space defined by mins and maxs.")
    range_vals = maxs - mins
    range_vals[range_vals == 0] = 1  # Avoid division by zero
    
    grid_indices = ((x - mins) / (range_vals) * (num_points_per_dim - 1)).astype(np.int_)
    return grid_indices

@njit
def unravel_random(arr, element):
    locations = np.where(arr == element)
    locations = np.concatenate(locations).reshape(len(locations),-1).T
    i = np.random.randint(len(locations))
    idx = locations[i]
    return idx

@njit
def sample_index_weighted(arr, alpha=1.0):
    flat_arr = arr.ravel()  # Efficient flattening
    if np.all(flat_arr == 0):
        raise ValueError("All elements are zero; cannot sample proportionally.")

    weights = np.power(flat_arr, alpha)
    probabilities = weights / np.sum(weights)

    element = random_choice(flat_arr, size = 1, p = probabilities)
    return unravel_random(arr, element)

@njit
def sample_index_inverse_weighted(arr, beta=1.0):
    flat_arr = arr.ravel()
    # Select only nonzero elements
    nonzero_indices = np.nonzero(flat_arr)[0]
    if nonzero_indices.size == 0:
        raise ValueError("All nonzero elements are zero; cannot sample inversely.")
    
    nonzero_values = flat_arr[nonzero_indices]
    max_val = nonzero_values.max()

    weights = np.power(max_val - nonzero_values + 1, beta)
    probabilities = weights / np.sum(weights)

    element = random_choice(nonzero_values, size = 1, p = probabilities)
    
    return unravel_random(arr, element)

@njit
def check_if_point_is_covered(x, mins, maxs):
    if np.any(x < mins) or np.any(x > maxs):
        return False
    else:
        return True

@njit
def compute_grid_count(archive_obj, mins, maxs, num_points_per_dim):
    grid_count = np.zeros((num_points_per_dim-1,num_points_per_dim-1, num_points_per_dim-1)) ############################ change this * num_objectives #####################################
    for x in archive_obj:
        index = locate_point_in_the_grid(x, mins, maxs, num_points_per_dim)
        grid_count[index[0], index[1], index[2]] += 1  ############################ change this * num_objectives #####################################
    return grid_count
    

@njit
def find_mins_maxs(normalized_objectives):
    num_columns = normalized_objectives.shape[1]
    mins = np.full(num_columns, np.inf)
    maxs = np.full(num_columns, -np.inf)
    
    for i in range(normalized_objectives.shape[0]):  # Iterate over rows
        for j in range(num_columns):  # Iterate over columns
            if normalized_objectives[i, j] < mins[j]:
                mins[j] = normalized_objectives[i, j]
            if normalized_objectives[i, j] > maxs[j]:
                maxs[j] = normalized_objectives[i, j]

    # Adjust mins and maxs by 1e-06
    mins -= 1e-06
    maxs += 1e-06

    return mins, maxs
    
@njit
def update_archive(archive, pareto_solutions, pareto_objectives, pf_indexes, archive_maxsize, num_points_per_dim, alpha):
    
    archive_solutions, archive_objectives, archive_size = archive
    
    if len(pf_indexes) <= archive_maxsize:
        archive_size = len(pf_indexes)
        archive_solutions[:archive_size] = pareto_solutions
        archive_objectives[:archive_size] = pareto_objectives
    else:
        normalized_objectives = normalize_objectives(pareto_objectives)
        mins, maxs = find_mins_maxs(normalized_objectives)
        grid_count = compute_grid_count(normalized_objectives, mins, maxs, num_points_per_dim)
        to_take = np.ones(len(pf_indexes))
        K = len(pf_indexes) - archive_maxsize
        
        while K > 0:
            index = sample_index_weighted(grid_count, alpha)[:len(mins)]
            elements = [i for i, x in enumerate(normalized_objectives)
                        if to_take[i] and np.all(locate_point_in_the_grid(x, mins, maxs, num_points_per_dim)[:len(mins)] == index)]
            
            if elements:
                i = np.random.choice(np.array(elements))
                index = locate_point_in_the_grid(normalized_objectives[i], mins, maxs, num_points_per_dim)[:len(mins)]
                grid_count[index[0], index[1], index[2]] -= 1  ############################ change this * num_objectives #####################################
                to_take[i] = 0
                K -= 1
        
        archive_size = archive_maxsize
        archive_solutions = pareto_solutions[to_take==1]
        archive_objectives = pareto_objectives[to_take==1]
    
    return archive_solutions, archive_objectives, archive_size


@njit
def Initialize_archive(archive, population, num_objectives, archive_maxsize, num_points_per_dim, alpha, other_args):
    archive_solutions, archive_objectives, archive_size = archive
    
    objective_values = evaluate_objectives(population, other_args)

    pf_indexes = find_pareto_front_indexes(objective_values)
    pareto_solutions = population[pf_indexes]
    pareto_objectives = objective_values[pf_indexes]

    archive = update_archive(archive, pareto_solutions, pareto_objectives, pf_indexes, archive_maxsize, num_points_per_dim, alpha)

    return archive


@njit
def Insert_new_solutions(archive, new_solutions, num_objectives, archive_maxsize, num_points_per_dim, alpha, other_args):
    archive_solutions, archive_objectives, archive_size = archive
    new_objective_values = evaluate_objectives(new_solutions, other_args)
    combined_solutions = np.concatenate((archive_solutions[:archive_size], new_solutions))
    combined_objective_values = np.concatenate((archive_objectives[:archive_size], new_objective_values))
    pf_indexes = find_pareto_front_indexes(combined_objective_values)

    pareto_solutions = combined_solutions[pf_indexes]
    pareto_objectives = combined_objective_values[pf_indexes]

    archive = update_archive(archive, pareto_solutions, pareto_objectives, pf_indexes, archive_maxsize, num_points_per_dim, alpha)

    return archive


@njit
def Insert_new_solutions_mopso(archive, new_solutions, new_objective_values, num_objectives, archive_maxsize, num_points_per_dim, alpha, other_args):
    archive_solutions, archive_objectives, archive_size = archive
    combined_solutions = np.concatenate((archive_solutions[:archive_size], new_solutions))
    combined_objective_values = np.concatenate((archive_objectives[:archive_size], new_objective_values))
    pf_indexes = find_pareto_front_indexes(combined_objective_values)

    pareto_solutions = combined_solutions[pf_indexes]
    pareto_objectives = combined_objective_values[pf_indexes]

    archive = update_archive(archive, pareto_solutions, pareto_objectives, pf_indexes, archive_maxsize, num_points_per_dim, alpha)

    return archive


@njit
def normalize_objectives(objective_values):
    population_size, num_objectives = objective_values.shape
    mins = np.zeros(num_objectives)
    maxs = np.zeros(num_objectives)
    
    # Compute min and max for each objective
    for i in range(num_objectives):
        mins[i] = np.min(objective_values[:, i])
        maxs[i] = np.max(objective_values[:, i])
    
    # Compute range and handle division by zero
    range_vals = maxs - mins
    range_vals[range_vals == 0] = 1  # Avoid division by zero

    return (objective_values - mins) / range_vals



@njit
def generate_intervals(a, b, min_pl, min_npl, k, L):
    # Check feasibility conditions
    if L < k * min_pl or (b - a) < L + (k - 1) * min_npl:
        raise ValueError("Infeasible")
    
    L_extra = L - k * min_pl
    G_extra = (b - a) - L - (k - 1) * min_npl
    
    # Generate interval length deltas
    if L_extra == 0:
        interval_deltas = np.zeros(k)
    else:
        interval_deltas = np.random.dirichlet(np.ones(k)) * L_extra
    
    # Generate gap length deltas
    if G_extra == 0:
        gap_deltas = np.zeros(k+1)
    else:
        gap_deltas = np.random.dirichlet(np.ones(k+1)) * G_extra

    intervals = np.zeros((k, 2))
    current = a + gap_deltas[0]
    for i in range(k):
        start = current
        end = start + min_pl + interval_deltas[i]
        intervals[i] = start, end
        current = end + min_npl + gap_deltas[i + 1]
     
    return intervals

@njit
def find_availability(j, schedule, Instance, min_distance_between_vehicles):
    requests, chargers, A, kappa, std_deviations = Instance
    n,m = A.shape
    
    sub_schedule = schedule[schedule[:,0]==j+1].copy()
    sub_schedule = sub_schedule[np.argsort(sub_schedule[:,1])]
    N = len(sub_schedule)
    availability = np.zeros((N+1,2))
    if N>0:
        availability[0] = np.array([chargers[j,0],sub_schedule[0,1]-min_distance_between_vehicles])
        for k in range(1,N):
            availability[k] = np.array([sub_schedule[k-1,2]+min_distance_between_vehicles,sub_schedule[k,1]-min_distance_between_vehicles])
        availability[N] = np.array([sub_schedule[N-1,2]+min_distance_between_vehicles,123456789])
    else:
        availability[0] = np.array([chargers[j,0],123456789])
    return availability

    
@njit
def find_candidate_charging_periods(i, j, availability, mean_duration_to_allocate, min_npl, Instance):
    requests, chargers, A, kappa, std_deviations = Instance
    n,m = A.shape
    candidate_charging_periods = []
    N = len(availability)
    for k in range(N):
        av_s, av_e = max(availability[k,0],requests[i,0]), availability[k,1]
        if av_e == 123456789:
            candidate_charging_periods.append(k)
        elif av_e-av_s>=mean_duration_to_allocate:
            candidate_charging_periods.append(k)
    return np.array(candidate_charging_periods)

@njit
def make_vehicle_charging_sessions(ei, wj, a, b, kappa, min_pl, min_npl, max_preemptions):
    L = ei/(wj*kappa)
    if b-a<L:
        raise ValueError("Not feasible b-a<L")
    else:
        if L > min_pl:
            ub = min( int(L/min_pl), int((b-a-L)/min_npl+1), max_preemptions )
            k = np.random.randint(1, ub+1)
            intervals = generate_intervals(a, b, min_pl, min_npl, k, L)
            ts = intervals[0,0]
            te = intervals[-1,1]
        else:
            ts = np.random.uniform(a, b-L)
            te = np.random.uniform(ts+L, b)
            intervals = np.array([[ts, te]])
    return ts, te, intervals

@njit
def place_vehicle_on_charger(i, j, schedule, sigma, Instance, min_percentage, min_distance_between_vehicles, min_pl, min_npl, max_preemptions):
    requests, chargers, A, kappa, std_deviations = Instance
    
    schedule = schedule.copy()

    e = ((min_percentage/100)+(1-(min_percentage/100))*np.random.rand())*requests[i, 2]

    mean_duration_to_allocate = e/(chargers[j,1]*kappa)

    availability = find_availability(j, schedule, Instance, min_distance_between_vehicles)

    candidate_charging_periods = find_candidate_charging_periods(i, j, availability, mean_duration_to_allocate, min_npl, Instance)

    k = random_choice(candidate_charging_periods)[0]

    a,b = max(availability[k,0],requests[i,0]), availability[k,1]

    if b == 123456789:
        b = a + mean_duration_to_allocate + np.abs(np.random.normal(0,sigma))

    ts, te, intervals = make_vehicle_charging_sessions(e, chargers[j, 1], a, b, kappa, min_pl, min_npl, max_preemptions)
    
    schedule[i,:4] = [j+1, ts, te, e]
    schedule[i, 4:]*=0
    schedule[i, 4:4+2*len(intervals)] = intervals.flatten()
    
    return schedule


@njit
def generate_random_schedule(Instance, sigma, min_percentage, min_distance_between_vehicles, min_pl, min_npl, max_preemptions):
    requests, chargers, A, kappa, std_deviations = Instance
    n,m = A.shape

    schedule = np.zeros((n, 4+2*max_preemptions))
    
    vehicles_indexes = np.arange(n)
    
    np.random.shuffle(vehicles_indexes)
    
    for i in vehicles_indexes:
        
        candidate_chargers = np.arange(m)[A[i,:] == 1]
        
        j = np.random.choice(candidate_chargers)
        
        schedule = place_vehicle_on_charger(i, j, schedule, sigma, Instance, min_percentage, min_distance_between_vehicles, min_pl, min_npl, max_preemptions)
        
    return schedule




    
def plot_schedule(Instance, schedule, figsize=(15, 7)):
    requests, chargers, A, kappa, std_deviations = Instance

    
    n,m = A.shape
    T = max(schedule[:, 2])
    start_index = min(chargers[:,0])
    
    fig = plt.figure(figsize=figsize)
    for j in range(m):
        plt.plot([chargers[j, 0], T + 1], [j+1, j+1], lw=1, c="black")

    text = ""
    for i in range(n):
        text += f" $V_{{{i+1}}}$: ({round(requests[i, 0], 2)}, {round(requests[i, 1], 2)}, {round(requests[i, 2], 2)}kWh), "
        if (i + 1) % 8 == 0:
            text += "\n"
        mid_point = (schedule[i, 1] + schedule[i, 2]) / 2

        plt.text(mid_point, schedule[i, 0] + 0.2, f"$V_{{{i+1}}}$", fontsize=8, ha='center')

        
        for x in schedule[i, 4:].reshape(-1,2):
            if x[0]==0:
                break
            plt.bar(x[0], 0.25, x[1] - x[0], schedule[i, 0]-0.25/2, align = "edge", color = "red")
        
        plt.bar(schedule[i, 1], 0.25, schedule[i, 2] - schedule[i, 1], 
        schedule[i, 0] - 0.25 / 2, align="edge", 
        edgecolor="black", facecolor="none", linewidth=2)
            
    # xticks = [min_to_time(t * kappa * 60) + "\n" + min_to_time((t + 1) * kappa * 60) for t in range(start_index, T + 2)]
    xticks = []
    yticks = [f"Charger {j+1} \n ({chargers[j,1]} kW)" for j in range(m)]
    
    # plt.xticks(np.arange(start_index, T + 2), xticks, rotation = 45)
    plt.xticks()
    plt.yticks(np.arange(1, m + 1), yticks)
    
    plt.xlim(start_index-0.5, T + 1.5)
    plt.ylim(0, m + 1)
    plt.grid()
    
    fig.text(0.5, -0.1, text, ha='center', va='center', fontsize=7)  # Adjust position as needed

    plt.show()
    # Save the figure
    # plt.close(fig)
    # return fig



@njit
def find_next_vehicle(i, schedule):
    j, _, te = schedule[i, :3]
    mask = (schedule[:, 0] == j) & (schedule[:, 1] > te)
    candidates = np.where(mask)[0]
    return candidates[np.argmin(schedule[candidates, 1])] if candidates.size > 0 else -1


@njit
def phi(x):
    """Standard normal PDF."""
    return (1 / np.sqrt(2 * np.pi)) * np.exp(-0.5 * x**2)

@njit
def cdf(x):
    """Standard normal CDF."""
    return 0.5 * (1 + erf(x / np.sqrt(2)))

@njit
def conditional_mean(mu, std, a, b, gamma):
    """Compute the expectation E[max(0, X - gamma) | a <= X <= b]."""
    # Standardize the bounds
    Ta = (a - mu) / std
    Tb = (b - mu) / std
    Tmax = (max(gamma,a) - mu) / std

    if Tb>Tmax:
        # Expectation term
        expectation = (mu-gamma)*(cdf(Tb)-cdf(Tmax)) + std*(phi(Tmax)-phi(Tb))
    
        # Probability P(a <= X <= b)
        prob = cdf(Tb) - cdf(Ta)
    
        return expectation / prob
    else:
        return 0
    
@njit
def actualise_schedule(schedule, std_deviations, min_distance_between_vehicles, max_preemptions, min_pl, min_npl):
    n = schedule.shape[0]
    actualised_schedule = schedule.copy()

    for i in range(n):
        next_vehicle = find_next_vehicle(i, schedule)
        j = int(schedule[i, 0]) - 1

        # Identify last nonzero index in the row
        for last_index in range(len(schedule[i])-1,-1,-1):
            if schedule[i,last_index]!=0:
                break
                
        flattened_intervals = schedule[i, 4:last_index + 1]
        starts, ends = flattened_intervals[::2], flattened_intervals[1::2]
        sessions_lengths = ends - starts
        total_duration = np.sum(sessions_lengths)
        portions = sessions_lengths / total_duration
        stds = np.sqrt(portions) * (0.1*total_duration)
        #std_deviations[i, j]#############################################################################################################

        new_intervals = np.zeros_like(flattened_intervals)

        for v in range(len(starts)):
            if sessions_lengths[v] > min_pl:
                a = starts[v] + min_pl
            else:
                a = starts[v]

            if v<len(starts)-1:
                b = starts[v + 1] - min_npl
            else:
                b = 123456789 if next_vehicle == -1 else schedule[next_vehicle, 1]-min_distance_between_vehicles
            
            # k = 0
            while True:
                # k+=1
                # if k==10000:
                #     print(a,b,starts[v] + sessions_lengths[v], stds[v], v, schedule[i])
                #     raise ValueError()
                r = np.random.normal(starts[v] + sessions_lengths[v], stds[v])
                if a<r<b:
                    break

            new_intervals[2 * v], new_intervals[2 * v + 1] = starts[v], r

        actualised_schedule[i, 1] = new_intervals[0]
        actualised_schedule[i, 2] = new_intervals[-1]
        actualised_schedule[i, 4:last_index + 1] = new_intervals

    return actualised_schedule


@njit
def grid_capacity(schedule, chargers):
    events = []
    s = len(schedule[0,4:])
    for x in schedule:
        power = chargers[int(x[0]) - 1, 1]
        for i in range(4, 4+s-1,2):
            if x[i] == 0:
                break
            else:
                events.append([x[i], power])
                events.append([x[i+1], -power])
    events = np.array(events)
    events = events[events[:, 0].argsort()]
    return np.max(np.cumsum(events[:, 1]))


@njit
def total_tardiness(schedule, requests, min_pl, min_distance_between_vehicles, std_deviations):
    TT = 0
    n = schedule.shape[0]

    for i in range(n):
        # Identify last nonzero index in the row
        for last_index in range(len(schedule[i])-1,-1,-1):
            if schedule[i,last_index]!=0:
                break

        flattened_intervals = schedule[i, 4:last_index + 1]
        starts, ends = flattened_intervals[::2], flattened_intervals[1::2]
        sessions_lengths = ends - starts
        total_duration = np.sum(sessions_lengths)

        if total_duration == 0:
            continue  # Skip if no valid session exists

        portions = sessions_lengths / total_duration
        last_session_length = sessions_lengths[-1]
        ts = starts[-1]
        ui = last_session_length >= min_pl
        a = ts + min_pl * ui

        next_vehicle = find_next_vehicle(i, schedule)
        b = 100000 if next_vehicle == -1 else schedule[next_vehicle, 1] - min_distance_between_vehicles

        gamma = requests[i, 1]
        j = int(schedule[i, 0]) - 1

        mu = ts + last_session_length
        std = np.sqrt(portions[-1]) * (0.1*total_duration)
        #std_deviations[i, j]#############################################################################################################

        TT += conditional_mean(mu, std, a, b, gamma)

    return TT


@njit
def objectives_calculation_MC(schedule, Instance, min_distance_between_vehicles, max_preemptions, min_pl, min_npl, num_replications):
    requests, chargers, A, kappa, std_deviations = Instance

    if num_replications == 0:
        GC = grid_capacity(schedule, chargers)
        TT = np.sum(np.maximum(schedule[:,2]-requests[:,1],0))
        P = np.sum(np.maximum((requests[:,2]-schedule[:,3])/requests[:,2],0))/len(schedule) 
    else:
        u = np.zeros(num_replications)
        for k in range(num_replications):
            actualised_schedule = actualise_schedule(schedule, std_deviations, min_distance_between_vehicles, max_preemptions, min_pl, min_npl)
            u[k] = grid_capacity(actualised_schedule, chargers)
    
        GC = np.mean(u)
        TT = total_tardiness(schedule, requests, min_pl, min_distance_between_vehicles, std_deviations)
        P = np.sum(np.maximum((requests[:,2]-schedule[:,3])/requests[:,2],0))/len(schedule)
    
    return GC,TT,100*P


@njit
def generalized_crossover(solutions, other_args):
    Instance, sigma, min_percentage, min_distance_between_vehicles, min_pl, min_npl, max_preemptions = other_args
    requests, chargers, A, kappa, std_deviations = Instance
    
    n,m = A.shape
    new_solution = np.zeros(solutions[0].shape, dtype = np.float64)
    placable = np.zeros(len(solutions), dtype = np.int8)
    vehicles_indexes = np.arange(n)
    np.random.shuffle(vehicles_indexes)
    
    first_one = True
    placed = []
    not_placed = []
    for i in vehicles_indexes:
        if first_one:
            s = np.random.randint(len(solutions))
            new_solution[i] = solutions[s][i].copy()
            placed.append(i)
            first_one = False
        else:
            placable = np.ones(len(solutions), dtype = np.int8)
            for s, solution in enumerate(solutions):
                for j in placed:
                    cond1 = solution[i,1]>=new_solution[j,2]+min_distance_between_vehicles
                    cond2 = new_solution[j,1]>=solution[i,2]+min_distance_between_vehicles
                    cond = cond1 or cond2
                    if not cond:
                        placable[s] = 0
                        break     
            if np.sum(placable)>0:
                s = np.random.choice(np.arange(len(solutions))[placable==1])
                new_solution[i] = solutions[s][i].copy()
                placed.append(i)
            else:
                not_placed.append(i)

    for i in not_placed:
        candidate_chargers = np.arange(m)[A[i,:] == 1]
        j = np.random.choice(candidate_chargers)
        new_solution = place_vehicle_on_charger(i, j, new_solution, sigma, Instance, min_percentage, min_distance_between_vehicles, min_pl, min_npl, max_preemptions)
        
    return new_solution


@njit
def Mutate(schedule, other_args):
    Instance, p_m, sigma, min_percentage, min_distance_between_vehicles, min_pl, min_npl, max_preemptions = other_args
    requests, chargers, A, kappa, std_deviations = Instance
    n,m = A.shape
    schedule = schedule.copy()
    candidate_vehicles = np.array([i for i in range(n) if np.sum(A[i,:])>1], dtype = np.int32)
    for i in candidate_vehicles:
        if np.random.rand()<p_m:
            candidate_chargers = np.array([j for j in range(m) if A[i,j]==1])
            j = np.random.choice(candidate_chargers)
            schedule = place_vehicle_on_charger(i, j, schedule, sigma, Instance, min_percentage, min_distance_between_vehicles, min_pl, min_npl, max_preemptions).copy()
    return schedule


@njit
def FN(schedule, other_args):
    Instance, pc, sigma, min_percentage, min_distance_between_vehicles, min_pl, min_npl, max_preemptions = other_args
    requests, chargers, A, kappa, std_deviations = Instance
    n,m = A.shape
    schedule = schedule.copy()
    candidate_vehicles = np.array([i for i in range(n) if np.sum(A[i,:])>1], dtype = np.int32)
    for i in candidate_vehicles:
        if np.random.rand()<pc:
            candidate_chargers = np.array([j for j in range(m) if A[i,j]==1])
            j = np.random.choice(candidate_chargers)
            schedule = place_vehicle_on_charger(i, j, schedule, sigma, Instance, min_percentage, min_distance_between_vehicles, min_pl, min_npl, max_preemptions).copy()
    return schedule




@njit
def Initialize_population(population_size, other_args):
    Instance, sigma, min_percentage, min_distance_between_vehicles, min_pl, min_npl, max_preemptions = other_args
    population = np.zeros((population_size, len(Instance[0]), 4+2*max_preemptions))
    for i in range(population_size):
        population[i] = generate_random_schedule(Instance, sigma, min_percentage, min_distance_between_vehicles, min_pl, min_npl, max_preemptions)
    return population


@njit
def evaluate_objectives(population, other_args):
    Instance, min_distance_between_vehicles, max_preemptions, min_pl, min_npl, num_replications = other_args
    objective_values = np.zeros((len(population), 3))
    for i in range(len(population)):
        objective_values[i] = objectives_calculation_MC(population[i], Instance, min_distance_between_vehicles, max_preemptions, min_pl, min_npl, num_replications)
    return objective_values


@njit
def selection_nsga2(population, ranks, distances):
    i1, i2 = random_choice(np.arange(len(population)), size=2, replace=False)
    if ranks[i1] < ranks[i2]:
        return population[i1]
    elif ranks[i2] < ranks[i1]:
        return population[i2]
    else:
        if distances[i1]>distances[i2]:
            return population[i1]
        elif distances[i1]<distances[i2]:
            return population[i2]
        else:
            i = np.random.choice(np.array([i1, i2]))
            return population[i]


@njit
def select_next_generation_nsga2(population, objective_values, population_size, max_preemptions):
    population = population.copy()

    fronts, ranks = fast_non_dominated_sort(objective_values)
    distances = calculate_crowding_distance(fronts, objective_values)
    
    choosen_indexes = np.zeros(population_size, dtype = np.int64)
    
    new_population = np.zeros((population_size,len(population[0]), 4+2*max_preemptions))
    new_objective_values = np.zeros((population_size,3))
    new_distances = np.zeros(population_size)
    new_ranks = np.zeros(population_size, dtype = np.int64)
    
    k = 0
    for front in fronts:
        F_l = front.copy()
        if k + len(front) < population_size:
            choosen_indexes[k:k + len(front)] = front
            k += len(front)
        else:
            break

    new_population[:k] = population[choosen_indexes[:k]]
    new_objective_values[:k] = objective_values[choosen_indexes[:k]]
    new_distances[:k] = distances[choosen_indexes[:k]]
    new_ranks[:k] = ranks[choosen_indexes[:k]]
    
    K = population_size - k
    if K > 0:
        sorted_indices = np.argsort(-distances[F_l])
        completing_indexes = F_l[sorted_indices][:K]
        
        new_population[k:] = population[completing_indexes]
        new_objective_values[k:] = objective_values[completing_indexes]
        new_distances[k:] = distances[completing_indexes]
        new_ranks[k:] = ranks[completing_indexes]
    
    return new_population, new_objective_values, new_distances, new_ranks

@njit
def NSGA2_main(Instance, other_args, inverse_signs, population_size, generations, mutation_rate, p_sel, show_progress):
    sigma, min_percentage, min_distance_between_vehicles, min_pl, min_npl, max_preemptions, num_replications, p_m = other_args
    
    new_population = np.zeros((population_size,len(Instance[0]), 4+2*max_preemptions))
    combined_population = np.zeros((2*population_size,len(Instance[0]), 4+2*max_preemptions))
    population = Initialize_population(population_size, other_args = (Instance, sigma, min_percentage, min_distance_between_vehicles, min_pl, min_npl, max_preemptions))
    population_objective_values = evaluate_objectives(population, other_args = (Instance, min_distance_between_vehicles, max_preemptions, min_pl, min_npl, num_replications))
    population, objective_values, distances, ranks = select_next_generation_nsga2(population, population_objective_values, population_size, max_preemptions)

    new_population_objective_values = np.zeros_like(population_objective_values)
    combined_objective_values = np.vstack((population_objective_values, new_population_objective_values))
    
    for gen in range(generations):
        for k in range(population_size):
            parent1 = selection_nsga2(population, ranks, distances)
            parent2 = selection_nsga2(population, ranks, distances)
            solutions = [parent1, parent2]
            child = generalized_crossover(solutions, other_args = (Instance, sigma, min_percentage, min_distance_between_vehicles, min_pl, min_npl, max_preemptions))
            if np.random.rand()<mutation_rate:
                child = Mutate(child, other_args = (Instance, p_m, sigma, min_percentage, min_distance_between_vehicles, min_pl, min_npl, max_preemptions))
            new_population[k] = child.copy()
            new_population_objective_values[k] = objectives_calculation_MC(new_population[k], Instance, min_distance_between_vehicles, max_preemptions, min_pl, min_npl, num_replications)
        
        combined_population[:population_size] = population.copy()
        combined_population[population_size:] = new_population.copy()

        combined_objective_values[:population_size] = population_objective_values.copy()
        combined_objective_values[population_size:] = new_population_objective_values.copy()
        
        population, objective_values, distances, ranks = select_next_generation_nsga2(combined_population, combined_objective_values, population_size, max_preemptions)

    pareto_front = population[ranks == 0]
    pov = objective_values[ranks == 0]
    
    pov *= inverse_signs
    return pareto_front, pov


def NSGA2(Instance, other_args, inverse_signs = np.array([1,1,1]), population_size = 100, generations = 200, mutation_rate=0.2, p_sel = 0.3, show_progress = True):
    
    pareto_front, objectives_values_pareto_front = NSGA2_main(Instance, other_args, inverse_signs, population_size, generations, mutation_rate, p_sel, show_progress)
    
    result = {
        "pareto_front":pareto_front,
        "objectives_values_pareto_front":objectives_values_pareto_front
    }
    return result




@njit
def selection_nsga3(population, ranks, association, distances, rho):
    i1, i2 = random_choice(np.arange(len(population)), size=2, replace=False)
    if ranks[i1] < ranks[i2]:
        return population[i1]
    elif ranks[i2] < ranks[i1]:
        return population[i2]
    else:
        a1 = association[i1]
        a2 = association[i2]
        if rho[a1]<rho[a2]:
            return population[i1]
        elif rho[a2]<rho[a1]:
            return population[i2]
        else:
            if distances[i1]<distances[i2]:
                return population[i1]
            else:
                return population[i2]

def generate_reference_points(num_objectives, divisions=12):
    possible_combinations = list(itertools.product(np.arange(divisions + 1), repeat=num_objectives))
    ref_points = [list(c) for c in possible_combinations if sum(c) == divisions]
    ref_points = np.array(ref_points)
    ref_points = ref_points / divisions
    return ref_points

@njit
def find_shortest_distance(A,u):
    return np.linalg.norm(A-((A.T@u)/(u.T@u))*u)

@njit
def find_closest_ref_line(solution, ref_points):
    return np.argmin(np.array([find_shortest_distance(solution,x) for x in ref_points]))

@njit
def associate_to_ref_point(normalized_objective_values, ref_points):
    population_size, num_objectives = normalized_objective_values.shape
    association = np.zeros(population_size, dtype = np.int16)
    distances = np.zeros(population_size)
    for i in range(population_size):
        association[i] = find_closest_ref_line(normalized_objective_values[i], ref_points)
        distances[i] = find_shortest_distance(normalized_objective_values[i],ref_points[association[i]])
    return association, distances

@njit
def find_number_of_associated_solutions(ref_points, association):
    num_ref_points = len(ref_points)
    return np.bincount(association, minlength=num_ref_points)

@njit
def select_next_generation_nsga3(population, objective_values, population_size, ref_points, max_preemptions):
    population = population.copy()

    fronts, ranks = fast_non_dominated_sort(objective_values)
    
    choosen_indexes = np.zeros(population_size, dtype = np.int64)
    
    new_population = np.zeros((population_size,len(population[0]), 4+2*max_preemptions))
    new_objective_values = np.zeros((population_size,3))
    new_ranks = np.zeros(population_size, dtype = np.int64)
    
    k = 0
    for front in fronts:
        F_l = front.copy()
        if k + len(front) < population_size:
            choosen_indexes[k:k + len(front)] = front
            k += len(front)
        else:
            break

    new_population[:k] = population[choosen_indexes[:k]]
    new_objective_values[:k] = objective_values[choosen_indexes[:k]]
    new_ranks[:k] = ranks[choosen_indexes[:k]]
    
    K = population_size - k
    
    if K > 0:
        S_t = np.concatenate((choosen_indexes[:k], F_l))
        normalized_objective_values = normalize_objectives(objective_values[S_t])
        association, distances = associate_to_ref_point(normalized_objective_values, ref_points)
        indices_before_F_l = np.arange(len(S_t) - len(front))
        indices_F_l = np.arange(len(S_t) - len(front), len(S_t))
        rho = find_number_of_associated_solutions(ref_points, association[indices_before_F_l])
        Z_r = np.arange(len(ref_points))
        selected = np.full(len(indices_F_l), False)
        while k < population_size:
            rho_min = np.min(rho[Z_r])
            J_min = Z_r[rho[Z_r] == rho_min]
            j = np.random.choice(J_min)
            I_j_bar = indices_F_l[(association[indices_F_l] == j) & (~selected)]
            if len(I_j_bar) > 0:
                if rho[j] == 0:
                    v = np.argmin(distances[I_j_bar])
                    s = I_j_bar[v]
                else:
                    s = np.random.choice(I_j_bar)
                    
                new_population[k] = population[S_t[s]]
                new_objective_values[k] = objective_values[S_t[s]]
                new_ranks[k] = ranks[S_t[s]]
                
                k += 1
                rho[j] += 1
                selected[np.where(indices_F_l == s)[0][0]] = True
            else:
                Z_r = Z_r[Z_r != j]

            if len(Z_r) == 0:
                break

    normalized_objective_values = normalize_objectives(new_objective_values)
    association, distances = associate_to_ref_point(normalized_objective_values, ref_points)
    rho = find_number_of_associated_solutions(ref_points, association)
    return new_population, new_objective_values, new_ranks, association, distances, rho


@njit
def NSGA3_main(Instance, other_args, inverse_signs, population_size, generations, mutation_rate, p_sel, show_progress):
    sigma, min_percentage, min_distance_between_vehicles, min_pl, min_npl, max_preemptions, num_replications, p_m, ref_points = other_args
    
    new_population = np.zeros((population_size,len(Instance[0]), 4+2*max_preemptions))
    combined_population = np.zeros((2*population_size,len(Instance[0]), 4+2*max_preemptions))
    population = Initialize_population(population_size, other_args = (Instance, sigma, min_percentage, min_distance_between_vehicles, min_pl, min_npl, max_preemptions))
    population_objective_values = evaluate_objectives(population, other_args = (Instance, min_distance_between_vehicles, max_preemptions, min_pl, min_npl, num_replications))
    population, objective_values, ranks, association, distances, rho = select_next_generation_nsga3(population, population_objective_values, population_size, ref_points, max_preemptions)

    new_population_objective_values = np.zeros_like(population_objective_values)
    combined_objective_values = np.vstack((population_objective_values, new_population_objective_values))
    
    for gen in range(generations):
        for k in range(population_size):
            parent1 = selection_nsga3(population, ranks, association, distances, rho)
            parent2 = selection_nsga3(population, ranks, association, distances, rho)
            solutions = [parent1, parent2]
            child = generalized_crossover(solutions, other_args = (Instance, sigma, min_percentage, min_distance_between_vehicles, min_pl, min_npl, max_preemptions))
            if np.random.rand()<mutation_rate:
                child = Mutate(child, other_args = (Instance, p_m, sigma, min_percentage, min_distance_between_vehicles, min_pl, min_npl, max_preemptions))
            new_population[k] = child.copy()
            new_population_objective_values[k] = objectives_calculation_MC(new_population[k], Instance, min_distance_between_vehicles, max_preemptions, min_pl, min_npl, num_replications)
        
        combined_population[:population_size] = population.copy()
        combined_population[population_size:] = new_population.copy()

        combined_objective_values[:population_size] = population_objective_values.copy()
        combined_objective_values[population_size:] = new_population_objective_values.copy()
        
        population, objective_values, ranks, association, distances, rho = select_next_generation_nsga3(combined_population, combined_objective_values, population_size, ref_points, max_preemptions)

    pareto_front = population[ranks == 0]
    pov = objective_values[ranks == 0]
    
    pov *= inverse_signs
    return pareto_front, pov


def NSGA3(Instance, other_args, inverse_signs = np.array([1,1,1]), population_size = 100, generations = 200, mutation_rate=0.2, p_sel = 0.3, show_progress = True):
    
    pareto_front, objectives_values_pareto_front = NSGA3_main(Instance, other_args, inverse_signs, population_size, generations, mutation_rate, p_sel, show_progress)
    
    result = {
        "pareto_front":pareto_front,
        "objectives_values_pareto_front":objectives_values_pareto_front
    }
    return result

    

@njit
def sort_population_according_to_domination(population, objective_values, population_size):
    population = population.copy()

    fronts, ranks = fast_non_dominated_sort(objective_values)
    
    choosen_indexes = np.argsort(ranks)[:population_size]

    return population[choosen_indexes], ranks[choosen_indexes], objective_values[choosen_indexes]

@njit
def MOCS_main(Instance, other_args, inverse_signs, population_size, generations, perc_abandoned, p_sel, show_progress):
    sigma, min_percentage, min_distance_between_vehicles, min_pl, min_npl, max_preemptions, num_replications, pc = other_args
    
    new_population = np.zeros((population_size,len(Instance[0]), 4+2*max_preemptions))
    combined_population = np.zeros((2*population_size,len(Instance[0]), 4+2*max_preemptions))
    population = Initialize_population(population_size, other_args = (Instance, sigma, min_percentage, min_distance_between_vehicles, min_pl, min_npl, max_preemptions))
    
    num_abandoned = int(perc_abandoned*population_size)

    N = ceil(p_sel*population_size)
    probabilities = np.zeros(N, dtype = np.float64)
    probabilities = (N-np.arange(N))/(N*(N+1)/2)

    population = Initialize_population(population_size, other_args = (Instance, sigma, min_percentage, min_distance_between_vehicles, min_pl, min_npl, max_preemptions))
    population_objective_values = evaluate_objectives(population, other_args = (Instance, min_distance_between_vehicles, max_preemptions, min_pl, min_npl, num_replications))
    population, ranks, objective_values = sort_population_according_to_domination(population, population_objective_values, population_size)

    new_population_objective_values = np.zeros_like(population_objective_values)
    combined_objective_values = np.vstack((population_objective_values, new_population_objective_values))
    
    for gen in range(generations):
        for i in range(population_size):
            k = random_choice(np.arange(N),p = probabilities)[0]
            new_population[i] = FN(population[k], other_args = (Instance, pc, sigma, min_percentage, min_distance_between_vehicles, min_pl, min_npl, max_preemptions))
            new_population_objective_values[i] = objectives_calculation_MC(new_population[i], Instance, min_distance_between_vehicles, max_preemptions, min_pl, min_npl, num_replications)

        for j in range(1,num_abandoned+1):
            population[-j] = generate_random_schedule(Instance, sigma, min_percentage, min_distance_between_vehicles, min_pl, min_npl, max_preemptions)
            population_objective_values[-j] = objectives_calculation_MC(population[-j], Instance, min_distance_between_vehicles, max_preemptions, min_pl, min_npl, num_replications)
            
        
        combined_population[:population_size] = population.copy()
        combined_population[population_size:] = new_population.copy()

        combined_objective_values[:population_size] = population_objective_values.copy()
        combined_objective_values[population_size:] = new_population_objective_values.copy()
        

        population, ranks, objective_values = sort_population_according_to_domination(combined_population, combined_objective_values, population_size)

    pareto_front = population[ranks == 0]
    pov = objective_values[ranks == 0]
    
    pov *= inverse_signs
    return pareto_front, pov

def MOCS(Instance, other_args, inverse_signs = np.array([1,1,1]), population_size = 100, generations = 200, perc_abandoned = 0.2, p_sel = 0.3, show_progress = True):
    
    pareto_front, objectives_values_pareto_front = MOCS_main(Instance, other_args, inverse_signs, population_size, generations, perc_abandoned, p_sel, show_progress)
    
    result = {
        "pareto_front":pareto_front,
        "objectives_values_pareto_front":objectives_values_pareto_front
    }
    return result



    

@njit
def select_leaders(archive, num_points_per_dim, beta = 1.0):
    archive_solutions, archive_objectives, archive_size = archive
    
    pareto_solutions = archive_solutions[:archive_size]
    pareto_objectives = archive_objectives[:archive_size]
    pf_indexes = np.arange(archive_size)
    
    normalized_objectives = normalize_objectives(pareto_objectives)
    mins, maxs = find_mins_maxs(normalized_objectives)
    grid_count = compute_grid_count(normalized_objectives, mins, maxs, num_points_per_dim)
    to_take = np.zeros(len(pf_indexes))
    K = 3
    while K > 0:
        index = sample_index_inverse_weighted(grid_count, beta)[:len(mins)]
        elements = [i for i, x in enumerate(normalized_objectives)
                    if not to_take[i] and np.all(locate_point_in_the_grid(x, mins, maxs, num_points_per_dim)[:len(mins)] == index)]
        if elements:
            i = np.random.choice(np.array(elements))
            index = locate_point_in_the_grid(normalized_objectives[i], mins, maxs, num_points_per_dim)[:len(mins)]
            grid_count[index[0], index[1], index[2]] -= 1
            to_take[i] = 1
            K -= 1
    
    return pareto_solutions[to_take==1]

@njit
def MOGWO_main(Instance, other_args, inverse_signs, population_size, generations, show_progress):
    archive_maxsize, num_points_per_dim, alpha, beta, num_replications, sigma, min_percentage, min_distance_between_vehicles, min_pl, min_npl, max_preemptions, mutation_rate, p_m = other_args
    
    
    num_objectives = len(inverse_signs)
    
    population = Initialize_population(population_size, other_args = (Instance, sigma, min_percentage, min_distance_between_vehicles, min_pl, min_npl, max_preemptions))
    
    archive_solutions = np.zeros((archive_maxsize,len(Instance[0]), 4+2*max_preemptions))
    archive_objectives = np.zeros((archive_maxsize, num_objectives))
    archive_size = 0
    archive = archive_solutions, archive_objectives, archive_size
    archive = Initialize_archive(archive, population, num_objectives, archive_maxsize, num_points_per_dim, alpha, other_args = (Instance, min_distance_between_vehicles, max_preemptions, min_pl, min_npl, num_replications))
    for gen in range(generations):
        for i in range(population_size):
            if archive[2]>=3:
                Alpha, Beta, Delta = select_leaders(archive, num_points_per_dim, beta)
            else:
                x = np.random.randint(0, archive[2], 3)
                Alpha, Beta, Delta = archive[0][x]
                
            solutions = [Alpha, Beta, Delta]
            population[i] = generalized_crossover(solutions, other_args = (Instance, sigma, min_percentage, min_distance_between_vehicles, min_pl, min_npl, max_preemptions))
            
            if np.random.rand() < mutation_rate:
                population[i] = Mutate(population[i], other_args = (Instance, p_m, sigma, min_percentage, min_distance_between_vehicles, min_pl, min_npl, max_preemptions))
                
        archive = Insert_new_solutions(archive, population, num_objectives, archive_maxsize, num_points_per_dim, alpha, other_args = (Instance, min_distance_between_vehicles, max_preemptions, min_pl, min_npl, num_replications))
    
    return archive[0][:archive[2]], archive[1][:archive[2]]*inverse_signs


def MOGWO(Instance, other_args, inverse_signs = np.array([1,1,1]), population_size = 100, generations = 200, show_progress = True):
    
    pareto_front, objectives_values_pareto_front = MOGWO_main(Instance, other_args, inverse_signs, population_size, generations, show_progress)
    
    result = {
        "pareto_front":pareto_front,
        "objectives_values_pareto_front":objectives_values_pareto_front
    }
    return result