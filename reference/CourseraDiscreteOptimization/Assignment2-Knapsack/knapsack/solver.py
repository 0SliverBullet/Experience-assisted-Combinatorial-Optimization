#!/usr/bin/python
# -*- coding: utf-8 -*-
import numpy as np
from collections import namedtuple
from queue import PriorityQueue

Item = namedtuple("Item", ['index', 'value', 'weight'])

class Node:
    def __init__(self, level, value, weight, bound, taken):
        self.level = level
        self.value = value
        self.weight = weight
        self.bound = bound
        self.taken = taken

    def __lt__(self, other):
        return self.bound > other.bound  # greater than for max heap

def calculate_bound(node, items, capacity):
    if node.weight >= capacity:
        return 0
    else:
        bound = node.value
        total_weight = node.weight
        j = node.level + 1
        while j < len(items) and total_weight + items[j].weight <= capacity:
            bound += items[j].value
            total_weight += items[j].weight
            j += 1
        if j < len(items):
            bound += (capacity - total_weight) * items[j].value / items[j].weight
        return bound

def solve_it_bb(input_data):
    # parse the input
    lines = input_data.split('\n')

    firstLine = lines[0].split()
    item_count = int(firstLine[0])
    capacity = int(firstLine[1])

    items = []

    for i in range(1, item_count+1):
        line = lines[i]
        parts = line.split()
        items.append(Item(i-1, int(parts[0]), int(parts[1])))

    items.sort(key=lambda x: x.value / x.weight, reverse=True)

    # Branch and Bound
    max_node = Node(-1, 0, 0, 0, [0]*len(items))
    priority_queue = PriorityQueue()
    priority_queue.put(max_node)

    while not priority_queue.empty():
        node = priority_queue.get()
        if node.bound >= max_node.value:


            left_child = Node(node.level + 1,
                              node.value + items[node.level + 1].value,
                              node.weight + items[node.level + 1].weight,
                              node.bound,
                              node.taken[:])
            if left_child.weight <= capacity and left_child.value > max_node.value:
                max_node = left_child
            left_child.bound = calculate_bound(left_child, items, capacity)
            left_child.taken[node.level + 1] = 1  # Item included in the knapsack
            if left_child.bound > max_node.value:
                priority_queue.put(left_child)
            
            
            right_child = Node(node.level + 1,
                               node.value,
                               node.weight,
                               node.bound - items[node.level + 1].value,
                               node.taken[:])
            right_child.bound = calculate_bound(right_child, items, capacity)
            right_child.taken[node.level + 1] = 0  # Item not included in the knapsack
            if right_child.bound > max_node.value:
                priority_queue.put(right_child)

    # prepare the solution in the specified output format
    output_data = str(max_node.value) + ' ' + str(0) + '\n'
    output_data += ' '.join(map(str, max_node.taken))
    return output_data

def solve_it(input_data):
    # Modify this code to run your optimization algorithm

    # parse the input
    lines = input_data.split('\n')

    firstLine = lines[0].split()
    item_count = int(firstLine[0])
    capacity = int(firstLine[1])

    items = []

    for i in range(1, item_count+1):
        line = lines[i]
        parts = line.split()
        items.append(Item(i-1, int(parts[0]), int(parts[1])))

    # # a trivial algorithm for filling the knapsack
    # # it takes items in-order until the knapsack is full
    # value = 0
    # weight = 0
    # taken = [0]*len(items)

    # for item in items:
    #     if weight + item.weight <= capacity:
    #         taken[item.index] = 1
    #         value += item.value
    #         weight += item.weight
        


    # # Dynamic programming using rolling array
    # dp = [0] * (capacity + 1)

    # for item in items:
    #     print(item.index)
    #     for w in range(capacity, item.weight - 1, -1):
    #         if dp[w - item.weight] + item.value > dp[w]:
    #             dp[w] = dp[w - item.weight] + item.value

    # # Retrieve the value
    # value = dp[capacity]

    # # Backtrack to find the items selected
    # taken = [0] * item_count
    # remaining_capacity = capacity

    # for item in reversed(items):
    #     if remaining_capacity >= item.weight and dp[remaining_capacity] == dp[remaining_capacity - item.weight] + item.value:
    #         taken[item.index] = 1
    #         remaining_capacity -= item.weight
        

    # Dynamic programming approach to solve knapsack problem
    # Initialize a table to store the maximum value that can be achieved for each subproblem
    # table[i][w] represents the maximum value that can be achieved with capacity w using items up to i
    table = [[0] * (capacity + 1) for _ in range(item_count + 1)]

    # Build up the table iteratively
    for i in range(1, item_count + 1):
        for w in range(1, capacity + 1):
            # If the current item can fit into the knapsack
            if items[i - 1].weight <= w:
                # Choose the maximum value between including the current item or excluding it
                table[i][w] = max(table[i - 1][w], table[i - 1][w - items[i - 1].weight] + items[i - 1].value)
            else:
                # If the current item cannot fit, just take the value from the previous row
                table[i][w] = table[i - 1][w]

    # Backtrack to find the items included in the optimal solution
    value = table[item_count][capacity]
    weight = capacity
    taken = [0] * item_count
    for i in range(item_count, 0, -1):
        if table[i][weight] != table[i - 1][weight]:
            taken[items[i - 1].index] = 1
            weight -= items[i - 1].weight
    # prepare the solution in the specified output format
    output_data = str(value) + ' ' + str(0) + '\n'
    output_data += ' '.join(map(str, taken))
    return output_data



def solve_it_ga(input_data):
    # parse the input
    lines = input_data.split('\n')

    firstLine = lines[0].split()
    item_count = int(firstLine[0])
    capacity = int(firstLine[1])

    items = []

    for i in range(1, item_count+1):
        line = lines[i]
        parts = line.split()
        items.append(Item(i-1, int(parts[0]), int(parts[1])))

    # Genetic Algorithm Parameters
    POPULATION_SIZE = 2
    GENERATIONS = 1000
    MUTATION_RATE = 0.3
    best_individual = [0]*item_count
    best_value = 0

    # Genetic Algorithm for Knapsack Problem
    def create_individual():
        individual=np.random.choice([0, 1], size=item_count)
        while fitness(individual)==0:
            individual=np.random.choice([0, 1], size=item_count)
        return individual

    def fitness(individual):
        total_value = sum(item.value * individual[i] for i, item in enumerate(items))
        total_weight = sum(item.weight * individual[i] for i, item in enumerate(items))
        if total_weight > capacity:
            return 0
        else:
            return total_value

    def crossover(parent1, parent2):
        crossover_point = np.random.randint(1, len(parent1) - 1)
        child1 = np.concatenate((parent1[:crossover_point], parent2[crossover_point:]))
        child2 = np.concatenate((parent2[:crossover_point], parent1[crossover_point:]))
        return child1, child2

    def mutate(individual):
        for i in range(len(individual)):
            if np.random.random() < MUTATION_RATE:
                individual[i] = 1 - individual[i]
        return individual

    # Initialize population
    population = [create_individual() for _ in range(POPULATION_SIZE)]

    # Main Genetic Algorithm loop
    for _ in range(GENERATIONS):
        print(_)
        # Select parents
        parents = []
        for i in range(POPULATION_SIZE):
            parent_indices = np.random.choice(range(POPULATION_SIZE), size=2, replace=False)
            parent1, parent2 = population[parent_indices[0]], population[parent_indices[1]]
            parents.append((parent1, parent2))

        # Create offspring via crossover and mutation
        offspring = []
        for parent1, parent2 in parents:
            child1, child2 = crossover(parent1, parent2)
            child1 = mutate(child1)
            child2 = mutate(child2)
            offspring.extend([child1, child2])

        # Select survivors for the next generation
        for i in range(POPULATION_SIZE):
            if fitness(offspring[i])>fitness(population[i]):
                population[i]=offspring[i]
            if fitness(population[i])>best_value:
                  best_individual=population[i]
                  best_value=fitness(population[i])


    # prepare the solution in the specified output format
    output_data = str(best_value) + ' ' + str(0) + '\n'
    output_data += ' '.join(map(str, best_individual))

    return output_data

if __name__ == '__main__':
    import sys
    if len(sys.argv) > 1:
        file_location = sys.argv[1].strip()
        with open(file_location, 'r') as input_data_file:
            input_data = input_data_file.read()
        print(solve_it(input_data))
    else:
        print('This test requires an input file.  Please select one from the data directory. (i.e. python solver.py ./data/ks_4_0)')

