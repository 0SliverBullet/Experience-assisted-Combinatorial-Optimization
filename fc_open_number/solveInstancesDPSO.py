import numpy as np
import sys
import random
import time
import copy

f = None

class Main:

    # Basic attribute of the UFLP problem
    m = None
    n = None
    f = None
    c = None

    generation = None
    goal_generation = None
    particle_list = None
    gbest_particle = None
    value_list = None
    
    start = None

    def __init__(self, f, c, goal_generation):
        Main.goal_generation = goal_generation
        Main.m, Main.n, Main.f, Main.c = len(f), len(c), f, c

    def DPSO(self):
        self.initialize()
        counts=[]
        anss=[]
        pop=[]
        while Main.generation < Main.goal_generation:
            Main.generation += 1
            self.update_pbest_gbest()
            counts.append(Main.generation)
            anss.append(Main.gbest_particle.fitness_value)
            Main.value_list = []
            for individual in Main.particle_list:
                individual.velocity()
                individual.cognition()
                individual.social()
                individual.calculate_fitness_value()
                Main.value_list.append(individual.fitness_value)
            self.local_search()
        for individual in Main.particle_list:
            pop.append(individual.facility_vector)
        return Main.gbest_particle.facility_vector

    def initialize(self):
        Main.start = time.time()
        Main.generation = 0
        Main.particle_list = []
        Main.value_list = []
        Main.gbest_particle = particle()
        Main.gbest_particle.fitness_value = int(sys.maxsize)
        for j in range(0, Main.m):
            initial_solution_particle = particle()
            Main.particle_list.append(initial_solution_particle)
            Main.value_list.append(initial_solution_particle.fitness_value)
            if initial_solution_particle.fitness_value < Main.gbest_particle.fitness_value:
                Main.gbest_particle = initial_solution_particle
        Main.gbest_particle = copy.deepcopy(Main.gbest_particle)

    def update_pbest_gbest(self):
        for individual in Main.particle_list:
            if individual.fitness_value < Main.gbest_particle.fitness_value:
                Main.gbest_particle = individual
            if individual.fitness_value < individual.pbest_fitness_value:
                individual.pbest_fitness_value = individual.fitness_value
                individual.pbest_facility_vector = individual.facility_vector
        Main.gbest_particle = copy.deepcopy(Main.gbest_particle)

    def local_search(self):
        s0 = np.copy(Main.gbest_particle.facility_vector)
        rand_facility_1_loc, rand_facility_2_loc = random.sample(
            range(0, Main.m), 2)
        s0[rand_facility_1_loc] = 1-s0[rand_facility_1_loc]
        s0[rand_facility_2_loc] = 1-s0[rand_facility_2_loc]
        s = s0
        fs = self.calculate_fitness_value(s)
        for l in range(0, Main.m):
            rand_facility_loc = random.randint(0, Main.m-1)
            s1 = np.copy(s)
            s1[rand_facility_loc] = 1-s1[rand_facility_loc]
            fs1 = self.calculate_fitness_value(s1)
            if fs1 < fs:
                s = s1
                fs = fs1
        if fs <= Main.gbest_particle.fitness_value:
            Main.gbest_particle.facility_vector = s
            Main.gbest_particle.fitness_value = fs
            Main.gbest_particle.pbest_facility_vector = s
            Main.gbest_particle.pbest_fitness_value = fs

    def calculate_fitness_value(self, facility_vector):
        result = 0

        # Calculate the opening cost
        result += np.inner(facility_vector, Main.f)
        
        open_facility_indexes = np.where(facility_vector)[0]
        if len(open_facility_indexes) == 0:
            return sys.maxsize
        temp = np.min(Main.c[:, open_facility_indexes], axis=1)
        result += np.sum(temp)

        return result

    def test(self):  # This is just a test for correctness verify
        print(Main.m, Main.n, Main.f, Main.c)


class particle:

    c1 = 0.5  # social probability
    c2 = 0.5  # cognitive probability
    w = 0.9  # inertia weight

    def __init__(self):
        self.facility_vector = None
        self.fitness_value = None
        self.pbest_facility_vector = None
        self.pbest_fitness_value = None
        
        # Initialize the facility vector with random 0/1
        self.facility_vector = np.random.randint(0, 2, Main.m)
        # Calculate the fitness value for the initial solution
        self.calculate_fitness_value()

        self.pbest_facility_vector = self.facility_vector
        self.pbest_fitness_value = self.fitness_value

    def calculate_fitness_value(self):
        result = 0

        # Calculate the opening cost
        result += np.inner(self.facility_vector, Main.f)
        
        open_facility_indexes = np.where(self.facility_vector)[0]
        if len(open_facility_indexes) == 0:
            self.fitness_value = sys.maxsize
            return
        temp = np.min(Main.c[:, open_facility_indexes], axis=1)
        result += np.sum(temp)

        self.fitness_value = result

    def velocity(self):
        r = random.random()
        if r < particle.w:
            rand_facility_1_loc, rand_facility_2_loc = random.sample(
                range(0, Main.m), 2)
            a = self.facility_vector[rand_facility_1_loc]
            b = self.facility_vector[rand_facility_2_loc]
            self.facility_vector[rand_facility_1_loc] = b
            self.facility_vector[rand_facility_2_loc] = a

    def cognition(self):
        r = random.random()
        if r < particle.c1:
            first_parant_lambda = np.copy(self.facility_vector)
            second_parant_pbest = np.copy(self.pbest_facility_vector)
            start_idx = 0
            end_idx = random.randint(1, Main.m-2)

            # One-cut crossover
            temp = np.copy(first_parant_lambda[start_idx:end_idx+1])
            first_parant_lambda[start_idx:end_idx +
                                1] = second_parant_pbest[start_idx:end_idx+1]
            second_parant_pbest[start_idx:end_idx+1] = temp

            self.facility_vector = random.choice(
                [first_parant_lambda, second_parant_pbest])

    def social(self):
        r = random.random()
        if r < particle.c2:
            first_parant_sigma = np.copy(self.facility_vector)
            second_parant_gbest = np.copy(Main.gbest_particle.facility_vector)
            start_idx = random.randint(1, Main.m-2)
            end_idx = random.randint(start_idx+1, Main.m-1)

            # Two-cut crossover
            temp = np.copy(first_parant_sigma[start_idx:end_idx+1])
            first_parant_sigma[start_idx:end_idx +
                               1] = second_parant_gbest[start_idx:end_idx+1]
            second_parant_gbest[start_idx:end_idx+1] = temp

            self.facility_vector = random.choice(
                [first_parant_sigma, second_parant_gbest])
