import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as ani
import numpy.random as npr

# 1. Population size: 20, 50, 100, 200.
# 2. Num of Gens: 50, 100, 200.
# 3. Crossover Probability : 0.80
# 4. Constants K and F for creating mutant vector: K = 0.5; for F, choose between -2 and +2 randomly across generations, but keep it same for all candidates within a generation.


def initialiseVectors(boundaries, Population_Size):
    vectors = []
    vectord = []
    for item in range(Population_Size):
        vectord = []
        for item2 in range(len(boundaries)):
            vectord.append(npr.normal((boundaries[item2][1] + boundaries[item2][0])/2, abs(
                boundaries[item2][1] - boundaries[item2][0])/5))
        vectors.append(vectord)
    # print(vectors)
    return vectors

def random_sample(arr: np.array, size: int = 1):
    return arr[np.random.choice(len(arr), size=size, replace=False)]


def generate_Mutant(vector, K, F, vectors):
    Xr1 = random_sample(vectors)[0]
    Xr2 = random_sample(vectors)[0]
    Xr3 = random_sample(vectors)[0]
    return vector + K*(Xr1 - vector) + F*(Xr2 - Xr3)


def Crossover(Crossover_probability, mutant, vector, K, F, vectors, boundaries):
    trial = []
    for item in range(len(mutant)):
        if npr.random_sample() > Crossover_probability:
            trial.append(mutant[item])
        else:
            trial.append(vector[item])
    flag = 0
    for index in range(len(trial)):
        if trial[index] > boundaries[index][0] or trial[index] < boundaries[index][1]:
            flag = 1
            mutant = generate_Mutant(vector, K, F, vectors)
            return Crossover(Crossover_probability, mutant, vector, K, F, vectors, boundaries)
    if flag == 0:
        return np.array(trial)

# EGGHOLDER FUNCTION


def objective_function_Egg(pop):
    # print(pop.shape[0])
    fitness = np.zeros(pop.shape[0])
    for i in range(pop.shape[0]):
        x = pop[i]
        fitness[i] = (-(x[1]+47)*np.sin(np.sqrt(abs(x[0]/2+(x[1]+47)))
                                        )-x[0]*np.sin(np.sqrt(abs(x[0]-(x[1]+47)))))
    return fitness

# HOLDER FUNCTION


def objective_function_Holder(pop):
    # print(pop.shape[0])
    fitness = np.zeros(pop.shape[0])
    for i in range(pop.shape[0]):
        x = pop[i]
        fitness[i] = -(abs(np.sin(x[0])*np.cos(x[1]) *
                           np.exp(abs(1 - ((np.sqrt(x[0]**2 + x[1]**2))/np.pi)))))
    return fitness


def bestfit(vectors, func):
    if func == 0:
        objective_function = objective_function_Egg
    else:
        objective_function = objective_function_Holder

    fitness = objective_function(vectors)
    # print(vectors[np.argmin(fitness)], fitness[np.argmin(fitness)], fitness)
    k = np.argmin(fitness)
    for item in range(len(fitness)):
        if item == (k + 1) or item == (k):
            print("\n")
        print(vectors[item], fitness[item])

    return vectors[k]


def plot(best, avg):
    x1 = best
    y1 = range(len(best))
    plt.plot(y1, x1, label="Best")
    x2 = avg
    y2 = range(len(avg))
    plt.plot(y2, x2, label="Average")
    plt.xlabel('Epochs')
    plt.ylabel('Fitness')
    plt.title('Fitness(best) and Fitness(avg) vs epochs')
    plt.legend()
    plt.show()


def Differential_Evolution(Population_Size, Total_Generations, Crossover_probability, K, func):
    # Xi,g = Xi,g + K*(Xr1,g - Xi,g) + F*(Xr2,g - Xr3,g)
    dimentions = int(input("\nBefore proceeding, we assign the boundary with respect to the objective function\n(here both objective functions(eggholder and holder table) require only two dimentions).\nPlease enter your dimention: "))

    boundaries = []
    c = input("Do you want to assign one higher bound and one lower bound for all dimentions? [y/n]:\n")

    if c == "y" or c == "yes":
        higher = float(input("Higher bound: "))
        lower = float(input("Lower bound: "))
        for item in range(dimentions):
            boundaries.append([higher, lower])

    else:
        for item in range(dimentions):
            boundaries.append([float(input("Dimention: " + str(item + 1) + " higher bound\n")),
                               float(input("Dimention: " + str(item + 1) + " lower bound\n"))])

    vectors = initialiseVectors(boundaries, Population_Size)
    vectors = np.array(vectors)

    print("Printing generations:\n")
    fitness_history_best = []
    fitness_history_average = []
    if func == 0:
        objective_function = objective_function_Egg
    else:
        objective_function = objective_function_Holder

    for generation in range(Total_Generations):
        print("Generation: ", generation)
        fitness = objective_function(vectors)
        fitness_history_best.append(fitness[np.argmin(fitness)])
        fitness_history_average.append(np.average(fitness))
        # for item in range(len(fitness)):
        #     print(vectors[item], fitness[item])
        F = random_sample(np.array([-2, 2]))[0]
        for vectorindex in range(len(vectors)):
            mutant = generate_Mutant(vectors[vectorindex], K, F, vectors)
            trial = Crossover(Crossover_probability, mutant,
                              vectors[vectorindex], K, F, vectors, boundaries)
            if objective_function(np.array([trial]))[0] < objective_function(np.array([vectors[vectorindex]]))[0]:
                vectors[vectorindex] = trial
    plot(fitness_history_best, fitness_history_average)
    return bestfit(vectors, func)


print("Input hyperparameters:")
Population_Size = int(input(("Enter Population:")))
Total_Generations = int(input(("Enter Total generations:")))
while True:
    Crossover_probability = float(input(("Enter crossover probability:")))
    if Crossover_probability <= 1 and Crossover_probability >= 0:
        break
    print("Probability lies between 0 and 1")
K = float(input(("Enter K:")))
print("(F is either -2 and +2 and is decided randomly for each generation): ")

print("\nEggholder Function: ")
Solution = Differential_Evolution(Population_Size, Total_Generations,
                                  Crossover_probability, K, 0)
print("Solution Eggholder Function: ", Solution)

print("\nHolder Function: ")
Solution = Differential_Evolution(Population_Size, Total_Generations,
                                  Crossover_probability, K, 1)
print("Solution Holder Function: ", Solution)
