import numpy as np

from solution import solution

def GOA(objf, lower_bound,upper_bound, dim, PopSize, iters, c_min, c_max):
    lb, ub = lower_bound, upper_bound
    # Initialize the grasshopper population
    population = np.random.uniform(lb, ub, (PopSize, dim))
    fitness = np.array([objf(ind) for ind in population])
    g_best = population[np.argmin(fitness)]


    s = solution()
    for epoch in range(iters):
        c = c_min + (c_max - c_min) * (iters - epoch) / iters
        for i in range(PopSize):
            new_position = np.zeros(dim)
            for j in range(PopSize):
                distance = np.linalg.norm(population[i] - population[j])
                r_ij_vector = (population[j] - population[i]) / (distance + 1e-20)
                new_position += r_ij_vector * np.exp(-distance / 1.5) * np.cos(distance * 2 * np.pi) * c
            new_position = g_best + new_position * c
            new_position = np.clip(new_position, lb, ub)
            population[i] = new_position
            fitness[i] = objf(new_position)

        # Update the global best
        idx_best = np.argmin(fitness)
        if fitness[idx_best] < objf(g_best):
            g_best = population[idx_best]
        
        s.y.append(objf(g_best))
        s.x.append(epoch)
        print(f"Iteration {epoch + 1}, gbestscore: {objf(g_best)}")

    s.best = objf(g_best) 
    s.bestIndividual = g_best.tolist()  

    return s


