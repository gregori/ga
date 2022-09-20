import numpy
import ga


def main():
    # Entradas da equação
    equation_inputs = [4, -2, 3.5, 5, -11, -4.7]
    # número de pesos que queremos otimizar
    num_weights = 6

    # Tamanho da população
    sol_per_pop = 8
    # a população terá sol_per_pop cromossomos, cada um com num_weights gens
    pop_size = (sol_per_pop, num_weights)

    # Criando a população inicial
    population = numpy.random.uniform(low=-4.0, high=4.0, size=pop_size)

    # print(population)

    num_generations = 5
    num_parents_mating = 4
    for generation in range(num_generations):
        # medir o fitness de cada cromossomo da geração
        fitness = ga.calc_fitness(equation_inputs, population)
        # print(fitness)
        # selecionar os melhores indivíduos para gerar a próx geração
        parents = ga.select_mating_pool(population, fitness, num_parents_mating)
        # print(parents)

        # gerar a próxima geração usando crossover
        offspring_crossover = ga.crossover(parents, (sol_per_pop - parents.shape[0], num_weights))
        # print(offspring_crossover)

        # adicionar variáveis aos filhos usando mutação
        offspring_mutation = ga.mutation(offspring_crossover)

        # criar nova população baseada nos pais
        population[0:parents.shape[0], :] = parents
        population[parents.shape[0]:, :] = offspring_mutation

        print("Melhor resultado : ", numpy.max(numpy.sum(population*equation_inputs, axis=1)))

    fitness = ga.calc_fitness(equation_inputs, population)
    best_match_idx = numpy.where(fitness == numpy.max(fitness))
    print("Melhor solução : ", population[best_match_idx, :])
    print("Fitness da melhor solução: ", fitness[best_match_idx])

if __name__ == "__main__":
    main()