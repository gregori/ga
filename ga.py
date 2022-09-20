import sys
import numpy

def calc_fitness(equation_inputs, population):
    # calcular o fitness na população atual
    # a função fitness calcula a soma dos produtos entre cada entrada e o peso correspondente
    return numpy.sum(population * equation_inputs, axis=1)


def select_mating_pool(population, fitness, num_parents):
    # selecionar os melhores indivíduos na geração atual como pais para gerar a próxima
    # geração

    # Cria um vetor com o tamanho de num_parents
    parents = numpy.empty((num_parents, population.shape[1]))

    # Preenche o vetor de progenitores
    for parent_num in range(num_parents):
        max_fitness_idx = numpy.where(fitness == numpy.max(fitness))
        max_fitness_idx = max_fitness_idx[0][0]
        parents[parent_num, :] = population[max_fitness_idx, :]
        fitness[max_fitness_idx] = -sys.maxsize - 1

    return parents


def crossover(parents, offspring_size):
    offspring = numpy.empty(offspring_size)

    # onde ocorrerá o crossover entre os pais? (geralmente na metade do vetor)
    crossover_point = numpy.uint8(offspring_size[1]/2)

    for idx in range(offspring_size[0]):
        # índice do primeiro genitor
        parent1_idx = idx % parents.shape[0]
        # índice do segundo genitor
        parent2_idx = (idx + 1) % parents.shape[0]

        # o novo filho terá a primeira metade dos genes do primeiro genitor
        offspring[idx, 0:crossover_point] = parents[parent1_idx, 0:crossover_point]

        # o novo filho terá a segunda metade dos genes do segundo genitor
        offspring[idx, crossover_point:] = parents[parent2_idx, crossover_point:]

    return offspring


def mutation(offspring_crossover):
    # muda um gene em cada filho aleatoriamente
    for idx in range(offspring_crossover.shape[0]):
        # o valor aleatório a ser adicionado ao gene
        random_value = numpy.random.uniform(-1.0, 1.0, 1)
        random_idx = numpy.random.randint(offspring_crossover.shape[1])
        offspring_crossover[idx, random_idx] = offspring_crossover[idx, random_idx] + random_value

    return offspring_crossover
