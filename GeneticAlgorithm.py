import numpy as np
import random
import threading


class Individual:
    LATEST_ID = 0

    def __init__(self, genome=None):
        self.genome = Individual.mutateGenome(
            genome) if genome != None else Individual.getRandomGenome()
        self.fitness = GeneticAlgorithm.getFitness(self)

        self.id = Individual.LATEST_ID
        Individual.LATEST_ID += 1

    def getRandomGenome():
        return random.random()

    def mutateGenome(genome):
        if (random.random() < GeneticAlgorithm.MUTATION_RATE):
            return Individual.getRandomGenome()

        return genome

    def cross_breed(A, B):
        return Individual(genome=random.choice((A, B)).genome)

    def __eq__(self, other):
        return self.id == other.id

    def __str__(self) -> str:
        return f'{self.id}: {self.fitness:.2f} - {self.genome}'


class GeneticAlgorithm:
    POP_SIZE = 1000
    MAX_GENERATIONS = 5
    MUTATION_RATE = 0.1

    TOURNAMENT_WHEEL_PROB = 0.5  # 0 - always wheel, 1 - always Tournament

    N_THREADS = 4

    def getFitness(individual: Individual):
        return individual.genome

    def exitCondition(self, gen):
        return gen < self.MAX_GENERATIONS

    def multidimensional_shifting(num_samples, sample_size, elements, probabilities):
        # replicate probabilities as many times as `num_samples`
        replicated_probabilities = np.tile(probabilities, (num_samples, 1))
        # get random shifting numbers & scale them correctly
        random_shifts = np.random.random(replicated_probabilities.shape)
        random_shifts /= random_shifts.sum(axis=1)[:, np.newaxis]
        # shift by numbers & find largest (by finding the smallest of the negative)
        shifted_probabilities = random_shifts - replicated_probabilities
        indices = np.argpartition(shifted_probabilities, sample_size, axis=1)[
            :, :sample_size]

        # get the elements
        return np.array([[elements[j] for j in i]for i in indices])

    def tournamentSelection(self):
        candidate1 = GeneticAlgorithm.multidimensional_shifting(
            1, 1, self.population, self.UNIFORM_PROBS)[0][0]
        candidate2 = GeneticAlgorithm.multidimensional_shifting(
            1, 1, self.population, self.UNIFORM_PROBS)[0][0]

        while (candidate1 == candidate2):
            candidate2 = GeneticAlgorithm.multidimensional_shifting(
                1, 1, self.population, self.UNIFORM_PROBS)[0][0]

        if (candidate1.fitness > candidate2.fitness):
            return candidate1

        return candidate2

    def updateBiesedWheelParameters(self):
        weights = [member.fitness for member in self.population]
        self.totalWeights = sum(weights)
        self.probabilities = [w / self.totalWeights for w in weights]

    def biasedWheelSelection(self):
        return GeneticAlgorithm.multidimensional_shifting(1, 1, self.population, self.probabilities)[0][0]

    def chooseParent(self):
        if (random.random() < GeneticAlgorithm.TOURNAMENT_WHEEL_PROB):
            return self.tournamentSelection()

        return self.biasedWheelSelection()

    def updateStatistics(self):
        self.fittest = self.population[0]
        self.averageFitness = sum(
            [member.fitness for member in self.population]) / GeneticAlgorithm.POP_SIZE
        self.stdFitness = np.std(
            [member.fitness for member in self.population])

    def logStatistics(self):
        # print number of generation, fittest, average fitness, standard deviation up to 2 decimal points
        print(
            f'Gen:{self.gen} - Fittest: [{self.fittest}] - Average Fitness:{self.averageFitness:.2f} - Standard Deviation:{self.stdFitness:.2f}')

    def trainLoop(self):
        newPopulation = []

        self.updateBiesedWheelParameters()

        def breed(n):
            for _ in range(n):
                newPopulation.append(Individual.cross_breed(
                    self.chooseParent(), self.chooseParent()))

        # multithreading breeding
        # threads = [threading.Thread(target=breed, args=(
        #     GeneticAlgorithm.POP_SIZE // GeneticAlgorithm.N_THREADS,)) for i in range(GeneticAlgorithm.N_THREADS)]

        # for thread in threads:
        #     thread.start()

        # for thread in threads:
        #     thread.join()

        breed(GeneticAlgorithm.POP_SIZE)

        # Add new population to old population
        self.population = self.population + newPopulation

        # Sort Population
        self.population = sorted(
            self.population, key=lambda x: x.fitness, reverse=True)

        # Kill the weak
        self.population = self.population[:GeneticAlgorithm.POP_SIZE]

        self.updateStatistics()

        self.logStatistics()

    def initializePopulation(self):
        self.population = []

        def addIndividual(n):
            for _ in range(n):
                self.population.append(Individual())

        # multithreading population initialization
        # threads = [threading.Thread(target=addIndividual, args=(
        #     GeneticAlgorithm.POP_SIZE // GeneticAlgorithm.N_THREADS,)) for i in range(GeneticAlgorithm.N_THREADS)]

        # for thread in threads:
        #     thread.start()

        # for thread in threads:
        #     thread.join()

        addIndividual(GeneticAlgorithm.POP_SIZE)

    def __init__(self):
        self.UNIFORM_PROBS = [
            1/GeneticAlgorithm.POP_SIZE for _ in range(GeneticAlgorithm.POP_SIZE)]

        # Instantiate Population
        self.initializePopulation()

        self.gen = 0

        while self.exitCondition(self.gen):
            # Train Loop
            self.trainLoop()

            self.gen += 1

        pass


if __name__ == "__main__":
    import cProfile
    import pstats

    cProfile.run('GeneticAlgorithm()', 'stats')
    p = pstats.Stats('stats')
    p.sort_stats('cumulative').print_stats(10)

    # python -m snakeviz stats.prof --server  # for visualizing the stats
    p.dump_stats('stats.prof')
