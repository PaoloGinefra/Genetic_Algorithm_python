import numpy as np
import random


class Individual:
    def __init__(self, genome=None):
        self.genome = Individual.mutateGenome(
            genome) if genome != None else Individual.getRandomGenome()
        self.fitness = GeneticAlgorithm.getFitness(self)

    def getRandomGenome():
        return random.random()

    def mutateGenome(genome):
        if (random.random() < GeneticAlgorithm.MUTATION_RATE):
            return Individual.getRandomGenome()

        return genome

    def cross_breed(A, B):
        return Individual(genome=A.genome), Individual(genome=B.genome)


class GeneticAlgorithm:
    POP_SIZE = 10
    MAX_GENERATIONS = 100
    MUTATION_RATE = 0.1

    TOURNAMENT_WHEEL_PROB = 0.5  # 0 - always wheel, 1 - always Tournament

    def getFitness(individual: Individual):
        return random.random()

    def exitCondition(self, gen):
        return gen < self.MAX_GENERATIONS

    def tournamentSelection(self):
        candidate1 = np.random.choice(self.population, 1)[0]
        candidate2 = np.random.choice(self.population, 1)[0]

        while (candidate1 == candidate2):
            candidate2 = np.random.choice(self.population, 1)[0]

        if (candidate1.fitness > candidate2.fitness):
            return candidate1

        return candidate2

    def biasedWheelSelection(self):
        weights = [member.fitness for member in self.population]
        totalWeights = sum(weights)
        probabilities = [w / totalWeights for w in weights]
        return np.random.choice(self.population, 1, probabilities)[0]

    def chooseParent(self):
        if (random.random() < GeneticAlgorithm.TOURNAMENT_WHEEL_PROB):
            return self.tournamentSelection()

        return self.biasedWheelSelection()

    def trainLoop(self):
        # Breeding
        for i in range(GeneticAlgorithm.POP_SIZE // 2):
            parent1, parent2 = self.chooseParent(), self.chooseParent()

            while (parent1 == parent2):
                parent2 = self.chooseParent()

            child1, child2 = Individual.cross_breed(
                parent1, parent2)

            self.population.append(child1)
            self.population.append(child2)

        # Sort Population
        self.population = sorted(
            self.population, key=lambda x: x.fitness, reverse=True)

        # Kill the weak
        self.population = self.population[:GeneticAlgorithm.POP_SIZE]

        self.fittest = self.population[0]

    def __init__(self):
        # Instantiate Population
        self.population = [Individual()
                           for i in range(GeneticAlgorithm.POP_SIZE)]

        self.gen = 0

        while self.exitCondition(self.gen):
            # Train Loop
            self.trainLoop()

            print(f'{self.gen} - {self.fittest.fitness}')

            self.gen += 1

        pass


ga = GeneticAlgorithm()
