import numpy as np
import random
import cv2


class Genome:
    def getRandomTraits():
        return (random.randint(0, GeneticAlgorithm.COLS-1),
                random.randint(0, GeneticAlgorithm.ROWS-1),
                random.randint(
                    0, (min(GeneticAlgorithm.ROWS, GeneticAlgorithm.COLS)-1)//2)
                )

    def mutateGenome(genome):
        if (random.random() < GeneticAlgorithm.MUTATION_RATE):
            return Genome(tuple([max(1, trait + random.randint(-GeneticAlgorithm.MUTATION_RADIUS, GeneticAlgorithm.MUTATION_RADIUS)) for trait in genome.traits]))

        return genome

    def __init__(self, traits=None):
        self.traits = traits if traits != None else Genome.getRandomTraits()


class Individual:
    def __init__(self, genome=None):
        self.genome = Genome.mutateGenome(
            genome) if genome != None else Genome()
        self.fitness = GeneticAlgorithm.getFitness(self)

    def cross_breed(A, B):
        newTraits = tuple([A.genome.traits[i] if random.random(
        ) < 0.5 else B.genome.traits[i] for i, _ in enumerate(A.genome.traits)])
        newGenome = Genome(newTraits)
        child1 = Individual(genome=newGenome)

        newTraits = tuple([A.genome.traits[i] if random.random(
        ) < 0.5 else B.genome.traits[i] for i, _ in enumerate(A.genome.traits)])
        newGenome = Genome(newTraits)
        child2 = Individual(genome=newGenome)

        return child1, child2


class GeneticAlgorithm:
    POP_SIZE = 10
    MAX_GENERATIONS = 20
    MUTATION_RATE = 0.1

    MUTATION_RADIUS = 20

    TOURNAMENT_WHEEL_PROB = 0.5  # 0 - always wheel, 1 - always Tournament

    canvas = []

    COLS, ROWS = 0, 0

    def getFitness(individual: Individual, draw=False, overide=False):
        newCanvas = GeneticAlgorithm.canvas.copy()

        cx, cy, radius = individual.genome.traits

        ul_corner_x, br_corner_x = max(
            cx - radius, 0), min(cx + radius, GeneticAlgorithm.COLS - 1)
        ul_corner_y, br_corner_y = max(
            cy - radius, 0), min(cy + radius, GeneticAlgorithm.ROWS - 1)

        # Get Avg Color
        avgColor = GeneticAlgorithm.source[ul_corner_y: br_corner_y, ul_corner_x: br_corner_x].mean(
            axis=0).mean(axis=0).astype(np.uint8).tolist()

        newCanvas = cv2.rectangle(
            newCanvas, (ul_corner_x, ul_corner_y), (br_corner_x, br_corner_y), tuple(avgColor), -1)

        if draw:
            cv2.imshow("newCanvas", newCanvas)
            cv2.waitKey(1)

        if overide:
            GeneticAlgorithm.canvas = newCanvas.copy()

        diff = cv2.subtract(GeneticAlgorithm.source / 255, newCanvas / 255)

        return 1/np.sum(np.square(diff))

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

    def __init__(self, sourceImage):
        GeneticAlgorithm.ROWS, GeneticAlgorithm.COLS, _ = sourceImage.shape
        GeneticAlgorithm.canvas = np.zeros(sourceImage.shape, np.uint8)
        GeneticAlgorithm.source = sourceImage

        self.move = 0
        while 1:
            self.gen = 0

            # Instantiate Population
            self.population = [Individual()
                               for i in range(GeneticAlgorithm.POP_SIZE)]

            while self.exitCondition(self.gen):
                # Train Loop
                self.trainLoop()

                GeneticAlgorithm.getFitness(self.fittest, True)
                print(
                    f'{self.move} - {self.gen}/{GeneticAlgorithm.MAX_GENERATIONS} - {self.fittest.fitness} - {self.fittest.genome.traits} - {len(self.population)}')

                self.gen += 1

            GeneticAlgorithm.getFitness(self.fittest, True, True)

            self.move += 1


sourceImage = cv2.imread('./source.jpg')
sourceImage = cv2.resize(sourceImage, (300, 300))

cv2.imshow("Source", sourceImage)

ga = GeneticAlgorithm(sourceImage)
