from functools import reduce
from operator import add
import random
from network import Network
from deap import creator, base, tools, algorithms


class Optimizer():

    def __init__(self, nn_param_population, retain=0.4,
                 random_select=0.1, mutate_chance=0.2):
       

        self.mutate_chance = mutate_chance
        self.random_select = random_select
        self.retain = retain
        self.nn_param_population = nn_param_population

    def create_population(self, count):
      
        pop = []
        for _ in range(0, count):
            # Create a random network.
            network = Network(self.nn_param_population)
            network.create_random()

            # Add the network to our population.
            pop.append(network)

        return pop

    def fitness(network):
        return network.accuracy

    def grade(self, pop):
  
        summed = reduce(add, (self.fitness(network) for network in pop))
        return summed / float((len(pop)))

    def breed(self, mother, father):
       
        children = []
        for _ in range(2)#two paretns generate one offspring:

            child = {}
            for param in self.nn_param_population:
                child[param] = random.choice(
                    [mother.network[param], father.network[param]]
                )

          
            network = Network(self.nn_param_population)
            network.create_set(child)
            if self.mutate_chance > random.random():
                network = self.mutate(network)

            children.append(network)

        return children

    def mutate(self, network):

        mutation = random.choice(list(self.nn_param_population.keys()))
        network.network[mutation] = random.choice(self.nn_param_population[mutation])

        return network

    def evolve(self, pop):
        graded = [(self.fitness(network), network) for network in pop]

        # Sort based on the scores we graded
        graded = [x[1] for x in sorted(graded, key=lambda x: x[0], reverse=True)]
        retain_length = int(len(graded)*self.retain)#top 40%
        parents = graded[:retain_length]
        for individual in graded[retain_length:]:
            if self.random_select > random.random():
                parents.append(individual) #6%

        parents_length = len(parents)
        desired_length = len(pop) - parents_length
        children = []

    
        while len(children) < desired_length:

            # Get a random mother and father 
            male = random.randint(0, parents_length-1)
            female = random.randint(0, parents_length-1)

          
            if male != female:
                male = parents[male]
                female = parents[female]

                # Breed
                babies = self.breed(male, female)
                for baby in babies:
                    if len(children) < desired_length:
                        children.append(baby)

        parents.extend(children)

        return parents
        
        
        
