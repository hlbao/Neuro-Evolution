import logging
from optimizer import Optimizer
from tqdm import tqdm

# Setup logging.
logging.basicConfig(
    format='%(asctime)s - %(levelname)s - %(message)s',
    datefmt='%m/%d/%Y %I:%M:%S %p',
    level=logging.DEBUG,
    filename='log.txt'
)

def train_networks(networks, dataset):
    pbar = tqdm(total=len(networks))
    for network in networks:
        network.train(dataset)
        pbar.update(1)
    pbar.close()

def get_average_accuracy(networks):
    total_accuracy = 0
    for network in networks:
        total_accuracy += network.accuracy

    return total_accuracy / len(networks)

def generate(generations, population, nn_param_population, dataset):
    
    optimizer = Optimizer(nn_param_population)
    networks = optimizer.create_population(population)

    # Evolve the generation.
    for i in range(generations):
        logging.info(i + 1, generations)

        # Train and get accuracy for networks.
        train_networks(networks, dataset)

        # Get the average accuracy for this generation.
        average_accuracy = get_average_accuracy(networks)

        # Print out the average accuracy each generation.
        logging.info(average_accuracy * 100)
        logging.info('-'*80)

        # Evolve, except on the last iteration.
        if i != generations - 1:
            # Do the evolution.
            networks = optimizer.evolve(networks)

    # Sort final population.
    networks = sorted(networks, key=lambda x: x.accuracy, reverse=True)
    print_networks(networks[:5])

def print_networks(networks):
    logging.info('-'*80)
    for network in networks:
        network.print_network()

def main():
    generations = 10  # Number of times to evole the population.
    population = 20  # Number of network population in each generation.
    dataset = 'fer2013'

    nn_param_population = {
        'nb_neurons': [32, 64, 128, 256, 512, 768, 1024],
        'nb_layers': [1, 2, 3, 4],
        'activation': ['elu', 'relu', 'sigmoid', 'tanh'],
        'optimizer': ['sgd', 'adam', 'adagrad',
                      'adadelta', 'adamax', 'nadam', 'rmsprop']
    }

    logging.info((generations, population))

    generate(generations, population, nn_param_population, dataset)

if __name__ == '__main__':
    main()
