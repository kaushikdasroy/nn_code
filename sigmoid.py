import math
import argparse


def sigmoid(x):
    t = 1 / (1+ math.exp (-float(x)))
    return t

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--value",
    )
    """
    parser.add_argument(
        '--resolution', default=1024
    )
    """
    hyperparams = parser.parse_args()

    x = hyperparams.value
    #resolution = hyperparams.resolution
    
    #x=0
    print(f'Sigmoid activation for {x} is {sigmoid(x)}')


    
        
