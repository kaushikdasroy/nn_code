import math
import argparse


def sigmoid(x):
    t1 = 1 / (1+ math.exp (-float(x)))
    t2 = 1 - t1
    return t1, t2

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
    t1 , t2 = sigmoid(x)
    print(f'Sigmoid activation for {x} to be part of one class is  {t1}')
    print(f'Sigmoid activation for {x} to be part of onother class is  {t2}')



    
        
