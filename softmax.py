import math as m
import argparse


def softmax(x0, x1):
    t1 = m.exp(float(x0)) / (m.exp(float(x0)) + m.exp(float(x1)))
    t2 = m.exp(float(x1)) / (m.exp(float(x1)) + m.exp(float(x1)))

    return t1, t2

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--value1",
    )
   
    parser.add_argument(
        '--value2',
    )
    
    hyperparams = parser.parse_args()

 
    x0 = hyperparams.value1
    x1 = hyperparams.value2
    #resolution = hyperparams.resolution
    
    #x=0
    t1 , t2 = softmax(x0, x1)
    print(f'Softmax activation for {x0} is {t1}')
    print(f'Softmax activation for {x1} is {t2}')


    
        
