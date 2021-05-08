import pandas as pd

weight1=1
weight2=1
bias=-1

inputs=[(0,0),(0,1),(1,0),(1,1)]
outputs=[False,True,True,True]

together=[]

for input, output in zip(inputs, outputs):
    print(input,output)
    result = weight1*input[0] + weight2*input[1]+bias
    x = int(result >= 0)
    """print(x)"""
    is_correct_string= 'Yes' if x == output else 'No'
    print(is_correct_string)
    together.append([input[0], input[1], result, x, is_correct_string])
output_frame= pd.DataFrame(together,columns=['Input 1','Input 2','Linear Combination','output','is_correct_string'])
print(output_frame)


