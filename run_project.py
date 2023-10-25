import os
from warnings import filterwarnings

import pretty_errors

filterwarnings(action='ignore')

print('\n1: Train')
print('2: Test')
print('3: Predict')

script_choice = input("\nTask to perform : ").strip()
try:
    script_choice = int(script_choice)
    if script_choice not in {1, 2, 3}:
        raise Exception
except:
    raise Exception("Input number 1, 2 or 3")

if script_choice == 1:
    os.system('python core/drivers/run_train.py')
elif script_choice == 2:
    os.system('python core/drivers/run_test.py')
elif script_choice == 3:
    os.system('python core/drivers/run_predict.py')