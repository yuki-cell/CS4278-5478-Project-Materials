'''
goal of milestone 1 is just to convert control txt files under testcases/milestone1_paths to control files with (float, float) format so can use it in simulation
ex: control txt file under root folder is the correct format we need to submit
so in milestone1, I only need to convert files
no need to implement algorithm
'''

import json
import numpy as np

# testcases/milestone1.json
testcases_path = './testcases/milestone1.json'
# Load the JSON file
with open(testcases_path, 'r') as file:
    testcases = json.load(file)

# Iterate through the items in the dictionary
for map_name, test_data in testcases.items():
    seed = test_data['seed'][0]
    start_pos = test_data['start']
    goal_pos = test_data['goal']

    # Load the given path txt file
    filename = f'{map_name}_seed{seed}_start_{start_pos[0]},{start_pos[1]}_goal_{goal_pos[0]},{goal_pos[1]}.txt'
    given_filepath = './testcases/milestone1_paths/' + filename
    given_moves = []
    with open(given_filepath, 'r') as file:
        # each line contains action in each step
        i = -1
        for line in file:
            # skip first action
            if i == -1:
                i += 1
                continue
            # ex: (1, 6), forward
            # after split: ' forward\n'
            # - extra space at front and \n at back
            move = line.split(',')[-1][1:-1]
            '''
            run.pyの[lin_vel, ang_vel]を参考にする
            if key_handler[key.UP]:
                action = np.array([0.44, 0.0])
            if key_handler[key.DOWN]:
                action = np.array([-0.44, 0])
            if key_handler[key.LEFT]:
                action = np.array([0.35, +1])
            if key_handler[key.RIGHT]:
                action = np.array([0.35, -1])
            if key_handler[key.SPACE]:
                action = np.array([0, 0])
            '''
            if move == 'forward':
                action = '1,0'
            elif move == 'left':
                action = '0,1'
            elif move == 'right':
                action = '0,-1'
            else:
                raise ValueError('invalid move in given path txt file') 
            given_moves.append(action)

    # create new controls txt file
    my_filepath = './milestone1_controls/' + filename
    with open(my_filepath, "w") as f:
        for move in given_moves:
            f.write(f'{move}\n')
    
    # try only first item
    break