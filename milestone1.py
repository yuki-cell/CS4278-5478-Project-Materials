'''
load test cases, milestone.json
loop each test cases
for each tese case
- load map
- do A* or other algo
- put actions to controls file and save
'''
'''
map1
- grassがobstacle?
map2
- floor (office floor)がobstacle? 歩道?
map3
- asphaltとgrassがobstacleっぽい
map4
- skip
map5
- skip
all available tile types are in 
https://github.com/yuki-cell/CS4278-5478-Project-Materials/tree/master/gym-duckietown#Map-File-Format
in Design, Map File Format
見た感じ
obstacle = empty, asphalt, grass, floor (office floor)
それ以外=valid path=straight, curve_left, curvee_right, 3way_left (3-way intersection), 3way_right, 4way (4-way intersection)
'''
import json
import yaml


def generate_actions(map_name, seed, start_pos, goal_pos):
    # load map
    map_path = './gym-duckietown/gym_duckietown/map_2021' + map_name + '.yaml'
    with open(map_path, "r") as stream:
        map_data = yaml.safe_load(stream)
    # can access like dictionary
    map_data['tiles']
    map_data['objects']
    map_data['tile_size']

testcases_path = './testcases/milestone1.json'
# Load the JSON file
with (testcases_path, 'r') as file:
    testcases = json.load(file)

# Iterate through the items in the dictionary
for map_name, test_data in testcases.items():
    seed = test_data['seed']
    start_pos = test_data['start']
    goal_pos = test_data['goal']
    actions = generate_actions(map_name, seed, start_pos, goal_pos)
    output_folder_path = './milestone1_controls'
    for action in actions:
        # write it to output file
        pass

    # try with only first test case to check code is working properly
    break 