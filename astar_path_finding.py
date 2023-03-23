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
import heapq
import numpy as np
import math

def generate_actions(map_name, seed, start_pos, goal_pos):
    # load map
    map_path = './gym-duckietown/gym_duckietown/map_2021/' + map_name + '.yaml'
    with open(map_path, "r") as stream:
        map_data = yaml.safe_load(stream)
    # can access like dictionary
    # map_data['tiles']
    # map_data['objects']
    # map_data['tile_size']
    map_tiles = map_data['tiles']
    '''
    duckieの向きが分からないので例えば上に行きたいときforwardなのかleftなのか分からない
    なのでまずup, left, right, downでaction取得
    その後最初のactionをforwardにしてforward, left, rightに
    その後forward, left, rightを(linear velocity, angular velocity)に
    1. (x, y): up=(0,1), down, left, right
    2. (lin_vel, ang_vel): forward=(l, a)...
    まず1で取得して2に変換
    '''
    '''
    coordinate system
    start_pos and goal_pos is in x, y
    map is in row, col
    need to convert x, y to row, col
    origin of x, y is top left
    row, col = y, x
    '''
    # up means one row up in 2d array
    # (row, col)
    up = (-1, 0)
    down = (1, 0)
    left = (0, -1)
    right = (0, 1)
    stay = (0, 0)
    actions = [up, down, left, right]
    # start_pos and goal_pos is in x, y so need to convert to row, col
    start_pos_rowcol = (start_pos[1], start_pos[0])
    goal_pos_rowcol = (goal_pos[1], goal_pos[0])
    # define tile types which are obstacle
    obstacles = ['empty', 'asphalt', 'grass', 'floor']

    # Node to store x, y, cost, heuristic, and parent
    # parent is needed to check path
    class Node:
        def __init__(self, state, action=stay, g=0, h=0):
            self.state = state
            # action took from parent to this node
            self.action = action
            self.g = g # cost
            self.h = h # heuristic
            self.parent = None
        
        def f(self):
            return self.g + self.h
        
        # check self is less than other
        def __lt__(self, other):
            return self.f() < other.f()

    # use priority queue to store states
    open_list = []
    closed_list = set()
    heapq.heappush(open_list, Node(start_pos_rowcol))
    action_seq = []
    # iterate each state in open_list 
    while open_list:
        # pick state with lowest value of cost + heuristics
        cur = heapq.heappop(open_list)

        # check if goal is reached
        cur_position = cur.state
        if cur_position == goal_pos_rowcol:
            # traver from leaf to parent, since order should parent->lead, reverse 
            reversed_action_seq = []
            while cur:
                reversed_action_seq.append(cur.action)
                cur = cur.parent
            action_seq = list(reversed(reversed_action_seq))   
            # root node doesnt contain action so remove it (because root dont have parent)             
            action_seq.pop(0)
            break

        closed_list.add(cur.state)           
        
        # consider every possible action (except stay)
        for action in actions:
            # get next post based on current tate and action
            next_pose = (cur.state[0] + action[0], cur.state[1] + action[1])

            # check next_pose is within in boundary
            if not (0 <= next_pose[0] < len(map_tiles) and 0 < next_pose[1] < len(map_tiles[0])):
                continue
            # check next_pose is not obstacle
            if map_tiles[next_pose[0]][next_pose[1]] in obstacles:
                continue
            # check is already visited
            if next_pose in closed_list:
                continue

            # calculate cost and heuristic
            cost = cur.g + 1
            heuristic = abs(next_pose[0] - goal_pos_rowcol[0]) + abs(next_pose[1] - goal_pos_rowcol[1])
            next_node = Node(next_pose, action, cost, heuristic)
            next_node.parent = cur
            # if same already exists and f is lower, ignore next_node
            lowerExists = False
            for node in open_list:
                if node.state == next_node.state and node.f() < next_node.f():
                    lowerExists = True
                    break
            if lowerExists:
                continue
            # add to openlist
            heapq.heappush(open_list, next_node)

    return action_seq

def convert_actions(actions):
    new_actions = []
    # forward = (0, 0)
    # left = (0, 0)
    # right = (0, 0)
    forward = "forward"
    left = "left"
    right = "right"
    prev_action = None
    for action in actions:
        # initial action
        if not prev_action:
            new_actions.append(forward)
        # if current action is same as prev action, go foward
        # ex: down -> down means no need to change direction
        elif prev_action == action:
            new_actions.append(forward)
        # if current action is different from prev action, need to rotate
        # ex: up -> right, need to change direction by 90 deg to right before moving forward
        else:
            # rotation
            # action is in [row, col] so convert to x, y (bottom left origin)
            # bottom left origin, not top left origin
            action_xy = (action[1], -action[0])
            prev_action_xy = (prev_action[1], -prev_action[0])
            # find angle between two vector
            vector1 = np.array(list(prev_action_xy))
            vector2 = np.array(list(action_xy))
            # Compute the angle between the two vectors
            dot_product = vector1[0]*vector2[0] + vector1[1]*vector2[1]
            magnitude_product = math.sqrt(vector1[0]**2 + vector1[1]**2) * math.sqrt(vector2[0]**2 + vector2[1]**2)
            angle = math.acos(dot_product / magnitude_product)
            # Compute the direction of the rotation (clockwise or counterclockwise)
            cross_product = vector1[0]*vector2[1] - vector1[1]*vector2[0]
            if cross_product < 0:
                # clockwise rotation
                angle_degrees = -int(np.degrees(angle))
            else:
                # counterclockwise rotation
                angle_degrees = int(np.degrees(angle))
            
            # Convert angle to degrees
            if angle_degrees == 90:
                new_actions.append(left)
            elif angle_degrees == -90:
                new_actions.append(right)
            else:
                raise ValueError("invalid degrees angle: " + str(angle_degrees)) 
            # go forward
            new_actions.append(forward)
        # update prev_action
        prev_action = action
    return new_actions

def compare_actions(actions, solution_path):
    with open(solution_path, 'r') as file:
        i = -1
        for line in file:
            # skip first action
            if i == -1:
                i += 1
                continue
            # ex: (1, 6), forward
            # after split: ' forward\n'
            # - extra space at front and \n at back
            correct_move = line.split(',')[-1][1:-1]
            if i >= len(actions):
                print('index is out of range of actions')
                return False
            my_move = actions[i]
            if correct_move != my_move:
                print('my move does not match with correct move')
                return False
            if my_move == 'forward':
                i += 1
            else: 
                '''
                solution: left = rotation + forward
                my move: left = rotation, forward
                in my move, left and right is followed by forward every time
                while forward is included inside left in solution
                so I need to skip forward after left to match with solution
                '''
                i += 2
    return True

# testcases/milestone1.json
testcases_path = './testcases/milestone1.json'
# Load the JSON file
with open(testcases_path, 'r') as file:
    testcases = json.load(file)

# Iterate through the items in the dictionary
for map_name, test_data in testcases.items():
    seed = test_data['seed']
    start_pos = test_data['start']
    goal_pos = test_data['goal']
    # generate action in up, down, left ,right
    actions = generate_actions(map_name, seed, start_pos, goal_pos)
    # convert actionfrom up... to lin_vel, ang_vel
    actions = convert_actions(actions)
    output_folder_path = './milestone1_controls'

    # check actions generated against solution(?)
    solution_filename = f'{map_name}_seed{seed[0]}_start_{start_pos[0]},{start_pos[1]}_goal_{goal_pos[0]},{goal_pos[1]}.txt'
    solution_path = 'testcases/milestone1_paths/' + solution_filename
    result = compare_actions(actions, solution_path)
    if result:
        print("success: " + solution_filename)
    else:
        print(solution_filename)
        print(actions)
        raise ValueError("test failed")

    # output   
    for action in actions:
        # print(action)
        # write it to output file
        pass

    # try with only first test case to check code is working properly
    # break 