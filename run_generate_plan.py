import argparse

import numpy as np
import pyglet
from gym_duckietown.envs import DuckietownEnv
from pyglet.window import key
import sys
import cv2

import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import math
import time

def str2bool(v):
    """
    Reads boolean value from a string
    """
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')

# declare the arguments
parser = argparse.ArgumentParser()

# Do not change this
parser.add_argument('--max_steps', type=int, default=1500, help='max_steps')

# You should set them to different map name and seed accordingly
parser.add_argument('--map-name', '-m', default="map4_0", type=str)
parser.add_argument('--seed', '-s', default=2, type=int)
parser.add_argument('--start-tile', '-st', default="1,13", type=str, help="two numbers separated by a comma")
parser.add_argument('--goal-tile', '-gt', default="3,3", type=str, help="two numbers separated by a comma")
parser.add_argument('--control_path', default='./map4_0_seed2_start_1,13_goal_3,3.txt', type=str,
                    help="the control file to run")
parser.add_argument('--manual', default=False, type=str2bool, help="whether to manually control the robot")
args = parser.parse_args()


# simulator instantiation
env = DuckietownEnv(
    domain_rand=False,
    max_steps=1500,
    map_name=args.map_name,
    seed=args.seed,
    user_tile_start=args.start_tile,
    goal_tile=args.goal_tile,
    randomize_maps_on_reset=False
)

# obs = env.reset() # WARNING: never call this function during testing
env.render()

map_img, goal, start_pos = env.get_task_info()
print("start tile:", start_pos, " goal tile:", goal)

# Show the map image
# White pixels are drivable and black pixels are not.
# Blue pixels indicate lan center
# Each tile has size 100 x 100 pixels
# Tile (0, 0) locates at left top corner.
cv2.imshow("map", map_img)
cv2.waitKey(200)

# save map (example)
# cv2.imwrite(env.map_name + ".png", env.get_occupancy_grid(env.map_data))

# main loop
if args.manual:
    # Register a keyboard handler
    key_handler = key.KeyStateHandler()
    env.unwrapped.window.push_handlers(key_handler)

    def update(dt):
        action = np.array([0.0, 0.0])

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

        # Speed boost when pressing shift
        if key_handler[key.LSHIFT]:
            action *= 3

        obs, reward, done, info = env.step(action)
        print(f"current pose = {info['curr_pos']}, step count = {env.unwrapped.step_count}, step reward = {reward:.3f}")

        env.render()

    pyglet.clock.schedule_interval(update, 1.0 / env.unwrapped.frame_rate)
    pyglet.app.run()

else:
    # remove every file in logs evertime before running
    import os, shutil
    folder = './milestone1_logs'
    # iterate every file in folder
    for filename in os.listdir(folder):
        file_path = os.path.join(folder, filename)
        try:
            os.unlink(file_path)
        except Exception as e:
            print('Failed to delete %s. Reason: %s' % (file_path, e))

    def adjust_rotation():
        # adjust initial pose of robot to face next tile
        # set limit to avoid infinite loop
        i = 0
        while i < 10:
            i += 1
            # access current robot's information
            speed = 0
            steering = 0
            obs, reward, done, info = env.step([speed, steering])

            # color filtering 
            # yellow lane for left lane
            # white lane for right lane
            # blur the image to remove the noise (ex: grass section)
            # map1: (7, 7) well for green noise
            img_blur = cv2.GaussianBlur(obs,(11,11),0)
            # It converts the BGR color space of image to HSV color space
            # BGR2HSVとRGB2HSVがある, BGRじゃなくRGBが正解っぽい
            hsv = cv2.cvtColor(img_blur, cv2.COLOR_RGB2HSV)
            
            # Threshold in HSV space
            # white lane, yellow lane
            # white (and gray)
            lower_white = np.array([0,0,0])
            upper_white = np.array([255,10,255])
            # yellow
            lower_yellow = np.array([20,80,80])
            upper_yellow = np.array([30,255,255])
        
            # preparing the mask to overlay
            mask_white = cv2.inRange(hsv, lower_white, upper_white)
            mask_yellow = cv2.inRange(hsv, lower_yellow, upper_yellow)

            
            # The black region in the mask has the value of 0,
            # so when multiplied with original image removes all non-blue regions
            result_white = cv2.bitwise_and(obs, obs, mask = mask_white)
            result_yellow = cv2.bitwise_and(obs, obs, mask = mask_yellow)
            # plt.figure()
            # plt.title('result white')
            # plt.imshow(result_white)
            # filename = 'milestone1_logs/white_' + str(time.time())+ '.png'
            # plt.savefig(filename)
            # plt.close()

            # plt.figure()
            # plt.title('result yellow')
            # plt.imshow(result_yellow)
            # filename = 'milestone1_logs/yellow_' + str(time.time())+ '.png'
            # plt.savefig(filename)
            # plt.close()

            # Call Canny Edge Detection here.
            # convert to grayscale here.
            # yellow lane for left lane
            yellow_gray_image = cv2.cvtColor(result_yellow, cv2.COLOR_RGB2GRAY)
            # white lane for right lane
            white_gray_image = cv2.cvtColor(result_white, cv2.COLOR_RGB2GRAY)
            # use blurred image to select threshold for canny
            def extract_canny_edges(img):
                img_blur = cv2.GaussianBlur(img,(11,11),0)
                med_val = np.median(img_blur)
                sigma = 0.33  # 0.33
                min_val = int(max(0, (1.0 - sigma) * med_val))
                max_val = int(max(255, (1.0 + sigma) * med_val))
                cannyed_img = cv2.Canny(img_blur, min_val, max_val)
                return cannyed_img
            
            yellow_cannyed_image = extract_canny_edges(yellow_gray_image)
            white_cannyed_image = extract_canny_edges(white_gray_image)
            # plt.figure()
            # plt.title('left cannyed imaged')
            # plt.imshow(left_cannyed_image)
            # plt.figure()
            # plt.title('right cannyed imaged')
            # plt.imshow(right_cannyed_image)
            # filename = 'milestone1_logs/' + str(time.time())+ '.png'
            # plt.savefig(filename)
            # plt.close()

            # 下半分だけcropする?
            def region_of_interest(img, vertices):
                mask = np.zeros_like(img)
                match_mask_color = 255 # <-- This line altered for grayscale.
                cv2.fillPoly(mask, vertices, match_mask_color)
                masked_image = cv2.bitwise_and(img, mask)
                return masked_image
            # middle half
            height, width = yellow_cannyed_image.shape[:2]
            region_of_interest_vertices = [
                # order of verticies is important
                # make sure that shape if drawn by connecting points in order of verticies
                # ex: top left -> bottom right: invalid square
                # ex: top left -> bottom left: valid square
                (0, 100),
                (width, 100),
                (width, 450),
                (0, 450)   
            ]
            yellow_cannyed_image = region_of_interest(
                yellow_cannyed_image,
                np.array([region_of_interest_vertices], np.int32)
            )
            white_cannyed_image = region_of_interest(
                white_cannyed_image,
                np.array([region_of_interest_vertices], np.int32)
            )
            # plt.figure()            
            # plt.title('cropped image')
            # plt.imshow(cropped_cannyed_image)

            # detect line from edges
            def detect_lines(img, minLineLength, maxLineGap):
                lines = cv2.HoughLinesP(
                    img,
                    rho=6,
                    theta=np.pi / 60,
                    # only accpect line with # of votes > threshold
                    # initial: 160
                    threshold=80,
                    lines=np.array([]),
                    #minLineLength: 元々40
                    #長さ増やすことでいい感じのlaneだけ取れる(画像によって変わりそうやから微調整は必要そう)
                    minLineLength=minLineLength,
                    # default: 25
                    maxLineGap=maxLineGap
                )
                if lines is None:
                    return []
                else:
                    return lines
            # different line setting for white and yellow lane
            # yellow lane is separated, gap exists
            # ここの値調整めんどくさい?
            # ex: map2 passするとmap4がfailしたり, vice versa
            # yellow_lines = detect_lines(yellow_cannyed_image, minLineLength=30, maxLineGap=25)
            # # white lane is connected, no gap
            # white_lines = detect_lines(white_cannyed_image, minLineLength=40, maxLineGap=10)
            
            # 上より緩い条件の値
            yellow_lines = detect_lines(yellow_cannyed_image, minLineLength=25, maxLineGap=25)
            white_lines = detect_lines(white_cannyed_image, minLineLength=25, maxLineGap=25)

            def draw_lines(img, lines, color=[255, 0, 0], thickness=3):
                # If there are no lines to draw, exit.
                if lines is None:
                    return
                # Make a copy of the original image.
                img = np.copy(img)
                # Create a blank image that matches the original in size.
                line_img = np.zeros(
                    (
                        img.shape[0],
                        img.shape[1],
                        3
                    ),
                    dtype=np.uint8,
                )
                # Loop over all lines and draw them on the blank image.
                for line in lines:
                    for x1, y1, x2, y2 in line:
                        cv2.line(line_img, (x1, y1), (x2, y2), color, thickness)
                # Merge the image with the lines onto the original.
                img = cv2.addWeighted(img, 0.8, line_img, 1.0, 0.0)
                # Return the modified image.
                return img
            
            # if len(left_lines) == 0 or len(right_lines) == 0:
            #     raise ValueError('all lines(unfiltered) is empty')
            yellow_line_image = draw_lines(obs, yellow_lines) 
            white_line_image = draw_lines(obs, white_lines)
            # plt.figure()
            # plt.title('All yellow lines')
            # plt.imshow(yellow_line_image)
            # filename = 'milestone1_logs/yellow_all_line_' + str(time.time())+ '.png'
            # plt.savefig(filename)
            # plt.close()

            # plt.figure()
            # plt.title('All white lines')
            # plt.imshow(white_line_image)
            # filename = 'milestone1_logs/white_all_line' + str(time.time())+ '.png'
            # plt.savefig(filename)
            # plt.close()
    
            # Filter lines by slop
            # group to left and right based on slop
            left_lines_filtered_by_slope = []
            right_lines_filtered_by_slope = []
            def split_lines_to_left_right(lines):
                def isSlopeValid(slope):
                    #どのslopeの値を無視するか微調整必要(initial: 0.5)
                    if math.fabs(slope) < 0.15: # too horizontal
                        return False
                    elif math.fabs(slope) > 3: # too vertical
                        return False
                    else:
                        return True
                left_line_x = []
                left_line_y = []
                right_line_x = []
                right_line_y = []
                # check every line
                for line in lines:
                    for x1, y1, x2, y2 in line:
                        if (x2 - x1) == 0:
                            continue
                        slope = (y2 - y1) / (x2 - x1) 
                        if not isSlopeValid(slope):
                            continue
                        # becareful: origin of img is at top left and not bottom left
                        # so slope value here starts from top left
                        # positive slope (top left origin) means negative slope in bottom left origin = right group
                        if slope <= 0: # If the slope is negative, left group.
                            left_lines_filtered_by_slope.append([[x1, y1, x2, y2]])
                            left_line_x.extend([x1, x2])
                            left_line_y.extend([y1, y2])
                        elif slope > 0: # If the slope is positive, right group
                            right_lines_filtered_by_slope.append([[x1, y1, x2, y2]])
                            right_line_x.extend([x1, x2])
                            right_line_y.extend([y1, y2])
                return left_line_x, left_line_y, right_line_x, right_line_y
            
            # print(yellow_lines)
            # print(white_lines)
            yellow_left_line_x, yellow_left_line_y, yellow_right_line_x, yellow_right_line_y = split_lines_to_left_right(yellow_lines)
            white_left_line_x, white_left_line_y, white_right_line_x, white_right_line_y = split_lines_to_left_right(white_lines)    
            
            if len(right_lines_filtered_by_slope) + len(left_lines_filtered_by_slope) == 0:
                # raise ValueError('lines(filtered by slope) is empty')
                print('lines(filtered by slope) is empty')
            left_line_image = draw_lines(
                obs,
                left_lines_filtered_by_slope,
                thickness=5,
            )
            right_line_image = draw_lines(
                obs,
                right_lines_filtered_by_slope,
                thickness=5,
            )
            # plt.figure()
            # plt.title('right Lines filtered by slope')
            # plt.imshow(right_line_image)
            # filename = 'milestone1_logs/' + str(time.time())+ '.png'
            # plt.savefig(filename)
            # plt.close()

            # plt.figure()
            # plt.title('left Lines filtered by slope')
            # plt.imshow(left_line_image)
            # filename = 'milestone1_logs/' + str(time.time())+ '.png'
            # plt.savefig(filename)
            # plt.close()

            # Creating a Single Linear Representation of each Line Group
            # min_y needs to be in int for draw_lines()?
            # min_y = int(obs.shape[0] * (3 / 5)) # <-- Just below the horizon
            min_y = 0
            max_y = obs.shape[0] # <-- The bottom of the image
            def get_line_representation(lines_x, lines_y):
                x_start = None
                x_end = None
                # group lines
                if len(lines_x) > 0 and len(lines_y) > 0:
                    poly_left = np.poly1d(np.polyfit(
                        lines_y,
                        lines_x,
                        deg=1
                    ))
                    x_start = int(poly_left(max_y))
                    x_end = int(poly_left(min_y))
                return x_start, x_end
            # for yellow, left/right group
            yellow_left_merge_x_start, yellow_left_merge_x_end = get_line_representation(yellow_left_line_x, yellow_left_line_y)
            yellow_right_merge_x_start, yellow_right_merge_x_end = get_line_representation(yellow_right_line_x, yellow_right_line_y)
            # for white, left/right group
            white_left_merge_x_start, white_left_merge_x_end = get_line_representation(white_left_line_x, white_left_line_y)
            white_right_merge_x_start, white_right_merge_x_end = get_line_representation(white_right_line_x, white_right_line_y)
            
            # check slope of merged line is valid
            merged_lines = []
            # assuming straight road, left is yellow and right is white
            yellow_left_merged_lines = None
            white_right_merged_lines = None
            if (yellow_left_merge_x_start != None and yellow_left_merge_x_end != None):
                # draw_linesの仕組み的にnested listじゃないといけない?
                # lines = cv2.HoughLinesPに合わせている?
                yellow_left_merged_lines = [[yellow_left_merge_x_start, max_y, yellow_left_merge_x_end, min_y]]
                # check slope is valid
                x1, y1, x2, y2 =  yellow_left_merged_lines[0]
                left_slope = (y2 - y1) / (x2 - x1)
                if left_slope < 0:
                    merged_lines.append(yellow_left_merged_lines)
            if (white_right_merge_x_start != None and white_right_merge_x_end != None):
                white_right_merged_lines = [[white_right_merge_x_start, max_y, white_right_merge_x_end, min_y]]
                x1, y1, x2, y2 = white_right_merged_lines[0]
                right_slope = (y2 - y1) / (x2 - x1)
                if right_slope > 0:
                    merged_lines.append(white_right_merged_lines)

            # edge case: merged lines does not exists
            # my initial algorithm was based on assumption that 
            # 1. initial position of robot is at right lane
            # 2. there is yellow left line and white right line infront of robot (so we can use intersection for rotation adjustment)
            # Unfortunately, I learned thaat these assumptions are not correct in later test cases. This lead to add lots of if-case to adapt for edge cases which is causing code to get messy...
            if len(merged_lines) == 0:
                print('merged_lines(representative line of left and right group) is empty')
                # merged lines doesnt exists means = left yellow and right white doesnt exists

                # instead of right white, if left white exists
                # or instead of left yellow, if right yellow exists
                # - it possibly means robot is at left lane
                if (white_left_merge_x_start is not None) or (yellow_right_merge_x_start is not None):
                    # rotate right and move from left to right lane
                    return 'rotate_right', 'rotate_right'
                # else: yellow and white line does not exist at all
                # - no lane, possibly curve lane
                return None, None
            
            line_image = draw_lines(
                obs,
                merged_lines,
                thickness=5,
            )
            # plt.figure()
            # plt.title('Representative line of left and right group')
            # plt.imshow(line_image)
            # filename = 'milestone1_logs/' + str(time.time())+ '.png'
            # plt.savefig(filename)
            # plt.close()

            # filename = 'milestone1_logs/' + 'representative_line_' + str(time.time())+ '.png'
            # plt.savefig(filename)
            # plt.close()
            # plt.show()

            # Make action decision based on representative line of left and right group
            # 1. left line doesnt exist => need to rotate left
            if yellow_left_merged_lines == None:
                print('yellow left doesnt exist')
                left = [0, 3]
                env.step(left)
            elif white_right_merged_lines == None:
                print('white right doesnt exist')
                right = [0, -3]
                env.step(right)
            # 2. left and right exists => need to rotate such that slope of left and right line matches (absolute of slope)
            elif yellow_left_merged_lines != None and white_right_merged_lines != None:
                print('yellow left and white right exist')
                x1, y1, x2, y2 = yellow_left_merged_lines[0]
                left_slope = (y2 - y1) / (x2 - x1)
                x1, y1, x2, y2 = white_right_merged_lines[0]
                right_slope = (y2 - y1) / (x2 - x1)
                # adjust rotation based on intersection of left and right lane
                def find_intersection(x1, y1, x2, y2, x3, y3, x4, y4):
                    # calculate slopes and y-intercepts of the two lines
                    slope1 = (y2 - y1) / (x2 - x1)
                    y_int1 = y1 - slope1 * x1
                    slope2 = (y4 - y3) / (x4 - x3)
                    y_int2 = y3 - slope2 * x3
                    
                    # check if lines are parallel
                    if slope1 == slope2:
                        return None  # no intersection
                    
                    # calculate intersection point
                    x_int = (y_int2 - y_int1) / (slope1 - slope2)
                    y_int = slope1 * x_int + y_int1
                    
                    return (x_int, y_int)
                x1, y1, x2, y2 = yellow_left_merged_lines[0]
                x3, y3, x4, y4 = white_right_merged_lines[0]
                inter_pt = find_intersection(x1, y1, x2, y2, x3, y3, x4, y4)
                inter_x = inter_pt[0]
                height, width = obs.shape[:2]
                # within left and right bound
                leftBoundX = width * 0.45
                rightBoundX = width * 0.55
                # based on intersection point between left and right line
                if leftBoundX <= inter_x <= rightBoundX:
                    break
                else:
                    # not facing straight to the next tile so need to rotate
                    if inter_x < leftBoundX:
                        # left slope is bigger, need to rotate to left
                        env.step([0, 0.5])
                    elif rightBoundX < inter_x:
                        # right slope is bigger, need to rotate to right
                        env.step([0, -0.5])     
                    else:
                        raise ValueError('invalid intersection point x between representative line of left and right group')

            # print(f"current pose = {info['curr_pos']}, step count = {env.unwrapped.step_count}, step reward = {reward:.3f}")

            # render the action taken by env.step()
            env.render()
        
        # return slopes, used later to adjust robot to middle of lane
        return left_slope, right_slope
    

    # Load the given path txt file
    filename = f'{args.map_name}_seed{args.seed}_start_{start_pos[0]},{start_pos[1]}_goal_{goal[0]},{goal[1]}.txt'
    given_filepath = './testcases/milestone1_paths/' + filename
    with open(given_filepath, 'r') as file:
        # each line contains action in each step
        # skip first action
        isFirst = True
        isSecond = True
        for line in file:
            if isFirst:
                isFirst = False
                continue

            # # adjust rotation before taking action in every step       
            # left_slope, right_slope = adjust_rotation()

            # ex: (1, 6), forward
            # after split: ' forward\n'
            # - extra space at front and \n at back
            next_tile = line.split(',')[:2]
            print(next_tile)
            # ex: '(12', remove first (
            next_x = int(next_tile[0][1:])
            # ex: ' 1)' remove first space and last )
            next_y = int(next_tile[1][1:-1])
            print(next_x)
            print(next_y)
            next_tile = (next_x, next_y)
            # get current tile
            speed = 0
            steering = 0
            obs, reward, done, info = env.step([speed, steering])
            initial_tile = info['curr_pos']
            current_tile = initial_tile
            # keep moving until robot gets to next tile
            move = line.split(',')[-1][1:-1]
            left_slope = 'init'
            right_slope = 'init'
            # if move == 'forward':
            print('adjust start')
            if move == 'forward':
                left_slope, right_slope = adjust_rotation()
            print('adjust finished')
            print(left_slope)
            # edge cases for initial robot pose
            if isSecond:
                isSecond = False
                if left_slope is None and right_slope is None:
                    print('probably curved lane')
                    while current_tile != next_tile:
                        # rotate left
                        action = [0.3,0.2]
                        # take action
                        obs, reward, done, info = env.step(action)
                        env.render()
                        # update current tile
                        current_tile = info['curr_pos']
                        if initial_tile != current_tile and current_tile != next_tile:
                            raise ValueError('moved to wrong tile')
                    # skip normal action execution in next while loop below
                    continue
                if left_slope == 'rotate_right': 
                    print('probably on right lane')
                    # probably on left lane (wrong lane, should be on right lane)
                    isFirst = True
                    while current_tile != next_tile:
                        if isFirst:
                            # rotate right
                            action = [0, -20]
                            isFirst = False
                        else:
                            action = [0.8, 0.2]
                        # take action
                        obs, reward, done, info = env.step(action)
                        env.render()
                        # update current tile
                        current_tile = info['curr_pos']
                        if initial_tile != current_tile and current_tile != next_tile:
                            raise ValueError('moved to wrong tile')
                    # skip normal action execution in next while loop below
                    continue
                
            # normal action execution (in correct line)
            while current_tile != next_tile:
                if move == 'forward':
                    # move slightly to push robot to middle of lane?
                    # rotate based on slope of left and right
                    rotate = None
                    if left_slope is None and right_slope is None:
                        rotate = 0
                    elif abs(left_slope) < abs(right_slope):
                        # right side of lane, move to left
                        rotate = 0.05
                    elif abs(left_slope) > abs(right_slope):
                        # left side of lane, move to rright
                        rotate = -0.05
                    else:
                        rotate = 0
                    # print('rotate: ' + str(rotate))
                    # lin_velが1だとなぜかrotateが効かなくなる?
                    action = [0.5, rotate]
                elif move == 'left':
                    # lin_vel, rotation
                    action = [0.3,0.4]
                elif move == 'right':
                    action = [0.1,-0.3]
                else:
                    raise ValueError('invalid move in given path txt file') 
                # take action
                obs, reward, done, info = env.step(action)
                env.render()
                # update current tile
                current_tile = info['curr_pos']
                if initial_tile != current_tile and current_tile != next_tile:
                    raise ValueError('moved to wrong tile')



    # dump the controls using numpy
    # np.savetxt(f'./{args.map_name}_seed{args.seed}_start_{start_pos[0]},{start_pos[1]}_goal_{goal[0]},{goal[1]}.txt',
    #            actions, delimiter=',')

env.close()

