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
            img_blur = cv2.GaussianBlur(obs,(7,7),0)
            # It converts the BGR color space of image to HSV color space
            # BGR2HSVとRGB2HSVがある, BGRじゃなくRGBが正解っぽい
            hsv = cv2.cvtColor(img_blur, cv2.COLOR_RGB2HSV)
            
            # Threshold in HSV space
            # white lane, yellow lane
            # white (and gray)
            lower_white = np.array([0,0,0])
            upper_white = np.array([255,10,255])
            # yellow
            lower_yellow = np.array([20,100,100])
            upper_yellow = np.array([30,255,255])
        
            # preparing the mask to overlay
            mask_white = cv2.inRange(hsv, lower_white, upper_white)
            mask_yellow = cv2.inRange(hsv, lower_yellow, upper_yellow)
            
            # The black region in the mask has the value of 0,
            # so when multiplied with original image removes all non-blue regions
            result_white = cv2.bitwise_and(obs, obs, mask = mask_white)
            result_yellow = cv2.bitwise_and(obs, obs, mask = mask_yellow)

            # Call Canny Edge Detection here.
            # convert to grayscale here.
            # yellow lane for left lane
            left_gray_image = cv2.cvtColor(result_yellow, cv2.COLOR_RGB2GRAY)
            # white lane for right lane
            right_gray_image = cv2.cvtColor(result_white, cv2.COLOR_RGB2GRAY)
            # use blurred image to select threshold for canny
            img_blur = cv2.GaussianBlur(obs,(7,7),0)
            med_val = np.median(img_blur)
            sigma = 0.33  # 0.33
            min_val = int(max(0, (1.0 - sigma) * med_val))
            max_val = int(max(255, (1.0 + sigma) * med_val))
            left_cannyed_image = cv2.Canny(left_gray_image, min_val, max_val)
            right_cannyed_image = cv2.Canny(right_gray_image, min_val, max_val)
            # plt.figure()
            # plt.title('left cannyed imaged')
            # plt.imshow(left_cannyed_image)
            # plt.figure()
            # plt.title('right cannyed imaged')
            # plt.imshow(right_cannyed_image)
            # plt.show()

            # 下半分だけcropする?
            def region_of_interest(img, vertices):
                mask = np.zeros_like(img)
                match_mask_color = 255 # <-- This line altered for grayscale.
                cv2.fillPoly(mask, vertices, match_mask_color)
                masked_image = cv2.bitwise_and(img, mask)
                return masked_image
            # middle half
            height, width = left_cannyed_image.shape[:2]
            region_of_interest_vertices = [
                # order of verticies is important
                # make sure that shape if drawn by connecting points in order of verticies
                # ex: top left -> bottom right: invalid square
                # ex: top left -> bottom left: valid square
                (0, 100),
                (width, 100),
                (width, 400),
                (0, 400)   
            ]
            left_cannyed_image = region_of_interest(
                left_cannyed_image,
                np.array([region_of_interest_vertices], np.int32)
            )
            right_cannyed_image = region_of_interest(
                right_cannyed_image,
                np.array([region_of_interest_vertices], np.int32)
            )
            # plt.figure()            
            # plt.title('cropped image')
            # plt.imshow(cropped_cannyed_image)

            # detect line from edges
            def detect_lines(img):
                lines = cv2.HoughLinesP(
                    img,
                    rho=6,
                    theta=np.pi / 60,
                    threshold=160,
                    lines=np.array([]),
                    #minLineLength: 元々40
                    #長さ増やすことでいい感じのlaneだけ取れる(画像によって変わりそうやから微調整は必要そう)
                    minLineLength=40,
                    # default: 25
                    maxLineGap=25
                )
                if lines is None:
                    return []
                else:
                    return lines
            left_lines = detect_lines(left_cannyed_image)
            right_lines = detect_lines(right_cannyed_image)

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
            left_line_image = draw_lines(obs, left_lines) # <---- Add this call.
            right_line_image = draw_lines(obs, right_lines)
            # plt.figure()
            # plt.title('All lines')
            # plt.imshow(line_image)
    
            # group line to left and right group
            left_line_x = []
            left_line_y = []
            right_line_x = []
            right_line_y = []
            lines_filtered_by_slope = []
            # filter by slope
            def isSlopeValid(slope):
                #どのslopeの値を無視するか微調整必要(initial: 0.5)
                    if math.fabs(slope) < 0.1: # <-- Only consider extreme slope
                        return False
                    else:
                        return True
            for line in left_lines:
                for x1, y1, x2, y2 in line:
                    slope = (y2 - y1) / (x2 - x1) 
                    if not isSlopeValid(slope):
                        continue
                    lines_filtered_by_slope.append([[x1, y1, x2, y2]])
                    # becareful: origin of img is at top left and not bottom left
                    # so slope value here starts from top left
                    # positive slope (top left origin) means negative slope in bottom left origin = right group
                    if slope <= 0: # If the slope is negative, left group.
                        left_line_x.extend([x1, x2])
                        left_line_y.extend([y1, y2])
            for line in right_lines:
                for x1, y1, x2, y2 in line:
                    slope = (y2 - y1) / (x2 - x1) 
                    if not isSlopeValid(slope):
                        continue
                    lines_filtered_by_slope.append([[x1, y1, x2, y2]])
                    if slope > 0: # If the slope is positive, right gorup
                        right_line_x.extend([x1, x2])
                        right_line_y.extend([y1, y2])
            
            if len(lines_filtered_by_slope) == 0:
                raise ValueError('lines(filtered by slope) is empty')
            line_image = draw_lines(
                obs,
                lines_filtered_by_slope,
                thickness=5,
            )
            plt.figure()
            plt.title('Lines filtered by slope')
            plt.imshow(line_image)

            # Creating a Single Linear Representation of each Line Group
            # min_y needs to be in int for draw_lines()?
            # min_y = int(obs.shape[0] * (3 / 5)) # <-- Just below the horizon
            min_y = 0
            max_y = obs.shape[0] # <-- The bottom of the image
            left_x_start = None
            left_x_end = None
            right_x_start = None
            right_x_end = None
            # group lines
            if len(left_line_x) > 0 and len(left_line_y) > 0:
                poly_left = np.poly1d(np.polyfit(
                    left_line_y,
                    left_line_x,
                    deg=1
                ))
                left_x_start = int(poly_left(max_y))
                left_x_end = int(poly_left(min_y))
            if len(right_line_x) > 0 and len(right_line_y) > 0:
                poly_right = np.poly1d(np.polyfit(
                    right_line_y,
                    right_line_x,
                    deg=1
                ))
                right_x_start = int(poly_right(max_y))
                right_x_end = int(poly_right(min_y))

            # check slope of merged line is valid
            merged_lines = []
            left_merged_lines = None
            right_merged_lines = None
            if (left_x_start != None and left_x_end != None):
                # draw_linesの仕組み的にnested listじゃないといけない?
                # lines = cv2.HoughLinesPに合わせている?
                left_merged_lines = [[left_x_start, max_y, left_x_end, min_y]]
                # check slope is valid
                x1, y1, x2, y2 = left_merged_lines[0]
                left_slope = (y2 - y1) / (x2 - x1)
                if left_slope < 0:
                    merged_lines.append(left_merged_lines)
            if (right_x_start != None and right_x_end != None):
                right_merged_lines = [[right_x_start, max_y, right_x_end, min_y]]
                x1, y1, x2, y2 = right_merged_lines[0]
                right_slope = (y2 - y1) / (x2 - x1)
                if right_slope > 0:
                    merged_lines.append(right_merged_lines)

            if len(merged_lines) == 0:
                raise ValueError('merged_lines(representative line of left and right group) is empty')
            line_image = draw_lines(
                obs,
                merged_lines,
                thickness=5,
            )
            plt.figure()
            plt.title('Representative line of left and right group')
            plt.imshow(line_image)
            filename = 'milestone1_logs/' + 'representative_line_' + str(time.time())+ '.png'
            plt.savefig(filename)
            plt.close()
            # plt.show()

            # Make action decision based on representative line of left and right group
            # 1. left line doesnt exist => need to rotate left
            if left_merged_lines == None:
                left = [0, 3]
                env.step(left)
            elif right_merged_lines == None:
                right = [0, -3]
                env.step(right)
            # 2. left and right exists => need to rotate such that slope of left and right line matches (absolute of slope)
            elif left_merged_lines != None and right_merged_lines != None:
                x1, y1, x2, y2 = left_merged_lines[0]
                left_slope = (y2 - y1) / (x2 - x1)
                x1, y1, x2, y2 = right_merged_lines[0]
                right_slope = (y2 - y1) / (x2 - x1)
                # check if absolute slope of left and line is about the same 
                # exactly the same will be difficult, create threshold for similarity?
                diff = abs(abs(left_slope) - abs(right_slope))
                # diffがthresholdいないだったらbreakするやり方うまくいかない、永遠にleftにrotateする、おそらくline representationの正確さが足りない
                # 試しにleftとright lineのintersectionで決める
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
                x1, y1, x2, y2 = left_merged_lines[0]
                x3, y3, x4, y4 = right_merged_lines[0]
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

                # based on diff
                # if diff < 0.05:
                #     break
                # else:
                #     # not facing straight to the next tile so need to rotate
                #     if abs(left_slope) - abs(right_slope) > 0:
                #         # left slope is bigger, need to rotate to left
                #         env.step([0, 0.5])
                #     else:
                #         # right slope is bigger, need to rotate to right
                #         env.step([0, -0.5])

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
        for line in file:
            if isFirst:
                isFirst = False
                continue

            # adjust rotation before taking action in every step       
            left_slope, right_slope = adjust_rotation()

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
            while current_tile != next_tile:
                move = line.split(',')[-1][1:-1]
                if move == 'forward':
                    # move slightly to push robot to middle of lane?
                    # rotate based on slope of left and right
                    rotate = None
                    if abs(left_slope) < abs(right_slope):
                        # right side of lane, move to left
                        rotate = 0.05
                    elif abs(left_slope) > abs(right_slope):
                        # left side of lane, move to rright
                        rotate = -0.05
                    # print('rotate: ' + str(rotate))
                    # lin_velが1だとなぜかrotateが効かなくなる?
                    action = [0.5, rotate]
                elif move == 'left':
                    action = [0,1]
                elif move == 'right':
                    action = [0,-1]
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

