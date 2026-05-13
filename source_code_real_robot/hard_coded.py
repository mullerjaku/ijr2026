#TODO: pridat goal do cur.txt
import random, time
import numpy as np
import os
import glob
import json
import torch
import torch.nn as nn
import torch.optim as optim
from classes.json_class import GoalManager
from classes.ml_class import UtilityModel
from classes.methods_class import Motivations
from classes.effectance_class import Effactance
from classes.curiosity_class import RNDModule
from classes.entropy_class import UtilityModelEntropy
from classes.predictor_class import AdaptivePredictor
from classes.world_model_class import WorldModel
from collections import deque
from fanucpy import Robot

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

current_dir = os.path.dirname(os.path.abspath(__file__))
final_goals = os.path.join(current_dir, 'data', 'Goals.txt')
final_data = os.path.join(current_dir, 'data', 'Data.json')
extrin_goals = os.path.join(current_dir, 'data', 'Extrin_goals.txt')
pnodes_file = os.path.join(current_dir, 'data', 'Pnodes.json')
ums_file = os.path.join(current_dir, 'data', 'UMs_vals.json')
traces_um = os.path.join(current_dir, 'data', 'Traces_UM.json')
AREA_MAX_DISTANCE = 774.28
FILE = final_goals
FILENAME = final_data
EXGOAL = extrin_goals
PNODES = pnodes_file
UMS = ums_file
TRACES = traces_um
EPOCH_STEPS = 100


def normalize(d):
    d_norm = d / AREA_MAX_DISTANCE
    return d_norm

def is_file_empty(file_path):
    # Check if file is empty
    return os.stat(file_path).st_size == 0

def softmax(logits, scale_factor=10):
    exp_values = np.exp(logits / scale_factor)
    return exp_values / np.sum(exp_values)

def choose_motivation(logits):
    softmax_values = softmax(logits)
    choice = np.random.choice(['exploration','improvement'], p=softmax_values)
    return choice

def remove_goal_from_txt(txt_filename, selected_goal):
    with open(txt_filename, "r") as f:
        lines = f.readlines()
    new_lines = []
    for line in lines:
        values = np.array([float(x) for x in line.strip().split()])
        if not np.allclose(values, selected_goal):
            new_lines.append(line)
    with open(txt_filename, "w") as f:
        f.writelines(new_lines)

def remove_goal_from_pnodes(filename, selected_goal):
    with open(filename, "r") as f:
        data = json.load(f)
    new_data = []
    for entry in data:
        if not np.allclose(np.array(entry["goal"]), selected_goal):
            new_data.append(entry)
    with open(filename, "w") as f:
        json.dump(new_data, f, indent=4)

def compute_slope(values):
    grad = np.gradient(values)
    acc = np.gradient(grad)
    final = values[-1]
    
    # Monotonic penalty (if prediction drops)
    monotonic_penalty = np.sum(np.diff(values) < 0)

    # Final composite score
    score = (
        2.0 * final              # reward how close to 1.0
        + 1.0 * np.mean(grad)    # reward overall growth
        + 1.0 * np.mean(acc)     # reward acceleration
        - 1.0 * monotonic_penalty # penalty for regressions
    )
    return score

def data_and_usg(data):
    n_cols = max(len(row) for block in data for row in block)
    sum_slopes = np.zeros(n_cols)
    for i, block in enumerate(data):
        arr = np.array(block)
        for col in range(arr.shape[1]):
            col_values = arr[:, col]
            if len(col_values) < 2:
                continue
            slope = compute_slope(col_values)
            sum_slopes[col] += slope

    print("Sum slopes:", sum_slopes)
    return np.argmax(sum_slopes)

def gaussian_nll(mu, logvar, target):
    var = torch.exp(logvar)
    return 0.5 * ((target - mu)**2 / var + logvar).mean()

def hit_the_glass_3d(walls, p1, p2):
    for wall in walls:
        t_min = 0.0
        t_max = 1.0
        hit = True
        
        for i in range(3):
            d = p2[i] - p1[i]
            
            if abs(d) < 1e-9: 
                if p1[i] < wall['min'][i] or p1[i] > wall['max'][i]:
                    hit = False
                    break
            else:
                t1 = (wall['min'][i] - p1[i]) / d
                t2 = (wall['max'][i] - p1[i]) / d
                
                t_enter = min(t1, t2)
                t_exit = max(t1, t2)
                
                t_min = max(t_min, t_enter)
                t_max = min(t_max, t_exit)
                
                if t_min > t_max:
                    hit = False
                    break
        
        if hit:
            return True

    return False

def distance_point_to_aabb(point, box_min, box_max):
    dx = np.maximum(0, np.maximum(box_min[0] - point[0], point[0] - box_max[0]))
    dy = np.maximum(0, np.maximum(box_min[1] - point[1], point[1] - box_max[1]))
    dz = np.maximum(0, np.maximum(box_min[2] - point[2], point[2] - box_max[2]))

    return np.sqrt(dx**2 + dy**2 + dz**2)


def main():
    robot = Robot(
        robot_model="Fanuc",
        host="192.168.1.3",
        port=18735,
        ee_DO_type="RDO",
        ee_DO_num=7,
    )

    robot.connect()

    robot.move(
        "pose",
        vals=[600.0, 200.0, -450.0, -1.646, 1.306, -64.23],
        velocity=10,
        acceleration=10,
        cnt_val=0,
        linear=False
    )
    
    with open(FILENAME, 'w') as filex:
        pass
    with open(FILE, 'w') as filex:
        pass
    with open(EXGOAL, 'w') as filex:
        pass
    with open(PNODES, 'w') as filex:
        pass 
    with open(UMS, 'w') as filex:
        pass
    with open(TRACES, 'w') as filex:
        pass
    folder = os.path.join(current_dir, 'models')
    files = glob.glob(os.path.join(folder, '*'))
    for f in files:
        if os.path.isfile(f):
            os.remove(f)
    

    obstacles_bounds = []
    obstacles_bounds.append({
        'min': np.array([491.247, -237.103, -760.372]),
        'max': np.array([589.671, 166.679, -465.61])
    }) #stěna1

    obstacles_bounds.append({
        'min': np.array([491.247,  -124.204, -760.372]),
        'max': np.array([810.815,  166.679, -465.61])
    }) #stěna2

    obstacles_bounds.append({
        'min': np.array([491.247,  309.423, -760.372]),
        'max': np.array([593.259, 430.00, -663.000])
    }) #box pro button

    obstacles_bounds.append({
        'min': np.array([714.138, -286.724, -760.372]),
        'max': np.array([810.815, -100.496, -663.000])
    }) #box pro button after change

    #Y: 166.679, 
    #BOX: 589.671, -124.204, -465.61


    static_obj_glass = np.array([540.459, 21.2375, -431.589])

    #X 491.247, 810.815
    #Y -237.103, 367.678
    #Z -760.372, -397.568
    limits_min = np.array([491.247, -237.103, -760.372])
    limits_max = np.array([810.815, 430.00, -397.568])

    # ox = random.uniform(1.05, 1.55)
    # oy = random.uniform(0.7, 1.1)
    # oz = random.uniform(0.5, 0.9)
    desired_pos_red = np.array([514.496, 426.146, -660.00])
    desired_pos_orange = np.array([760.476, 242.05, -757.078])

    pozice_orange = np.array([
    [760.476, 242.05, -757.078],   # 1. pozice
    [676.898, 192.303, -757.078],   # 2. pozice
    [627.341, 283.309, -757.078],   # 3. pozice
    [731.565, 310.941, -757.078],   # 4. pozice
    [539.745, 206.497, -757.078]   # 5. pozice
])

    robot_pos = robot.get_curpos()[:3]
    robot_pos = np.array(robot_pos)
    
    #Definitions
    p = 0 #epoch counter
    i = 0 #loop counter
    index_count = 0
    change = False
    results_array = []

    motivations = Motivations(FILE, FILENAME)
    json_manager = GoalManager(FILENAME)

    selected_goal = desired_pos_orange

    while True:
        if change == True:
            desired_pos_red = np.array([791.613, -189.437, -660.00])
            # else:
            #     ox = random.uniform(1.05, 1.55)
            #     oy = random.uniform(0.7, 1.1)
            #     oz = random.uniform(0.5, 0.6)
            #     desired_pos_red = np.array([ox, oy, oz])
            #     env.unwrapped.goal = desired_pos_red.copy()

        robot_pos = robot.get_curpos()[:3]
        robot_pos = np.array(robot_pos)

        #Definations
        i = 0
        loop_count = []
        actions_paths = []
        perception_paths = []
        best_um = []

        while True:
            future_point_pose_list = []
            point_move_list = []
            i +=1
            thats_the_goal = False

            print("Number of iterations: ",+i)
            if i == EPOCH_STEPS:
                break

            robot_pos = robot.get_curpos()[:3]
            robot_pos = np.array(robot_pos)
            points = 0

            while points < 1000:
                points += 1
                #Pose move
                pose_plan = robot_pos.copy()
                # poslední hodnota (gripper) bude vždy 0.0
                vel_action = np.concatenate([
                    np.random.uniform(low=-40, high=40, size=3)
                ])

                predicted_pos = pose_plan + vel_action[:3]
                future_point_pose = np.array([predicted_pos[0], predicted_pos[1], predicted_pos[2]])
                if hit_the_glass_3d(obstacles_bounds, pose_plan, future_point_pose):
                    continue

                distances = []

                distance = np.linalg.norm(future_point_pose - selected_goal)
                norm_distance = normalize(distance)
                distances.append(norm_distance)
                prediction = norm_distance


                if np.any(future_point_pose < limits_min) or np.any(future_point_pose > limits_max):
                    prediction = float("inf")
                else:
                    future_point_pose_list.append(future_point_pose)

                to_list = vel_action.tolist() + [prediction]

                point_move_list.append(to_list)

            #Checking if list is empty or no
            if not point_move_list:
                print("Empty list")
                continue

            #Sorting the list
            serazene_pole = sorted(point_move_list, key=lambda x: x[-1], reverse=False)
            vybrane_pole = serazene_pole[:10]
                
            for pole in vybrane_pole:
                [x, y, z, prediction]=pole
                prev_pos = robot.get_curpos()[:3]
                prev_pos_array = np.array(prev_pos)
                prev_pos[0] += x
                prev_pos[1] += y
                prev_pos[2] += z

                robot.move(
                    "pose",
                    vals=[prev_pos[0], prev_pos[1], prev_pos[2], -1.646, 1.306, -64.23],
                    velocity=50,
                    acceleration=50,
                    cnt_val=0,
                    linear=False
                )

                new_pos  = robot.get_curpos()[:3]
                new_pos = np.array(new_pos)
                threshold = 1e-6
                if np.linalg.norm(new_pos - prev_pos_array) > threshold:
                    distances = []

                    distance = np.linalg.norm(new_pos - selected_goal)
                    norm_distance = normalize(distance)
                    distances.append(norm_distance)

                    """
                    Here is when it check all the effactances
                    """
                    if norm_distance < 0.005:
                        thats_the_goal = True
                    else:
                        thats_the_goal = False

                    break

                else:
                    continue

            if thats_the_goal == True:
                if norm_distance<0.005 and np.any(np.all(selected_goal == pozice_orange, axis=1)):
                    robot.move(
                        "pose",
                        vals=[prev_pos[0], prev_pos[1], -771.587, -1.646, 1.306, -64.23],
                        velocity=5,
                        acceleration=5,
                        cnt_val=0,
                        linear=False
                    )

                    robot.call_prog("GRIPPER")
                    time.sleep(1)

                    robot.move(
                        "pose",
                        vals=[prev_pos[0], prev_pos[1], -740.587, -1.646, 1.306, -64.23],
                        velocity=5,
                        acceleration=5,
                        cnt_val=0,
                        linear=False
                    )

                    robot.move(
                        "pose",
                        vals=[734.438, 510.445, -748.428, -1.646, 1.306, -64.23],
                        velocity=5,
                        acceleration=5,
                        cnt_val=0,
                        linear=False
                    )

                    robot.call_prog("GRIPPER")
                    time.sleep(1)

                    desired_pos_orange = pozice_orange[index_count % len(pozice_orange)]
                    index_count += 1
                break
        
        results_array.append([p, i, str(selected_goal)])
        np.savetxt("fanuc_hard_4.txt", results_array, fmt="%s")

        random_decision = np.random.randint(2)
        if random_decision == 0:
            selected_goal = desired_pos_orange
        else: 
            selected_goal = desired_pos_red

        
        robot.move(
            "pose",
            vals=[prev_pos[0], prev_pos[1], -450.0, -1.646, 1.306, -64.23],
            velocity=50,
            acceleration=50,
            cnt_val=0,
            linear=False
        )

        robot.move(
            "pose",
            vals=[600.0, 200.0, -450.0, -1.646, 1.306, -64.23],
            velocity=50,
            acceleration=50,
            cnt_val=0,
            linear=False
        )

        p+=1
        print("Number of repeats: ",+p)

        if p == 10:
            change = True

        if p == 250:
            break

if __name__ == '__main__':
    try:
        main()
    except KeyboardInterrupt:
        print("Shutting down the program")