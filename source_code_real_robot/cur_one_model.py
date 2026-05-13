#TODO:
import random
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
import gymnasium as gym
import gymnasium_robotics
import mujoco

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

current_dir = os.path.dirname(os.path.abspath(__file__))
final_goals = os.path.join(current_dir, 'data', 'Goals.txt')
final_data = os.path.join(current_dir, 'data', 'Data.json')
extrin_goals = os.path.join(current_dir, 'data', 'Extrin_goals.txt')
pnodes_file = os.path.join(current_dir, 'data', 'Pnodes.json')
ums_file = os.path.join(current_dir, 'data', 'UMs_vals.json')
traces_um = os.path.join(current_dir, 'data', 'Traces_UM.json')
AREA_MAX_DISTANCE = 1 #0.995
FILE = final_goals
FILENAME = final_data
EXGOAL = extrin_goals
PNODES = pnodes_file
UMS = ums_file
TRACES = traces_um
EPOCH_STEPS = 200


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
    """
    Vypočítá nejmenší vzdálenost bodu od AABB kvádru.
    Pokud je bod uvnitř, vrátí 0.
    """
    # Vypočítáme vzdálenost v každé ose (X, Y, Z)
    # np.maximum(0, ...) zajistí, že pokud jsme "mezi" min a max, je složka 0.
    dx = np.maximum(0, np.maximum(box_min[0] - point[0], point[0] - box_max[0]))
    dy = np.maximum(0, np.maximum(box_min[1] - point[1], point[1] - box_max[1]))
    dz = np.maximum(0, np.maximum(box_min[2] - point[2], point[2] - box_max[2]))
    
    # Celková vzdálenost je délka vektoru (dx, dy, dz)
    return np.sqrt(dx**2 + dy**2 + dz**2)


def main():
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

    # Initialize the environment
    env = gym.make('FetchReach-v4', render_mode='human', max_episode_steps=200)
    observation, info = env.reset()
    limits_min = np.array([1.05, 0.4, 0.4])
    limits_max = np.array([1.55, 1.1, 0.9])
    

    #walls = [(20, 39, 50, 41), (70, 45, 72, 75)]
    obstacles_bounds = []
    obstacles_bounds.append({
        'min': np.array([1.239 - 0.04, 0.6 - 0.13, 0.52 - 0.15]), #+0.03 všude
        'max': np.array([1.239 + 0.04, 0.6 + 0.13, 0.52 + 0.15])
    })

    obstacles_bounds.append({
        'min': np.array([1.35 - 0.13,  0.69 - 0.04, 0.52 - 0.15]),
        'max': np.array([1.35 + 0.13,  0.69 + 0.04, 0.52 + 0.15])
    })

    static_obj_glass = np.array([1.2945, 0.645, 0.52])

    ox = random.uniform(1.05, 1.55)
    oy = random.uniform(0.7, 1.1)
    oz = random.uniform(0.5, 0.9)
    desired_pos = np.array([ox, oy, oz])
    env.unwrapped.goal = desired_pos.copy()
    robot_pos = observation["achieved_goal"]
    
    #Definitions
    dt = 0.04  # FetchReach uses 25 Hz → dt ~0.04 s
    p = 0 #epoch counter
    i = 0 #loop counter
    x_exploration = 60
    x_improvement = 40
    choice = 'exploration'
    change = False
    size_perception_space = (1 + 1 + 1) #distance + distance glasses + obj in grip
    selected_goals_list = np.empty((0, size_perception_space))
    learned_goals_list = np.empty((0, size_perception_space))
    results_array = []
    points_list_poses = np.empty((0, size_perception_space))
    points_list_frustration = np.empty((0, size_perception_space))
    already_trained = np.empty((0, size_perception_space))
    memory_goals = []
    memory_colors = []
    selected_color = None
    obj_in_grip = 0

    motivations = Motivations(FILE, FILENAME)
    json_manager = GoalManager(FILENAME)
    effactance_class = Effactance()
    rnd = RNDModule(input_size=size_perception_space)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    world_model = WorldModel().to(device)
    optimizer = torch.optim.Adam(world_model.parameters(), lr=3e-4, weight_decay=1e-5)
    criterion = nn.MSELoss() 

    while True:
        if i == EPOCH_STEPS or (choice != "exploration" and thats_the_goal == True):
            if change == True:
                ox = random.uniform(1.4, 1.45)
                oy = random.uniform(0.45, 0.5)
                oz = random.uniform(0.55, 0.6)
                desired_pos = np.array([ox, oy, oz])
                env.unwrapped.goal = desired_pos.copy()
            else:
                ox = random.uniform(1.05, 1.55)
                oy = random.uniform(0.7, 1.1)
                oz = random.uniform(0.5, 0.6)
                desired_pos = np.array([ox, oy, oz])
                env.unwrapped.goal = desired_pos.copy()

        robot_pos = observation["achieved_goal"]

        #Definations
        i = 0
        loop_count = []
        actions_paths = []
        perception_paths = []
        best_um = []
        goal_found = False
        predictor = AdaptivePredictor() #Reset predictor for weights for novelty-frustration and UM functions
        points_list_poses = np.empty((0, size_perception_space))
        points_list_frustration = np.empty((0, size_perception_space))
        
        #Start distance
        distances = []

        distance = np.linalg.norm(robot_pos - desired_pos)
        norm_distance = normalize(distance)
        distances.append(norm_distance)
        
        distance = np.linalg.norm(robot_pos - static_obj_glass)
        norm_distance = normalize(distance)
        distances.append(norm_distance)

        start_point = np.array([*distances, obj_in_grip])
        points_list_poses = np.append(points_list_poses, [start_point], axis=0)
        reach_point_real = start_point.copy()

        while True:
            future_point_pose_list = []
            point_move_list = []
            i +=1
            obj_in_grip = 0
            thats_the_goal = False

            print("Number of iterations: ",+i)
            if i == EPOCH_STEPS:
                if choice == 'exploration' and goal_found == False:
                    if x_exploration != 0:
                        x_exploration -= 10
                        x_improvement += 10
                    if x_improvement > 100:
                        x_improvement = 100
                break

            if choice == 'exploration':
                competence = deque([0] * 20, maxlen=20)
                delta_competence = None

            robot_pos = observation["achieved_goal"]
            points = 0

            while points < 1000:
                points += 1
                obj_in_grip_per = 0
                #Pose move
                pose_plan = robot_pos.copy()
                # poslední hodnota (gripper) bude vždy 0.0
                vel_action = np.concatenate([
                    np.random.uniform(low=-1, high=1, size=3),
                    np.array([0.0])
                ])


                predicted_pos = pose_plan + vel_action[:3] * dt
                future_point_pose = np.array([predicted_pos[0], predicted_pos[1], predicted_pos[2]])
                if hit_the_glass_3d(obstacles_bounds, pose_plan, future_point_pose):
                    continue

                distances = []
                distances_move = []
                distances_stat = []

                distance = np.linalg.norm(future_point_pose - desired_pos)
                norm_distance = normalize(distance)
                distances.append(norm_distance)
                distances_move.append(norm_distance)

                distance = np.linalg.norm(future_point_pose - static_obj_glass)
                norm_distance = normalize(distance)
                distances.append(norm_distance)
                distances_stat.append(norm_distance)

                dist_array = np.array(distances)
                dist_array_move = np.array(distances_move)
                dist_array_stat = np.array(distances_stat)

                if np.any(dist_array_move < 0.05):
                    obj_in_grip_per = 1

                reach_point = np.array([*distances, obj_in_grip_per])

                if choice == 'exploration':
                    method = 'cur'
                    prediction = rnd.compute_curiosity(reach_point)
                    #prediction = motivations.novelty(reach_point, points_list_poses)
                elif choice == 'exploration_path':
                    method = 'nu'
                    #prediction = motivations.novelty(reach_point, points_list_poses)
                    nov_number = motivations.novelty(reach_point, points_list_poses)
                    um_val = um_entropy.predict_one_model([reach_point])
                    prediction = predictor.predict(nov_number, um_val)
                else:
                    method = 'um'
                    prediction = trained_model.predict([reach_point])

                if np.any(future_point_pose < limits_min) or np.any(future_point_pose > limits_max):
                    prediction = float("-inf")
                else:
                    future_point_pose_list.append(future_point_pose)
    
                to_list = vel_action.tolist() + [prediction]

                point_move_list.append(to_list)

            #Checking if list is empty or no
            if not point_move_list:
                print("Empty list")
                rand_action = np.random.uniform(low=-1.0, high=1.0, size=4)
                rand_action[3] = 0.0
                observation, reward, terminated, truncated, info = env.step(rand_action)
                continue

            #Sorting the list
            serazene_pole = sorted(point_move_list, key=lambda x: x[-1], reverse=True)
            vybrane_pole = serazene_pole[:10]
                
            for pole in vybrane_pole:
                [x, y, z, gripp, prediction]=pole
                prev_pos = observation["achieved_goal"]

                observation, reward, terminated, truncated, info = env.step(np.array([x, y, z, gripp]))

                new_pos  = observation["achieved_goal"]
                threshold = 1e-6
                if np.linalg.norm(new_pos - prev_pos) > threshold:
                    distances = []
                    distances_move = []
                    distances_stat = []

                    distance = np.linalg.norm(new_pos - desired_pos)
                    norm_distance = normalize(distance)
                    distances.append(norm_distance)
                    distances_move.append(norm_distance)
                    
                    distance = np.linalg.norm(new_pos - static_obj_glass)
                    norm_distance = normalize(distance)
                    distances.append(norm_distance)
                    distances_stat.append(norm_distance)
                    
                    dist_array_move = np.array(distances_move)
                    dist_array = np.array(distances)
                    dist_array_stat = np.array(distances_stat)

                    """
                    Here is when it check all the effactances
                    """
                    if reward == 0.0 and np.any(dist_array_move < 0.05):
                        obj_in_grip = 1
                        thats_the_goal = True
                    else:
                        obj_in_grip = 0
                    reach_point_real = np.array([*distances, obj_in_grip])
                    

                    """Testing UM functions"""
                    if choice == 'exploration_path':
                        nov_number = motivations.novelty(reach_point_real, points_list_poses)
                        frust_number = motivations.compute_frustration(points_list_frustration, reach_point_real)
                        ums_values_functions = um_entropy.predict_ums([reach_point_real])
                        best_um.append(ums_values_functions)

                    # if choice == 'exploration':
                    #     points_list_poses = np.append(points_list_poses, [reach_point_real], axis=0) #Add the expected point in the list, not just points of this looop
                    #     if points_list_poses.shape[0] > 50:
                    #         points_list_poses = np.delete(points_list_poses, 0, axis=0)

                    if choice == 'exploration_path':
                        points_list_poses = np.append(points_list_poses, [reach_point_real], axis=0) #Add the expected point in the list, not just points of this looop
                        frust_number = motivations.compute_frustration(points_list_frustration, reach_point_real)
                        predictor.update_weights(frust_number)
                        points_list_frustration = np.append(points_list_frustration, [reach_point_real], axis=0)
                        if points_list_poses.shape[0] > 50:
                            points_list_poses = np.delete(points_list_poses, 0, axis=0)
                        if points_list_frustration.shape[0] > 5:
                            points_list_frustration = np.delete(points_list_frustration, 0, axis=0)

                    perception_paths.append(reach_point_real)
                    action_path = np.array([x, y, z, gripp])
                    actions_paths.append(action_path)
                    loop_count.append(i)
                    if choice == 'exploration':
                        rnd.train_step(reach_point_real)
                    break

                else:
                    continue
        
            #Saving the goal
            json_manager = GoalManager(FILENAME)
            json_manager_pnodes = GoalManager(PNODES)
            if thats_the_goal == True and choice == 'exploration':
                """Saving the goal as txt and unique goal"""
                if is_file_empty(FILE):
                    goal_in = False
                    with open (FILE, 'a' ) as f:
                        goal_found = True
                        np.savetxt(f, reach_point_real, fmt='%s', newline=' ')
                        f.write('\n') 
                else:
                    data_file = np.loadtxt(FILE, dtype=float)
                    goal_in = False
                    if len(data_file.shape) == 1:
                        position_data = np.where(data_file < 0.05)[0]
                        position_reach = np.where(reach_point_real < 0.05)[0]
                        if np.array_equal(position_data, position_reach):
                            goal_in = True      
                    else:
                        for q in range(len(data_file)):
                            position_data = np.where(data_file[q] < 0.05)[0]
                            position_reach = np.where(reach_point_real < 0.05)[0]
                            if np.array_equal(position_data, position_reach): #and easy_count0 == easy_count1:
                                goal_in = True
                                break
                    if goal_in == False:
                        with open (FILE, 'a' ) as f:
                            goal_found = True
                            np.savetxt(f, reach_point_real, fmt='%s', newline=' ')
                            f.write('\n')
                
                """Saving the new goal to json"""
                if goal_in == False:
                    cp = 1
                    competence.append(cp)
                    delta_competence = motivations.competence_func(competence, PT=10)
                    selected_goal = reach_point_real
                    new_perceptions = np.array(perception_paths)
                    new_paths = np.array(actions_paths)
                    new_competence = np.array(competence)
                    json_manager.add_new_goal(selected_goal, new_perceptions, new_paths, new_competence, delta_competence)
                    json_manager_pnodes.add_p_node(selected_goal)
                    perception_paths = []
                    actions_paths = []
                    perception_paths.append(reach_point_real)
                    action_path = np.array([x, y, z])


            if thats_the_goal == True and choice != "exploration":
                break
        

        """Saving data to txt for see the results"""
        if choice == "exploration":
            results_array.append([p, choice, i, "None"])
        else:
            with open(PNODES, 'r') as m:
                data_model = json.load(m)
            model = None
            for entry in data_model:
                entry_goal_np = np.array(entry['goal'])
                if np.array_equal(entry_goal_np, selected_goal):
                    model = entry.get('model', None)
                    break
            results_array.append([p, choice, i, model])
        np.savetxt("cur_mlp_9.txt", results_array, fmt="%s")

        """Adding paths to existing goals"""
        json_manager = GoalManager(FILENAME)
        json_manager_ums = GoalManager(UMS)
        json_manager_traces = GoalManager(TRACES)
        if thats_the_goal == True and choice == "exploration_path": #Changed for exploration_path only because of overlearning
            cp = 1
            competence.append(cp)
            delta_competence = motivations.competence_func(competence, PT=10)
            new_perceptions = np.array(perception_paths)
            new_paths = np.array(actions_paths)
            new_competence = np.array(competence)
            json_manager.add_new_goal(selected_goal, new_perceptions, new_paths, new_competence, delta_competence)
            json_manager_ums.add_um_vals(selected_goal, best_um)

        if choice !="exploration" and i == 200 and thats_the_goal == False:
            cp = 0
            competence.append(cp)
            delta_competence = motivations.competence_func(competence, PT=10)
            new_perceptions = np.array(perception_paths)
            new_paths = np.array(actions_paths)
            new_competence = np.array(competence)
            json_manager.add_new_goal(selected_goal, new_perceptions, new_paths, new_competence, delta_competence)

        if choice == "improvement" and thats_the_goal == True:
            new_perceptions = np.array(perception_paths)
            json_manager_traces.add_new_traces(selected_goal, new_perceptions)

        if choice != "exploration" and len(competence_for_goal)>=20:
            last = np.array(competence_for_goal[-1])
            zeros_count = np.sum(last == 0)
            if zeros_count >= 15 and cp == 0: #Maybe change to 10
                rnd = RNDModule(input_size=size_perception_space)
                remove_goal_from_txt(FILE, selected_goal)
                remove_goal_from_txt(EXGOAL, selected_goal)
                remove_goal_from_pnodes(PNODES, selected_goal)
                learned_goals_list = [item for item in learned_goals_list if not np.allclose(item[0], selected_goal)]

        logits = np.array([x_exploration, x_improvement])
        choice = choose_motivation(logits)

        data = np.loadtxt(FILE, dtype=float)
        if data.shape[0] > 0:
            choice = 'improvement'
        else:
            choice = 'exploration'

        motivations = Motivations(FILE, FILENAME)
        if choice == 'improvement':
            selected_goal, selected_goals_list = motivations.select_goal(selected_goals_list)        

        if choice != 'exploration':
            old_choice = choice
            json_manager = GoalManager(FILENAME)
            json_manager_ums = GoalManager(UMS)
            json_manager_pnodes = GoalManager(PNODES)
            um = UtilityModel(FILENAME)
            um_entropy = UtilityModelEntropy(FILENAME)

            perception_for_goal, paths_for_goal, competence_for_goal, delta_competences_for_goal = json_manager.collect_competence(selected_goal)
            perception_for_goal_, paths_for_goal_, competence_for_goal_, delta_competences_for_goal_ = json_manager.collect_similar_goals(selected_goal)

            last_delta_competence = delta_competences_for_goal[-1]
            last_values = competence_for_goal[-1]
            competence = deque(last_values, maxlen=20)
            reached_last = last_values[-1]

            if last_delta_competence >= 0.8:
                if not any(np.allclose(selected_goal, goal) for goal in already_trained):
                    already_trained = np.vstack((already_trained, selected_goal))
                    values_test = json_manager_ums.get_ums_for_goal(selected_goal)
                    position_of_model = data_and_usg(values_test)
                    position_of_model = 3
                    json_manager_pnodes.add_model_to_goal(selected_goal, position_of_model)
                    train_model = um.neural_network(perception_for_goal_, position_of_model)
                    model_path = os.path.join(current_dir, 'models', f"model_goal_{str(selected_goal)}.pkl")
                    um.save_model(train_model, model_path)

                    learned_goals_list = np.append(learned_goals_list, [selected_goal], axis=0)

            elif last_delta_competence <= 0.2 and len(delta_competences_for_goal)>=10:
                last = np.array(competence_for_goal[-1])
                zeros_count = np.sum(last == 0)
                ones_count = np.sum(last == 1)
                if not any(np.allclose(selected_goal, goal) for goal in already_trained) and zeros_count < ones_count:
                    already_trained = np.vstack((already_trained, selected_goal))
                    values_test = json_manager_ums.get_ums_for_goal(selected_goal)
                    position_of_model = data_and_usg(values_test)
                    position_of_model = 3
                    json_manager_pnodes.add_model_to_goal(selected_goal, position_of_model)
                    train_model = um.neural_network(perception_for_goal_, position_of_model)
                    model_path = os.path.join(current_dir, 'models', f"model_goal_{str(selected_goal)}.pkl")
                    um.save_model(train_model, model_path)

                    learned_goals_list = np.append(learned_goals_list, [selected_goal], axis=0)
            
            else:
                choice = 'exploration_path'

            json_manager_pnodes = GoalManager(PNODES)

            if np.any(np.all(already_trained == selected_goal, axis=1)) and reached_last == 1:
                choice = old_choice
                model_path = os.path.join(current_dir, 'models', f"model_goal_{str(selected_goal)}.pkl")
                trained_model = um.load_model(model_path)

            else:
                if reached_last == 0:
                    mask = ~np.all(already_trained == selected_goal, axis=1)
                    already_trained = already_trained[mask]
                choice = 'exploration_path'
                train_model = um_entropy.train_ensemble(perception_for_goal_) #Fitting the models!
        
        observation, info = env.reset()
        p+=1
        print("Number of repeats: ",+p)

        if p == 150:
            change = True

        if p == 251:
            env.close()
            break

if __name__ == '__main__':
    try:
        main()
    except KeyboardInterrupt:
        print("Shutting down the program")