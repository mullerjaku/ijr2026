import json
import numpy as np
import os
 
class GoalManager:
    def __init__(self, filename):
        self.filename = filename
        self.data = self.load_from_json()
 
    def add_new_goal(self, goal, perception, paths, competence, delta_competence):
        new_goal = {
            "goal": goal.tolist(),
            "perception": [pe.tolist() for pe in perception],
            "paths": [pa.tolist() for pa in paths],
            "competence": competence.tolist(),
            "delta_competence": delta_competence
        }
        self.data.append(new_goal)
        self.save_to_json()

    def add_new_traces(self, goal, perception):
        new_goal = {
            "goal": goal.tolist(),
            "perception": [pe.tolist() for pe in perception]
        }
        self.data.append(new_goal)
        self.save_to_json()

    def add_p_node(self, goal, subgoal=None, model=None):
        new_goal = {
        "goal": goal.tolist(),
        "in_training": False,
        }
        if subgoal is not None:
            new_goal["subgoal"] = subgoal.tolist()
        if model is not None:
            new_goal["model"] = model
        self.data.append(new_goal)
        self.save_to_json()

    def add_um_vals(self, goal, ums_vals):
        new_goal = {
        "goal": goal.tolist(),
        "utility_values": ums_vals.tolist() if isinstance(ums_vals, np.ndarray) else ums_vals
        }
        self.data.append(new_goal)
        self.save_to_json()

    def get_ums_for_goal(self, selected_goal):
        ums_list = []
        for entry in self.data:
            if np.allclose(np.array(entry["goal"]), np.array(selected_goal)):
                if "utility_values" in entry:
                    ums_list.append(entry["utility_values"])
        return ums_list
    
    def add_subgoal_to_goal(self, goal, subgoal):
        for entry in self.data:
            if np.allclose(np.array(entry["goal"]), np.array(goal)):
                if "subgoal" not in entry:
                    entry["subgoal"] = subgoal.tolist() if isinstance(subgoal, np.ndarray) else subgoal
                    entry["in_training"] = False
                    self.save_to_json()
                return
            
    def add_model_to_goal(self, goal, model_type):
        for entry in self.data:
            if np.allclose(np.array(entry["goal"]), np.array(goal)):
                entry["model"] = int(model_type)
                self.save_to_json()
                return
            
    def get_subgoal_for_goal(self, goal):
        for entry in self.data:
            if np.allclose(np.array(entry["goal"]), np.array(goal)):
                return entry.get("subgoal", None)
        return None
    
    def subgoal_chain(self, array_subgoals, goal_name):
        current_goal = goal_name
        while True:
            subgoal = self.get_subgoal_for_goal(current_goal)
            if not subgoal:
                break
            array_subgoals.append(subgoal)
            current_goal = subgoal
        return array_subgoals
            
    def set_in_training(self, goal, val):
        for entry in self.data:
            if np.allclose(np.array(entry["goal"]), np.array(goal)):
                entry["in_training"] = val
                self.save_to_json()

    def filter_peception(self, perception_path, sub_goal): #Not ready yet!
        perception_filtred = []
        for path in perception_path:
            points_filtered = []
            for point in path:
                point = np.array(point)
                position_point = np.where(point < 0.006)[0]
                position_goal = np.where(sub_goal < 0.006)[0]
                if np.array_equal(position_goal, position_point):
                    points_filtered.append(point)
            last_point = np.array(path[-1])
            points_filtered.append(last_point)
            perception_filtred.append(points_filtered)
        return perception_filtred

    def get_traces_for_goal(self, goal):
        perception_array = []
        for item in self.data:
            if np.allclose(np.array(item['goal']), goal, atol=0.0):
                perception_array.append([np.array(pe) for pe in item["perception"]])
        return perception_array

    def save_to_json(self):
        with open(self.filename, 'w') as f:
            json.dump(self.data, f, indent=4)
 
    def load_from_json(self):
        if os.path.exists(self.filename) and os.path.getsize(self.filename) > 0:
            with open(self.filename, 'r') as f:
                return json.load(f)
        return []
 
    def collect_similar_goals(self, goal, tolerance=0.0):
        collected_perception = []
        collected_paths = []
        collected_competence = []
        collected_delta_competences = []
        for item in self.data:
            if np.allclose(np.array(item['goal']), goal, atol=tolerance) and (item['competence'][-1] == 1):
                collected_perception.append([np.array(pe) for pe in item['perception']])
                collected_paths.append([np.array(pa) for pa in item['paths']])
                #collected_competence.append([np.array(cp) for cp in item['competence']])
                collected_competence.append(item['competence'])
                collected_delta_competences.append(item['delta_competence'])
        return collected_perception, collected_paths, collected_competence, np.array(collected_delta_competences)
    
    def collect_competence(self, goal, tolerance=0.0):
        collected_perception = []
        collected_paths = []
        collected_competence = []
        collected_delta_competences = []
        for item in self.data:
            if np.allclose(np.array(item['goal']), goal, atol=tolerance):
                collected_perception.append([np.array(pe) for pe in item['perception']])
                collected_paths.append([np.array(pa) for pa in item['paths']])
                #collected_competence.append([np.array(cp) for cp in item['competence']])
                collected_competence.append(item['competence'])
                collected_delta_competences.append(item['delta_competence'])
        return collected_perception, collected_paths, collected_competence, np.array(collected_delta_competences)
 