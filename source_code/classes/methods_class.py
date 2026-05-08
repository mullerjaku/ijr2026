import numpy as np
from collections import deque
from classes.json_class import GoalManager

class Motivations:
    def __init__(self, goals_file, json_file, epsilon=0.3):
        self.goals_file = goals_file
        self.json_manager = GoalManager(json_file)
        self.epsilon = epsilon
        self.competence = deque([0] * 20, maxlen=20)
        self.goals_interest = np.loadtxt(goals_file, dtype=float)

    def diversification(self, point, goals):
        distances = []
        for goal in goals:
            if not np.array_equal(goal, point):
                dist = np.linalg.norm(goal - point)
                distances.append(dist)
        return np.mean(distances) if distances else 0

    def novelty(self, point, points_poses_list):
        novelty_list = []
        for one_point in points_poses_list:
            d_novelty_points = np.linalg.norm(one_point - point)
            novelty_list.append(d_novelty_points)
        return (sum(novelty_list)) / len(points_poses_list)

    def competence_func(self, history, PT=10):
        history = np.array(history)
        recent_avg = np.mean(history[-PT:])
        past_avg = np.mean(history[-2*PT:-PT])
        return recent_avg - past_avg

    def interest(self, point, points_poses_list, history, goals, weights=(0.5, 0.5, 0.1)): #Hlavní 0.5, 0.25, 0.1
        novelty_score = self.novelty(point, points_poses_list)
        competence_score = self.competence_func(history)
        diversification_score = self.diversification(point, goals)

        interest_score = (
            weights[0] * novelty_score +
            weights[1] * competence_score +
            weights[2] * (1 - diversification_score)
        )

        return interest_score
    
    def compute_frustration(self, original_states, new_state):
        updated_states = np.vstack((original_states, new_state))
        M = updated_states.shape[0]
        mean_state = np.mean(updated_states, axis=0)
        variance = np.sum(np.sum((updated_states - mean_state) ** 2, axis=1)) / M
        std_deviation = np.sqrt(variance)

        # # Frustrace (F)
        frustration = 1 - std_deviation
        return frustration

    def select_goal(self, selected_goals_list):
        if self.goals_interest.size == 0:
            return None

        if len(self.goals_interest.shape) == 1:
            selected_goal = self.goals_interest
            selected_goals_list = np.append(selected_goals_list, [selected_goal], axis=0)
            print("Selected goal: ", selected_goal)
            return selected_goal, selected_goals_list

        if np.random.random() < self.epsilon:
            selected_goal = self.goals_interest[np.random.randint(0, len(self.goals_interest))]
        else:
            prev_interest = -np.inf

            if selected_goals_list.size == 0:
                for goal in self.goals_interest:
                    selected_goals_list = np.append(selected_goals_list, [goal], axis=0)

            for goal in self.goals_interest:
                per, path, compete, delta = self.json_manager.collect_similar_goals(goal)
                flattened_list = [item for sublist in compete for item in sublist]
                last_values = flattened_list[-20:]
                self.competence = deque(last_values, maxlen=20)

                if len(self.competence) == 0:
                    self.competence = deque([0] * 20, maxlen=20)

                interest_value = self.interest(goal, selected_goals_list, self.competence, self.goals_interest)
                if interest_value > prev_interest:
                    prev_interest = interest_value
                    selected_goal = goal
                #print("Curiosity value: ", interest_value)

        selected_goals_list = np.append(selected_goals_list, [selected_goal], axis=0)
        print("Selected goal: ", selected_goal)
        return selected_goal, selected_goals_list
    
    def select_goal_input_learned(self, selected_goals_list, learned_goals_list=None):
        if self.goals_interest.size == 0:
            return None

        if learned_goals_list is not None and learned_goals_list.size > 0:
            filtered_goals = []
            for goal in self.goals_interest:
                if not any(np.array_equal(goal, learned) for learned in learned_goals_list):
                    filtered_goals.append(goal)
            self.goals_interest = np.array(filtered_goals) if filtered_goals else np.empty((0, self.goals_interest.shape[-1]))

        if np.random.random() < self.epsilon:
            selected_goal = self.goals_interest[np.random.randint(0, len(self.goals_interest))]
        else:
            prev_interest = -np.inf
            if selected_goals_list.size == 0:
                for goal in self.goals_interest:
                    selected_goals_list = np.append(selected_goals_list, [goal], axis=0)

            for goal in self.goals_interest:
                per, path, compete, delta = self.json_manager.collect_similar_goals(goal)
                flattened_list = [item for sublist in compete for item in sublist]
                last_values = flattened_list[-20:]
                self.competence = deque(last_values, maxlen=20)

                if len(self.competence) == 0:
                    self.competence = deque([0] * 20, maxlen=20)

                interest_value = self.interest(goal, selected_goals_list, self.competence, self.goals_interest)
                if interest_value > prev_interest:
                    prev_interest = interest_value
                    selected_goal = goal

        selected_goals_list = np.append(selected_goals_list, [selected_goal], axis=0)
        print("Selected goal:", selected_goal)
        return selected_goal, selected_goals_list

