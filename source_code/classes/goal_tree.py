"""TODO: přidat jak se vytvoří strom a následně případně řešit i více sub-goalů než 2"""
import json
import os
import torch
import random
from similarity_class import Similarity
from similarity_class import SeedManager

class GoalTree:
    def __init__(self, json_path="data/Goal_Tree.json", pnodes_path="data/PNodes.json"):
        self.json_path = json_path
        self.pnodes_path = pnodes_path
        self.tree = self.build_tree_from_pnodes()
        self.save_goal_tree()

    def build_tree_from_pnodes(self):
        """Načti PNodes.json a vytvoř strom goal/subgoal struktury."""
        if not os.path.exists(self.pnodes_path):
            return {"goals": []}
    
        with open(self.pnodes_path, "r", encoding="utf-8") as f:
            pnodes = json.load(f)
    
        tree = {"goals": []}
    
        # pomocná rekurzivní funkce pro subgoaly
        def build_subgoals(subnode, parent_name, level=1):
            """
            subnode může být dict nebo list:
            - dict: obsahuje vector a případně 'subgoals'
            - list: více subgoalů
            """
            subgoals_list = []
    
            if isinstance(subnode, list):
                # víc subgoalů
                for i, s in enumerate(subnode, start=1):
                    sub_name = f"{parent_name} → Subgoal {i}"
                    subgoals_list.append({
                        "name": sub_name,
                        "vector": s.get("goal") if isinstance(s, dict) else s,
                        "subgoals": build_subgoals(s.get("subgoal", []), sub_name, level+1)
                    })
            elif isinstance(subnode, dict):
                # jeden subgoal
                sub_name = f"{parent_name} → Subgoal {level}"
                subgoals_list.append({
                    "name": sub_name,
                    "vector": subnode.get("goal"),
                    "subgoals": build_subgoals(subnode.get("subgoal", []), sub_name, level+1)
                })
            return subgoals_list
    
        # hlavní gólová smyčka
        for idx, node in enumerate(pnodes, start=1):
            goal_name = f"Goal {idx}"
            goal_vec = node["goal"]
    
            goal_entry = {
                "name": goal_name,
                "vector": goal_vec,
                "subgoals": build_subgoals(node.get("subgoal", []), goal_name)
            }
    
            tree["goals"].append(goal_entry)
    
        return tree

    def save_goal_tree(self):
        with open(self.json_path, "w", encoding="utf-8") as f:
            json.dump(self.tree, f, ensure_ascii=False, indent=4)

    def merge_similar_goals(self, similarity_matrix, sorted_pairs, threshold=0.999):
        """Sloučí góly se similarity > threshold."""
        merged = set()

        for i, j, sim in sorted_pairs:
            if sim < threshold:
                break
            if i in merged or j in merged:
                continue

            goal_i = self.tree["goals"][i]
            goal_j = self.tree["goals"][j]

            new_name = f"{goal_i['name']} + {goal_j['name']}"
            new_subgoals = goal_i.get("subgoals", []) + goal_j.get("subgoals", [])

            merged_goal = {
                "name": new_name,
                "vector": goal_i["vector"],  # můžeme vzít první
                "subgoals": new_subgoals
            }

            self.tree["goals"][i] = merged_goal
            self.tree["goals"][j] = None
            merged.update([i, j])

        self.tree["goals"] = [g for g in self.tree["goals"] if g is not None]
        self.save_goal_tree()

test = GoalTree()
sim = Similarity()

all_similarity_matrices = []
for i in range(100):
    SeedManager.set_seed(random.randint(0, 10000))
    tri = Similarity()
    emb = tri.get_traces()
    sim_matrix = tri.compute_similarity_matrix(emb)
    if sim_matrix is not None:
        all_similarity_matrices.append(sim_matrix)

if all_similarity_matrices:
    stacked_matrices = torch.stack(all_similarity_matrices)
    average_similarity_matrix = torch.mean(stacked_matrices, dim=0)

    print("\nPrůměrná similarity matrix ze 100 běhů:")
    print(average_similarity_matrix)

    sorted_pairs = tri.get_sorted_pairs(average_similarity_matrix)
    print(f"\nNejpodobnější dvojice goalů:")
    for i, (idx1, idx2, similarity) in enumerate(sorted_pairs[:10]):
        print(f"{i+1}. Goal {idx1} - Goal {idx2}: {similarity:.4f}")

    # vytvoř strom a slouč pokud je potřeba
    test.merge_similar_goals(average_similarity_matrix, sorted_pairs, threshold=0.999)

else:
    print("Nepodařilo se vypočítat žádné similarity matrices")