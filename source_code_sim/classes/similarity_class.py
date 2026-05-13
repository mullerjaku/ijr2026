import numpy as np
import random
import torch
import torch.nn as nn
import torch.nn.functional as F
from json_class import GoalManager

# ===== Nastavení náhodného semene pro opakovatelnost =====
class SeedManager:
    @staticmethod
    def set_seed(seed=42):
        """Nastaví všechna náhodná semena pro reprodukovatelnost."""
        np.random.seed(seed)
        random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

#set_seed(42)

class MLPEncoder(nn.Module):
    def __init__(self, input_dim=10, embedding_dim=5):
        super(MLPEncoder, self).__init__()
        self.fc1 = nn.Linear(input_dim, 32)
        self.fc2 = nn.Linear(32, 16)
        self.fc3 = nn.Linear(16, embedding_dim)

    def forward(self, x):
        x = F.relu(self.fc1(x))   # 10 → 32
        x = F.relu(self.fc2(x))   # 32 → 16
        x = self.fc3(x)           # 16 → embedding_dim
        return x

class Similarity:
    def __init__(self):
        self.model = MLPEncoder(input_dim=7, embedding_dim=5)

        self.goals = np.loadtxt('data/Goals.txt', dtype=float)
        self.json_manager = GoalManager('data/Traces_UM.json')

        # ===================== SIMILARITY =====================
    def remove_traces_before_goal(self, traces, current_goal):
        if traces is None or len(traces) == 0:
            return traces
            
        # Najdeme všechny goals kromě current_goal
        pos_current_goal = np.where(current_goal < 0.006)[0]
        other_goals = [goal for goal in self.goals if not np.array_equal(np.where(goal < 0.006)[0], pos_current_goal)]
        
        last_goal_index = -1
        
        # Projdeme traces od konce a hledáme poslední výskyt jakéhokoli goalu
        for i in range(len(traces) - 1, -1, -1):
            trace = traces[i]
            
            # Porovnáme s každým goalem (kromě current_goal)
            for goal in other_goals:
                pos_trace = np.where(trace < 0.006)[0]
                pos_goal = np.where(goal < 0.006)[0]
                if np.array_equal(pos_trace, pos_goal):
                    last_goal_index = i
                    break
            
            # Pokud jsme našli goal, ukončíme hledání
            if last_goal_index != -1:
                break
        
        # Pokud jsme našli goal, vrátíme traces od následující pozice
        if last_goal_index != -1:
            return traces[last_goal_index + 1:]
        
        # Pokud nebyl nalezen žádný goal, vrátíme původní traces
        return traces


    def get_traces(self):
        embeddings = []
        for goal in self.goals:
            traces = self.json_manager.get_traces_for_goal(goal)
            if traces:
                just_check = np.vstack(traces)
                cleaned_traces = []
                for trace in traces:
                    cleaned_trace = self.remove_traces_before_goal(trace, goal)
                    if cleaned_trace is not None and len(cleaned_trace) > 0:
                        cleaned_traces.append(cleaned_trace)
                if cleaned_traces:
                    traces = np.vstack(cleaned_traces)
                    embedding = self.model(torch.tensor(traces, dtype=torch.float32))
                    embedding = torch.mean(embedding, dim=0)
                    embeddings.append(embedding)
        return embeddings
    
    def compute_similarity_matrix(self, embeddings):
        if not embeddings:
            return None
        
        # Stack all embeddings into a matrix [N, D]
        emb_tensor = torch.stack(embeddings)  # shape: [N, D]

        # Normalize each embedding vector (L2 norm)
        emb_tensor = F.normalize(emb_tensor, p=2, dim=1)

        # Compute cosine similarity as dot product of normalized vectors
        similarity_matrix = torch.matmul(emb_tensor, emb_tensor.T)  # shape: [N, N]
        return similarity_matrix
    
    def get_sorted_pairs(self, similarity_matrix):
        if similarity_matrix is None:
            return []
        
        pairs = []
        n = similarity_matrix.shape[0]
        
        # Projdeme všechny dvojice kromě diagonály
        for i in range(n):
            for j in range(i + 1, n):  # j > i, abychom nebrali duplicity
                value = similarity_matrix[i, j].item()
                pairs.append((i, j, value))
        
        # Seřadíme podle hodnoty sestupně
        pairs.sort(key=lambda x: x[2], reverse=True)
        
        return pairs

# all_similarity_matrices = []

# print("Počítám similarity matrices pro 100 různých seedů...")

# for i in range(100):
#     # Nastavení jiného seeda pro každý cyklus
#     set_seed(random.randint(0, 10000))
#     tri = Similarity()
    
#     # Přepočítání embeddings s novým seedem
#     emb = tri.get_traces()
    
#     # Výpočet similarity matrix
#     sim_matrix = tri.compute_similarity_matrix(emb)
    
#     if sim_matrix is not None:
#         all_similarity_matrices.append(sim_matrix)

# # Výpočet průměrné similarity matrix
# if all_similarity_matrices:
#     # Stack všech matrices a vypočítání průměru
#     stacked_matrices = torch.stack(all_similarity_matrices)
#     average_similarity_matrix = torch.mean(stacked_matrices, dim=0)
    
#     print("\nPrůměrná similarity matrix ze 100 běhů:")
#     print(average_similarity_matrix)

#     sorted_pairs = tri.get_sorted_pairs(average_similarity_matrix)
#     print(f"\nNejpodobnější dvojice goalů:")
#     for i, (idx1, idx2, similarity) in enumerate(sorted_pairs[:10]):
#         print(f"{i+1}. Goal {idx1} - Goal {idx2}: {similarity:.4f}")
# else:
#     print("Nepodařilo se vypočítat žádné similarity matrices")
