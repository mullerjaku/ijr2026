import torch
import torch.nn as nn
import torch.optim as optim

from world_model_class import WorldModel 

def load_training_data(filename):
    data = []
    with open(filename, "r") as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) == 0:
                continue
            data.append([float(x) for x in parts])

    data = torch.tensor(data, dtype=torch.float32)
    
    # Kontrola, zda sedí data (přizpůsobte dle potřeby)
    # assert data.shape[1] == 15, f"Neočekávaný počet sloupců: {data.shape[1]}"

    states = data[:, 0:2]
    # predicted_states = data[:, 4:8]   # nepoužívá se
    actions = data[:, 8:10]
    next_states = data[:, 11:13]

    return states, actions, next_states

# --- KONFIGURACE ---
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
state_dim = 2
action_dim = 2

# 1. ZMĚNA: Inicializace nového modelu
# in_dim = state + action (2+2=4), out_dim = state (2)
model = WorldModel(in_dim=state_dim + action_dim, hidden_dim=256, out_dim=state_dim).to(device)
criterion = nn.GaussianNLLLoss(full=True)

optimizer = torch.optim.Adam(model.parameters(), lr=3e-4, weight_decay=1e-5)
metric_mae = nn.L1Loss(reduction="none")

model.train()

# Načtení dat
states, actions, next_states = load_training_data("train.txt")
val_states, val_actions, val_next_states = load_training_data("val.txt")

# Přesun na device
states = states.to(device)
actions = actions.to(device)
next_states = next_states.to(device)
val_states = val_states.to(device)
val_actions = val_actions.to(device)
val_next_states = val_next_states.to(device)

epochs = 500
batch_size = 64
dataset_size = states.shape[0]

best_metric = float("inf")
best_model_path = "/home/citic_gii/my_intrinsic_motivation/model_wm/world_model.pth"

print(f"Start tréninku na zařízení: {device}")

for epoch in range(epochs):
    model.train() # Ujistíme se, že jsme v train módu
    perm = torch.randperm(dataset_size, device=device)
    
    total_loss = 0

    for i in range(0, dataset_size, batch_size):
        idx = perm[i:i + batch_size]

        s = states[idx]
        a = actions[idx]
        ns = next_states[idx]

        mu, var = model(s, a)
        target_delta = ns - s
        loss = criterion(mu, target_delta, var)
        loss = loss + 1e-4 * (mu.pow(2).mean() + var.pow(2).mean())
        optimizer.zero_grad()
        loss.backward()
        
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        
        optimizer.step()
        total_loss += loss.item()

    # --- VALIDACE ---
    if epoch % 20 == 0:
        model.eval()
        with torch.no_grad():
            # Získáme parametry distribuce pro validační set
            val_mu, val_logvar = model(val_states, val_actions)
            
            # 7. ZMĚNA: Rekonstrukce absolutního stavu pro metriku
            # Predikovaný stav = Starý stav + Průměrná změna (mu)
            val_pred_abs = val_states + val_mu
            
            # MAE počítáme mezi "rekonstruovanou predikcí" a "reálným dalším stavem"
            per_dim_error = torch.mean(torch.abs(val_pred_abs - val_next_states), dim=0)
            
            # Metrika pro uložení modelu (součet chyb ve všech dimenzích)
            val_metric = per_dim_error.sum().item()
            
            # Průměrná loss za epochu
            avg_train_loss = total_loss / (dataset_size / batch_size)

        print(f"\nEpoch {epoch}")
        print(f"Train NLL Loss: {avg_train_loss:.6f}")
        print(f"Val metric (sum MAE): {val_metric:.6f}")
        print(f"Per-dim MAE: {per_dim_error.tolist()}")
        
        # Výpis nejistoty modelu (volitelné, ale užitečné)
        avg_std = torch.exp(0.5 * val_logvar).mean().item()
        print(f"Avg Model Uncertainty (std): {avg_std:.6f}\n")

        if val_metric < best_metric:
            best_metric = val_metric

            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "val_mae_per_dim": per_dim_error.cpu().numpy().tolist(),
                    "val_metric": best_metric,
                    "epoch": epoch
                },
                best_model_path
            )

            print(f"--> Uložen nový BEST MODEL! Metric = {best_metric:.6f}")

print("\nTrénink dokončen.")
print(f"Nejlepší model uložen zde:\n{best_model_path}")

# import torch
# import torch.nn as nn
# import torch.optim as optim

# from world_model_class import WorldModel 

# def load_training_data(filename):
#     data = []
#     with open(filename, "r") as f:
#         for line in f:
#             parts = line.strip().split()
#             if len(parts) == 0:
#                 continue
#             data.append([float(x) for x in parts])

#     data = torch.tensor(data, dtype=torch.float32)

#     assert data.shape[1] == 15, f"Neočekávaný počet sloupců: {data.shape[1]}"

#     states = data[:, 0:2]
#     predicted_states = data[:, 4:8]   # nepoužívá se
#     actions = data[:, 8:10]
#     next_states = data[:, 11:13]

#     return states, actions, next_states

# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# state_dim = 2
# action_dim = 2

# model = WorldModel(state_dim=state_dim, action_dim=action_dim).to(device)
# optimizer = optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)

# criterion = nn.SmoothL1Loss()   # loss pro trénování
# metric_mae = nn.L1Loss(reduction="none")   # MAE, bez redukce – použijeme vlastní výpočet

# model.train()

# states, actions, next_states = load_training_data("train.txt")
# val_states, val_actions, val_next_states = load_training_data("val.txt")

# # přesun na device
# states = states.to(device)
# actions = actions.to(device)
# next_states = next_states.to(device)
# val_states = val_states.to(device)
# val_actions = val_actions.to(device)
# val_next_states = val_next_states.to(device)

# epochs = 500
# batch_size = 64
# dataset_size = states.shape[0]

# best_metric = float("inf")
# best_model_path = "/home/citic_gii/my_intrinsic_motivation/model_wm/world_model.pth"

# for epoch in range(epochs):

#     perm = torch.randperm(dataset_size, device=device)

#     for i in range(0, dataset_size, batch_size):
#         idx = perm[i:i + batch_size]

#         s = states[idx]
#         a = actions[idx]
#         ns = next_states[idx]

#         pred = model(s, a)
#         loss = criterion(pred, ns)

#         optimizer.zero_grad()
#         loss.backward()
#         optimizer.step()

#     if epoch % 20 == 0:
#         model.eval()
#         with torch.no_grad():
#             val_pred = model(val_states, val_actions)
#             # 1) MAE per dimension  -> shape: [4]
#             per_dim_error = torch.mean(torch.abs(val_pred - val_next_states), dim=0)
#             # 2) Metrika pro výběr nejlepšího modelu
#             val_metric = per_dim_error.sum().item()

#         print(f"\nEpoch {epoch}")
#         print(f"Train Loss: {loss.item():.6f}")
#         print(f"Val metric (sum of dim MAE): {val_metric:.6f}")
#         print(f"Per-dim MAE: {per_dim_error.tolist()}\n")

#         if val_metric < best_metric:
#             best_metric = val_metric

#             torch.save(
#                 {
#                     "model_state_dict": model.state_dict(),
#                     "optimizer_state_dict": optimizer.state_dict(),
#                     "val_mae_per_dim": per_dim_error.cpu().numpy().tolist(),
#                     "val_metric": best_metric,
#                     "epoch": epoch
#                 },
#                 best_model_path
#             )

#             print(f"Uložen nový BEST MODEL! Metric = {best_metric:.6f}, epoch = {epoch}")

#         model.train()

# print("\nTrénink dokončen.")
# print(f"Nejlepší model uložen zde:\n{best_model_path}")