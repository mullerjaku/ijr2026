import json
import numpy as np
from sklearn.svm import SVR
from sklearn.neural_network import MLPRegressor
from sklearn.kernel_ridge import KernelRidge
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import GradientBoostingRegressor

class UtilityModelEntropy:
    def __init__(self, json_file_path):
        self.data = self.initialize_data(json_file_path)
        
        self.models = [
            LinearRegression(),
            SVR(kernel='rbf', C=1.0, epsilon=0.1),
            GradientBoostingRegressor(n_estimators=200, learning_rate=0.1, max_depth=5),
            MLPRegressor(hidden_layer_sizes=(16,), max_iter=500, early_stopping=False)
        ]

    def initialize_data(self, json_file_path):
        with open(json_file_path, 'r') as file:
            data = json.load(file)
        return data

    def prepare_training_data(self, paths):
        y = np.array([])
        gamma = 0.95 

        for sublist in paths:
            length = len(sublist)
            if length == 0:
                continue

            exponents = np.arange(length)[::-1]
            val = np.power(gamma, exponents)
            y = np.append(y, val)

        flattened_paths = np.concatenate(paths)
        
        return flattened_paths, y

    def train_ensemble(self, paths):
        X, y = self.prepare_training_data(paths)
        
        if len(X) == 0:
            print("Warning: No data to train on.")
            return

        for model in self.models:
            model.fit(X, y)

    def predict_entropy(self, perception_vector):
        preds = []
        for model in self.models:
            pred = model.predict(perception_vector)
            preds.append(pred[0])
            
        std = np.std(preds)
        mean_val = np.mean(preds)
        
        result = mean_val - std
        return result
    
    def predict_one_model(self, perception_vector, num_model=3):
        pred = self.models[num_model].predict(perception_vector)
        return pred
    
    def predict_ums(self, perception_vector):
        preds = []
        for model in self.models:
            pred = model.predict(perception_vector)
            preds.append(pred[0])
        return preds
    

    
# import json
# import numpy as np
# from sklearn.svm import SVR
# from sklearn.neural_network import MLPRegressor
# from sklearn.kernel_ridge import KernelRidge
# from sklearn.linear_model import LinearRegression
# from sklearn.tree import DecisionTreeRegressor
# from sklearn.ensemble import GradientBoostingRegressor


# class UtilityModelEntropy:
#     def __init__(self, json_file_path):
#         self.data = self.initialize_data(json_file_path)
#         # self.models = [
#         #     SVR(kernel='rbf', C=1.0, epsilon=0.1),
#         #     SVR(kernel='poly', degree=2, C=1.0, epsilon=0.1),
#         #     MLPRegressor(hidden_layer_sizes=(32,), max_iter=500),
#         #     KernelRidge(kernel='rbf', alpha=1.0)
#         # ]
#         self.models = [
#             LinearRegression(),
#             SVR(kernel='rbf', C=1.0, epsilon=0.1),
#             GradientBoostingRegressor(n_estimators=200, learning_rate=0.1, max_depth=5),
#             MLPRegressor(hidden_layer_sizes=(16,), max_iter=500, early_stopping=False)
#         ]

#     def initialize_data(self, json_file_path):
#         with open(json_file_path, 'r') as file:
#             data = json.load(file)
#         return data

#     def prepare_training_data(self, paths):
#         # Zkracuje cesty na délku 18
#         for i, sublist in enumerate(paths):
#             if len(sublist) > 18:
#                 paths[i] = sublist[-18:]

#         y = np.array([])
#         for sublist in paths:
#             #val = np.array([0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,1.0])
#             val = np.array([0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 1.0])
#             length = len(sublist)
#             if length < 18:
#                 delete = 18 - length
#                 val = val[delete:]
#                 y = np.append(y, val)
#             else:
#                 y = np.append(y, val)

#         # Sloučíme všechny stavy do jednoho trénovacího pole
#         flattened_paths = np.concatenate(paths)
#         return flattened_paths, y

#     def train_ensemble(self, paths):
#         X, y = self.prepare_training_data(paths)
#         for model in self.models:
#             model.fit(X, y)

#     def predict_entropy(self, perception_vector):
#         #preds = np.array([model.predict(perception_vector)[0] for model in self.models])
#         preds = []
#         for model in self.models:
#             pred = model.predict(perception_vector)
#             preds.append(pred[0])
#         std = np.std(preds)
#         mean_val = np.mean(preds)
#         result = mean_val - std
#         return result
    
#     def predict_ums(self, perception_vector):
#         #preds = np.array([model.predict(perception_vector)[0] for model in self.models])
#         preds = []
#         for model in self.models:
#             pred = model.predict(perception_vector)
#             preds.append(pred[0])
#         return preds