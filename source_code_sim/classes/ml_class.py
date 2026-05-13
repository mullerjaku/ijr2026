import json
import joblib
import numpy as np
from sklearn.svm import SVR
from sklearn.neural_network import MLPRegressor
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import GradientBoostingRegressor

class UtilityModel:
    def __init__(self, json_file_path):
        self.data = self.initialize_data(json_file_path)

    def initialize_data(self, json_file_path):
        with open(json_file_path, 'r') as file:
            data = json.load(file)
        return data

    def neural_network(self, paths, model_type):
        if model_type == 0:
            model = LinearRegression()
        elif model_type == 1:
            model = SVR(kernel='rbf', C=1.0, epsilon=0.1)
        elif model_type == 2:
            model = GradientBoostingRegressor(n_estimators=200, learning_rate=0.1, max_depth=5)
        elif model_type == 3:
            model = MLPRegressor(hidden_layer_sizes=(16,), max_iter=500, early_stopping=False)
        else:
            model = SVR(kernel='rbf', C=1.0, epsilon=0.1)

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

        if len(flattened_paths) > 0:
            model.fit(flattened_paths, y)
            
        return model
    
    def save_model(self, trained_model, file_path):
        with open(file_path, 'wb') as file:
            joblib.dump(trained_model, file, protocol=5)

    def load_model(self, file_path):
        with open(file_path, 'rb') as file:
            selected_model = joblib.load(file_path)
        return selected_model
    

# import json
# import joblib
# import numpy as np
# from sklearn.svm import SVR
# from sklearn.neural_network import MLPRegressor
# from sklearn.linear_model import LinearRegression
# from sklearn.ensemble import GradientBoostingRegressor

# class UtilityModel:
#     def __init__(self, json_file_path):
#         self.data = self.initialize_data(json_file_path)

#     def initialize_data(self, json_file_path):
#         with open(json_file_path, 'r') as file:
#             data = json.load(file)
#         return data

#     def neural_network(self, paths, model_type):
#         if model_type == 0:
#             model = LinearRegression()
#         elif model_type == 1:
#             model = SVR(kernel='rbf', C=1.0, epsilon=0.1)
#         elif model_type == 2:
#             model = GradientBoostingRegressor(n_estimators=200, learning_rate=0.1, max_depth=5)
#         elif model_type == 3:
#             model = MLPRegressor(hidden_layer_sizes=(16,), max_iter=500, early_stopping=False)
#         else:
#             model = SVR(kernel='rbf', C=1.0, epsilon=0.1)

#         # Zkracuje cesty na 9
#         for i, sublist in enumerate(paths):
#             if len(sublist) > 18:
#                 paths[i] = sublist[-18:]

#         y = np.array([])
#         for i, sublist in enumerate(paths):
#             #val = np.array([0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,1.0]) 
#             val = np.array([0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 1.0])
#             length = len(sublist)
#             if length < 18:
#                 delete = 18 - length
#                 val = val[delete:]
#                 y = np.append(y, val)
#             else:
#                 y = np.append(y, val)

#         # Převod všech vnořených seznamů numpy polí na numpy pole
#         flattened_paths = np.concatenate(paths)
#         model.fit(flattened_paths, y)
#         return model
    
#     def save_model(self, trained_model, file_path):
#         with open(file_path, 'wb') as file:
#             joblib.dump(trained_model, file, protocol=5)

#     def load_model(self, file_path):
#         with open(file_path, 'rb') as file:
#             selected_model = joblib.load(file_path)
#         return selected_model
