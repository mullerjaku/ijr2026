import numpy as np

class AdaptivePredictor:
    def __init__(self, initial_w_nov=0.0, initial_w_um=1.0, alpha=0.05):
        self.frust_history = []
        self.window = 5
        self.w_nov = initial_w_nov
        self.w_um = initial_w_um
        self.alpha = alpha
        self.mode = "um"
        self.novelty_steps = 0  # Počet kroků, kdy je aktivní novelty

    def update_weights(self, frust_number):
        self.frust_history.append(frust_number)
        if len(self.frust_history) > self.window:
            self.frust_history.pop(0)

        if len(self.frust_history) == self.window:
            diffs = np.diff(self.frust_history)
            if np.all(diffs > 0):
                self.mode = "novelty"
                self.w_nov = 0.7
                self.w_um = 0.3
            elif np.all(diffs < 0):
                self.mode = "um"
                self.w_nov = 0.3
                self.w_um = 0.7

    def predict(self, nov_number, um_val):
        prediction = self.w_nov * nov_number + self.w_um * um_val
        return prediction