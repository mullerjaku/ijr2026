from skimage.metrics import structural_similarity
import cv2
import numpy as np
from ollama import Client
import re


class Effactance:
    def __init__(self):
        self.before = None
        self.after = None
        self.old_color = 'purple'

    def load_images(self, before_image, after_image):
        # Načíst obrázky z cest
        self.before = cv2.imread(before_image)
        self.after = cv2.imread(after_image)

    def compute_difference(self):
        # Zkontrolujte, zda byly obrázky načteny
        if self.before is None or self.after is None:
            raise ValueError("Obrázky nejsou načteny. Použijte load_images pro načtení obrázků.")

        # Převod obrázků na odstíny šedi
        before_gray = cv2.cvtColor(self.before, cv2.COLOR_BGR2GRAY)
        after_gray = cv2.cvtColor(self.after, cv2.COLOR_BGR2GRAY)

        # Výpočet SSIM mezi dvěma obrázky
        (score, diff) = structural_similarity(before_gray, after_gray, full=True)
        print(f"Image Similarity: {score * 100:.4f}%")

        # Obrázek rozdílů obsahuje skutečné rozdíly mezi obrázky
        diff = (diff * 255).astype("uint8")

        # Prahování rozdílového obrázku a nalezení kontur
        thresh = cv2.threshold(diff, 0, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)[1]
        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        # Vytvoření masky
        mask = np.zeros(self.before.shape, dtype='uint8')

        for c in contours:
            area = cv2.contourArea(c)
            if area > 40:  # Ignoruj malé oblasti
                cv2.drawContours(mask, [c], 0, (255,255,255), -1)

        # Počet pixelů s rozdílem
        difference_pixels = np.sum(mask > 0)  # Počet pixelů, které nejsou černé (0)

        # Celkový počet pixelů
        total_pixels = mask.shape[0] * mask.shape[1]

        # Výpočet procenta rozdílových pixelů
        percentage_difference = (difference_pixels / total_pixels) * 100
        return percentage_difference
    
    def do_boxes_overlap(self, box1, box2):
        x_min1, y_min1, x_max1, y_max1 = box1
        x_min2, y_min2, x_max2, y_max2 = box2

        if x_min1 <= x_max2 and x_max1 >= x_min2 and y_min1 <= y_max2 and y_max1 >= y_min2:
            return True
        return False

    def check_overlap_with_robot(self, bb_robot, bounding_boxes):
        for i, box in enumerate(bounding_boxes):
            if self.do_boxes_overlap(bb_robot, box):
                #print(f"Robot (bb_robot) se překrývá s Boxem {i}.")
                return True
        return False
    
    def check_overlap_with_robot_and_multiple_boxes(self, bb_robot, bounding_boxes):
        """
        Vrací True, pokud se robot překrývá s alespoň dvěma různými boxy ze seznamu bounding_boxes.
        """
        overlap_count = 0
        for i, box in enumerate(bounding_boxes):
            if self.do_boxes_overlap(bb_robot, box):
                overlap_count += 1
            if overlap_count >= 2:
                #print("Robot se překrývá s alespoň dvěma boxy.")
                return True
        return False
    
    def change_color(self, color):
        if color == self.old_color:
            return False
        else:
            self.old_color = color
            return True
        
    def llm(self, all_objects_info, user_task):
        client = Client()

        prompt = self.build_prompt(all_objects_info, user_task)

        response = client.generate(
            #model='qwen3:8b',
            model='gpt-oss:20b',
            prompt=prompt
        )

        def extract_true_false(response_text: str) -> str:
            # Remove any content between <think>...</think> tags (including the tags)
            cleaned = re.sub(r'<think>.*?</think>', '', response_text, flags=re.DOTALL | re.IGNORECASE)

            # Extract "True" or "False" from the cleaned response
            match = re.search(r'\b(True|False)\b', cleaned, re.IGNORECASE)
            return match.group(1).capitalize() if match else "Unknown"

        answer = extract_true_false(response['response'])

        if "true" in answer.lower():
            print("Odpověď je True")
            return True
        elif "false" in answer.lower():
            print("Odpověď je False")
            return False
        else:
            print("Odpověď není jednoznačná:", answer)
            return None
        
    def build_prompt(self, environment, user_task):
        env_description = "The environment contains the following objects:\n"
        for i, [obj, color] in enumerate(environment):
            xmin, ymin, xmax, ymax = obj
            env_description += (
                f"Object {i+1}: color={color}, xmin={xmin}, xmax={xmax}, "
                f"ymin={ymin}, ymax={ymax}\n"
            )
        env_description += (
            "\nNotes:\n"
            "- The black object is the robot.\n"
            "- Green and red objects are graspable.\n"
            "- Blue and gray objects are non-graspable.\n"
            "- The purple/brown object is a button and is changing the color.\n"
            "- Partial overlap is considered as full overlap.\n"
            "- If more objects overlap with the robot than the user intends to interact with, the answer must be False. The robot itself is not considered an object.\n"
        )

        prompt = f"""{env_description}
    User task:
    {user_task}

    Question:
    Did the environment reflect what the user task is?

    Answer only 'True' or 'False' based on the environment and task above.
    """
        return prompt

