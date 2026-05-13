import numpy as np
from fanucpy import Robot
import time

#Josef Arenštein, Adam Kič#

robot = Robot(
    robot_model="Fanuc",
    host="192.168.1.3",
    port=18735,
    ee_DO_type="RDO",
    ee_DO_num=7,
)

robot.connect()

#robot.call_prog("GRIPPER")

# get robot state
print("Current poses: ")
cur_pos = robot.get_curpos()
cur_jpos = robot.get_curjpos()
print(f"Current pose: {cur_pos}")
print(f"Current joints: {cur_jpos}")

#X 491.247, 810.815
#Y -237.103, 367.678
#Z -760.372, -397.568