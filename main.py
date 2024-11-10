import numpy as np
import pandas as pd
import math
from environment import Environment, Robot
from EKF import StateEstimator
from rhoPOMDP import SeqDec


def main():
    Env = Environment()
    EKF = StateEstimator()
    Controller = SeqDec()

    while not Env.reached_goal_:
        sensor_readings = Env.sensor()
        s_t_1 = EKF.s_t_1()      #To update the values at the end where current state estimated - EKF.py
        prev_action = Controller.prev_actions() #To update the values at the end where current action calculated -POMDP.py

        '''
        Estimate the current state and pass it to 
        '''
        cur_es_state = EKF.estimate(sensor_readings, s_t_1, prev_action)

        action_next = Controller.decision(cur_es_state)
        Env.update()



if __name__ == '__main__':
    main()

