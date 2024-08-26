from enum import Enum
import math
import numpy as np
import random
import torch
from torchvision.transforms import functional as F

from .models import PuckDetector, load_model


def q_to_yaw(q):
    """
    Convert quaternion to euler yaw angle (-180 to 180)
    """
    y, z, x, w = [float(i) for i in q]
    return math.atan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z)) / math.pi * 180.0


def distance_between_points(p1, p2):
    return math.sqrt(math.pow(p1[0] - p2[0], 2) + math.pow(p1[1] - p2[1], 2))


def clamp(value, min_, max_):
    return max(min_, min(max_, value))


def angle_to_point(origin, destination):
    """
    Return the angle (in degrees) to the destiination point from the origin
    """
    return math.atan2(destination[0] - origin[0], destination[1] - origin[1]) / math.pi * 180.0


class HockeyPlayer:
    """
       Your ice hockey player. You may do whatever you want here. There are three rules:
        1. no calls to the pystk library (your code will not run on the tournament system if you do)
        2. There needs to be a deep network somewhere in the loop
        3. You code must run in 100 ms / frame on a standard desktop CPU (no for testing GPU)
        
        Try to minimize library dependencies, nothing that does not install through pip on linux.
    """

    """
       You may request to play with a different kart.
       Call `python3 -c "import pystk; pystk.init(pystk.GraphicsConfig.ld()); print(pystk.list_karts())"` to see all values.
       ['adiumy', 'amanda', 'beastie', 'emule', 'gavroche', 'gnu', 'hexley', 'kiki', 'konqi', 'nolok', 'pidgin', 'puffy', 'sara_the_racer', 'sara_the_wizard', 'suzanne', 'tux', 'wilber', 'xue']
    """
    kart = "hexley"

    def __init__(self, player_id=0):
        """
        Set up a soccer player.
        The player_id starts at 0 and increases by one for each player added. You can use the player id to figure out your team (player_id % 2), or assign different roles to different agents.
        """
        self.player_id = player_id
        self.frame_number = 0
        self.net = load_model().eval()
        self.goal_coords = [0.0, -65.0] if player_id % 2 else [0.0, 65.0]
        self.stuck_frames = 0
        self.last_puck_guess = [0, 0]
        self.reverse_maneuver = False
        self.puck_detected = False
        self.last_kart_location = [0.0, 0.0]
        self.start_maneuver = False
        self.reset_distance = 10.0  # distance kart moves that is considered a reset (start of new game or rescue)
        self.controlling_puck = False
        self.games_played = 0
        self.rescue = False
        self.attempted_reverse = False
        self.search = False
        self.search_start_angle = 0.0
        self.search_direction = 0  # 0 for clockwise, 1 for counter-clockwise
        self.last_kart_angle = 0.0
        self.last_action = {'acceleration': 0, 'brake': False, 'drift': False, 'nitro': False, 'rescue': False, 'steer': 0}
        # integral used to help fix deadband in long-distance seeking
        self.integral_gain = 10.0
        self.integral_steering = 0.0
        self.integral_max = 0.25
        self.integral_reset = 0.5
        self.search_frames = 0
        self.last_aim_point = [0.0, 0.0]

        #
        self.random_start_steer = clamp(10 * (random.random() - 0.5), -0.5, 0.5)

    def act(self, image: np.ndarray, player_info):
        """
        Set the action given the current image
        :param image: numpy array of shape (300, 400, 3)
        :param player_info: pystk.Player object for the current kart.
        return: Dict describing the action
        """
        self.frame_number += 1

        action = {'acceleration': 0, 'brake': False, 'drift': False, 'nitro': False, 'rescue': False, 'steer': 0}

        # run image through network
        in_tensor = F.to_tensor(image).unsqueeze(0)
        aim_point = self.net(in_tensor).squeeze(0).detach().tolist()

        # kart properties
        kart_location = player_info.kart.location[0], player_info.kart.location[-1]
        kart_angle = q_to_yaw(player_info.kart.rotation)

        # static parameters
        puck_ahead_y_min = -0.33
        puck_ahead_y_max = 0.0
        puck_beside_y_min = -0.1
        puck_beside_y_max = 0.75
        target_velocity = 50.0
        min_velocity = 10.0
        steering_power = 3
        steering_gain = 1000.0
        accel_threshold = 0.5
        brake_threshold = 0.9
        drift_threshold = 0.6
        nitro_threshold = 0.2
        max_puck_offset = 0.05

        # calculate aim values
        aim_magnitude = abs(aim_point[0])
        aim_direction = aim_point[0] / aim_magnitude  # -1 or 1

        # scalar velocity, without direction
        current_vel = np.linalg.norm(player_info.kart.velocity)

        ###################### SET MODE #####################
        if distance_between_points(self.last_kart_location, kart_location) >= self.reset_distance:
            if not self.rescue:
                print(f'ENTER START ({self.games_played})')
                self.games_played += 1
                self.start_maneuver = True
                self.rescue = False
                self.stuck_frames = 0

        ################## SPECIAL MANEUVERS #################

        # ----------------- REVERSE MANEUVER -----------------
        # rescue if stuck
        elif current_vel >= 1.0:
            # not stuck anymore--reset stuck frames
            self.stuck_frames = 0
        else:
            self.stuck_frames += 1
        if self.stuck_frames >= 10:
            print(f'{self.player_id} | =+=+=+=+=RESCUE=+=+=+=+=')
            action['rescue'] = True
            self.stuck_frames = 0

        # ----------------- START MANEUVER ------------------
        if self.start_maneuver:
            # accelerate straight to the center at full speed
            if current_vel >= 20:
                # if we have nitro, use it to get to the puck faster
                action['nitro'] = True
            angle_to_center = math.atan2(0 - kart_location[0], 0 - kart_location[1]) / math.pi * 180.0
            print(f'{self.frame_number}|{self.player_id}| angle to center: {angle_to_center}')
            print(f'{self.frame_number}|{self.player_id}| IN START')
            action['acceleration'] = 1.0
            action['steer'] = clamp(0.1 * (angle_to_center - kart_angle) ** 3, -1.0, 1.0)
            # # TODO: THIS IS JUST FOR TESTING! REMOVE WHEN DONE!
            # if abs(kart_location[-1]) >= 25:
            #     action['steer'] = self.random_start_steer
            # # TODO: END

            if abs(kart_location[-1]) <= 30:
                print(f'{self.frame_number}|{self.player_id} | EXIT START')
                self.start_maneuver = False
            else:
                print(f'{self.frame_number}|{self.player_id} | start action: {action}')
                print(f'{self.frame_number}|{self.player_id} | kart | pos: {[round(x, 3) for x in kart_location]} vel: {int(current_vel)} | aim: {[round(x, 3) for x in aim_point]} ')
                self.last_kart_location = kart_location

                # return early to ensure nothing overrides this
                return action

        # ---------------- SEARCH MANEUVER ------------------
        if self.search:
            # break out of search maneuver
            if puck_ahead_y_min <= aim_point[1] <= 0.5:
                self.search = False
            else:
                print(f"{self.frame_number}|============ SEARCH ===============")
                target_speed = 10.0  # turning radius is smaller with lower speed
                action['acceleration'] = 1.0 if current_vel < target_speed else 0.1
                action['steer'] = -1.0
                action['drift'] = True

        # ############################# CALCULATIONS ###########################

        # ---------------------------- PUCK IN CONTROL -------------------------
        # we are in control of the puck (inside window of control
        if abs(aim_point[0]) <= 0.2 and abs(aim_point[1]) <= 0.05:
            print(f'{self.frame_number}|********* IN CONTROL ***********')
            self.controlling_puck = True
            self.last_puck_guess = kart_location
            # the goal here is to get the kart's direction toward the goal while steering the kart based on
            # the predicted puck location. Basically, keep the puck between the kart and goal.
            angle_to_goal = angle_to_point(kart_location, self.goal_coords)
            control_gain = abs(angle_to_goal - kart_angle)

            action['acceleration'] = clamp(1 - (abs(aim_point[0]) * 10.0), 0.2, 1.0)

            print(f'control_gain: {control_gain}')
            print(f'goal angle: {angle_to_goal}')

            if kart_angle > angle_to_goal:
                offset_aim_point = aim_point[0] + clamp(control_gain * 0.001, 0, max_puck_offset)
            else:
                offset_aim_point = aim_point[0] - clamp(control_gain * 0.001, 0, max_puck_offset)

            print(f'{self.player_id} | kart | pos: {[round(x, 3) for x in kart_location]} | vel: {int(current_vel)} | aim: {[round(x, 3) for x in aim_point]} | offset x: {round(offset_aim_point, 3)} | angle: {kart_angle}')

            offset_aim_magnitude = abs(offset_aim_point)
            offset_aim_direction = offset_aim_point / aim_magnitude  # -1 or 1

            # set the aim point offset to steer toward the goal
            # action['steer'] = clamp(offset_aim_direction * (control_gain * abs(offset_aim_magnitude ** steering_power) + 0.5 * offset_aim_magnitude), -1.0, 1.0)
            action['steer'] = clamp(offset_aim_direction * 1000.0 * abs(offset_aim_magnitude ** steering_power), -1.0, 1.0)
            print(f'{self.player_id} | action: {action}')
            self.last_kart_location = kart_location
            return action

        # -------------- PUCK IN FORWARD VIEW ------------------
        # we can see the puck in front of us, and can turn toward it
        elif puck_ahead_y_min <= aim_point[1] <= puck_ahead_y_max:
            self.integral_steering = clamp(self.integral_steering + aim_point[0] * self.integral_gain, -1.0 * self.integral_max, self.integral_max)

            print(f'{self.player_id} | puck in view | integral: {self.integral_steering}')
            action['steer'] = steering_gain * aim_direction * abs(aim_magnitude ** steering_power) + self.integral_steering
            if action['steer'] >= self.integral_reset:
                self.integral_steering = 0.0
            action['acceleration'] = 1.0 if aim_magnitude <= accel_threshold and current_vel <= target_velocity else 0.25
            action['brake'] = True if aim_magnitude >= brake_threshold \
                                      and current_vel > 5.0 \
                                      and aim_point[1] >= puck_ahead_y_min else False
            action['drift'] = True if aim_magnitude >= drift_threshold else False
            action['nitro'] = True if aim_magnitude <= nitro_threshold and current_vel >= 20.0 else False

            if current_vel < min_velocity or action['nitro']:
                action['acceleration'] = 1.0

            # guess distance to puck
            puck_distance = 3.0 * np.exp(-12.0 * aim_point[1])
            print(f'{self.frame_number}|Puck guess distance: {puck_distance}')

        # -------------- PUCK IN SIDE VIEW ------------------
        # we can see the puck in front of us, and can turn toward it
        elif puck_beside_y_min <= aim_point[1] <= puck_beside_y_max:
            print(f'{self.frame_number}| Puck in side view!')
            # steer toward puck and brake, then reverse away until puck is back in front
            action['acceleration'] = 0.0
            action['steer'] = aim_direction
            # we need to tell if we're reversing so we can reverse steering direction

        else:  # we can't find the puck, head to last guessed location. Wait 2 frames in case of an anomaly frame
            self.search_frames += 1
            if self.search_frames >= 3:
                self.search = True
                self.search_frames = 0
            else:
                action = self.last_action

        # clamp values
        action['steer'] = clamp(action['steer'], -1.0, 1.0)
        action['acceleration'] = clamp(action['acceleration'], 0.0, 1.0)

        print(f'{self.frame_number}|{self.player_id}| kart | pos: {[round(x, 3) for x in kart_location]} | vel: {int(current_vel)} | aim: {[round(x, 3) for x in aim_point]} | angle: {kart_angle}')
        print(f'{self.frame_number}|{self.player_id}| action: {action}')

        self.last_kart_location = kart_location
        self.last_kart_angle = 0.0
        self.last_action = action
        self.last_aim_point = aim_point

        return action

