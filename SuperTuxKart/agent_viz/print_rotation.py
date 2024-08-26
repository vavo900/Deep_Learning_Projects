#!/root/miniconda3/envs/env36/bin/python

import json
import math
import os


def q_to_e(q):
    """
    Convert quaternion to euler angles (yaw, pitch, roll)
    """
    # w, x, y, z = [float(i) for i in q]
    # w, x, z, y = [float(i) for i in q]
    # w, y, x, z = [float(i) for i in q]
    # w, y, z, x = [float(i) for i in q]  # ********
    # w, z, x, y = [float(i) for i in q]
    # w, z, x, y = [float(i) for i in q]

    # x, w, y, z = [float(i) for i in q]
    # x, w, z, y = [float(i) for i in q]
    # x, y, w, z = [float(i) for i in q]
    # x, y, z, w = [float(i) for i in q]
    # x, z, w, y = [float(i) for i in q]
    # x, z, y, w = [float(i) for i in q]  # **********

    # y, w, x, z = [float(i) for i in q]
    # y, w, z, x = [float(i) for i in q]
    # y, x, w, z = [float(i) for i in q]
    # y, x, z, w = [float(i) for i in q]
    y, z, x, w = [float(i) for i in q]  # **********
    # y, z, w, x = [float(i) for i in q]

    # z, w, x, y = [float(i) for i in q]
    # z, w, y, x = [float(i) for i in q]
    # z, x, w, y = [float(i) for i in q]
    # z, x, y, w = [float(i) for i in q]
    # z, y, x, w = [float(i) for i in q]
    # z, y, w, x = [float(i) for i in q]  # **********

    sinr_cosp = 2.0 * (w * x * y * z)
    cosr_cosp = 1.0 - 2.0 * (x * x + y * y)
    roll = math.atan2(sinr_cosp, cosr_cosp) / math.pi * 180

    sinp = 2.0 * (w * y - z * x)
    if abs(sinp) >= 1:
        pitch = math.copysign(math.pi / 2, sinp) / math.pi * 180
    else:
        pitch = math.asin(sinp) / math.pi * 180

    siny_cosp = 2.0 * (w * z + x * y)
    cosy_cosp = 1.0 - 2.0 * (y * y + z * z)
    yaw = math.atan2(siny_cosp, cosy_cosp) / math.pi * 180

    return yaw, pitch, roll


data_path = '/root/cs342/final/continuous_rotation/'
for i in (0, 26, 55, 75, 98, 119, 136, 155, 176):  # range(len(os.listdir(data_path))):
    f = os.path.join(data_path, f'player00_{i:0>5}.json')
    with open(f, 'r') as file_:
        j = json.loads(file_.read())
        q = j['players'][0]['kart']['rotation']
        # print(f'{i}: {q}  =  {q_to_yaw(q)}')
        print(f'{[f"{round(x, 5): >10}" for x in q]}  =  {[f"{round(x, 5): >10}" for x in q_to_e(q)]}')

