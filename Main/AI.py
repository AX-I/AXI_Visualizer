# Basic behaviors

import numpy as np
from math import atan2

def follow(target, current, radius, tolerance, isMoving):
    """target, current as (x,y,z) => move, direction"""
    r = atan2(*(target[2::-2] - current[2::-2]))
    if not isMoving:
        if np.linalg.norm(target - current) > radius + tolerance:
            return (1, r)
        else: return (0, r)
    else:
        if np.linalg.norm(target - current) > radius - tolerance:
            return (1, r)
        else: return (0, r)
