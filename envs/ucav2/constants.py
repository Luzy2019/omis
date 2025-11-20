"""
Constants module for the Harfang3D aircraft simulation environment.

This module defines normalization constants used to scale various aircraft and missile
state variables to a standardized range for machine learning training and inference.
"""

import math

# Mathematical constants
pi = math.pi

# FPS monitoring variables for network client frequency checking
# These variables are used to track timing for performance monitoring
t = 0    # Current time counter
t0 = 0   # Initial time reference
t1 = 0   # Previous time reference

# Aircraft and Missile State Normalization Constants
# These values define the maximum expected ranges for each state variable
# Used to normalize raw sensor data to [-1, 1] or [0, 1] range for neural networks
NormStates = {
    # Aircraft (Plane) state normalization factors
    "Plane_position"            : 10000,  # Maximum position in meters (10km range)
    "Plane_Euler_angles"        : pi,     # Maximum Euler angles in radians (180 degrees)
    "Plane_heading"             : 360,    # Maximum heading in degrees (full circle)
    "Plane_pitch_attitude"      : 180,    # Maximum pitch attitude in degrees (±90 degrees)
    "Plane_roll_attitude"       : 180,    # Maximum roll attitude in degrees (±90 degrees)
    "Plane_thrust_level"        : 100,    # Maximum thrust level percentage (0-100%)
    "Plane_horizontal_speed"    : 800,    # Maximum horizontal speed in m/s
    "Plane_vertical_speed"      : 800,    # Maximum vertical speed in m/s
    "Plane_move_vector"         : 1,      # Normalization factor for unit vectors
    "Plane_linear_acceleration" : 50,     # Maximum linear acceleration in m/s²
    
    # Missile state normalization factors
    "Missile_position"          : 20000,  # Maximum missile position in meters (20km range)
    "Missile_Euler_angles"      : 360,    # Maximum missile Euler angles in degrees
    "Missile_move_vector"       : 1,      # Normalization factor for missile unit vectors
    "Missile_heading"           : 360,    # Maximum missile heading in degrees
    "Missile_pitch_attitude"    : 180,    # Maximum missile pitch attitude in degrees
    "Missile_roll_attitude"     : 180,    # Maximum missile roll attitude in degrees
    "Missile_horizontal_speed"  : 2000,   # Maximum missile horizontal speed in m/s
    "Missile_vertical_speed"    : 2000    # Maximum missile vertical speed in m/s
}
