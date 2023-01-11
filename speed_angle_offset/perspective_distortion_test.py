import numpy as np
import cv2
from math import cos, sin, radians, atan2, degrees
import math
import matplotlib.pyplot as plt


def rescale_frame(frame, scale):
    width = int(frame.shape[1] * scale)
    height = int(frame.shape[0] * scale)
    dimension = (width, height)
    return cv2.resize(frame, dimension, interpolation=cv2.INTER_AREA)


def angle_delta(vec_0, vec_1):
    return abs(degrees(atan2(vec_0[0], vec_0[1])) - degrees(atan2(vec_1[0], vec_1[1])))


# settings
rescale_setting = 0.6
canvas_shape = (1000, 1000, 3)

# constants
grav_const = 6.67430e-11
rad_earth = 6371000
earth_mass = 5.972e+24

# satellite data
exposure_time = 6
height = 200 * 1000
satellite_velocity = math.sqrt((grav_const * earth_mass) / (rad_earth + height))

# sky top view
canvas = np.zeros(canvas_shape, dtype='uint8')

# satellite_plot
satellite_length = 100
satellite_direction_angle = 0 # counter-clockwise rotation from [1, 0]
satellite_position = np.array([50, 100])
# [450, 303]
trace_data = []
undistorted_trace_data = []
declination_data = []

for x_pos in range(0, 450, 10):
    satellite_position[0] = x_pos

    satellite_direction_vector = np.array([cos(radians(satellite_direction_angle)), sin(radians(satellite_direction_angle))])
    satellite_vector = np.around(satellite_direction_vector * satellite_length).astype(int)
    satellite_position_canvas = satellite_position, satellite_position + satellite_vector
    cv2.line(canvas, satellite_position_canvas[0], satellite_position_canvas[1], (255, 255, 255), thickness=2)

    # trace calculation
    canvas_center = round(canvas_shape[0] / 2), round(canvas_shape[1] / 2)
    vector_0 = np.subtract(canvas_center, satellite_position_canvas[0])
    vector_1 = np.subtract(canvas_center, satellite_position_canvas[1])

    cv2.line(canvas, canvas_center, satellite_position_canvas[0], (255, 100, 100), thickness=1)
    cv2.line(canvas, canvas_center, satellite_position_canvas[1], (255, 100, 100), thickness=1)

    trace_angle_delta = angle_delta(vector_0, vector_1)
    trace_data.append(trace_angle_delta)

    # distortion calculation
    zero_angle_vector = np.array([satellite_vector[1],  - satellite_vector[0]])
    trace_center_vector = np.subtract((satellite_position + (satellite_vector * 0.5)).astype(int), canvas_center)
    horizontal_declination = angle_delta(zero_angle_vector, trace_center_vector) % 180
    if horizontal_declination > 90:
        horizontal_declination = 90 - (horizontal_declination - 90)

    trace_test_value = trace_angle_delta / (0.5 * cos(radians(2 * horizontal_declination)) + 0.5)
    print(trace_test_value)
    # undistorted_trace = trace_angle_delta + trace_test_value
    undistorted_trace_data.append(trace_test_value)
    declination_data.append(horizontal_declination)

    cv2.line(canvas, canvas_center, canvas_center + zero_angle_vector, (255, 255, 00), thickness=2)
    cv2.line(canvas, canvas_center, canvas_center + trace_center_vector, (255, 255, 00), thickness=2)

    # cv2.imshow("Satellite Simmulation", rescale_frame(canvas, rescale_setting))
    # cv2.waitKey()

plt.scatter(declination_data, trace_data, c="b", s=5)
plt.scatter(declination_data, undistorted_trace_data, c='r', s=5)
plt.show()
