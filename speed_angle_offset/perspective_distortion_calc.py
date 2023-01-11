''' This is a simple sim. of a satellite passing over the observer. It calculates how  satellite (with previously set
    speeed and relative position) appears to the camera in long exposure images and then tries to calculate back
    the input speed, height and position. This was used to check if the perspective calculation is correct'''


import numpy as np
import cv2
from math import cos, sin, radians, atan2, degrees, atan, acos
import math
import matplotlib.pyplot as plt
from scipy import optimize
from mpl_toolkits.mplot3d import Axes3D


def set_axes_equal(ax):

    x_limits = ax.get_xlim3d()
    y_limits = ax.get_ylim3d()
    z_limits = ax.get_zlim3d()

    x_range = abs(x_limits[1] - x_limits[0])
    x_middle = np.mean(x_limits)
    y_range = abs(y_limits[1] - y_limits[0])
    y_middle = np.mean(y_limits)
    z_range = abs(z_limits[1] - z_limits[0])
    z_middle = np.mean(z_limits)

    # The plot bounding box is a sphere in the sense of the infinity
    # norm, hence I call half the max range the plot radius.
    plot_radius = 0.5*max([x_range, y_range, z_range])

    ax.set_xlim3d([x_middle - plot_radius, x_middle + plot_radius])
    ax.set_ylim3d([y_middle - plot_radius, y_middle + plot_radius])
    ax.set_zlim3d([z_middle - plot_radius, z_middle + plot_radius])


def rescale_frame(frame, scale):
    width = int(frame.shape[1] * scale)
    height = int(frame.shape[0] * scale)
    dimension = (width, height)
    return cv2.resize(frame, dimension, interpolation=cv2.INTER_AREA)


def unit_vector(vector):
    return vector / np.linalg.norm(vector)


def angle_delta(v1, v2):
    v1_u = unit_vector(v1)
    v2_u = unit_vector(v2)
    return degrees(np.arccos(np.clip(np.dot(v1_u, v2_u), -1.0, 1.0)))


def translate_angles_vector(theta, phi):    # theta = right assertion, phi = declination
    x = cos(radians(theta)) * cos(radians(phi))
    z = sin(radians(theta)) * cos(radians(phi))
    y = sin(radians(phi))
    return x, y, z


def translate_angles_point(r, theta, phi):    # theta = right assertion, phi = declination
    x = r * cos(radians(theta)) * cos(radians(phi))
    z = r * sin(radians(theta)) * cos(radians(phi))
    y = r * sin(radians(phi))
    return x, y, z


def translate_vector_angles(vector):
    if vector[0] > 0:
        right_assertion = degrees(atan(vector[1]/vector[0]))
        declination = degrees(acos(vector[2]/np.linalg.norm(vector)))
        return right_assertion, declination
    else:
        declination = degrees(acos(vector[2] / np.linalg.norm(vector)))
        return None, declination


def line_format(point_1, point_2):
    x_list = []
    y_list = []
    z_list = []

    x_list.append(point_1[0])
    x_list.append(point_2[0])

    y_list.append(point_1[1])
    y_list.append(point_2[1])

    z_list.append(point_1[2])
    z_list.append(point_2[2])

    return x_list, y_list, z_list


def get_foot(p, a, b):
    ap = p - a
    ab = b - a
    result = a + np.dot(ap, ab)/np.dot(ab, ab) * ab
    return result


# settings
rescale_setting = 0.6
canvas_shape = (1000, 1000, 3)

# constants
grav_const = 6.67430e-11
rad_earth = 6371000
earth_mass = 5.972e+24

# satellite data
exposure_time = 6
height = 200
satellite_velocity = math.sqrt((grav_const * earth_mass) / (rad_earth + (height * 1000)))
print(f'Satellite_Velocity: {satellite_velocity}')

# sky top view
canvas = np.zeros(canvas_shape, dtype='uint8')
canvas_center = round(canvas.shape[1] / 2), round(canvas.shape[0] / 2)

print(satellite_velocity * exposure_time)
# satellite_plot
satellite_length = (satellite_velocity * exposure_time) / 1000
satellite_direction_angle = 0 # counter-clockwise rotation from [1, 0]
satellite_position = np.array([480 + 300, 200, height])

# data
trace_data = []

# calculation
satellite_direction_vector = np.array([cos(radians(satellite_direction_angle)), sin(radians(satellite_direction_angle)), 0])
print(np.linalg.norm(satellite_direction_vector))
satellite_vector = satellite_direction_vector * satellite_length
satellite_position_endpoint = satellite_position[0] + satellite_vector[0], satellite_position[1] + satellite_vector[1], height
print(f'satellite_position_endpoint: {satellite_position_endpoint}')

# trace calculation
observer_position = np.array([canvas_center[0], canvas_center[1], 0])
vector_0 = np.subtract(satellite_position_endpoint, observer_position)
vector_1 = np.subtract(satellite_position, observer_position)

trace_angle_delta = angle_delta(vector_0, vector_1)
print(trace_angle_delta)

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

# plot satellite path
path_point_0 = (satellite_vector * - 5) + satellite_position
path_point_1 = (satellite_vector * 5) + satellite_position

sat_data_x, sat_data_y, sat_data_z = line_format(path_point_0, path_point_1)
ax.plot(sat_data_x, sat_data_y, sat_data_z, linewidth=0.7)

# plot satellite
sat_data_x, sat_data_y, sat_data_z = line_format(satellite_position, satellite_position_endpoint)
ax.plot(sat_data_x, sat_data_y, sat_data_z, c='red')

# foot point
path_foot_point = get_foot(observer_position, satellite_position, satellite_position_endpoint)
sat_data_x, sat_data_y, sat_data_z = line_format(path_foot_point, observer_position)
ax.plot(sat_data_x, sat_data_y, sat_data_z, linewidth=0.9, c='orange')

# trace center point
trace_center = (satellite_position + (satellite_vector * 0.5)).astype(int)
sat_data_x, sat_data_y, sat_data_z = line_format(trace_center, observer_position)
ax.plot(sat_data_x, sat_data_y, sat_data_z, linewidth=0.9, c='orange')

sat_data_x, sat_data_y, sat_data_z = line_format(satellite_position, observer_position)
ax.plot(sat_data_x, sat_data_y, sat_data_z, linewidth=0.4, c='lime')

sat_data_x, sat_data_y, sat_data_z = line_format(satellite_position_endpoint, observer_position)
ax.plot(sat_data_x, sat_data_y, sat_data_z, linewidth=0.4, c='lime')

# helper lines
trace_center_ground = (satellite_position + (satellite_vector * 0.5)).astype(int)[:2]
trace_center_ground = trace_center_ground[0], trace_center_ground[1], 0
sat_data_x, sat_data_y, sat_data_z = line_format(trace_center_ground, observer_position)
ax.plot(sat_data_x, sat_data_y, sat_data_z, linewidth=0.4, c='royalblue')

sat_data_x, sat_data_y, sat_data_z = line_format(trace_center, trace_center_ground)
ax.plot(sat_data_x, sat_data_y, sat_data_z, linewidth=0.4, c='royalblue')

path_foot_point_ground = path_foot_point[0], path_foot_point[1], 0
sat_data_x, sat_data_y, sat_data_z = line_format(path_foot_point_ground, path_foot_point)
ax.plot(sat_data_x, sat_data_y, sat_data_z, linewidth=0.4, c='royalblue')

sat_data_x, sat_data_y, sat_data_z = line_format(path_foot_point_ground, observer_position)
ax.plot(sat_data_x, sat_data_y, sat_data_z, linewidth=0.4, c='royalblue')

reference_line_vector = np.array([1, -1, 0])
scalar = observer_position[1] - satellite_position[1]
reference_line_point = observer_position + (reference_line_vector * scalar)
sat_data_x, sat_data_y, sat_data_z = line_format(reference_line_point, observer_position)
ax.plot(sat_data_x, sat_data_y, sat_data_z, linewidth=0.4, c='gray')


# calculation of ground declination
zero_angle_vector_ground = np.subtract(path_foot_point[:2], observer_position[:2])
trace_center_vector = np.subtract((satellite_position + (satellite_vector * 0.5))[:2], observer_position[:2])
horizontal_declination = angle_delta(zero_angle_vector_ground, trace_center_vector) % 180
if horizontal_declination > 90:
    horizontal_declination = 90 - (horizontal_declination - 90)
print(f'horizontal_declination: {horizontal_declination}')

# calculation of tilted declination
# in the real image we would need to translate ethe declination and right assertion to a vector
zero_angle_vector = np.subtract(path_foot_point, observer_position)
trace_center_vector = np.subtract(trace_center, observer_position)

tilted_declination = angle_delta(zero_angle_vector, trace_center_vector)
print(f'tilted_declination: {tilted_declination}')

# distortion calculation
perspective_correction = trace_angle_delta / (0.5 * cos(radians(2 * tilted_declination)) + 0.5)
print(f'perspective correction: {perspective_correction}')


def velocity_bisect(expected_height):

    angles_start = translate_vector_angles(satellite_position)[0], 90 - translate_vector_angles(satellite_position)[1]
    satellite_start_distance = expected_height / sin(radians(angles_start[1]))
    angles_end = translate_vector_angles(satellite_position_endpoint)[0], 90 - translate_vector_angles(satellite_position_endpoint)[1]
    satellite_end_distance = expected_height / sin(radians(angles_end[1]))

    satellite_calc_position_start = translate_angles_point(satellite_start_distance, angles_start[0], angles_start[1])
    satellite_calc_position_end = translate_angles_point(satellite_end_distance, angles_end[0], angles_end[1])

    satellite_calc_distance = np.linalg.norm(np.subtract(satellite_calc_position_start, satellite_calc_position_end))
    satellite_calc_velocity = satellite_calc_distance / exposure_time

    satellite_orbit_velocity = math.sqrt((grav_const * earth_mass) / (rad_earth + (expected_height)))

    return satellite_calc_velocity - satellite_orbit_velocity


satellite_height = optimize.bisect(velocity_bisect, 1, 20000000)
print(f'satellite_height: {satellite_height}')

ax.scatter(observer_position[0], observer_position[1], observer_position[2], s=5)
ax.set_zlabel('h')
set_axes_equal(ax)
plt.xlabel('x')
plt.ylabel('y')
ax.set_zlim(0, height + 100)
plt.show()#


# settings
rescale_setting = 0.6
canvas_shape = (1000, 1000, 3)

# constants
grav_const = 6.67430e-11
rad_earth = 6371000
earth_mass = 5.972e+24

# satellite data
exposure_time = 6
height = 200
satellite_velocity = math.sqrt((grav_const * earth_mass) / (rad_earth + (height * 1000)))
print(f'Satellite_Velocity: {satellite_velocity}')

# sky top view
canvas = np.zeros(canvas_shape, dtype='uint8')
canvas_center = round(canvas.shape[1] / 2), round(canvas.shape[0] / 2)

print(satellite_velocity * exposure_time)
# satellite_plot
satellite_length = (satellite_velocity * exposure_time) / 1000
satellite_direction_angle = 0 # counter-clockwise rotation from [1, 0]
satellite_position = np.array([480 + 300, 200, height])

# data
trace_data = []

# calculation
satellite_direction_vector = np.array([cos(radians(satellite_direction_angle)), sin(radians(satellite_direction_angle)), 0])
print(np.linalg.norm(satellite_direction_vector))
satellite_vector = satellite_direction_vector * satellite_length
satellite_position_endpoint = satellite_position[0] + satellite_vector[0], satellite_position[1] + satellite_vector[1], height
print(f'satellite_position_endpoint: {satellite_position_endpoint}')

# trace calculation
observer_position = np.array([canvas_center[0], canvas_center[1], 0])
vector_0 = np.subtract(satellite_position_endpoint, observer_position)
vector_1 = np.subtract(satellite_position, observer_position)

trace_angle_delta = angle_delta(vector_0, vector_1)
print(trace_angle_delta)

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

# plot satellite path
path_point_0 = (satellite_vector * - 5) + satellite_position
path_point_1 = (satellite_vector * 5) + satellite_position

sat_data_x, sat_data_y, sat_data_z = line_format(path_point_0, path_point_1)
ax.plot(sat_data_x, sat_data_y, sat_data_z, linewidth=0.7)

# plot satellite
sat_data_x, sat_data_y, sat_data_z = line_format(satellite_position, satellite_position_endpoint)
ax.plot(sat_data_x, sat_data_y, sat_data_z, c='red')

# foot point
path_foot_point = get_foot(observer_position, satellite_position, satellite_position_endpoint)

path_foot_point_sky = translate_vector_angles(np.subtract(path_foot_point, observer_position))

trace_center_sky = translate_vector_angles(np.subtract(trace_center, observer_position))
sat_data_x, sat_data_y, sat_data_z = line_format(path_foot_point, observer_position)
ax.plot(sat_data_x, sat_data_y, sat_data_z, linewidth=0.9, c='orange')

# trace center point
trace_center = (satellite_position + (satellite_vector * 0.5)).astype(int)

sat_data_x, sat_data_y, sat_data_z = line_format(trace_center, observer_position)
ax.plot(sat_data_x, sat_data_y, sat_data_z, linewidth=0.9, c='orange')

sat_data_x, sat_data_y, sat_data_z = line_format(satellite_position, observer_position)
ax.plot(sat_data_x, sat_data_y, sat_data_z, linewidth=0.4, c='lime')

sat_data_x, sat_data_y, sat_data_z = line_format(satellite_position_endpoint, observer_position)
ax.plot(sat_data_x, sat_data_y, sat_data_z, linewidth=0.4, c='lime')

# helper lines
trace_center_ground = (satellite_position + (satellite_vector * 0.5)).astype(int)[:2]
trace_center_ground = trace_center_ground[0], trace_center_ground[1], 0
sat_data_x, sat_data_y, sat_data_z = line_format(trace_center_ground, observer_position)
ax.plot(sat_data_x, sat_data_y, sat_data_z, linewidth=0.4, c='royalblue')

sat_data_x, sat_data_y, sat_data_z = line_format(trace_center, trace_center_ground)
ax.plot(sat_data_x, sat_data_y, sat_data_z, linewidth=0.4, c='royalblue')

path_foot_point_ground = path_foot_point[0], path_foot_point[1], 0
sat_data_x, sat_data_y, sat_data_z = line_format(path_foot_point_ground, path_foot_point)
ax.plot(sat_data_x, sat_data_y, sat_data_z, linewidth=0.4, c='royalblue')

sat_data_x, sat_data_y, sat_data_z = line_format(path_foot_point_ground, observer_position)
ax.plot(sat_data_x, sat_data_y, sat_data_z, linewidth=0.4, c='royalblue')

reference_line_vector = np.array([1, -1, 0])
scalar = observer_position[1] - satellite_position[1]
reference_line_point = observer_position + (reference_line_vector * scalar)
sat_data_x, sat_data_y, sat_data_z = line_format(reference_line_point, observer_position)
ax.plot(sat_data_x, sat_data_y, sat_data_z, linewidth=0.4, c='gray')


# calculation of ground declination
zero_angle_vector_ground = np.subtract(path_foot_point[:2], observer_position[:2])
trace_center_vector = np.subtract((satellite_position + (satellite_vector * 0.5))[:2], observer_position[:2])
horizontal_declination = angle_delta(zero_angle_vector_ground, trace_center_vector) % 180
if horizontal_declination > 90:
    horizontal_declination = 90 - (horizontal_declination - 90)
print(f'horizontal_declination: {horizontal_declination}')

# calculation of tilted declination
# in the real image we would need to translate ethe declination and right assertion to a vector
zero_angle_vector = np.subtract(path_foot_point, observer_position)
trace_center_vector = np.subtract(trace_center, observer_position)

tilted_declination = angle_delta(zero_angle_vector, trace_center_vector)
print(f'tilted_declination: {tilted_declination}')

# distortion calculation
perspective_correction = trace_angle_delta / (0.5 * cos(radians(2 * tilted_declination)) + 0.5)
print(f'perspective correction: {perspective_correction}')


def velocity_bisect(expected_height):
    satellite_foot_declination = 90 - path_foot_point_sky[1]

    satellite_vector_distance = expected_height / sin(radians(satellite_foot_declination))

    satellite_calc_distance = (satellite_vector_distance * sin(radians(perspective_correction / 2))) * 2

    satellite_calc_velocity = satellite_calc_distance / exposure_time

    satellite_orbit_velocity = math.sqrt((grav_const * earth_mass) / (rad_earth + (expected_height)))

    return satellite_calc_velocity - satellite_orbit_velocity


satellite_height = optimize.bisect(velocity_bisect, 1, 20000000)
print(f'satellite_height: {satellite_height}')

ax.scatter(observer_position[0], observer_position[1], observer_position[2], s=5)
ax.set_zlabel('h')
set_axes_equal(ax)
plt.xlabel('x')
plt.ylabel('y')
ax.set_zlim(0, height + 100)
plt.show()#
