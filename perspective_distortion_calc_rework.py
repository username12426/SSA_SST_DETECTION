''' This is a simple sim. of a satellite passing over the observer. It calculates how  satellite (with previously set
    speeed and relative position) appears to the camera in long exposure images and then tries to calculate back
    the input speed, height and position. This was used to check if the perspective calculation is correct'''


import numpy as np

from math import cos, sin, radians, atan2, degrees, atan, acos
import math
import matplotlib.pyplot as plt
from scipy import optimize


def set_axes_equal(ax):

    # function for the plot

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


def unit_vector(vector):
    return vector / np.linalg.norm(vector)


def angle_delta(v1, v2):
    v1_u = unit_vector(v1)
    v2_u = unit_vector(v2)
    return degrees(np.arccos(np.clip(np.dot(v1_u, v2_u), -1.0, 1.0)))


def translate_angles_vector(theta, phi):    # theta = Azimuth, phi = elevation
    x = cos(radians(theta)) * cos(radians(phi))
    z = sin(radians(theta)) * cos(radians(phi))
    y = sin(radians(phi))
    return x, y, z


def translate_angles_point(r, theta, phi):    # theta = Azimuth, phi = elevation
    x = r * cos(radians(theta)) * cos(radians(phi))
    z = r * sin(radians(theta)) * cos(radians(phi))
    y = r * sin(radians(phi))
    return x, y, z


def translate_vector_angles(vector):
    if vector[0] > 0:
        azimuth = degrees(atan(vector[1]/vector[0]))
        elevation = degrees(acos(vector[2]/np.linalg.norm(vector)))
        return azimuth, elevation
    else:
        elevation = degrees(acos(vector[2] / np.linalg.norm(vector)))
        return None, elevation


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


# // plot settings //
canvas_shape = (1000, 1000, 3)  # this is basically the floor of the plot
# Setup basically the floor of the 3d plot, observer is in the middle
canvas = np.zeros(canvas_shape, dtype='uint8')      # make a 2d array from the canvas shape
canvas_center = round(canvas.shape[1] / 2), round(canvas.shape[0] / 2)  # the observer is in the middle of this plane

# / constants /
grav_const = 6.67430e-11
rad_earth = 6371000
earth_mass = 5.972e+24

# // satellite data //
exposure_time = 6
height = 100
satellite_position = np.array([770, 200, height])       # [distance in x from observer, distance y, height]
satellite_direction_angle = 0  # counter-clockwise rotation from [1, 0]

# / calculate the expected speed and travelled distance of the satellite with given inputs /
satellite_velocity = math.sqrt((grav_const * earth_mass) / (rad_earth + (height * 1000)))   # in m/s
print(f'Satellite_Velocity: {satellite_velocity}')

# How far the satellite travels in the exposure time, this is the trace visible in long exposure images
satellite_length = (satellite_velocity * exposure_time) / 1000      # /1000 to get result in km
print(f'Satellite_Travelled_Distance: {satellite_velocity * exposure_time}')    # result is in m


# / This simply calculates were and how long the satellite trace is in the 3d plot /

# the angle is only important if you want to let the satellite travel at an angle to the coordinate system
satellite_direction_vector = np.array([cos(radians(satellite_direction_angle)), sin(radians(satellite_direction_angle)), 0])
print(f'Satellite_Flight_Vector: {np.linalg.norm(satellite_direction_vector)}')     # this is just the direction
satellite_vector = satellite_direction_vector * satellite_length        # this is the actual vector the satellite moves

# calculate the endpoint the satellite will end up in the plot with your input data
satellite_position_endpoint = satellite_position[0] + satellite_vector[0], satellite_position[1] + satellite_vector[1], height
print(f'satellite_position_endpoint: {satellite_position_endpoint}')

# / trace calculation /
observer_position = np.array([canvas_center[0], canvas_center[1], 0])

# calculate the vector between the observer and the satellite startpoint and endpoint
vector_0 = np.subtract(satellite_position_endpoint, observer_position)
vector_1 = np.subtract(satellite_position, observer_position)

# This trace angle is basically what the camera sees in the sky. A line in the sky can always be interpreted
# as an angle this specific line spans across the sky. I am calculating what the camera would see of this
# manually defined satellite
trace_angle_delta = angle_delta(vector_0, vector_1)     # angle between these two vectors
print(f'Trace (angle) the Camera sees: {trace_angle_delta}')


# // 3d PLot //

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')


# plot satellite path (line the satellite travels on)
path_point_0 = (satellite_vector * - 6) + satellite_position
path_point_1 = (satellite_vector * 6) + satellite_position

sat_data_x, sat_data_y, sat_data_z = line_format(path_point_0, path_point_1)
ax.plot(sat_data_x, sat_data_y, sat_data_z, linewidth=0.4, c='black')

# plot satellite
# start and endpoint the satellite is in the beginning and in the end of the long exposure
sat_data_x, sat_data_y, sat_data_z = line_format(satellite_position, satellite_position_endpoint)
ax.plot(sat_data_x, sat_data_y, sat_data_z, c='red')

# foot point (Closest point between the observer and the line the satellite travels on)
path_foot_point = get_foot(observer_position, satellite_position, satellite_position_endpoint)
sat_data_x, sat_data_y, sat_data_z = line_format(path_foot_point, observer_position)
ax.plot(sat_data_x, sat_data_y, sat_data_z, linewidth=1.2, c='orange')

# trace center point
trace_center = (satellite_position + (satellite_vector * 0.5)).astype(int)
sat_data_x, sat_data_y, sat_data_z = line_format(trace_center, observer_position)
ax.plot(sat_data_x, sat_data_y, sat_data_z, linewidth=1.2, c='orange')

sat_data_x, sat_data_y, sat_data_z = line_format(satellite_position, observer_position)
ax.plot(sat_data_x, sat_data_y, sat_data_z, linewidth=0.4, c='lime')

sat_data_x, sat_data_y, sat_data_z = line_format(satellite_position_endpoint, observer_position)
ax.plot(sat_data_x, sat_data_y, sat_data_z, linewidth=0.4, c='lime')

# helper lines
trace_center_ground = (satellite_position + (satellite_vector * 0.5)).astype(int)[:2]
trace_center_ground = trace_center_ground[0], trace_center_ground[1], 0
sat_data_x, sat_data_y, sat_data_z = line_format(trace_center_ground, observer_position)
ax.plot(sat_data_x, sat_data_y, sat_data_z, linewidth=0.7, c='pink')

sat_data_x, sat_data_y, sat_data_z = line_format(trace_center, trace_center_ground)
ax.plot(sat_data_x, sat_data_y, sat_data_z, linewidth=0.6, c='royalblue', linestyle='dashed')

path_foot_point_ground = path_foot_point[0], path_foot_point[1], 0
sat_data_x, sat_data_y, sat_data_z = line_format(path_foot_point_ground, path_foot_point)
ax.plot(sat_data_x, sat_data_y, sat_data_z, linewidth=0.6, c='royalblue', linestyle='dashed')

sat_data_x, sat_data_y, sat_data_z = line_format(path_foot_point_ground, observer_position)
ax.plot(sat_data_x, sat_data_y, sat_data_z, linewidth=1.3, c='pink')

reference_line_vector = np.array([1, -1, 0])
scalar = observer_position[1] - satellite_position[1]
reference_line_point = observer_position + (reference_line_vector * scalar)
sat_data_x, sat_data_y, sat_data_z = line_format(reference_line_point, observer_position)
ax.plot(sat_data_x, sat_data_y, sat_data_z, linewidth=0.4, c='gray')

# // Angle Calculations //

# before here were some other values like (horizontal declination), I removed them for simplicity
# they were used for testing the accuracy but are not important for the accuracy in general.

# in the real image we would need to translate the elevation and Azimuth to a vector
zero_angle_vector = np.subtract(path_foot_point, observer_position)
trace_center_vector = np.subtract(trace_center, observer_position)

# In the real image it is very simple to calculate this angle. This angle is the Angle between the
# two orange lines in the plot. This is the important angle when calculating the "distortion" by perspective
tilted_elevation = angle_delta(zero_angle_vector, trace_center_vector)
print(f'tilted_elevation: {tilted_elevation}')

# distortion calculation (this actually corrects the visible trace)
perspective_correction = trace_angle_delta / (0.5 * cos(radians(2 * tilted_elevation)) + 0.5)
print(f'perspective correction: {perspective_correction}')


def velocity_bisect(expected_height):
    # convert startpoint to azimuth and elevation
    # this takes the three coordinates of the satellite in space and calculates azimuth and elevation angles from it.
    angles_start = translate_vector_angles(satellite_position)[0], 90 - translate_vector_angles(satellite_position)[1]

    # takes elevation and a expected height and calculates the distance between the observer and the satellite with that height
    satellite_start_distance = expected_height / sin(radians(angles_start[1]))
    # convert endpoint ro azimuth and elevation and distance
    angles_end = translate_vector_angles(satellite_position_endpoint)[0], 90 - \
                 translate_vector_angles(satellite_position_endpoint)[1]
    satellite_end_distance = expected_height / sin(radians(angles_end[1]))

    # from this guessed distance we can now calculate were the satellite would be if we hadn't known its position in space
    satellite_calc_position_start = translate_angles_point(satellite_start_distance, angles_start[0], angles_start[1])
    satellite_calc_position_end = translate_angles_point(satellite_end_distance, angles_end[0], angles_end[1])

    # now you can calculate the distance between these two points and see how far it it, divide it by the
    # exposure time and you get a speed the satellite would travel if it were at that height.
    satellite_calc_distance = np.linalg.norm(np.subtract(satellite_calc_position_start, satellite_calc_position_end))
    satellite_calc_velocity = satellite_calc_distance / exposure_time

    # Now we calculate the orbital speed a satellite would have in the expected height.
    satellite_orbit_velocity = math.sqrt((grav_const * earth_mass) / (rad_earth + (expected_height)))

    # Only if the speed a satellite would have and the speed we calculated match up, we know the satellite is at that
    # specific height.
    return satellite_calc_velocity - satellite_orbit_velocity


# This can be done using taylor approximation as well
satellite_height = optimize.bisect(velocity_bisect, 1, 20000000)
print(f'satellite_height: {satellite_height}')


def velocity_bisect(expected_height):

    # when we correct the satellite trace it is as if the satellite would be at its closest point to us
    # (in the foot_point) so we need to calculate the height of the satellite using this altitude.
    satellite_vector_distance = expected_height / sin(radians(foot_point_elevation))

    # this calculates how far two points are apart if I know the distance to them and the angle between them
    satellite_calc_length = (satellite_vector_distance * sin(radians(perspective_correction / 2))) * 2

    # we now calculate how fast the satellite would be with the guessed height, trace it leaves in the sky
    # alt altitude (angle) it passes through
    satellite_calc_velocity = satellite_calc_length / exposure_time

    # This is a different velocity, the velocity a satellite would need to have if it were to fly at that height
    satellite_orbit_velocity = math.sqrt((grav_const * earth_mass) / (rad_earth + (expected_height)))

    # if both of these velocities are the same we know we have found the height the satellite flies at.
    return satellite_calc_velocity - satellite_orbit_velocity


# This can be done using taylor approximation as well
satellite_height = optimize.bisect(velocity_bisect, 1, 20000000)
print(f'satellite_height: {satellite_height}')


ax.scatter(observer_position[0], observer_position[1], observer_position[2], s=5)
ax.set_zlabel('h')
set_axes_equal(ax)
plt.xlabel('x')
plt.ylabel('y')
ax.set_zlim(0, height + 100)
plt.show()


