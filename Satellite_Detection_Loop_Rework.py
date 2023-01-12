''' This is a test of a full detection code. '''

# 24.11 This is the same script as "main_imp_imgloop_test", just with more comments


from tkinter import *
import numpy as np
import datetime
import astropy.units as u
from astropy.time import Time
from astropy.coordinates import SkyCoord, EarthLocation, AltAz
import pytz
import time as T
import cv2
from math import radians, degrees, sin, cos, atan2, tan, atan, acos
from scipy import optimize
import matplotlib.pyplot as plt

from objects.Star_Calc_2_oop import CalcStarData
from objects.Star_Img_oop_2 import Star
from objects.satellite_oop import Satellite
import math
import os
from PIL import Image


def get_date_taken(path):
    return Image.open(path)._getexif()[36867]


def rescale_frame(frame, scale):
    width = int(frame.shape[1] * scale)
    height = int(frame.shape[0] * scale)
    dimension = (width, height)
    return cv2.resize(frame, dimension, interpolation=cv2.INTER_AREA)


def line_direction(x1, y1, x2, y2):
    orientation = (y1 - y2) / (x1 - x2)
    if orientation >= 0:
        return math.degrees(math.atan(orientation))
    else:
        return math.degrees(math.atan(abs(orientation))) + 90


def rotate_point(theta, vector):  # rotation clockwise
    theta_rad = np.radians(theta)
    co, s = np.cos(theta_rad), np.sin(theta_rad)
    rot_matrix = np.array([[co, -s], [s, co]])
    vector_flip = np.array([vector[0], vector[1]])
    return np.around(np.dot(rot_matrix, vector_flip))


def cal_angle(vector):  # direction is rotated pi degrees
    angle = degrees(atan2(vector[0], - vector[1])) % 360
    return angle


def print_time_delta(last_time):
    time = T.time() - last_time
    print(time)
    return T.time()


# /// SETTINGS ///

file_path = r'C:\Users\Admin\Desktop\Test_Images_Phone_12.7.22'
position = 52.35933, 13.56694   # in decimal degrees
height = 70 # height above ground
zenith_pos = 2181, 1492     # zenith in the image (calibration)
north_offset = 245.32713262988753   # from calibration
distortion_parameters = -0.0234756793128147, 3.920912444488422e-05, 1.6978300496190803e-08, 0.06685635181503462 # from calibration
rescale_setting = 0.3   # size of the output image (so the plots fit your screen)
arcsec = 70.6   # (arcecond / pixel) from astrometry.net (calibration)
satellite_positioning_thresh = 900  # radius or area for stars used for poitioning of the satellite
exposure_time = 6   # exposure time of the taken image
image_timezone = "Europe/Berlin"

# filter the small stars (the very dim ones)
magnitude_threshold = 5     # the higher the number the more stars are used from the database (to many ar not good)

# // Initial Lists and plot sizes //

# Lists storing data from "perspective correction" for plot
declination_list = []   # declination of satellite above sky
visible_trace = []      # length of trace in the sky
calc_trace = []     # perspective corrected trace
height_list = []    # calculated satellite height

sim_satellite_pos = []
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

trace_blank = np.zeros((3000, 4000, 3), dtype='uint8')  # empty image for image plot

# plot count
count = 0


# /// START OF DETECTION LOOP ///


# loops over all images given in the dataset (file_path)
if not os.path.exists(file_path):
    raise ValueError("Add Valid File Path")
for path in os.listdir(file_path):

    # check if current path is a valid file
    if not os.path.isfile(os.path.join(file_path, path)):
        print(f"ERROR: Cant open File {os.path.join(file_path, path)}")
        continue

    print(f'File Path: {path}')
    image_path = os.path.join(file_path, path)
    # extract the time and date from the image
    data = get_date_taken(image_path).split(" ")
    current_date = [int(dig) for dig in data[0].split(":")]
    current_time = [int(dig) for dig in data[1].split(":")]
    img = cv2.imread(image_path)
    image_center = (round(img.shape[1] * 0.5), round(img.shape[0] * 0.5))
    # !! This is the correct indexing, I use the format [x, y] when having coordinates but the size of the image
    # is given in [height, width] so you have to flip it because (width is the x) and (height the y)

    # /// STAR AND SATELLITE IMAGE DETECTION ///

    # these are some simple filter steps for detecting lines and sars in the image
    # grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # blur
    blur = cv2.GaussianBlur(gray, (3, 3), 0)
    # edge cascade (used for star detection later)
    canny = cv2.Canny(blur, 125, 175)  # 125, 175
    # line detection
    lines = cv2.HoughLinesP(canny, 1, np.pi / 180, 15, None, 50, 10)

    # /// satellite detection ///

    # // Sorting all the line segments in groups of similar lines //

    # the segments (detection sort) and (distance sort) simply sort all detected lines in the image to groups
    # this is because HoughLines is detecting one line as several line bits, as well this sort is needed if there are
    # more than one satellites in the image at the same time.

    # (needs upgrade!)

    satellite_objects = []

    # / direction sort /
    # sorts lines by similar direction
    if not isinstance(lines, type(None)):
        lines_list = lines.tolist()
        sorted_line_groups = []     # list of lines with similar direction
        for i_, line in enumerate(lines_list):
            line_points = line[0]
            line_dir = line_direction(line_points[0], line_points[1], line_points[2], line_points[3])
            line.append(line_dir)
            match_found_flag = False    # used to check if line segment was matched or not
            if not sorted_line_groups:
                sorted_line_groups.append([i_])
            else:
                for line_group in sorted_line_groups:
                    group_orientation = lines_list[line_group[0]][1]
                    if group_orientation + 2 > line_dir > group_orientation - 2 or \
                    (360 - group_orientation) + 2 > line_dir > (360 - group_orientation) - 2:   # 2 means 2 degrees of error is allowed
                        line_group.append(i_)
                        match_found_flag = True
                if not match_found_flag:
                    sorted_line_groups.append([i_])

        # / distance sort /
        sorted_lines = []
        for i_, line_group in enumerate(sorted_line_groups):
            # sort the line groups based on distance
            fix_points = lines_list[line_group[0]][0]
            p1 = np.array([fix_points[0], fix_points[1]])
            p2 = np.array([fix_points[2], fix_points[3]])
            sorted_lines.append([])
            for line in line_group:
                p3 = np.array([lines_list[line][0][0], lines_list[line][0][1]])
                dist = np.linalg.norm(np.cross(p2 - p1, p1 - p3)) / np.linalg.norm(p2 - p1)
                if dist < 10:
                    sorted_lines[i_].append(lines_list[line][0])

        # / save satellites to objects /
        # It is really important that the satellites are saved to the objetcts without correction!
        # Later in the code : SATELLITE POSITIONING AND CALCULATION the positions will un-distort the coordinates.
        satellite_objects = []  # list to store objects if there are several satellites
        sorted_lines_np = np.array(sorted_lines)
        for satellite_groups in sorted_lines:
            # init the object
            satellite_obj = Satellite(sorted_lines_np, arcsec, distortion_parameters, image_center)
            satellite_objects.append(satellite_obj)

        if len(satellite_objects) < 0:
            break

        # /// STAR POS CALCULATION ///
        # firstly all stars from the database are calculated in horizontal coordinates, then the image detection
        # finds stars in the image and calculates their horizontal coordinates. Then the database stars and the image
        # stars are compared and matched

        # Set the Location and Time object of the astropy library
        lat, long = position
        location_obj = EarthLocation(lat=lat * u.deg, lon=long * u.deg, height=height * u.m)  # *. u.deg is unit system
        # utcoffset = +2 * u.hour
        year, month, day = current_date
        hour, minute, second = current_time
        datetime_obj = datetime.datetime(year, month, day, hour, minute, second)
        timezone = pytz.timezone(image_timezone)
        aware = timezone.localize(datetime_obj)
        offset = int(aware.utcoffset().total_seconds() / 3600)
        time_obj = Time(datetime_obj) - offset * u.hour

        def translate_px_radec(star_pos, time_obj):  # convert an image coordinate to sky coordinates
            # only pass undistorted points !!!!
            # This funcion uses calibration settings from the beginning
            stars_direction_vector = np.subtract(star_pos, zenith_pos)  # order is very important
            translated_star_vector = rotate_point(north_offset, stars_direction_vector)

            stars_azimuth = 360 - cal_angle(translated_star_vector)  # minus 360 for counter-clockwise results

            stars_dist = np.linalg.norm(np.subtract(star_pos, zenith_pos))
            stars_elevation = 90 - ((stars_dist * arcsec) / 3600)

            newAltAzcoordiantes = SkyCoord(alt=stars_elevation * u.degree, az=stars_azimuth * u.degree,
                                           obstime=time_obj,
                                           frame='altaz',
                                           location=location_obj).icrs

            return newAltAzcoordiantes.ra.degree, newAltAzcoordiantes.dec.degree


        def translate_altaz_px(azimuth, altitude):  # translate sky to image coordinates
            # This function uses the settings from the beginning
            radec_dist_zenith = (((90 - altitude) * 3600) / arcsec)     # 3600 hour second conversion
            angle = (-(360 - azimuth) + 90) % 360
            azimuth_vector = np.array([cos(radians(angle)), -sin(radians(angle))])
            zenith_star_vector_off = (radec_dist_zenith * azimuth_vector)
            zenith_star_pos = rotate_point(360 - north_offset, zenith_star_vector_off) + zenith_pos
            translated_coordinates = round(zenith_star_pos[0]), round(zenith_star_pos[1])
            return translated_coordinates


        # // PARAMETERS FOR RADEC FILTER //
        # Radec means (right assertion and declination), is is used that not all stars from the database are checked

        # find the RaDec_0 image position
        radec = SkyCoord(ra=0 * u.degree, dec=90 * u.degree, frame='icrs')
        radec_altaz = radec.transform_to(AltAz(obstime=time_obj, location=location_obj))

        azimuth_0, altitude_0 = radec_altaz.az.degree, radec_altaz.alt.degree
        # this is the image coordinate were ra = 90 and dec = 0, used to position the filter grid
        translated_coordinates_latlong = translate_altaz_px(azimuth_0, altitude_0)

        # find the four points in which the satellite is (for filtering e.g + dist_thresh)
        satellite_img_pos = satellite_objects[0].corrected_positions

        sat_radec_vector = np.subtract(satellite_img_pos[0], translated_coordinates_latlong)
        sat_radec_dist = np.linalg.norm(sat_radec_vector)
        scalar = (satellite_positioning_thresh + 200) / np.linalg.norm(sat_radec_vector)

        dec_box_position_1 = translated_coordinates_latlong + (sat_radec_vector * (1 + scalar))
        dec_box_position_1 = int(dec_box_position_1[0]), int(dec_box_position_1[1])
        dec_min_filter = translate_px_radec(dec_box_position_1, time_obj)[1]

        dec_box_position_2 = translated_coordinates_latlong + (sat_radec_vector * (1 - scalar))
        dec_box_position_2 = int(dec_box_position_2[0]), int(dec_box_position_2[1])
        dec_max_filter = translate_px_radec(dec_box_position_2, time_obj)[1]

        sat_radec_vector_norm = np.subtract(satellite_img_pos[0], translated_coordinates_latlong)
        sat_radec_vector_norm = np.array([sat_radec_vector_norm[1], - sat_radec_vector_norm[0]])

        # max/min is determined by how you change the vector to get the norm vector
        ra_box_position_1 = satellite_img_pos[0] + (sat_radec_vector_norm * scalar)
        ra_box_position_1 = int(ra_box_position_1[0]), int(ra_box_position_1[1])
        ra_min_filter = translate_px_radec(ra_box_position_1, time_obj)[0]

        ra_box_position_2 = satellite_img_pos[0] - (sat_radec_vector_norm * scalar)
        ra_box_position_2 = int(ra_box_position_2[0]), int(ra_box_position_2[1])
        ra_max_filter = translate_px_radec(ra_box_position_2, time_obj)[0]

        # open the database with right-assertion and declination

        try:
            ra = open('../star_database/hygfull (2) (1).txt', 'r')
            dec = open('../star_database/hygfull (1) (1).txt', 'r')
            mag = open('../star_database/hygfull (2) (2).txt', 'r')
        except Exception as e:
            raise Exception(f"Download the star database and change the file path in line: {e.__traceback__.tb_lineno}")

        ra = open('../star_database/hygfull (2) (1).txt', 'r')
        dec = open('../star_database/hygfull (1) (1).txt', 'r')
        mag = open('../star_database/hygfull (2) (2).txt', 'r')


        # read all data
        star_calc_objects = []
        ra_contents = ra.readlines()
        dec_contents = dec.readlines()

        ra.close()
        dec.close()

        # // IMAGE STAR IDENTIFICATION //

        # check all stars from the database and if they are found in the image

        for i, line in enumerate(mag):
            mag_float = float(line)
            if mag_float < magnitude_threshold:
                star_name = None

                right_assertion = float(ra_contents[i]) * 15    # 15 to convert hours to degree
                declination = float(dec_contents[i])

                # satellite distance to the radec_0 point will decide wich filter to use
                if sat_radec_dist > satellite_positioning_thresh:
                    if dec_min_filter < declination < dec_max_filter and ra_min_filter < right_assertion < ra_max_filter:

                        star_obj = CalcStarData(star_name, right_assertion, declination, position, current_date,
                                                current_time)
                        star_obj.star_viewing_angles(time_obj, location_obj)
                        # translate angle to image coordinates
                        star_obj.translate_coordinates(arcsec, north_offset, zenith_pos)
                        star_calc_objects.append(star_obj)
                else:
                    if dec_min_filter < declination:
                        star_obj = CalcStarData(star_name, right_assertion, declination, position, current_date,
                                                current_time)
                        star_obj.star_viewing_angles(time_obj, location_obj)
                        # translate angle to image coordinates
                        star_obj.translate_coordinates(arcsec, north_offset, zenith_pos)
                        star_calc_objects.append(star_obj)

        # // IMAGE DETECTION OF STARS //
        # find contours of bright objects
        contours, _ = cv2.findContours(canny, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

        # / Sort the Stars /
        # Circle detection on all detected contours, sort them by size
        image_detected_stars = []
        image_detected_radius = []
        for i, c in enumerate(contours):
            # don't create star objects in here to get rid of double stars
            contours_poly = cv2.approxPolyDP(c, 3, False)  # try true and false
            center, radi = cv2.minEnclosingCircle(contours_poly)
            if 1 < radi < 7:  # Filter for only bigger objects (stars)
                image_detected_stars.append(center)
                image_detected_radius.append(radi)

        # initialise all image detected stars (detected_circles) as objects from Star class
        Star.arcseconds_per_px = arcsec
        Star.north_offset = north_offset
        Star.image_zenith = zenith_pos

        star_img_objects = []
        image_center = (round(img.shape[1] * 0.5), round(img.shape[0] * 0.5))
        # really odd is that the width of the image is index 1 and not 0! This caused quite some bugs so be careful
        for star_center in image_detected_stars:
            img_star_object = Star(star_center, image_center)
            img_star_object.translate_coordinates(distortion_parameters)
            star_img_objects.append(img_star_object)

        # /// Match the Star Positions and indentify stars ///

        matched_stars = []
        for i, img_star_obj in enumerate(star_img_objects):
            old_dist = 10 ** 10  # random high number to have a starting value
            img_star_obj.star_size = image_detected_radius[i]   # append the image size of the star to the star class (can be used to classify stars)
            position_trans = img_star_obj.translated_coordinate
            for star in star_calc_objects:
                # given in degrees not px-pos
                # The 0.8 in both cases is the variation in degrees that will still accept a detection
                # This isn't the lowest value but sufficient if the magnitude filter is adjusted, can be optimized
                if star.star_pos[0] - 0.8 <= position_trans[0] <= star.star_pos[0] + 0.8:  # Still need to hande stars close to north!!!!
                    # az is way more accurate than elevation (deviation = 1)
                    if star.star_pos[1] - 0.8 <= position_trans[1] <= star.star_pos[1] + 0.8:  # el is less accurate thus (deviation = 2)
                        img_star_obj.detected_Flag = True
                        dist = np.linalg.norm(np.subtract(star.star_pos, position_trans))
                        if dist < old_dist:
                            name = star.star_name
                            coord = star.star_pos
                        old_dist = dist
            if img_star_obj.detected_Flag:
                img_star_obj.star_name = name
                img_star_obj.exact_sky_coordinates = coord

        # /// Satellite Positioning and Database///


        def get_foot(p, a, b):  # shortest point between line and two points
            ap = p - a
            ab = b - a
            result = a + np.dot(ap, ab) / np.dot(ab, ab) * ab
            return result


        def unit_vector(vector):
            return vector / np.linalg.norm(vector)


        def angle_delta(v1, v2):
            v1_u = unit_vector(v1)
            v2_u = unit_vector(v2)
            return degrees(np.arccos(np.clip(np.dot(v1_u, v2_u), -1.0, 1.0)))


        def translate_angles_vector(theta, phi):  # theta = right assertion, phi = declination
            # this function takes normal declination -90 degrees
            x = cos(radians(theta)) * sin(radians(phi))
            y = sin(radians(theta)) * sin(radians(phi))
            z = cos(radians(phi))
            return x, y, z


        # settings for plot
        canvas_shape = (1000, 1000, 3)
        canvas = np.zeros(canvas_shape, dtype='uint8')
        canvas_center = round(canvas.shape[1] / 2), round(canvas.shape[0] / 2)

        # // SATELLITE POSITIONING AND CALCULATION //
        # calculation
        test_satellite_obj = satellite_objects[0]

        satellite_pos = test_satellite_obj.corrected_positions  # undistort the satellite positions (start and end point)
        img_coordinates_start, img_coordinates_end = np.array(satellite_pos[0]), np.array(satellite_pos[1])

        # This is calculating the satellites position in sky (horizontal) coordinates from the image coordinates.
        satellite_sky_coordinates = test_satellite_obj.positioning(star_img_objects, zenith_pos,
                                                                   arcseconds_per_px=arcsec,
                                                                   dist_thresh=satellite_positioning_thresh)

        # this is used to toggle the satellite positioning (read about this in the report), but in later testing the
        # positioning proved more accurate tan expected (still far from perfect)
        # satellite_sky_coordinates = star_img_objects[0].translate_coordinates(distortion_parameters, satellite_pos[0]), star_img_objects[0].translate_coordinates(distortion_parameters, satellite_pos[1])

        sky_coordinates_start = np.array(satellite_sky_coordinates[0])
        sky_coordinates_end = np.array(satellite_sky_coordinates[1])


        # // perspective distortion correction //

        # this is important to calculate the correct height of the satellite later
        # This is the same algorithm as in (Perspective_Distortion_calc)
        satellite_vector = np.array(np.subtract(img_coordinates_end, img_coordinates_start))
        print("")
        print(f'Satellite image vector: {np.linalg.norm(satellite_vector)}')

        # trace calculation
        lambda_ang = abs(sky_coordinates_start[0] - sky_coordinates_end[0])
        # calculate the angle distance of the two points in the sky
        trace_angle_delta = math.degrees(math.acos(
            (sin(radians(sky_coordinates_start[1])) * sin(radians(sky_coordinates_end[1]))) + (
                    cos(radians(sky_coordinates_start[1])) * cos(radians(sky_coordinates_end[1])) * cos(
                radians(lambda_ang)))))

        visible_trace.append(trace_angle_delta)
        print(f'trace angle: {trace_angle_delta}')

        # calculation of ground declination

        path_foot_point = get_foot(zenith_pos, img_coordinates_start, img_coordinates_end)

        zero_angle_vector_ground = np.subtract(path_foot_point, zenith_pos)
        img_trace_center = img_coordinates_start + (satellite_vector * 0.5)
        img_trace_center_vector = np.subtract(img_trace_center, zenith_pos)
        horizontal_declination = angle_delta(zero_angle_vector_ground, img_trace_center_vector) % 180
        if horizontal_declination > 90:
            horizontal_declination = 90 - (horizontal_declination - 90)
        print(f'horizontal_declination: {horizontal_declination}')

        # calculation of tilted declination
        sky_foot_point = test_satellite_obj.positioning(star_img_objects, zenith_pos,
                                                        arcseconds_per_px=arcsec,
                                                        dist_thresh=satellite_positioning_thresh,
                                                        manual_positions=[path_foot_point,
                                                                          path_foot_point])[0]

        sky_foot_point_vector = translate_angles_vector(sky_foot_point[0],
                                                        90 - (sky_foot_point[1]))

        sky_trace_center_point = test_satellite_obj.positioning(star_img_objects, zenith_pos,
                                                                arcseconds_per_px=arcsec,
                                                                dist_thresh=satellite_positioning_thresh,
                                                                manual_positions=[img_trace_center,
                                                                                  img_trace_center])

        declination_list.append(horizontal_declination)

        sky_trace_center_vector = translate_angles_vector(sky_trace_center_point[0][0],
                                                          90 - (sky_trace_center_point[0][1]))

        tilted_declination = angle_delta(sky_foot_point_vector, sky_trace_center_vector)
        print(f'tilted_declination: {tilted_declination}')

        # / perspective correction calculation /
        perspective_correction = trace_angle_delta / (0.5 * cos(radians(2 * tilted_declination)) + 0.5)     # both 0.5 values were tested by try and error
        calc_trace.append(perspective_correction)
        print(f'perspective correction: {perspective_correction}')

        # constants
        grav_const = 6.67430e-11
        rad_earth = 6371000
        earth_mass = 5.972e+24


        def translate_vector_angles(vector):
            if vector[0] > 0:
                right_assertion = degrees(atan(vector[1] / vector[0]))
                declination = degrees(acos(vector[2] / np.linalg.norm(vector)))
                return right_assertion, declination
            else:
                declination = degrees(acos(vector[2] / np.linalg.norm(vector)))
                return None, declination


        def translate_angles_point(r, theta, phi):  # theta = right assertion, phi = declination
            x = r * cos(radians(theta)) * cos(radians(phi))
            z = r * sin(radians(theta)) * cos(radians(phi))
            y = r * sin(radians(phi))
            return x, y, z


        def velocity_bisect(expected_height):

            satellite_vector_length = expected_height / sin(radians(sky_foot_point[1]))      # this was just a test but i think its correct like this
            # satellite_vector_length = expected_height / sin(radians(satellite_center_declination))

            satellite_calc_distance = (satellite_vector_length * sin(radians(perspective_correction / 2))) * 2

            satellite_calc_velocity = satellite_calc_distance / exposure_time

            satellite_orbit_velocity = math.sqrt((grav_const * earth_mass) / (rad_earth + (expected_height)))

            return satellite_calc_velocity - satellite_orbit_velocity


        # This can be done using taylor approximation as well
        satellite_height = optimize.bisect(velocity_bisect, 1, 20000000)
        print(f'satellite_height: {satellite_height}')
        print("")

        height_list.append(satellite_height)

        # / calculate data for the plot /

        canvas_position_dist_start = (satellite_height / 1000) / tan(radians(sky_coordinates_start[1]))
        canvas_position_dist_end = (satellite_height / 1000) / tan(radians(sky_coordinates_end[1]))

        canvas_position_vector_start = np.array(
            [cos(radians(sky_coordinates_start[0])), sin(radians(sky_coordinates_start[0]))])
        canvas_position_vector_end = np.array(
            [cos(radians(sky_coordinates_end[0])), sin(radians(sky_coordinates_end[0]))])

        canvas_position_start = canvas_position_vector_start * canvas_position_dist_start
        canvas_position_start = np.array(
            [canvas_position_start[0], canvas_position_start[1], satellite_height / 1000])

        canvas_position_end = canvas_position_vector_end * canvas_position_dist_end
        canvas_position_end = np.array(
            [canvas_position_end[0], canvas_position_end[1], satellite_height / 1000])

        satellite_vector = np.subtract(canvas_position_start, canvas_position_end)


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


        # // Calculate Velocity, Height, Oribtal Period //

        satellite_velocity = math.sqrt((grav_const * earth_mass) / (rad_earth + satellite_height))  # function to calculate orbital speed in specific height
        orbit_length = 2 * math.pi * (rad_earth + satellite_height)
        orbital_period = (orbit_length / satellite_velocity) / 60  # /60 to et minutes

        pos_vector = np.subtract(img_coordinates_start, img_coordinates_end)
        outside_points = (pos_vector * 400) + img_coordinates_start, (pos_vector * - 400) + img_coordinates_start  # 400 random large number

        satellite_center_vector = np.subtract(img_coordinates_end, img_coordinates_start) * 0.5
        satellite_center = (img_coordinates_start + satellite_center_vector).astype(int)

        cv2.line(trace_blank, satellite_center, zenith_pos, (200, 200, 200), thickness=1)

        # // 3D Satellite Plot //

        observer = [0, 0, 0]

        # plot satellite path
        path_point_0 = (satellite_vector * - 5) + canvas_position_start
        path_point_1 = (satellite_vector * 5) + canvas_position_start

        sat_data_x, sat_data_y, sat_data_z = line_format(path_point_0, path_point_1)
        ax.plot(sat_data_x, sat_data_y, sat_data_z, linewidth=0.7)

        # plot satellite
        sat_data_x, sat_data_y, sat_data_z = line_format(canvas_position_start, canvas_position_end)
        ax.plot(sat_data_x, sat_data_y, sat_data_z, c='red')

        sim_satellite_pos.append([sat_data_x, sat_data_y, sat_data_z])

        # foot point
        path_foot_point = get_foot(observer, canvas_position_start, canvas_position_end)
        sat_data_x, sat_data_y, sat_data_z = line_format(path_foot_point, observer)
        ax.plot(sat_data_x, sat_data_y, sat_data_z, linewidth=0.9, c='orange')

        # trace center point
        trace_center = (canvas_position_start - (satellite_vector * 0.5)).astype(int)
        sat_data_x, sat_data_y, sat_data_z = line_format(trace_center, observer)
        ax.plot(sat_data_x, sat_data_y, sat_data_z, linewidth=0.9, c='orange')

        sat_data_x, sat_data_y, sat_data_z = line_format(canvas_position_start, observer)
        ax.plot(sat_data_x, sat_data_y, sat_data_z, linewidth=0.4, c='lime')

        sat_data_x, sat_data_y, sat_data_z = line_format(canvas_position_end, observer)
        ax.plot(sat_data_x, sat_data_y, sat_data_z, linewidth=0.4, c='lime')

        # helper lines
        trace_center_ground = (canvas_position_start - (satellite_vector * 0.5)).astype(int)[:2]
        trace_center_ground = trace_center_ground[0], trace_center_ground[1], 0
        sat_data_x, sat_data_y, sat_data_z = line_format(trace_center_ground, observer)
        ax.plot(sat_data_x, sat_data_y, sat_data_z, linewidth=0.6, c='magenta')

        sat_data_x, sat_data_y, sat_data_z = line_format(trace_center, trace_center_ground)
        ax.plot(sat_data_x, sat_data_y, sat_data_z, linewidth=0.4, c='royalblue')

        path_foot_point_ground = path_foot_point[0], path_foot_point[1], 0
        sat_data_x, sat_data_y, sat_data_z = line_format(path_foot_point_ground, path_foot_point)
        ax.plot(sat_data_x, sat_data_y, sat_data_z, linewidth=0.4, c='royalblue')

        sat_data_x, sat_data_y, sat_data_z = line_format(path_foot_point_ground, observer)
        ax.plot(sat_data_x, sat_data_y, sat_data_z, linewidth=0.6, c='magenta')

        reference_line_vector = np.array([1, -1, 0])
        scalar = observer[1] - canvas_position_start[1]
        reference_line_point = observer + (reference_line_vector * scalar)
        sat_data_x, sat_data_y, sat_data_z = line_format(reference_line_point, observer)
        ax.plot(sat_data_x, sat_data_y, sat_data_z, linewidth=0.4, c='gray')

        # direction of travel, direction relative to north
        star_direction_vector = np.subtract(satellite_pos[0], satellite_pos[1])
        translated_star_vector = rotate_point(north_offset, star_direction_vector)
        flight_path = 360 - cal_angle(translated_star_vector)  # minus 360 for counter-clockwise results

        # // Trace Blank Image //

        cv2.line(trace_blank, outside_points[0], outside_points[1], (120, 120, 0), thickness=2)
        cv2.line(trace_blank, img_coordinates_start, img_coordinates_end, (0, 255, 0), thickness=5)

        # title
        cv2.putText(trace_blank, f'Satellite Detection {current_date}',
                    (80, 90), cv2.FONT_HERSHEY_SIMPLEX, 2,
                    (255, 255, 255), 2, 2)

        # /// Visualisation ///

        blank = np.zeros(img.shape, dtype='uint8')

        cv2.line(blank, translated_coordinates_latlong, image_center, (250, 250, 250), thickness=2)

        # Image center
        cv2.circle(blank, image_center, 14, (255, 255, 0), thickness=4)

        # box-points
        cv2.circle(blank, dec_box_position_1, 14, (0, 0, 250), thickness=10)
        cv2.circle(blank, dec_box_position_2, 14, (0, 0, 250), thickness=10)

        cv2.circle(blank, ra_box_position_1, 14, (0, 0, 250), thickness=10)
        cv2.circle(blank, ra_box_position_2, 14, (0, 0, 250), thickness=10)

        # Satellite
        cv2.line(blank, satellite_pos[0], satellite_pos[1], (0, 255, 0), thickness=6)

        cv2.circle(blank, satellite_pos[0], satellite_positioning_thresh, (0, 255, 0), thickness=2)
        cv2.circle(blank, satellite_pos[1], satellite_positioning_thresh, (0, 255, 0), thickness=2)
        cv2.line(blank, outside_points[0], outside_points[1], (150, 150, 150), thickness=2)

        # Image detected stars
        for img_star in star_img_objects:
            pos = img_star.image_coordinates
            pos = round(pos[0]), round(pos[1])
            if img_star.detected_Flag:
                cv2.circle(blank, pos, round(img_star.star_size) * 2, (255, 255, 255), thickness=2)
                cv2.circle(blank, pos, 18, (0, 255, 0), thickness=1)
                cv2.line(blank, pos, image_center, (255, 255, 0), thickness=1)

            else:
                cv2.circle(blank, pos, round(img_star.star_size) * 2, (255, 255, 255), thickness=4)

        # Image corrected Stars
        for img_star in star_img_objects:
            pos = img_star.corrected_coordinates
            pos = round(pos[0]), round(pos[1])
            if img_star.detected_Flag:
                cv2.circle(blank, pos, 14, (255, 255, 0), thickness=4)

        # Calc stars
        for calc_star in star_calc_objects:
            pos = calc_star.translated_coordinates
            cv2.circle(blank, pos, 12, (0, 0, 255), thickness=2)

        # Plot the coordinate system
        for elevation in range(0, 18):
            if elevation % 2 == 0:
                cv2.circle(blank, zenith_pos, round(((elevation * 5) * 3600) / arcsec), (200, 200, 200),
                           thickness=2)
            else:
                cv2.circle(blank, zenith_pos, round(((elevation * 5) * 3600) / arcsec), (255, 255, 255),
                           thickness=1)
        cv2.circle(blank, zenith_pos, 20, (255, 255, 255), thickness=2)

        for declination in range(0, 18):
            cv2.circle(blank, translated_coordinates_latlong, round(((declination * 5) * 3600) / arcsec), (200, 200, 200),
                        thickness=1)

        cv2.circle(blank, zenith_pos, 20, (255, 255, 255), thickness=2)

        # toggle for testing

        blank_r = rescale_frame(blank, rescale_setting)
        cv2.imshow("Star Identification ", blank_r)
        cv2.waitKey(1)


# height errorr

average_height = sum(height_list) / len(height_list)

print(f'Average height: {average_height}')
print(f'Error max: {(max(height_list) - average_height) / average_height}')
print(f'Error max: {abs(min(height_list) - average_height)/ average_height}')


cv2.imshow('trace_blank', rescale_frame(trace_blank, 0.35))
cv2.imwrite(r'C:\Users\niels\Desktop\satellite_time_plot_6.10.JPG', trace_blank)
cv2.waitKey()

plt.title("3D Satellite Model")
ax.set_zlabel('h')
plt.xlabel('x')
plt.ylabel('y')
plt.show()

xs, ys = zip(*sorted(zip(declination_list[1:], visible_trace[1:])))
xs_1, ys_1 = zip(*sorted(zip(declination_list[1:], calc_trace[1:])))


plt.plot(xs, ys, label='apparent trace')
plt.plot(xs_1, ys_1, label='corrected trace')
plt.xlabel('horizontal declination (degrees)')
plt.ylabel('satellite trace (degrees)')
plt.title('Perspective correction')
plt.legend(loc='lower left')
plt.show()
