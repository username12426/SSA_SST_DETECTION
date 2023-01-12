from math import sqrt, degrees, atan2
import numpy as np


class Satellite:
    # stores data from the detected satellites and calculates position in the sky
    # it was intended that this class includes the calculations of the speed and height of the satellite too
    # currently this calculation is performed in the main scripts itself (Satellite Detection loop)
    def __init__(self, position_points, arcseconds_per_px, distortion_coef, image_center):
        # position_points need to be np.array
        self.position_points = position_points  # image points the satellite passes through (lots of small lines)
        self.distortion_coef = distortion_coef  # calibration data (distortion function coefficients)
        self.arcsecond_per_px = arcseconds_per_px  # value from the calibration using astrometry website
        self.image_center = image_center
        self.endpoints = self.calc_endpoints()  # calculates endpoints from the lines (position points)
        self.corrected_positions = self.distortion(self.endpoints[0]), self.distortion(
            self.endpoints[1])  # undistortes the passed points
        self.sky_coordinates = []  # coordinates of the satellite in the sky
        self.close_stars = []  # stars that are close to the satellite

    def calc_endpoints(self):
        # The passed endpoints (position points) are lots of lines that are not connected, this function finds
        # the furthest apart points of all these lines. These two points are now the points the satellite is
        # characterised with.

        # These are the points directly from the image, so the satellite points are not undistorted!!

        for i, similar_lines in enumerate(self.position_points):

            line_point_dir = similar_lines[0]
            direction = (line_point_dir[1] - line_point_dir[3]) / (line_point_dir[0] - line_point_dir[2])

            if direction < 0:
                satellite_trace_ymax = np.min(self.position_points[0, :, 1:4:2])
                satellite_trace_ymin = np.max(self.position_points[0, :, 1:4:2])
            else:
                satellite_trace_ymax = np.max(self.position_points[0, :, 1:4:2])
                satellite_trace_ymin = np.min(self.position_points[0, :, 1:4:2])

            satellite_trace_xmax = np.max(self.position_points[0, :, 0:3:2])
            satellite_trace_xmin = np.min(self.position_points[0, :, 0:3:2])

        self.position = [[satellite_trace_xmax, satellite_trace_ymax], [satellite_trace_xmin, satellite_trace_ymin]]
        self.coordinate_trajectory = direction
        return self.position

    @staticmethod
    # distance between two points
    def get_distance(pos_1, pos_2):
        x_1, y_1 = pos_1[:]
        x_2, y_2 = pos_2[:]
        distance = sqrt((x_1 - x_2) ** 2 + (y_1 - y_2) ** 2)
        return distance

    @staticmethod
    # make vector from points
    def get_vector(pos_1, pos_2):
        vector = np.subtract(pos_1, pos_2)
        return vector

    def distortion(self, image_coordinates):  # center (width/2, height/2)
        # takes in an image coordinate (px) and undistorts it with the given distortion parameters (distortion coef)
        center = self.image_center
        star_vector = np.subtract(image_coordinates, center)
        dist = np.linalg.norm(star_vector)  # distance from image center to the input coordinate
        # polynomial distortion function
        distortion = (self.distortion_coef[0] * dist) + (self.distortion_coef[1] * (dist ** 2)) + (
                self.distortion_coef[2]
                * (dist ** 3)) + (self.distortion_coef[3])

        # scalar is how much the original point needs to be moved so it is in its undistorted position
        scalar = distortion / dist
        distortion_vector = star_vector - (star_vector * scalar)

        corrected_point = center + distortion_vector
        corrected_point = round(corrected_point[0]), round(corrected_point[1])
        return corrected_point

    def get_close_stars(self, star_objects, satellite_position, dist_thresh=300):  # dist_thresh = max distance
        # this returns the close stars of the star passed list, but saves all stars to self.close_stars
        # 1 and 2 stand for the starting and endpoint of the satellite
        close_satellite_1 = []
        close_satellite_2 = []
        for star in star_objects:
            satellite_pos_1 = satellite_position[0]
            satellite_pos_2 = satellite_position[1]
            star_pos = star.corrected_coordinates
            if dist_thresh >= self.get_distance(star_pos, satellite_pos_1):
                close_satellite_1.append(star)
            if dist_thresh >= self.get_distance(star_pos, satellite_pos_2):
                close_satellite_2.append(star)
        return close_satellite_1, close_satellite_2

    @staticmethod
    def relative_pos(image_zenith, star_pos, star_cord, sat_pos, arcseconds_per_px):
        # only pass undistorted images other classes sometimes undistort the passed positions automatically.
        # This one does not.
        def points_angle(pos_1, pos_2):  # (start, end)
            vector = np.subtract(pos_2, pos_1)  # this order of vector direction
            angle = degrees(atan2(vector[0], - vector[1])) % 360
            return angle

        angle_delta = points_angle(image_zenith, star_pos) - points_angle(image_zenith, sat_pos)
        dist_delta = np.linalg.norm(np.subtract(image_zenith, star_pos)) - np.linalg.norm(
            np.subtract(image_zenith, sat_pos))
        elevation_delta = (dist_delta * arcseconds_per_px) / 3600
        satellite_cord = abs((star_cord[0] + angle_delta) % 360), star_cord[1] + elevation_delta
        return satellite_cord

    def positioning(self, star_objects, image_zenith, arcseconds_per_px, dist_thresh=300, manual_positions=None):
        if not isinstance(manual_positions, type(None)):
            # satellite_img_pos = self.distortion(manual_positions[0]), self.distortion(manual_positions[1])
            satellite_img_pos = manual_positions[0], manual_positions[1]
            coordinates_save_list = []
        else:
            satellite_img_pos = self.distortion(self.position[0]), self.distortion(self.position[1])
        close_star_list = self.get_close_stars(star_objects, satellite_img_pos, dist_thresh)
        for satellite_index, close_satellite_stars in enumerate(close_star_list):  # satellite_1 = index 0
            satellite_az = []
            satellite_alt = []
            for star in close_satellite_stars:
                satellite_pos = satellite_img_pos[satellite_index]
                if star.detected_Flag:
                    # star_pos = star.corrected_coordinates
                    star_pos = star.image_coordinates
                    star_cord = star.exact_sky_coordinates
                    satellite_cord_ = self.relative_pos(image_zenith, star_pos, star_cord, satellite_pos,
                                                        arcseconds_per_px)
                    satellite_az.append(satellite_cord_[0])
                    satellite_alt.append(satellite_cord_[1])
            if len(satellite_az) > 0 and len(satellite_alt) > 0:
                satellite_cord = (sum(satellite_az) / len(satellite_az), sum(satellite_alt) / len(satellite_alt))
            else:
                print("Error:")
                print(satellite_az, satellite_alt)
                print(len(close_star_list))
                satellite_cord = (
                sum(satellite_az) / (len(satellite_az) + 1), sum(satellite_alt) / (len(satellite_alt) + 1))

            if manual_positions:
                coordinates_save_list.append(satellite_cord)
            else:
                self.sky_coordinates.append(satellite_cord)

        if manual_positions:
            return coordinates_save_list
        return self.sky_coordinates













