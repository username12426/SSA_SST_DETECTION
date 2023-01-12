import numpy as np
from math import sin, cos, degrees, atan2



class Star:
    # this class stores every star as a single object and as the counterpart to Star_Calc_oop it
    # takes in image coordinates and can calculate sky coordinates from it (horizontal coordinates)
    # and is used to undistort coordinates.
    def __init__(self, image_coordinates, image_center):
        self.image_coordinates = image_coordinates
        self.image_center = image_center
        self.translated_coordinate = None   # converted coordinate (image -> sky)
        self.exact_sky_coordinates = None   # used to store data from outside this class
        self.corrected_coordinates = None   # undistorted coordinate
        self.star_name = None
        self.star_size = None   # (magnitude)
        image_zenith = None     # values that are assigned once and stored in all objects
        north_offset = None
        arcseconds_per_px = None
        self.detected_Flag = False  # is star successfully detected?


    @staticmethod
    def rotate_point(theta, vector):  # rotation clockwise
        # rotate vector around an angle
        theta_rad = np.radians(theta)
        co, s = np.cos(theta_rad), np.sin(theta_rad)
        rot_matrix = np.array([[co, -s], [s, co]])
        vector_flip = np.array([vector[0], vector[1]])
        return np.around(np.dot(rot_matrix, vector_flip))

    @staticmethod
    def cal_angle(vector):  # direction is rotated pi degrees
        angle = degrees(atan2(vector[0], - vector[1])) % 360    # needs unit testing (%360 probably not needed)
        return angle

    def correct_coordinates(self, distortion_coef, img_input_coordinates=None):
        # this corrects (distortion) image coordinates using the distortion function from calibration
        center = self.image_center
        if not isinstance(img_input_coordinates, type(None)):
            # it is possible to input a coordinate and correct its position
            star_vector = np.subtract(img_input_coordinates, center)
        else:
            star_vector = np.subtract(self.image_coordinates, center)

        dist = np.linalg.norm(star_vector)  # dist ti image center

        distortion = (distortion_coef[0] * dist) + (distortion_coef[1] * (dist ** 2)) + (distortion_coef[2] * (dist ** 3)) + (distortion_coef[3])

        scalar = distortion / dist  # using scalar how far the coordinate needs to be corrected
        distortion_vector = star_vector - (star_vector * scalar)

        corrected_point = center + distortion_vector
        corrected_point = round(corrected_point[0]), round(corrected_point[1])

        if isinstance(img_input_coordinates, type(None)):
            self.corrected_coordinates = corrected_point

        return corrected_point

    def translate_coordinates(self, distortion_coef, position=None):

        # this is taking in a image coordinate and corrects (undistortes) its position and converts
        # the image coordinate to a sky coordinate using the data from the calibration.

        if not isinstance(position, type(None)):
            star_pos = self.correct_coordinates(distortion_coef, position)
        else:
            star_pos = self.correct_coordinates(distortion_coef)

        # Rotate the point around the offset
        stars_direction_vector = np.subtract(star_pos, self.image_zenith)  # order is very important
        translated_star_vector = self.rotate_point(self.north_offset, stars_direction_vector)

        stars_azimuth = 360 - self.cal_angle(translated_star_vector)  # minus 360 for counter-clockwise results

        stars_dist = np.linalg.norm(translated_star_vector)
        stars_elevation = 90 - ((stars_dist * self.arcseconds_per_px) / 3600)   # 3600 conversion hour second
        # the 90 is only there because i once defined the zero angle is at the 12o clock position in the image
        # most of the time if this 90 is somewhere in the code it is this.

        if isinstance(position, type(None)):
            self.translated_coordinate = np.array([stars_azimuth, stars_elevation])

        return np.array([stars_azimuth, stars_elevation])
