
from math import radians, sin, cos
import numpy as np
import astropy.units as u
from astropy.coordinates import SkyCoord, AltAz


# always pass numpy arrays, if list
# star_ra in decimal degrees, star_dec in decimal degrees
class CalcStarData:
    def __init__(self, star_name, star_ra, star_dec, current_position, current_date, current_time):
        self.star_name = star_name
        self.star_ra = star_ra  # convert decimal degrees to decimal hours
        self.star_dec = star_dec
        self.translated_coordinates = None  # converted coordinates (image -> sky)
        self.true_coordinates = None
        self.error_vector = None    # not used in here bus used to store data
        self.error_length = None    # not used in here bus used to store data
        self.current_position = current_position
        self.current_date = current_date
        self.current_time = current_time
        self.star_size = None   # not used in here bus used to store data
        self.stars_dist_zenith = None
        self.height = 0
        self.star_pos = None

    # current_position: type(decimal-degrees), 0 < long < 180, East(Positive) West(Negative)
    # current_date: type(y, m, d), current_time(UTC): type(h, m, s), time_offset: type(minutes)

    # viewing angles are the same as horizontal coordinates or sky coordinates!
    def star_viewing_angles(self, time_obj, location_obj):
        # tales in a time, and location object (from outside the class, because its faster)
        # and calculates the position of a star in the sky (horizontal coord)
        # at given time and date

        # This is very slow!!! use  Celest library or raw numpy
        star = SkyCoord(ra=self.star_ra * u.degree, dec=self.star_dec * u.degree, frame='icrs')
        # read docs about astropy.coordinates
        star_altaz = star.transform_to(AltAz(obstime=time_obj, location=location_obj))

        azimuth, altitude = star_altaz.az.degree, star_altaz.alt.degree
        self.star_pos = np.array([azimuth, altitude])

        return azimuth, altitude

    @staticmethod
    def rotate_point(theta, vector):  # rotation clockwise
        # function to rotate a vector around an angle
        theta_rad = np.radians(theta)
        co, s = np.cos(theta_rad), np.sin(theta_rad)
        rot_matrix = np.array([[co, -s], [s, co]])
        vector_flip = np.array([vector[0], vector[1]])
        return np.around(np.dot(rot_matrix, vector_flip))

    def translate_coordinates(self, arcseconds_per_px, north_offset, image_zenith):
        # arcseconds_per_px, north_offset, image_zenith ara all from the calibration
        # this function coverts image coordinates to horizontal coordinates
        if arcseconds_per_px and north_offset and any(image_zenith) is not None:
            #  just distance from point in the image to the image center
            self.stars_dist_zenith = (((90 - self.star_pos[1]) * 3600) / arcseconds_per_px)     # 3600 conversion between hour and second, -90 because the angle didn't match up
            angle = (-(360 - self.star_pos[0]) + 90) % 360  # this stuff is because of the different layouts of polar coordinate systems
            azimuth_vector = np.array([cos(radians(angle)), -sin(radians(angle))])
            zenith_star_vector_off = (self.stars_dist_zenith * azimuth_vector)
            zenith_star_pos = self.rotate_point(360 - north_offset, zenith_star_vector_off) + image_zenith
            self.translated_coordinates = round(zenith_star_pos[0]), round(zenith_star_pos[1])
            return round(zenith_star_pos[0]), round(zenith_star_pos[1])
        else:
            return None



