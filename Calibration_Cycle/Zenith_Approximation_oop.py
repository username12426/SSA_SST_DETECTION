# calculates the Zenith based on the angles between known stars
# Trying to fix the zenith-position and distortion issue
# 26.7.2022

# rename Zenith_Alignment_Angles_oop 26.8.2022

import numpy as np
import cv2
import math


class ZenithAlignment:
    def __init__(self, approx_zenith_pos, star_angles, star_positions, index=None):
        self.approx_zenith_pos = approx_zenith_pos
        self.approx_zenith_pos_loop = approx_zenith_pos
        self.star_angles = star_angles
        self.star_positions = star_positions
        self.distortion_star_positions = None
        self.zenith_error = None
        self.zenith_pos = None
        self.circle_objects = None
        self.average_zenith_intersection = None
        self.north_vector = None

    class Circle:
        def __init__(self, center, radius):
            self.center = center
            self.radius = radius

    def distortion(self, frame):
        # function has no affect at the moment
        for i, star in enumerate(self.star_positions):
            mid_point = frame.shape
            zenith_vector = np.subtract(star, (mid_point[1] / 2, mid_point[0] / 2))
            zenith_vector = np.around(np.array([zenith_vector[0], zenith_vector[1]]))  # flip y-axis because inversion
            vector_length = np.linalg.norm(zenith_vector)
            new_pos = zenith_vector + np.array([mid_point[1] / 2, mid_point[0] / 2])
            self.star_positions[i] = new_pos
        return new_pos

    @staticmethod
    def rescale_frame(frame, scale):
        width = int(frame.shape[1] * scale)
        height = int(frame.shape[0] * scale)
        dimension = (width, height)
        return cv2.resize(frame, dimension, interpolation=cv2.INTER_AREA)

    @staticmethod
    def angle_circle(pos_1, pos_2, theta):
        # pos_1 start, pos_2 end
        points_vector = np.subtract(pos_2, pos_1)
        norm_points_vector = np.array([points_vector[1], - points_vector[0]])
        points_mid = (points_vector * 0.5) + pos_1
        mid_length = math.tan(math.radians(90 - theta)) * np.linalg.norm(points_vector * 0.5)
        radius = round(np.linalg.norm(points_vector * 0.5) / math.cos(math.radians(90 - theta)))
        mid_vector_scalar = mid_length / np.linalg.norm(norm_points_vector)
        mid_point_1 = [mid_vector_scalar * norm_points_vector[0], mid_vector_scalar * norm_points_vector[1]] + points_mid
        mid_point_1 = [round(mid_point_1[0]), round(mid_point_1[1])]
        mid_point_2 = [- mid_vector_scalar * norm_points_vector[0], - mid_vector_scalar * norm_points_vector[1]] + points_mid
        mid_point_2 = [round(mid_point_2[0]), round(mid_point_2[1])]
        return mid_point_1, mid_point_2, radius, points_mid

    @staticmethod
    def get_intersections(x0, y0, r0, x1, y1, r1):
        # circle 1: (x0, y0), radius r0
        # circle 2: (x1, y1), radius r1

        d = math.sqrt((x1 - x0) ** 2 + (y1 - y0) ** 2)

        # non-intersecting
        if d > r0 + r1:
            return None
        # One circle within other
        if d < abs(r0 - r1):
            return None
        # coincident circles
        if d == 0 and r0 == r1:
            return None
        else:
            a = (r0 ** 2 - r1 ** 2 + d ** 2) / (2 * d)
            h = math.sqrt(r0 ** 2 - a ** 2)
            x2 = x0 + a * (x1 - x0) / d
            y2 = y0 + a * (y1 - y0) / d
            x3 = x2 + h * (y1 - y0) / d
            y3 = y2 - h * (x1 - x0) / d

            x4 = x2 - h * (y1 - y0) / d
            y4 = y2 + h * (x1 - x0) / d

            return x3, y3, x4, y4

    def zenith_alignment(self):

        while True:     # loop helps make the error value for the zenith guess bigger

            circle_objects = []

            for i, angle in enumerate(self.star_angles):
                for p, sub_angle in enumerate(self.star_angles):
                    if not angle == sub_angle:
                        theta = abs(angle - sub_angle)
                        if theta > 180:
                            theta = 360 - theta     # (theta -360)
                        if theta < 90:      # this will filter the too big angle-circles
                            circle_1, circle_2, radius, mid = self.angle_circle(self.star_positions[i], self.star_positions[p], theta)

                            dist_1 = np.linalg.norm(np.subtract(circle_1, self.approx_zenith_pos_loop))
                            dist_2 = np.linalg.norm(np.subtract(circle_2, self.approx_zenith_pos_loop))

                            if dist_1 < dist_2:
                                circle_objects.append(self.Circle(circle_1, radius))
                            else:
                                circle_objects.append(self.Circle(circle_2, radius))

                        else:
                            circle_1, circle_2, radius, mid = self.angle_circle(self.star_positions[i], self.star_positions[p], theta)

                            dist_1 = np.linalg.norm(np.subtract(circle_1, self.approx_zenith_pos_loop))
                            dist_2 = np.linalg.norm(np.subtract(circle_2, self.approx_zenith_pos_loop))

                            if dist_1 < dist_2:

                                circle_objects.append(self.Circle(circle_2, radius))
                            else:
                                circle_objects.append(self.Circle(circle_1, radius))

            self.circle_objects = circle_objects

            zenith_x_pos = []
            zenith_y_pos = []
            zenith_dist_error = []

            for i, circle in enumerate(circle_objects):
                for p, sub_circle in enumerate(circle_objects):
                    if not i == p:
                        pos_1 = circle.center
                        pos_2 = sub_circle.center
                        intersections = self.get_intersections(pos_1[0], pos_1[1], round(circle.radius), pos_2[0], pos_2[1], round(sub_circle.radius))
                        if intersections:

                            intersect_1 = intersections[0:2]
                            intersect_2 = intersections[2:4]

                            dist_1 = abs(np.linalg.norm(np.subtract(intersect_1, self.approx_zenith_pos_loop)))
                            dist_2 = abs(np.linalg.norm(np.subtract(intersect_2, self.approx_zenith_pos_loop)))

                            if dist_1 < dist_2 and dist_1 < 200:
                                zenith_intersection = intersect_1
                                zenith_dist_error.append(dist_1)

                                zenith_x_pos.append(zenith_intersection[0])
                                zenith_y_pos.append(zenith_intersection[1])

                            elif dist_1 > dist_2 and dist_2 < 200:
                                zenith_intersection = intersect_2
                                zenith_dist_error.append(dist_2)

                                zenith_x_pos.append(zenith_intersection[0])
                                zenith_y_pos.append(zenith_intersection[1])

            self.average_zenith_intersection = np.array([round(sum(zenith_x_pos) / len(zenith_x_pos)), round(sum(zenith_y_pos) / len(zenith_y_pos))])
            self.zenith_error = sum(zenith_dist_error) / len(zenith_dist_error)

            if np.linalg.norm(np.subtract(self.average_zenith_intersection, self.approx_zenith_pos_loop)) > 3:
                self.approx_zenith_pos_loop = self.average_zenith_intersection

            else:
                return self.average_zenith_intersection

    @staticmethod
    def rotate_point(theta, vector):  # rotation clockwise
        theta_rad = np.radians(theta)
        co, s = np.cos(theta_rad), np.sin(theta_rad)
        rot_matrix = np.array([[co, -s], [s, co]])
        vector_flip = np.array([vector[0], vector[1]])
        return np.around(np.dot(rot_matrix, vector_flip))

    def get_north_vector(self):
        max_angle = max(self.star_angles)
        max_angle_delta = 360 - max_angle
        max_pos = self.star_positions[self.star_angles.index(max_angle)]
        max_pos_vector = np.subtract(max_pos, self.average_zenith_intersection)
        max_pos_rotate = self.rotate_point(- max_angle_delta, max_pos_vector)  # negative rotation (anticlockwise)

        min_angle = min(self.star_angles)
        min_pos = self.star_positions[self.star_angles.index(min_angle)]
        min_pos_vector = np.subtract(min_pos, self.average_zenith_intersection)
        min_pos_rotate = self.rotate_point(min_angle, min_pos_vector)

        self.north_vector = np.array([round((max_pos_rotate[0] + min_pos_rotate[0]) / 2), round((max_pos_rotate[1] + min_pos_rotate[1]) / 2)])
        return self.north_vector

    def get_arc_per_px(self, aligned_points, elevation_difference):
        # elevation_difference decimal degrees
        vector_length = np.linalg.norm(np.subtract(aligned_points[0], aligned_points[1]))
        arc_per_px = elevation_difference * 3600 / vector_length
        return arc_per_px

    def visualize(self, blank, show=None):

        for circle in self.circle_objects:
            center = circle.center
            radius = circle.radius
            cv2.circle(blank, center, radius, (255, 200, 0), thickness=2)

        intersect = self.average_zenith_intersection

        cv2.circle(blank, intersect, 20, (255, 255, 0), thickness=6)
        cv2.circle(blank, self.approx_zenith_pos, 20, (0, 0, 255), thickness=6)

        if self.north_vector is not None:
            cv2.line(blank, intersect, intersect + self.north_vector, (255, 255, 255), thickness=4)

        if show:
            blank_rescale = self.rescale_frame(blank, 0.25)
            cv2.imshow("Alignment: ", blank_rescale)
            cv2.waitKey()
            return blank
        else:
            return blank
