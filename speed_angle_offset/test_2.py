# test how the perspective will change the visible arcsec

from math import tan, degrees, radians, atan, sin
import math
import cv2
import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import curve_fit
from numpy import arange

blank = np.zeros((500, 1000, 3), dtype='uint8')

height = 200
view_angle = 8

# this is the visible (pixel-length) in the zenith
view_length = height * tan(radians(view_angle/2))

print(view_length)

y_ang = []
y_ang_2 = []
y_ang_3 = []
y_ang_4 = []
y_ang_5 = []
x_ang = []

for angle in range(0, 90, 2):
    angle = 90 - angle
    middle_point = round(height / tan(radians(angle))), height
    # cv2.circle(blank, middle_point, 1, (255, 255, 255), thickness=1)
    satellite_pt_1 = round(middle_point[0] - view_length), height
    satellite_pt_2 = round(middle_point[0] + view_length), height
    if satellite_pt_1[0] > 0 and satellite_pt_2[0] > 0:
        angle_1 = degrees(atan(height / satellite_pt_1[0]))
        angle_2 = degrees(atan(height / satellite_pt_2[0]))

        angle_delta = abs(angle_2 - angle_1)

        x_ang.append(int(90 - angle))
        y_ang.append(angle_delta)

        cv2.line(blank, satellite_pt_1, satellite_pt_2, (255, 255, 255), thickness=1)
        cv2.line(blank, satellite_pt_1, (0, 0), (255, 255, 255), thickness=1)
        cv2.line(blank, satellite_pt_2, (0, 0), (255, 255, 255), thickness=1)


def objective(x, a, b, c, d):
    return (a * x) + (b * x ** 2) + (c * x ** 3) + d


def objective2(x, ang):
    return (0.5 * math.cos(radians(2 * x)) + 0.5) * ang


def objective3(x):
    return -0.5 * math.cos(radians(2 * x)) + 0.5


coef, _ = curve_fit(objective, x_ang, y_ang)
a, b, c, d = coef
print(f'{a}, {b}, {c}, {d}')
x_line = arange(min(x_ang), max(x_ang), 1)
y_line = objective(x_line, a, b, c, d)
plt.plot(x_line, y_line, '--', color='red', linewidth=1)


x_ = []
y_ = []
for x in range(0, 90, 2):
    x_.append(x)
    y_.append(objective3(x))
    plt.scatter(x, objective3(x), color='b', s=4)

plt.plot(x_, y_, linewidth=1, c="b")

x_ = []
y_ = []
for x in range(0, 90, 2):
    x_.append(x)
    y_.append(objective3(x))
    plt.scatter(x, objective3(x), color='b', s=4)

plt.scatter(x_ang, y_ang, s=4, c='r')

plt.title('Perspective Distortion')
plt.ylabel('red = [visible angle] blue = [relative angle distortion]')
plt.xlabel('Declination')

blank = np.flipud(blank)
cv2.imshow("Satellite_sim: ", blank)
plt.show()
cv2.waitKey()


