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

for angle in range(5, 90, 5):
    angle = 90 - angle
    middle_point = round(height / tan(radians(angle))), height
    # cv2.circle(blank, middle_point, 1, (255, 255, 255), thickness=1)
    satellite_pt_1 = round(middle_point[0] - view_length), height
    angle_1 = degrees(atan(height / satellite_pt_1[0]))
    satellite_pt_2 = round(middle_point[0] + view_length), height
    angle_2 = degrees(atan(height / satellite_pt_2[0]))

    angle_delta = abs(angle_2 - angle_1)

    x_ang.append(int(90 - angle))
    y_ang.append(angle_delta)

    cv2.line(blank, satellite_pt_1, satellite_pt_2, (255, 255, 255), thickness=1)
    cv2.line(blank, satellite_pt_1, (0, 0), (255, 255, 255), thickness=1)
    cv2.line(blank, satellite_pt_2, (0, 0), (255, 255, 255), thickness=1)

height = 1000
view_length = height * tan(radians(view_angle/2))
for angle in range(5, 90, 5):
    angle = 90 - angle
    middle_point = round(height / tan(radians(angle))), height
    # cv2.circle(blank, middle_point, 1, (255, 255, 255), thickness=1)
    satellite_pt_1 = round(middle_point[0] - view_length), height
    angle_1 = degrees(atan(height / satellite_pt_1[0]))
    satellite_pt_2 = round(middle_point[0] + view_length), height
    angle_2 = degrees(atan(height / satellite_pt_2[0]))

    angle_delta = abs(angle_2 - angle_1)
    y_ang_3.append(angle_delta)

    cv2.line(blank, satellite_pt_1, satellite_pt_2, (255, 255, 255), thickness=1)
    cv2.line(blank, satellite_pt_1, (0, 0), (255, 255, 255), thickness=1)
    cv2.line(blank, satellite_pt_2, (0, 0), (255, 255, 255), thickness=1)

height = 400
view_angle = 4
view_length = height * tan(radians(view_angle/2))
for angle in range(5, 90, 5):
    angle = 90 - angle
    middle_point = round(height / tan(radians(angle))), height
    # cv2.circle(blank, middle_point, 1, (255, 255, 255), thickness=1)
    satellite_pt_1 = round(middle_point[0] - view_length), height
    angle_1 = degrees(atan(height / satellite_pt_1[0]))
    satellite_pt_2 = round(middle_point[0] + view_length), height
    angle_2 = degrees(atan(height / satellite_pt_2[0]))

    angle_delta = abs(angle_2 - angle_1)
    y_ang_2.append(angle_delta)

    cv2.line(blank, satellite_pt_1, satellite_pt_2, (255, 255, 255), thickness=1)
    cv2.line(blank, satellite_pt_1, (0, 0), (255, 255, 255), thickness=1)
    cv2.line(blank, satellite_pt_2, (0, 0), (255, 255, 255), thickness=1)

height = 400
view_angle = 2
view_length = height * tan(radians(view_angle/2))
for angle in range(5, 90, 5):
    angle = 90 - angle
    middle_point = round(height / tan(radians(angle))), height
    # cv2.circle(blank, middle_point, 1, (255, 255, 255), thickness=1)
    satellite_pt_1 = round(middle_point[0] - view_length), height
    angle_1 = degrees(atan(height / satellite_pt_1[0]))
    satellite_pt_2 = round(middle_point[0] + view_length), height
    angle_2 = degrees(atan(height / satellite_pt_2[0]))

    angle_delta = abs(angle_2 - angle_1)
    y_ang_4.append(angle_delta)

    cv2.line(blank, satellite_pt_1, satellite_pt_2, (255, 255, 255), thickness=1)
    cv2.line(blank, satellite_pt_1, (0, 0), (255, 255, 255), thickness=1)
    cv2.line(blank, satellite_pt_2, (0, 0), (255, 255, 255), thickness=1)

height = 400
view_angle = 6
view_length = height * tan(radians(view_angle/2))
for angle in range(5, 90, 5):
    angle = 90 - angle
    middle_point = round(height / tan(radians(angle))), height
    # cv2.circle(blank, middle_point, 1, (255, 255, 255), thickness=1)
    satellite_pt_1 = round(middle_point[0] - view_length), height
    angle_1 = degrees(atan(height / satellite_pt_1[0]))
    satellite_pt_2 = round(middle_point[0] + view_length), height
    angle_2 = degrees(atan(height / satellite_pt_2[0]))

    angle_delta = abs(angle_2 - angle_1)
    y_ang_5.append(angle_delta)

    cv2.line(blank, satellite_pt_1, satellite_pt_2, (255, 255, 255), thickness=1)
    cv2.line(blank, satellite_pt_1, (0, 0), (255, 255, 255), thickness=1)
    cv2.line(blank, satellite_pt_2, (0, 0), (255, 255, 255), thickness=1)


def objective(x, a, b, c, d):
    return (a * x) + (b * x ** 2) + (c * x ** 3) + d

def objective2(x, ang):
    return (ang/2) * math.cos(radians(2 * x)) + (ang/2)


coef, _ = curve_fit(objective, x_ang, y_ang)
a, b, c, d = coef
print(f'{a}, {b}, {c}, {d}')
x_line = arange(min(x_ang), max(x_ang), 1)
y_line = objective(x_line, a, b, c, d)
plt.plot(x_line, y_line, '--', color='red')

coef, _ = curve_fit(objective, x_ang, y_ang_2)
a, b, c, d = coef
print(f'{a}, {b}, {c}, {d}')
x_line = arange(min(x_ang), max(x_ang), 1)
y_line = objective(x_line, a, b, c, d)
plt.plot(x_line, y_line, '--', color='red')

coef, _ = curve_fit(objective, x_ang, y_ang_3)
a, b, c, d = coef
print(f'{a}, {b}, {c}, {d}')
x_line = arange(min(x_ang), max(x_ang), 1)
y_line = objective(x_line, a, b, c, d)
plt.plot(x_line, y_line, '--', color='red')

coef, _ = curve_fit(objective, x_ang, y_ang_4)
a, b, c, d = coef
print(f'{a}, {b}, {c}, {d}')
x_line = arange(min(x_ang), max(x_ang), 1)
y_line = objective(x_line, a, b, c, d)
plt.plot(x_line, y_line, '--', color='red')

coef, _ = curve_fit(objective, x_ang, y_ang_5)
a, b, c, d = coef
print(f'{a}, {b}, {c}, {d}')
x_line = arange(min(x_ang), max(x_ang), 1)
y_line = objective(x_line, a, b, c, d)
plt.plot(x_line, y_line, '--', color='red')

for x in range(0, 90, 5):
    plt.plot(x, objective2(x, 8), '--', color='red')
    plt.scatter(x, objective2(x, 8), color='red')

for x in range(0, 90, 5):
    plt.plot(x, objective2(x, 4), '--', color='blue')
    plt.scatter(x, objective2(x, 4), color='blue')

for x in range(0, 90, 5):
    plt.plot(x, objective2(x, 2), '--', color='green')
    plt.scatter(x, objective2(x, 2),color='green')

for x in range(0, 90, 5):
    plt.plot(x, objective2(x, 6), '--', color='black')
    plt.scatter(x, objective2(x, 6),color='black')

plt.scatter(x_ang, y_ang)
plt.scatter(x_ang, y_ang_2)
plt.scatter(x_ang, y_ang_3)
plt.scatter(x_ang, y_ang_4)
plt.scatter(x_ang, y_ang_5)


blank = np.flipud(blank)
cv2.imshow("Satellite_sim: ", blank)
plt.show()
cv2.waitKey()


