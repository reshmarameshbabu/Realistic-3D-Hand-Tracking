import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

xmain = np.asarray([1, 0, 0])
ymain = np.asarray([0, 1, 0])
zmain = np.asarray([0, 0, 1])
origin = np.asarray([0, 0, 0])
tensor01 = np.asarray([0, 1])
tensor10 = np.asarray([1, 0])
tensor00 = np.asarray([0, 0])
xy_thres = 100
depth_thres = 150
cropSize = 176
fx = 475.065948
fy = 475.065857
u0 = 315.944855
v0 = 245.287079

def normalize(v):
    norm = np.linalg.norm(v)
    if norm == 0:
        return v
    return v / norm


def plot3DJ(y_values, x=0, y=0, z=0):
    fig = plt.figure()
    ax = plt.axes(projection='3d')
    y_predRot = y_values
    for i in range(0, 6):
        ax.scatter3D(y_predRot[i][0], y_predRot[i][1], y_predRot[i][2], color="k")
    for i in range(6, 9):
        ax.scatter3D(y_predRot[i][0], y_predRot[i][1], y_predRot[i][2], color="r")
    for i in range(9, 12):
        ax.scatter3D(y_predRot[i][0], y_predRot[i][1], y_predRot[i][2], color="g")
    for i in range(12, 15):
        ax.scatter3D(y_predRot[i][0], y_predRot[i][1], y_predRot[i][2], color="b")
    for i in range(15, 18):
        ax.scatter3D(y_predRot[i][0], y_predRot[i][1], y_predRot[i][2], color="m")
    for i in range(18, 21):
        ax.scatter3D(y_predRot[i][0], y_predRot[i][1], y_predRot[i][2], color="c")
    ax.plot3D(np.asarray([y_predRot[0][0], y_predRot[0][0] + 5]),
              np.asarray([y_predRot[0][1], y_predRot[0][1]]),
              np.asarray([y_predRot[0][2], y_predRot[0][2]]), 'r')
    ax.plot3D(np.asarray([y_predRot[0][0], y_predRot[0][0]]),
              np.asarray([y_predRot[0][1], y_predRot[0][1] + 5]),
              np.asarray([y_predRot[0][2], y_predRot[0][2]]), 'g')
    ax.plot3D(np.asarray([y_predRot[0][0], y_predRot[0][0]]),
              np.asarray([y_predRot[0][1], y_predRot[0][1]]),
              np.asarray([y_predRot[0][2], y_predRot[0][2] + 5]), 'b')

    ax.plot3D(np.asarray([y_predRot[x][0], y_predRot[y][0]]),
              np.asarray([y_predRot[x][1], y_predRot[y][1]]),
              np.asarray([y_predRot[x][2], y_predRot[y][2]]), 'k')
    ax.plot3D(np.asarray([y_predRot[y][0], y_predRot[z][0]]),
              np.asarray([y_predRot[y][1], y_predRot[z][1]]),
              np.asarray([y_predRot[y][2], y_predRot[z][2]]), 'k')
    ax.plot3D(np.asarray([y_predRot[z][0], y_predRot[x][0]]),
              np.asarray([y_predRot[z][1], y_predRot[x][1]]),
              np.asarray([y_predRot[z][2], y_predRot[x][2]]), 'k')
    plt.show()


def plot3DJLine(y_values, x=0, y=0, z=0, name = "temp.png"):
    fig = plt.figure(figsize=(5, 5))
    ax = plt.axes(projection='3d')
    ax.view_init(elev=-173., azim=93.)
    # ax.view_init(elev=177., azim=87.)

    y_predRot = y_values
    for i in range(0, 6):
        ax.scatter3D(y_predRot[i][0], y_predRot[i][1], y_predRot[i][2], color="k")
    for i in range(6, 9):
        ax.scatter3D(y_predRot[i][0], y_predRot[i][1], y_predRot[i][2], color="r")
    for i in range(9, 12):
        ax.scatter3D(y_predRot[i][0], y_predRot[i][1], y_predRot[i][2], color="g")
    for i in range(12, 15):
        ax.scatter3D(y_predRot[i][0], y_predRot[i][1], y_predRot[i][2], color="b")
    for i in range(15, 18):
        ax.scatter3D(y_predRot[i][0], y_predRot[i][1], y_predRot[i][2], color="m")
    for i in range(18, 21):
        ax.scatter3D(y_predRot[i][0], y_predRot[i][1], y_predRot[i][2], color="c")
    ax.plot3D(np.asarray([y_predRot[0][0], y_predRot[0][0] + 5]),
              np.asarray([y_predRot[0][1], y_predRot[0][1]]),
              np.asarray([y_predRot[0][2], y_predRot[0][2]]), 'r')
    ax.plot3D(np.asarray([y_predRot[0][0], y_predRot[0][0]]),
              np.asarray([y_predRot[0][1], y_predRot[0][1] + 5]),
              np.asarray([y_predRot[0][2], y_predRot[0][2]]), 'g')
    ax.plot3D(np.asarray([y_predRot[0][0], y_predRot[0][0]]),
              np.asarray([y_predRot[0][1], y_predRot[0][1]]),
              np.asarray([y_predRot[0][2], y_predRot[0][2] + 5]), 'b')

    ax.plot3D(np.asarray([y_predRot[x][0], y_predRot[y][0]]),
              np.asarray([y_predRot[x][1], y_predRot[y][1]]),
              np.asarray([y_predRot[x][2], y_predRot[y][2]]), 'k')
    ax.plot3D(np.asarray([y_predRot[y][0], y_predRot[z][0]]),
              np.asarray([y_predRot[y][1], y_predRot[z][1]]),
              np.asarray([y_predRot[y][2], y_predRot[z][2]]), 'k')
    ax.plot3D(np.asarray([y_predRot[z][0], y_predRot[x][0]]),
              np.asarray([y_predRot[z][1], y_predRot[x][1]]),
              np.asarray([y_predRot[z][2], y_predRot[x][2]]), 'k')
    line3D(y_predRot, 0, 1, ax)
    line3D(y_predRot, 0, 2, ax)
    line3D(y_predRot, 0, 3, ax)
    line3D(y_predRot, 0, 4, ax)
    line3D(y_predRot, 0, 5, ax)
    line3D(y_predRot, 1, 6, ax)
    line3D(y_predRot, 6, 7, ax)
    line3D(y_predRot, 7, 8, ax)
    line3D(y_predRot, 2, 9, ax)
    line3D(y_predRot, 9, 10, ax)
    line3D(y_predRot, 10, 11, ax)
    line3D(y_predRot, 3, 12, ax)
    line3D(y_predRot, 12, 13, ax)
    line3D(y_predRot, 13, 14, ax)
    line3D(y_predRot, 4, 15, ax)
    line3D(y_predRot, 15, 16, ax)
    line3D(y_predRot, 16, 17, ax)
    line3D(y_predRot, 5, 18, ax)
    line3D(y_predRot, 18, 19, ax)
    line3D(y_predRot, 19, 20, ax)
    plt.savefig(name, format='png')
    plt.show()


def plot3DJLine2(y_values, x=0, y=0, z=0, name = "temp.png"):
    fig = plt.figure(figsize=(5, 5))
    ax = plt.axes(projection='3d')
    ax.view_init(elev=-173., azim=-91.)
    # ax.view_init(elev=-173., azim=93.)
    # ax.set_xlim3d(18, 40)
    # ax.set_ylim3d(-80, 20)
    # ax.set_zlim3d(420, 500)
    ax.set_xlim3d(-90, -120)
    ax.set_ylim3d(-60, 40)
    ax.set_zlim3d(520, 620)
    y_predRot = y_values
    for i in range(1, 2):
        ax.scatter3D(y_predRot[i][0], y_predRot[i][1], y_predRot[i][2], color="k")
    for i in range(6, 9):
        ax.scatter3D(y_predRot[i][0], y_predRot[i][1], y_predRot[i][2], color="r")
    # for i in range(9, 12):
    #     ax.scatter3D(y_predRot[i][0], y_predRot[i][1], y_predRot[i][2], color="g")
    # for i in range(12, 15):
    #     ax.scatter3D(y_predRot[i][0], y_predRot[i][1], y_predRot[i][2], color="b")
    # for i in range(15, 18):
    #     ax.scatter3D(y_predRot[i][0], y_predRot[i][1], y_predRot[i][2], color="m")
    # for i in range(18, 21):
    #     ax.scatter3D(y_predRot[i][0], y_predRot[i][1], y_predRot[i][2], color="c")
    ax.plot3D(np.asarray([y_predRot[1][0], y_predRot[1][0] + 2]),
              np.asarray([y_predRot[1][1], y_predRot[1][1]]),
              np.asarray([y_predRot[1][2], y_predRot[1][2]]), 'r')
    ax.plot3D(np.asarray([y_predRot[1][0], y_predRot[1][0]]),
              np.asarray([y_predRot[1][1], y_predRot[1][1] + 18]),
              np.asarray([y_predRot[1][2], y_predRot[1][2]]), 'g')
    ax.plot3D(np.asarray([y_predRot[1][0], y_predRot[1][0]]),
              np.asarray([y_predRot[1][1], y_predRot[1][1]]),
              np.asarray([y_predRot[1][2], y_predRot[1][2] + 8]), 'b')
    #
    # ax.plot3D(np.asarray([y_predRot[x][0], y_predRot[y][0]]),
    #           np.asarray([y_predRot[x][1], y_predRot[y][1]]),
    #           np.asarray([y_predRot[x][2], y_predRot[y][2]]), 'k')
    # ax.plot3D(np.asarray([y_predRot[y][0], y_predRot[z][0]]),
    #           np.asarray([y_predRot[y][1], y_predRot[z][1]]),
    #           np.asarray([y_predRot[y][2], y_predRot[z][2]]), 'k')
    # ax.plot3D(np.asarray([y_predRot[z][0], y_predRot[x][0]]),
    #           np.asarray([y_predRot[z][1], y_predRot[x][1]]),
    #           np.asarray([y_predRot[z][2], y_predRot[x][2]]), 'k')
    # line3D(y_predRot, 0, 1, ax)
    # line3D(y_predRot, 0, 2, ax)
    # line3D(y_predRot, 0, 3, ax)
    # line3D(y_predRot, 0, 4, ax)
    # line3D(y_predRot, 0, 5, ax)
    line3D(y_predRot, 1, 6, ax)
    line3D(y_predRot, 6, 7, ax)
    line3D(y_predRot, 7, 8, ax)
    # line3D(y_predRot, 2, 9, ax)
    # line3D(y_predRot, 9, 10, ax)
    # line3D(y_predRot, 10, 11, ax)
    # line3D(y_predRot, 3, 12, ax)
    # line3D(y_predRot, 12, 13, ax)
    # line3D(y_predRot, 13, 14, ax)
    # line3D(y_predRot, 4, 15, ax)
    # line3D(y_predRot, 15, 16, ax)
    # line3D(y_predRot, 16, 17, ax)
    # line3D(y_predRot, 5, 18, ax)
    # line3D(y_predRot, 18, 19, ax)
    # line3D(y_predRot, 19, 20, ax)
    plt.savefig(name, format='png')
    plt.show()


def line3D(y_predRot, x, y, ax):
    ax.plot3D(np.asarray([y_predRot[x][0], y_predRot[y][0]]),
              np.asarray([y_predRot[x][1], y_predRot[y][1]]),
              np.asarray([y_predRot[x][2], y_predRot[y][2]]), 'k')


def PointRotate3D(p1, p2, p0, theta):
    # Translate so axis is at origin
    p = p0 - p1
    # Initialize point q
    N = (p2 - p1)
    # Rotation axis unit vector
    n = normalize(N)

    # Matrix common factors
    c = np.cos(theta)
    t = 1 - np.cos(theta)
    s = np.sin(theta)
    X = n[0]
    Y = n[1]
    Z = n[2]

    # Matrix 'M'
    d11 = t * X * X + c
    d12 = t * X * Y - s * Z
    d13 = t * X * Z + s * Y
    d21 = t * X * Y + s * Z
    d22 = t * Y * Y + c
    d23 = t * Y * Z - s * X
    d31 = t * X * Z - s * Y
    d32 = t * Y * Z + s * X
    d33 = t * Z * Z + c

    #            |p.x|
    # Matrix 'M'*|p.y|
    #            |p.z|
    q = np.zeros(3)
    q[0] = d11 * p[0] + d12 * p[1] + d13 * p[2]
    q[1] = d21 * p[0] + d22 * p[1] + d23 * p[2]
    q[2] = d31 * p[0] + d32 * p[1] + d33 * p[2]

    # Translate axis and rotated point back to original location
    return q + p1


def unit_vector(vector):
    """ Returns the unit vector of the vector.  """
    return vector / np.linalg.norm(vector)


def angle_between(v1, v2):
    v1_u = unit_vector(v1)
    v2_u = unit_vector(v2)
    rad = np.arccos(np.clip(np.dot(v1_u, v2_u), -1.0, 1.0))
    return rad


def angle_2d(a, b, c):
    v1 = a - b
    v2 = c - b
    v1norm = normalize(v1)
    v2norm = normalize(v2)
    res = v1norm[0] * v2norm[0] + v1norm[1] * v2norm[1]
    angle_rad = np.arccos(res)
    if np.isnan(angle_rad):
        return 360.
    else:
        return np.degrees(angle_rad)


def getAngleFrom(x, y, z, y_predRot, indices, bias=tensor00):  # for shorthand
    return angle_2d(y_predRot[x][indices],
                    y_predRot[y][indices],
                    y_predRot[z][indices] + bias)


def rotator(jointsToRot, alignment, angle, adder):
    sizeOfArray = np.shape(jointsToRot)[0]
    values = np.zeros((sizeOfArray, 3))
    for j in range(0, sizeOfArray):
        values[j, :] = (PointRotate3D(jointsToRot[alignment],
                                      jointsToRot[alignment] + adder,
                                      jointsToRot[j],
                                      np.radians(angle)))
    return values


def aligner(joints, alignJoint1, alignJoint2, alignJoint3):
    # Remove the other two components (caveman checking and seeing if the rotation works ;_;)
    indices = [0, 1]
    angleFirst = getAngleFrom(alignJoint2, alignJoint1, alignJoint1, joints, indices, tensor10)
    # print(angleFirst)
    test = PointRotate3D(joints[alignJoint1],
                         joints[alignJoint1] + zmain,
                         joints[alignJoint2],
                         np.radians(angleFirst))

    testAngle = angle_2d(test[indices],
                         joints[alignJoint1][indices],
                         joints[alignJoint1][indices] + tensor10)

    if np.round(testAngle) == 0:
        joints2 = rotator(joints, alignJoint1, angleFirst, zmain)
    else:
        joints2 = rotator(joints, alignJoint1, 360 - angleFirst, zmain)

    indices = [0, 2]
    angleSecond = getAngleFrom(alignJoint2, alignJoint1, alignJoint1, joints2, indices, tensor10)
    # print(angleSecond)

    test = PointRotate3D(joints2[alignJoint1],
                         joints2[alignJoint1] + ymain,
                         joints2[alignJoint2],
                         np.radians(angleSecond))

    testAngle = angle_2d(test[indices],
                         joints2[alignJoint1][indices],
                         joints2[alignJoint1][indices] + tensor10)
    # print(testAngle)
    # temp3 = testAngle

    if np.round(testAngle) == 0:
        joints3 = rotator(joints2, alignJoint1, angleSecond, ymain)
    else:
        joints3 = rotator(joints2, alignJoint1, 360 - angleSecond, ymain)

    indices = [1, 2]
    # Align with another joint for reference
    finalRotation = getAngleFrom(alignJoint3, alignJoint1, alignJoint1, joints3, indices, tensor10)
    test = PointRotate3D(joints3[alignJoint1],
                         joints3[alignJoint1] + xmain,
                         joints3[alignJoint3],
                         np.radians(finalRotation))

    testAngle = angle_2d(test[1:3],
                         joints3[alignJoint1][1:3],
                         joints3[alignJoint1][1:3] + tensor10)

    if np.round(testAngle) == 0:
        jointsR = rotator(joints3, alignJoint1, finalRotation, xmain)
    else:
        jointsR = rotator(joints3, alignJoint1, 360 - finalRotation, xmain)
    return jointsR


def fingerShorthand(y, a, b, c, indices):
    y_rot = aligner(y, a, b, c)
    val = 180 - getAngleFrom(a, b, c, y_rot, indices)
    if y_rot[c][1] < y_rot[b][1]:
        return -1 * val
    else:
        return val


def boundcheck(y_predRot1, mainAngle, maxAngle, minAngle, joints, axis, alpha):
    if mainAngle > maxAngle:
        return np.abs(np.abs(mainAngle) - np.abs(maxAngle))
    elif mainAngle < minAngle:
        return np.abs(np.abs(mainAngle) - np.abs(minAngle))
    else:
        return 0.0


def fixer(y_predRot1, mainAngle, maxAngle, minAngle, joints, axis, alpha):
    if mainAngle > maxAngle:
        # print("Doing max")
        angle = alpha * (mainAngle - maxAngle)
        # print("angle fix is " + str(angle))
        y_predRot1[joints] = rotator(y_predRot1[joints], 0, angle, axis)
    elif mainAngle < minAngle:
        # print("Doing min")
        angle = alpha * (mainAngle - minAngle)
        y_predRot1[joints] = rotator(y_predRot1[joints], 0, 360 + angle, axis)
    return y_predRot1[joints]


def fixerV(y_predRot1, mainAngle, refAngle, maxVel, joints, axis, alpha):
    vel = (mainAngle - refAngle) / (1/15.0)
    # print(vel)
    if vel > maxVel:
        diff = maxVel - vel
        angle = alpha * (diff * (1/15.0))
        y_predRot1[joints] = rotator(y_predRot1[joints], 0, angle, axis)
    elif vel < -maxVel:
        diff = maxVel - vel
        angle = alpha * (diff * (1/15.0))
        y_predRot1[joints] = rotator(y_predRot1[joints], 0, 360 - angle, axis)
    return y_predRot1[joints]


def coplanar(y_values, x, y, z, alpha):
    y_predRot = aligner(y_values, x, y, z)
    indices = [0, 2]
    val = 90 - angle_2d(y_predRot[y][indices] + tensor01,
                        y_predRot[y][indices],
                        y_predRot[y + 1][indices])
    joints = [y, y + 1]
    y_predRot[joints] = fixer(y_predRot, val, val / 2, val / 2, joints, ymain, alpha)

    joints = [y + 1, y]
    y_predRot[joints] = fixer(y_predRot, val, 0, 0, joints, ymain, alpha)
    return y_predRot


def correction(y_values, alpha):
    y_predRot = coplanar(y_values, 1, 6, 8, alpha)
    y_predRot = coplanar(y_predRot, 2, 9, 11, alpha)
    y_predRot = coplanar(y_predRot, 3, 12, 14, alpha)
    y_predRot = coplanar(y_predRot, 4, 15, 17, alpha)
    y_predRot = coplanar(y_predRot, 5, 18, 20, alpha)

    # Thumb
    y_predRot = aligner(y_predRot, 1, 2, 3)
    indices = [0, 1]
    val = getAngleFrom(3, 1, 6, y_predRot, indices) - getAngleFrom(3, 1, 2, y_predRot, indices)
    thumb = [1, 6, 7, 8]
    y_predRot[thumb] = fixer(y_predRot, val, 45, -20, thumb, zmain, alpha)

    indices = [0, 2]
    val = getAngleFrom(2, 1, 6, y_predRot, indices)
    if y_predRot[6][2] < y_predRot[1][2]:
        val = 360. - val
    y_predRot[thumb] = fixer(y_predRot, val, 45, 0, thumb, ymain, alpha)

    y_predRot = aligner(y_predRot, 1, 6, 2)
    val = 180 - getAngleFrom(1, 6, 7, y_predRot, indices)
    if y_predRot[7][2] < y_predRot[6][2]:
        val = val * -1
    thumb = [6, 7, 8]
    y_predRot[thumb] = fixer(y_predRot, val, 80, 0, thumb, ymain, alpha)

    indices = [0, 1]
    val = 90 - getAngleFrom(7, 6, 6, y_predRot, indices, tensor01)
    y_predRot[thumb] = fixer(y_predRot, val, 7, -12, thumb, zmain, alpha)

    indices = [1, 2]
    val = getAngleFrom(8, 7, 7, y_predRot, indices, tensor01)
    thumb = [7, 8]
    y_predRot[thumb] = fixer(y_predRot, val, 0, 0, thumb, -xmain, alpha)
    val = getAngleFrom(8, 7, 7, y_predRot, indices, tensor01)


    indices = [0, 2]
    val = 180 - getAngleFrom(6, 7, 8, y_predRot, indices)
    if y_predRot[8][2] < y_predRot[7][2]:
        val = -1 * val
    thumb = [7, 8]
    y_predRot[thumb] = fixer(y_predRot, val, 90, -30, thumb, ymain, alpha)

    # Finger 2 - Index
    y_predRot = aligner(y_predRot, 0, 2, 9)
    indices = [0, 1]
    val = 180 - getAngleFrom(0, 2, 9, y_predRot, indices)
    if y_predRot[9][1] < y_predRot[2][1]:
        val = -1 * val
    index = [2, 9, 10, 11]
    y_predRot[index] = fixer(y_predRot, val, 90, -40, index, -zmain, alpha)

    y_predRot = aligner(y_predRot, 3, 2, 9)
    val = 90 - getAngleFrom(3, 2, 9, y_predRot, indices)
    y_predRot[index] = fixer(y_predRot, val, 15, -15, index, -zmain, alpha)

    indices = [0, 2]
    y_predRot = aligner(y_predRot, 2, 9, 10)
    val = 180 - getAngleFrom(2, 9, 10, y_predRot, indices)
    if y_predRot[10][1] < y_predRot[9][1]:
        val = -1 * val
    index = [9, 10, 11]
    y_predRot[index] = fixer(y_predRot, val, 130, 0, index, ymain, alpha)

    indices = [0, 1]
    y_predRot = aligner(y_predRot, 9, 10, 11)
    val = 180 - getAngleFrom(9, 10, 11, y_predRot, indices)
    if y_predRot[11][1] > y_predRot[10][1]:
        val = -1 * val
    index = [10, 11]
    y_predRot[index] = fixer(y_predRot, val, 90, -30, index, -zmain, alpha)

    # Finger 3 - Middle
    y_predRot = aligner(y_predRot, 0, 3, 12)
    val = 180 - getAngleFrom(0, 3, 12, y_predRot, indices)
    if y_predRot[12][1] < y_predRot[3][1]:
        val = -1 * val
    middle = [3, 12, 13, 14]
    y_predRot[middle] = fixer(y_predRot, val, 90, -40, middle, -zmain, alpha)

    y_predRot = aligner(y_predRot, 4, 3, 12)
    val = 90 - getAngleFrom(4, 3, 12, y_predRot, indices)
    y_predRot[middle] = fixer(y_predRot, val, 15, -15, middle, -zmain, alpha)

    indices = [0, 2]
    y_predRot = aligner(y_predRot, 3, 12, 13)
    val = 180 - getAngleFrom(3, 12, 13, y_predRot, indices)
    if y_predRot[13][1] < y_predRot[12][1]:
        val = -1 * val
    middle = [12, 13, 14]
    y_predRot[middle] = fixer(y_predRot, val, 130, 0, middle, ymain, alpha)

    indices = [0, 1]
    y_predRot = aligner(y_predRot, 12, 13, 14)
    val = 180 - getAngleFrom(12, 13, 14, y_predRot, indices)
    if y_predRot[14][1] < y_predRot[13][1]:
        val = -1 * val
    middle = [13, 14]
    y_predRot[middle] = fixer(y_predRot, val, 90, -30, middle, ymain, alpha)

    # Finger 4 - Ring
    y_predRot = aligner(y_predRot, 0, 4, 15)
    val = 180 - getAngleFrom(0, 4, 15, y_predRot, indices)
    if y_predRot[15][1] < y_predRot[4][1]:
        val = -1 * val
    ring = [4, 15, 16, 17]
    y_predRot[ring] = fixer(y_predRot, val, 90, -40, ring, -zmain, alpha)

    y_predRot = aligner(y_predRot, 5, 4, 15)
    val = 90 - getAngleFrom(5, 4, 15, y_predRot, indices)
    y_predRot[ring] = fixer(y_predRot, val, 15, -15, ring, -zmain, alpha)

    indices = [0, 2]
    y_predRot = aligner(y_predRot, 4, 15, 16)
    val = 180 - getAngleFrom(4, 15, 16, y_predRot, indices)
    if y_predRot[16][1] < y_predRot[15][1]:
        val = -1 * val
    ring = [15, 16, 17]
    y_predRot[ring] = fixer(y_predRot, val, 130, 0, ring, ymain, alpha)

    indices = [0, 1]
    y_predRot = aligner(y_predRot, 15, 16, 17)
    val = 180 - getAngleFrom(15, 16, 17, y_predRot, indices)
    if y_predRot[17][1] > y_predRot[16][1]:  # greater
        val = -1 * val
    ring = [16, 17]
    y_predRot[ring] = fixer(y_predRot, val, 90, -30, ring, zmain, alpha)

    # Finger 5 Pinky
    y_predRot = aligner(y_predRot, 0, 5, 18)
    val = 180 - getAngleFrom(0, 5, 18, y_predRot, indices)
    if y_predRot[18][1] < y_predRot[5][1]:
        val = -1 * val
    pinky = [5, 18, 19, 20]
    y_predRot[pinky] = fixer(y_predRot, val, 90, -40, pinky, -zmain, alpha)

    y_predRot = aligner(y_predRot, 4, 5, 18)
    val = getAngleFrom(4, 5, 18, y_predRot, indices) - 90
    y_predRot[pinky] = fixer(y_predRot, val, 15, -15, pinky, zmain, alpha)

    indices = [0, 1]
    y_predRot = aligner(y_predRot, 5, 18, 19)
    val = 180 - getAngleFrom(5, 18, 19, y_predRot, indices)
    if y_predRot[19][1] < y_predRot[18][1]:
        val = -1 * val
    pinky = [18, 19, 20]
    y_predRot[pinky] = fixer(y_predRot, val, 130, 0, pinky, ymain, alpha)

    indices = [0, 1]
    y_predRot = aligner(y_predRot, 18, 19, 20)
    val = 180 - getAngleFrom(18, 19, 20, y_predRot, indices)
    if y_predRot[20][1] > y_predRot[19][1]:
        val = -1 * val
    pinky = [19, 20]
    y_predRot[pinky] = fixer(y_predRot, val, 90, -30, pinky, zmain, alpha)

    # Match former alignment
    translate = y_values[0] - y_predRot[0]
    y_predRot = y_predRot + translate

    v1 = y_predRot[3] - y_predRot[0]
    v2 = y_values[3] - y_predRot[0]
    rad = angle_between(v1, v2)
    d = np.degrees(rad)
    axisPoint = np.cross(v1, v2)

    test = PointRotate3D(y_predRot[0],
                         y_predRot[0] + unit_vector(axisPoint),
                         y_predRot[3],
                         np.radians(d))
    values = np.zeros((21, 3))

    if np.linalg.norm(np.round(y_values[3] - test)) == 0:
        for j in range(0, 21):
            values[j, :] = PointRotate3D(y_predRot[0],
                                         y_predRot[0] + unit_vector(axisPoint),
                                         y_predRot[j],
                                         np.radians(d))
    else:
        for j in range(0, 21):
            deg = 360 - d
            values[j, :] = PointRotate3D(y_predRot[0],
                                         y_predRot[0] + unit_vector(axisPoint),
                                         y_predRot[j],
                                         np.radians(deg))

    y_predRot = values

    # Final Rotation
    def getProjpoint(point, orig, normal):
        v = point - orig
        dist = np.dot(v, normal)
        projected_point = point - dist * normal
        return projected_point

    norm = unit_vector(y_predRot[3] - y_predRot[0])
    a = getProjpoint(y_predRot[4], y_predRot[0], norm)
    b = getProjpoint(y_values[4], y_predRot[0], norm)
    v1 = a - y_predRot[0]
    v2 = b - y_predRot[0]
    rad1 = angle_between(v1, v2)
    d = np.degrees(rad1)

    test = PointRotate3D(y_predRot[0],
                         y_predRot[3],
                         y_predRot[4],
                         np.radians(d))

    values = np.zeros((21, 3))
    c1 = np.linalg.norm(np.round(y_values[4] - test))

    if c1 == 0:
        for j in range(0, 21):
            values[j, :] = PointRotate3D(y_predRot[0],
                                         y_predRot[3],
                                         y_predRot[j],
                                         np.radians(d))
    else:
        for j in range(0, 21):
            deg1 = 360 - d
            values[j, :] = PointRotate3D(y_predRot[0],
                                         y_predRot[3],
                                         y_predRot[j],
                                         np.radians(deg1))
    y_predRot = values

    return y_predRot


def correctionV(y_values, alpha, previousPose):
    # y_predRot = coplanar(y_values, 1, 6, 8, alpha)
    # y_predRot = coplanar(y_predRot, 2, 9, 11, alpha)
    # y_predRot = coplanar(y_predRot, 3, 12, 14, alpha)
    # y_predRot = coplanar(y_predRot, 4, 15, 17, alpha)
    # y_predRot = coplanar(y_predRot, 5, 18, 20, alpha)
    refAngleVel = 350
    # Thumb
    y_predRot = aligner(y_values, 1, 2, 3)

    indices = [0, 1]
    val = getAngleFrom(3, 1, 6, y_predRot, indices) - getAngleFrom(3, 1, 2, y_predRot, indices)
    thumb = [1, 6, 7, 8]
    y_predRot[thumb] = fixerV(y_predRot, val, previousPose[0], refAngleVel,  thumb, zmain, alpha)

    indices = [0, 2]
    val = getAngleFrom(2, 1, 6, y_predRot, indices)
    if y_predRot[6][2] < y_predRot[1][2]:
        val = 360. - val

    y_predRot[thumb] = fixerV(y_predRot, val, previousPose[1], refAngleVel, thumb, ymain, alpha)

    y_predRot = aligner(y_predRot, 1, 6, 2)
    val = 180 - getAngleFrom(1, 6, 7, y_predRot, indices)
    if y_predRot[7][2] < y_predRot[6][2]:
        val = val * -1
    thumb = [6, 7, 8]
    y_predRot[thumb] = fixerV(y_predRot, val, previousPose[2], refAngleVel, thumb, ymain, alpha)

    indices = [0, 1]
    val = 90 - getAngleFrom(7, 6, 6, y_predRot, indices, tensor01)
    y_predRot[thumb] = fixerV(y_predRot, val, previousPose[3], refAngleVel, thumb, zmain, alpha)

    indices = [1, 2]
    val = getAngleFrom(8, 7, 7, y_predRot, indices, tensor01)
    thumb = [7, 8]
    y_predRot[thumb] = fixerV(y_predRot, val, previousPose[4], refAngleVel, thumb, -xmain, alpha)

    indices = [0, 2]
    val = 180 - getAngleFrom(6, 7, 8, y_predRot, indices)
    if y_predRot[8][2] < y_predRot[7][2]:
        val = -1 * val
    thumb = [7, 8]
    y_predRot[thumb] = fixerV(y_predRot, val, previousPose[5], refAngleVel, thumb, ymain, alpha)

    # Finger 2 - Index
    y_predRot = aligner(y_predRot, 0, 2, 9)
    indices = [0, 1]
    val = 180 - getAngleFrom(0, 2, 9, y_predRot, indices)
    if y_predRot[9][1] < y_predRot[2][1]:
        val = -1 * val
    index = [2, 9, 10, 11]
    y_predRot[index] = fixerV(y_predRot, val, previousPose[6], refAngleVel, index, -zmain, alpha)

    y_predRot = aligner(y_predRot, 3, 2, 9)
    val = 90 - getAngleFrom(3, 2, 9, y_predRot, indices)
    y_predRot[index] = fixerV(y_predRot, val, previousPose[7], refAngleVel, index, -zmain, alpha)

    indices = [0, 2]
    y_predRot = aligner(y_predRot, 2, 9, 10)
    val = 180 - getAngleFrom(2, 9, 10, y_predRot, indices)
    if y_predRot[10][1] < y_predRot[9][1]:
        val = -1 * val
    index = [9, 10, 11]
    y_predRot[index] = fixerV(y_predRot, val, previousPose[8], refAngleVel, index, ymain, alpha)

    indices = [0, 1]
    y_predRot = aligner(y_predRot, 9, 10, 11)
    val = 180 - getAngleFrom(9, 10, 11, y_predRot, indices)
    if y_predRot[11][1] > y_predRot[10][1]:
        val = -1 * val
    index = [10, 11]
    y_predRot[index] = fixerV(y_predRot, val, previousPose[9], refAngleVel, index, -zmain, alpha)

    # Finger 3 - Middle
    y_predRot = aligner(y_predRot, 0, 3, 12)
    val = 180 - getAngleFrom(0, 3, 12, y_predRot, indices)
    if y_predRot[12][1] < y_predRot[3][1]:
        val = -1 * val
    middle = [3, 12, 13, 14]
    y_predRot[middle] = fixerV(y_predRot, val, previousPose[10], refAngleVel, middle, -zmain, alpha)

    y_predRot = aligner(y_predRot, 4, 3, 12)
    val = 90 - getAngleFrom(4, 3, 12, y_predRot, indices)
    y_predRot[middle] = fixerV(y_predRot, val, previousPose[11], refAngleVel, middle, -zmain, alpha)

    indices = [0, 2]
    y_predRot = aligner(y_predRot, 3, 12, 13)
    val = 180 - getAngleFrom(3, 12, 13, y_predRot, indices)
    if y_predRot[13][1] < y_predRot[12][1]:
        val = -1 * val
    middle = [12, 13, 14]
    y_predRot[middle] = fixerV(y_predRot, val, previousPose[12], refAngleVel, middle, ymain, alpha)

    indices = [0, 1]
    y_predRot = aligner(y_predRot, 12, 13, 14)
    val = 180 - getAngleFrom(12, 13, 14, y_predRot, indices)
    if y_predRot[14][1] > y_predRot[13][1]:
        val = -1 * val
    middle = [13, 14]
    y_predRot[middle] = fixerV(y_predRot, val, previousPose[13], refAngleVel, middle, ymain, alpha)

    # Finger 4 - Ring
    y_predRot = aligner(y_predRot, 0, 4, 15)
    val = 180 - getAngleFrom(0, 4, 15, y_predRot, indices)
    if y_predRot[15][1] < y_predRot[4][1]:
        val = -1 * val
    ring = [4, 15, 16, 17]
    y_predRot[ring] = fixerV(y_predRot, val, previousPose[14], refAngleVel, ring, -zmain, alpha)

    y_predRot = aligner(y_predRot, 5, 4, 15)
    val = 90 - getAngleFrom(5, 4, 15, y_predRot, indices)
    y_predRot[ring] = fixerV(y_predRot, val, previousPose[15], refAngleVel, ring, -zmain, alpha)

    indices = [0, 2]
    y_predRot = aligner(y_predRot, 4, 15, 16)
    val = 180 - getAngleFrom(4, 15, 16, y_predRot, indices)
    if y_predRot[16][1] < y_predRot[15][1]:
        val = -1 * val
    ring = [15, 16, 17]
    y_predRot[ring] = fixerV(y_predRot, val, previousPose[16], refAngleVel, ring, ymain, alpha)

    indices = [0, 1]
    y_predRot = aligner(y_predRot, 15, 16, 17)
    val = 180 - getAngleFrom(15, 16, 17, y_predRot, indices)
    if y_predRot[17][1] > y_predRot[16][1]:  # greater
        val = -1 * val
    ring = [16, 17]
    y_predRot[ring] = fixerV(y_predRot, val, previousPose[17], refAngleVel, ring, zmain, alpha)

    # Finger 5 Pinky
    y_predRot = aligner(y_predRot, 0, 5, 18)
    val = 180 - getAngleFrom(0, 5, 18, y_predRot, indices)
    if y_predRot[18][1] < y_predRot[5][1]:
        val = -1 * val
    pinky = [5, 18, 19, 20]
    y_predRot[pinky] = fixerV(y_predRot, val, previousPose[18], refAngleVel, pinky, -zmain, alpha)

    y_predRot = aligner(y_predRot, 4, 5, 18)
    val = getAngleFrom(4, 5, 18, y_predRot, indices) - 90
    y_predRot[pinky] = fixerV(y_predRot, val, previousPose[19], refAngleVel, pinky, zmain, alpha)

    indices = [0, 1]
    y_predRot = aligner(y_predRot, 5, 18, 19)
    val = 180 - getAngleFrom(5, 18, 19, y_predRot, indices)
    if y_predRot[19][1] < y_predRot[18][1]:
        val = -1 * val
    pinky = [18, 19, 20]
    y_predRot[pinky] = fixerV(y_predRot, val, previousPose[20], refAngleVel, pinky, ymain, alpha)

    indices = [0, 1]
    y_predRot = aligner(y_predRot, 18, 19, 20)
    val = 180 - getAngleFrom(18, 19, 20, y_predRot, indices)
    if y_predRot[20][1] > y_predRot[19][1]:
        val = -1 * val
    pinky = [19, 20]
    y_predRot[pinky] = fixerV(y_predRot, val, previousPose[21], refAngleVel, pinky, zmain, alpha)

    # Match former alignment
    translate = y_values[0] - y_predRot[0]
    y_predRot = y_predRot + translate

    v1 = y_predRot[3] - y_predRot[0]
    v2 = y_values[3] - y_predRot[0]
    rad = angle_between(v1, v2)
    d = np.degrees(rad)
    axisPoint = np.cross(v1, v2)

    test = PointRotate3D(y_predRot[0],
                         y_predRot[0] + unit_vector(axisPoint),
                         y_predRot[3],
                         np.radians(d))
    values = np.zeros((21, 3))

    if np.linalg.norm(np.round(y_values[3] - test)) == 0:
        for j in range(0, 21):
            values[j, :] = PointRotate3D(y_predRot[0],
                                         y_predRot[0] + unit_vector(axisPoint),
                                         y_predRot[j],
                                         np.radians(d))
    else:
        for j in range(0, 21):
            deg = 360 - d
            values[j, :] = PointRotate3D(y_predRot[0],
                                         y_predRot[0] + unit_vector(axisPoint),
                                         y_predRot[j],
                                         np.radians(deg))

    y_predRot = values

    # Final Rotation
    def getProjpoint(point, orig, normal):
        v = point - orig
        dist = np.dot(v, normal)
        projected_point = point - dist * normal
        return projected_point

    norm = unit_vector(y_predRot[3] - y_predRot[0])
    a = getProjpoint(y_predRot[4], y_predRot[0], norm)
    b = getProjpoint(y_values[4], y_predRot[0], norm)
    v1 = a - y_predRot[0]
    v2 = b - y_predRot[0]
    rad1 = angle_between(v1, v2)
    d = np.degrees(rad1)

    test = PointRotate3D(y_predRot[0],
                         y_predRot[3],
                         y_predRot[4],
                         np.radians(d))

    values = np.zeros((21, 3))
    c1 = np.linalg.norm(np.round(y_values[4] - test))

    if c1 == 0:
        for j in range(0, 21):
            values[j, :] = PointRotate3D(y_predRot[0],
                                         y_predRot[3],
                                         y_predRot[j],
                                         np.radians(d))
    else:
        for j in range(0, 21):
            deg1 = 360 - d
            values[j, :] = PointRotate3D(y_predRot[0],
                                         y_predRot[3],
                                         y_predRot[j],
                                         np.radians(deg1))
    y_predRot = values

    return y_predRot


def DoF(y_values):
    Dofs = np.zeros(22)

    y_predRot = aligner(y_values, 1, 2, 3)
    indices = [0, 1]
    Dofs[0] = getAngleFrom(3, 1, 6, y_predRot, indices) - getAngleFrom(3, 1, 2, y_predRot, indices)

    indices = [0, 2]
    val = getAngleFrom(2, 1, 6, y_predRot, indices)
    if y_predRot[6][2] < y_predRot[1][2]:
        val = 360. - val
    Dofs[1] = val

    y_predRot = aligner(y_predRot, 1, 6, 2)
    val = 180 - getAngleFrom(1, 6, 7, y_predRot, indices)
    if y_predRot[7][2] < y_predRot[6][2]:
        val = val * -1
    Dofs[2] = val

    indices = [0, 1]
    val = 90 - getAngleFrom(7, 6, 6, y_predRot, indices, tensor01)
    Dofs[3] = val

    indices = [1, 2]
    val = getAngleFrom(8, 7, 7, y_predRot, indices, tensor01)
    Dofs[4] = val

    indices = [0, 2]
    val = 180 - getAngleFrom(6, 7, 8, y_predRot, indices)
    if y_predRot[8][2] < y_predRot[7][2]:
        val = -1 * val
    Dofs[5] = val

    # Finger 2 - Index
    y_predRot = aligner(y_predRot, 0, 2, 9)
    indices = [0, 1]
    val = 180 - getAngleFrom(0, 2, 9, y_predRot, indices)
    if y_predRot[9][1] < y_predRot[2][1]:
        val = -1 * val
    Dofs[6] = val

    y_predRot = aligner(y_predRot, 3, 2, 9)
    val = 90 - getAngleFrom(3, 2, 9, y_predRot, indices)
    Dofs[7] = val

    indices = [0, 2]
    y_predRot = aligner(y_predRot, 2, 9, 10)
    val = 180 - getAngleFrom(2, 9, 10, y_predRot, indices)
    if y_predRot[10][1] < y_predRot[9][1]:
        val = -1 * val
    Dofs[8] = val

    indices = [0, 1]
    y_predRot = aligner(y_predRot, 9, 10, 11)
    val = 180 - getAngleFrom(9, 10, 11, y_predRot, indices)
    if y_predRot[11][1] > y_predRot[10][1]:
        val = -1 * val
    Dofs[9] = val

    # Finger 3 - Middle
    y_predRot = aligner(y_predRot, 0, 3, 12)
    val = 180 - getAngleFrom(0, 3, 12, y_predRot, indices)
    if y_predRot[12][1] < y_predRot[3][1]:
        val = -1 * val
    Dofs[10] = val

    y_predRot = aligner(y_predRot, 4, 3, 12)
    val = 90 - getAngleFrom(4, 3, 12, y_predRot, indices)
    Dofs[11] = val

    indices = [0, 2]
    y_predRot = aligner(y_predRot, 3, 12, 13)
    val = 180 - getAngleFrom(3, 12, 13, y_predRot, indices)
    if y_predRot[13][1] < y_predRot[12][1]:
        val = -1 * val
    Dofs[12] = val

    indices = [0, 1]
    y_predRot = aligner(y_predRot, 12, 13, 14)
    val = 180 - getAngleFrom(12, 13, 14, y_predRot, indices)
    if y_predRot[14][1] > y_predRot[13][1]:
        val = -1 * val
    Dofs[13] = val

    # Finger 4 - Ring
    y_predRot = aligner(y_predRot, 0, 4, 15)
    val = 180 - getAngleFrom(0, 4, 15, y_predRot, indices)
    if y_predRot[15][1] < y_predRot[4][1]:
        val = -1 * val
    Dofs[14] = val

    y_predRot = aligner(y_predRot, 5, 4, 15)
    val = 90 - getAngleFrom(5, 4, 15, y_predRot, indices)
    Dofs[15] = val

    indices = [0, 2]
    y_predRot = aligner(y_predRot, 4, 15, 16)
    val = 180 - getAngleFrom(4, 15, 16, y_predRot, indices)
    if y_predRot[16][1] < y_predRot[15][1]:
        val = -1 * val
    Dofs[16] = val

    indices = [0, 1]
    y_predRot = aligner(y_predRot, 15, 16, 17)
    val = 180 - getAngleFrom(15, 16, 17, y_predRot, indices)
    if y_predRot[17][1] > y_predRot[16][1]:  # greater
        val = -1 * val
    Dofs[17] = val

    # Finger 5 Pinky
    y_predRot = aligner(y_predRot, 0, 5, 18)
    val = 180 - getAngleFrom(0, 5, 18, y_predRot, indices)
    if y_predRot[18][1] < y_predRot[5][1]:
        val = -1 * val
    Dofs[18] = val

    y_predRot = aligner(y_predRot, 4, 5, 18)
    val = getAngleFrom(4, 5, 18, y_predRot, indices) - 90
    Dofs[19] = val

    indices = [0, 1]
    y_predRot = aligner(y_predRot, 5, 18, 19)
    val = 180 - getAngleFrom(5, 18, 19, y_predRot, indices)
    if y_predRot[19][1] < y_predRot[18][1]:
        val = -1 * val
    Dofs[20] = val

    indices = [0, 1]
    y_predRot = aligner(y_predRot, 18, 19, 20)
    val = 180 - getAngleFrom(18, 19, 20, y_predRot, indices)
    if y_predRot[20][1] > y_predRot[19][1]:
        val = -1 * val
    Dofs[21] = val
    return Dofs


def minmaxCheck(y_values, alpha):
    realError = 0.0
    print("Start")
    debug = True
    # Thumb
    y_predRot = aligner(y_values, 1, 2, 3)
    indices = [0, 1]
    val = getAngleFrom(3, 1, 6, y_predRot, indices) - getAngleFrom(3, 1, 2, y_predRot, indices)
    thumb = [1, 6, 7, 8]
    realError += boundcheck(y_predRot, val, 45, -20, thumb, zmain, alpha)
    if debug:
        print(realError)

    indices = [0, 2]
    val = getAngleFrom(2, 1, 6, y_predRot, indices)
    if y_predRot[6][2] < y_predRot[1][2]:
        val = 360. - val
    realError += boundcheck(y_predRot, val, 45, 0, thumb, ymain, alpha)
    if debug:
        print(realError)

    y_predRot = aligner(y_predRot, 1, 6, 2)
    val = 180 - getAngleFrom(1, 6, 7, y_predRot, indices)
    if y_predRot[7][2] < y_predRot[6][2]:
        val = val * -1
    thumb = [6, 7, 8]
    realError += boundcheck(y_predRot, val, 80, 0, thumb, ymain, alpha)
    if debug:
        print(realError)

    indices = [0, 1]
    val = 90 - getAngleFrom(7, 6, 6, y_predRot, indices, tensor01)
    realError += boundcheck(y_predRot, val, 7, -12, thumb, zmain, alpha)
    if debug:
        print(realError)

    indices = [1, 2]
    val = getAngleFrom(8, 7, 7, y_predRot, indices, tensor01)
    thumb = [7, 8]
    realError += boundcheck(y_predRot, val, 0, 0, thumb, -xmain, alpha)
    if debug:
        print(realError)

    indices = [0, 2]
    val = 180 - getAngleFrom(6, 7, 8, y_predRot, indices)
    if y_predRot[8][2] < y_predRot[7][2]:
        val = -1 * val
    thumb = [7, 8]
    realError += boundcheck(y_predRot, val, 90, -30, thumb, ymain, alpha)
    if debug:
        print(realError)

    # Finger 2 - Index
    y_predRot = aligner(y_predRot, 0, 2, 9)
    indices = [0, 1]
    val = 180 - getAngleFrom(0, 2, 9, y_predRot, indices)
    if y_predRot[9][1] < y_predRot[2][1]:
        val = -1 * val
    index = [2, 9, 10, 11]
    realError += boundcheck(y_predRot, val, 90, -40, index, -zmain, alpha)
    if debug:
        print(realError)

    y_predRot = aligner(y_predRot, 3, 2, 9)
    val = 90 - getAngleFrom(3, 2, 9, y_predRot, indices)
    realError += boundcheck(y_predRot, val, 15, -15, index, -zmain, alpha)
    if debug:
        print(realError)

    indices = [0, 2]
    y_predRot = aligner(y_predRot, 2, 9, 10)
    val = 180 - getAngleFrom(2, 9, 10, y_predRot, indices)
    if y_predRot[10][1] < y_predRot[9][1]:
        val = -1 * val
    index = [9, 10, 11]
    realError += boundcheck(y_predRot, val, 130, 0, index, ymain, alpha)
    if debug:
        print(realError)

    indices = [0, 1]
    y_predRot = aligner(y_predRot, 9, 10, 11)
    val = 180 - getAngleFrom(9, 10, 11, y_predRot, indices)
    if y_predRot[11][1] > y_predRot[10][1]:
        val = -1 * val
    index = [10, 11]
    realError += boundcheck(y_predRot, val, 90, -30, index, -zmain, alpha)
    if debug:
        print(realError)

    # Finger 3 - Middle
    y_predRot = aligner(y_predRot, 0, 3, 12)
    val = 180 - getAngleFrom(0, 3, 12, y_predRot, indices)
    if y_predRot[12][1] < y_predRot[3][1]:
        val = -1 * val
    middle = [3, 12, 13, 14]
    realError += boundcheck(y_predRot, val, 90, -40, middle, -zmain, alpha)
    if debug:
        print(realError)

    y_predRot = aligner(y_predRot, 4, 3, 12)
    val = 90 - getAngleFrom(4, 3, 12, y_predRot, indices)
    realError += boundcheck(y_predRot, val, 15, -15, middle, -zmain, alpha)
    if debug:
        print(realError)

    indices = [0, 2]
    y_predRot = aligner(y_predRot, 3, 12, 13)
    val = 180 - getAngleFrom(3, 12, 13, y_predRot, indices)
    if y_predRot[13][1] < y_predRot[12][1]:
        val = -1 * val
    middle = [12, 13, 14]
    realError += boundcheck(y_predRot, val, 130, 0, middle, ymain, alpha)
    if debug:
        print(realError)

    indices = [0, 1]
    y_predRot = aligner(y_predRot, 12, 13, 14)
    val = 180 - getAngleFrom(12, 13, 14, y_predRot, indices)
    if y_predRot[14][1] < y_predRot[13][1]:
        val = -1 * val
    middle = [13, 14]
    realError += boundcheck(y_predRot, val, 90, -30, middle, ymain, alpha)
    if debug:
        print(realError)

    # Finger 4 - Ring
    y_predRot = aligner(y_predRot, 0, 4, 15)
    val = 180 - getAngleFrom(0, 4, 15, y_predRot, indices)
    if y_predRot[15][1] < y_predRot[4][1]:
        val = -1 * val
    ring = [4, 15, 16, 17]
    realError += boundcheck(y_predRot, val, 90, -40, ring, -zmain, alpha)
    if debug:
        print(realError)

    y_predRot = aligner(y_predRot, 5, 4, 15)
    val = 90 - getAngleFrom(5, 4, 15, y_predRot, indices)
    realError += boundcheck(y_predRot, val, 15, -15, ring, -zmain, alpha)
    if debug:
        print(realError)

    indices = [0, 2]
    y_predRot = aligner(y_predRot, 4, 15, 16)
    val = 180 - getAngleFrom(4, 15, 16, y_predRot, indices)
    if y_predRot[16][1] < y_predRot[15][1]:
        val = -1 * val
    print("Ring 2 is " + str(val))
    ring = [15, 16, 17]
    realError += boundcheck(y_predRot, val, 130, 0, ring, ymain, alpha)
    if debug:
        print(realError)

    indices = [0, 1]
    y_predRot = aligner(y_predRot, 15, 16, 17)
    val = 180 - getAngleFrom(15, 16, 17, y_predRot, indices)
    if y_predRot[17][1] > y_predRot[16][1]:  # greater
        val = -1 * val
    ring = [16, 17]
    realError += boundcheck(y_predRot, val, 90, -30, ring, zmain, alpha)
    if debug:
        print(realError)

    # Finger 5 Pinky
    y_predRot = aligner(y_predRot, 0, 5, 18)
    val = 180 - getAngleFrom(0, 5, 18, y_predRot, indices)
    if y_predRot[18][1] < y_predRot[5][1]:
        val = -1 * val
    pinky = [5, 18, 19, 20]
    realError += boundcheck(y_predRot, val, 90, -40, pinky, -zmain, alpha)
    if debug:
        print(realError)

    y_predRot = aligner(y_predRot, 4, 5, 18)
    val = getAngleFrom(4, 5, 18, y_predRot, indices) - 90
    realError += boundcheck(y_predRot, val, 15, -15, pinky, zmain, alpha)
    if debug:
        print(realError)

    indices = [0, 1]
    y_predRot = aligner(y_predRot, 5, 18, 19)
    val = 180 - getAngleFrom(5, 18, 19, y_predRot, indices)
    if y_predRot[19][1] < y_predRot[18][1]:
        val = -1 * val
    pinky = [18, 19, 20]
    realError += boundcheck(y_predRot, val, 130, 0, pinky, ymain, alpha)
    if debug:
        print(realError)

    indices = [0, 1]
    y_predRot = aligner(y_predRot, 18, 19, 20)
    val = 180 - getAngleFrom(18, 19, 20, y_predRot, indices)
    if y_predRot[20][1] > y_predRot[19][1]:
        val = -1 * val
    pinky = [19, 20]
    realError += boundcheck(y_predRot, val, 90, -30, pinky, zmain, alpha)
    if debug:
        print(realError)
    return realError


def correctionDebug(y_values, alpha):
    y_predRot = coplanar(y_values, 1, 6, 8, alpha)
    y_predRot = coplanar(y_predRot, 2, 9, 11, alpha)
    y_predRot = coplanar(y_predRot, 3, 12, 14, alpha)
    y_predRot = coplanar(y_predRot, 4, 15, 17, alpha)
    y_predRot = coplanar(y_predRot, 5, 18, 20, alpha)

    # Thumb
    y_predRot = aligner(y_predRot, 1, 2, 3)
    indices = [0, 1]
    val1 = getAngleFrom(3, 1, 6, y_predRot, indices) - getAngleFrom(3, 1, 2, y_predRot, indices)
    thumb = [1, 6, 7, 8]
    temp = y_predRot.copy()
    y_predRot[thumb] = fixer(y_predRot, val1, 45, -20, thumb, zmain, alpha)
    val = getAngleFrom(3, 1, 6, y_predRot, indices) - getAngleFrom(3, 1, 2, y_predRot, indices)
    checker = np.round(boundcheck(y_predRot, val, 45, -20, thumb, zmain, alpha))
    if checker != 0:
        y_predRot = temp
        y_predRot[thumb] = fixer(y_predRot, val1, 45, -20, thumb, zmain, -alpha)

    indices = [0, 2]
    val1 = getAngleFrom(2, 1, 6, y_predRot, indices)
    if y_predRot[6][2] < y_predRot[1][2]:
        val1 = 360. - val1
    temp = y_predRot.copy()
    y_predRot[thumb] = fixer(y_predRot, val1, 45, 0, thumb, ymain, alpha)
    val = getAngleFrom(2, 1, 6, y_predRot, indices)
    if y_predRot[6][2] < y_predRot[1][2]:
        val = 360. - val
    checker = np.round(boundcheck(y_predRot, val, 45, 0, thumb, ymain, alpha))
    if checker != 0:
        y_predRot = temp
        y_predRot[thumb] = fixer(y_predRot, 2 * val, 45, 0, thumb, ymain, -alpha)


    y_predRot = aligner(y_predRot, 1, 6, 2)
    val1 = 180 - getAngleFrom(1, 6, 7, y_predRot, indices)
    if y_predRot[7][2] < y_predRot[6][2]:
        val1 = val1 * -1
    thumb = [6, 7, 8]
    temp = y_predRot.copy()
    y_predRot[thumb] = fixer(y_predRot, val1, 80, 0, thumb, ymain, alpha)
    val = 180 - getAngleFrom(1, 6, 7, y_predRot, indices)
    if y_predRot[7][2] < y_predRot[6][2]:
        val = val * -1
    checker = np.round(boundcheck(y_predRot, val, 80, 0, thumb, ymain, alpha))
    if checker != 0:
        y_predRot = temp
        y_predRot[thumb] = fixer(y_predRot, val1, 80, 0, thumb, ymain, -alpha)

    indices = [0, 1]
    val1 = 90 - getAngleFrom(7, 6, 6, y_predRot, indices, tensor01)
    temp = y_predRot.copy()
    y_predRot[thumb] = fixer(y_predRot, val1, 7, -12, thumb, zmain, alpha)
    val = 90 - getAngleFrom(7, 6, 6, y_predRot, indices, tensor01)
    checker = np.round(boundcheck(y_predRot, val, 7, -12, thumb, zmain, alpha))
    if checker != 0:
        y_predRot = temp
        y_predRot[thumb] = fixer(y_predRot, val1, 7, -12, thumb, zmain, -alpha)


    indices = [1, 2]
    val1 = getAngleFrom(8, 7, 7, y_predRot, indices, tensor01)
    thumb = [7, 8]
    temp = y_predRot.copy()
    y_predRot[thumb] = fixer(y_predRot, val1, 0, 0, thumb, -xmain, alpha)
    val = getAngleFrom(8, 7, 7, y_predRot, indices, tensor01)
    checker = np.round(boundcheck(y_predRot, val, 0, 0, thumb, -xmain, alpha))
    # print("The check is " + str(checker))
    if checker != 0:
        y_predRot = temp
        y_predRot[thumb] = fixer(y_predRot, val1, 0, 0, thumb, -xmain, -alpha)
        # val = getAngleFrom(8, 7, 7, y_predRot, indices, tensor01)
        # print("Now " + str(np.round(boundcheck(y_predRot, val, 0, 0, thumb, -xmain, alpha))))

    indices = [0, 2]
    val1 = 180 - getAngleFrom(6, 7, 8, y_predRot, indices)
    if y_predRot[8][2] < y_predRot[7][2]:
        val1 = -1 * val1
    thumb = [7, 8]
    temp = y_predRot.copy()
    y_predRot[thumb] = fixer(y_predRot, val1, 90, -30, thumb, ymain, alpha)
    val = 180 - getAngleFrom(6, 7, 8, y_predRot, indices)
    if y_predRot[8][2] < y_predRot[7][2]:
        val = -1 * val
    checker = np.round(boundcheck(y_predRot, val, 90, -30, thumb, ymain, alpha))
    if checker != 0:
        y_predRot = temp
        y_predRot[thumb] = fixer(y_predRot, val1, 90, -30, thumb, ymain, -alpha)


    # Finger 2 - Index
    y_predRot = aligner(y_predRot, 0, 2, 9)
    indices = [0, 1]
    val1 = 180 - getAngleFrom(0, 2, 9, y_predRot, indices)
    if y_predRot[9][1] < y_predRot[2][1]:
        val1 = -1 * val1
    index = [2, 9, 10, 11]
    temp = y_predRot.copy()
    y_predRot[index] = fixer(y_predRot, val1, 90, -40, index, -zmain, alpha)
    val = 180 - getAngleFrom(0, 2, 9, y_predRot, indices)
    if y_predRot[9][1] < y_predRot[2][1]:
        val = -1 * val
    checker = np.round(boundcheck(y_predRot, val, 90, -40, index, -zmain, alpha))
    if checker != 0:
        y_predRot = temp
        y_predRot[index] = fixer(y_predRot, val1, 90, -40, index, -zmain, -alpha)


    y_predRot = aligner(y_predRot, 3, 2, 9)
    val1 = 90 - getAngleFrom(3, 2, 9, y_predRot, indices)
    temp = y_predRot.copy()
    y_predRot[index] = fixer(y_predRot, val1, 15, -15, index, -zmain, alpha)
    val = 90 - getAngleFrom(3, 2, 9, y_predRot, indices)
    checker = np.round(boundcheck(y_predRot, val, 15, -15, index, -zmain, alpha))
    if checker != 0:
        y_predRot = temp
        y_predRot[index] = fixer(y_predRot, val1, 15, -15, index, -zmain, -alpha)


    indices = [0, 2]
    y_predRot = aligner(y_predRot, 2, 9, 10)
    val1 = 180 - getAngleFrom(2, 9, 10, y_predRot, indices)
    if y_predRot[10][1] < y_predRot[9][1]:
        val1 = -1 * val1
    index = [9, 10, 11]
    temp = y_predRot.copy()
    y_predRot[index] = fixer(y_predRot, val1, 130, 0, index, ymain, alpha)
    val = 180 - getAngleFrom(2, 9, 10, y_predRot, indices)
    if y_predRot[10][1] < y_predRot[9][1]:
        val = -1 * val
    checker = np.round(boundcheck(y_predRot, val, 130, 0, index, ymain, alpha))
    if checker != 0:
        y_predRot = temp
        y_predRot[index] = fixer(y_predRot, val1, 130, 0, index, ymain, -alpha)


    indices = [0, 1]
    y_predRot = aligner(y_predRot, 9, 10, 11)
    val1 = 180 - getAngleFrom(9, 10, 11, y_predRot, indices)
    if y_predRot[11][1] > y_predRot[10][1]:
        val1 = -1 * val1
    index = [10, 11]
    temp = y_predRot.copy()
    y_predRot[index] = fixer(y_predRot, val1, 90, -30, index, -zmain, alpha)
    val = 180 - getAngleFrom(9, 10, 11, y_predRot, indices)
    if y_predRot[11][1] > y_predRot[10][1]:
        val = -1 * val
    checker = np.round(boundcheck(y_predRot, val, 90, -30, index, -zmain, alpha))
    if checker != 0:
        y_predRot = temp
        y_predRot[index] = fixer(y_predRot, val1, 90, -30, index, -zmain, -alpha)


    # Finger 3 - Middle
    y_predRot = aligner(y_predRot, 0, 3, 12)
    val1 = 180 - getAngleFrom(0, 3, 12, y_predRot, indices)
    if y_predRot[12][1] < y_predRot[3][1]:
        val1 = -1 * val1
    middle = [3, 12, 13, 14]
    temp = y_predRot.copy()
    y_predRot[middle] = fixer(y_predRot, val1, 90, -40, middle, -zmain, alpha)
    val = 180 - getAngleFrom(0, 3, 12, y_predRot, indices)
    if y_predRot[12][1] < y_predRot[3][1]:
        val = -1 * val
    checker = np.round(boundcheck(y_predRot, val, 90, -40, middle, -zmain, alpha))
    if checker != 0:
        y_predRot = temp
        y_predRot[middle] = fixer(y_predRot, val1, 90, -40, middle, -zmain, -alpha)


    y_predRot = aligner(y_predRot, 4, 3, 12)
    val1 = 90 - getAngleFrom(4, 3, 12, y_predRot, indices)
    temp = y_predRot.copy()
    y_predRot[middle] = fixer(y_predRot, val1, 15, -15, middle, -zmain, alpha)
    val = 90 - getAngleFrom(4, 3, 12, y_predRot, indices)
    checker = np.round(boundcheck(y_predRot, val, 15, -15, middle, -zmain, alpha))
    if checker != 0:
        y_predRot = temp
        y_predRot[middle] = fixer(y_predRot, 2 * val, 15, -15, middle, -zmain, -alpha)


    indices = [0, 2]
    y_predRot = aligner(y_predRot, 3, 12, 13)
    val1 = 180 - getAngleFrom(3, 12, 13, y_predRot, indices)
    if y_predRot[13][1] < y_predRot[12][1]:
        val1 = -1 * val1
    middle = [12, 13, 14]
    temp = y_predRot.copy()
    y_predRot[middle] = fixer(y_predRot, val1, 130, 0, middle, ymain, alpha)
    val = 180 - getAngleFrom(3, 12, 13, y_predRot, indices)
    if y_predRot[13][1] < y_predRot[12][1]:
        val = -1 * val
    checker = np.round(boundcheck(y_predRot, val, 130, 0, middle, ymain, alpha))
    if checker != 0:
        y_predRot = temp
        y_predRot[middle] = fixer(y_predRot, val1, 130, 0, middle, ymain, -alpha)


    indices = [0, 1]
    y_predRot = aligner(y_predRot, 12, 13, 14)
    val1 = 180 - getAngleFrom(12, 13, 14, y_predRot, indices)
    if y_predRot[14][1] < y_predRot[13][1]:
        val1 = -1 * val1
    middle = [13, 14]
    temp = y_predRot.copy()
    y_predRot[middle] = fixer(y_predRot, val1, 90, -30, middle, ymain, alpha)
    val = 180 - getAngleFrom(12, 13, 14, y_predRot, indices)
    if y_predRot[14][1] < y_predRot[13][1]:
        val = -1 * val
    checker = np.round(boundcheck(y_predRot, val, 90, -30, middle, ymain, alpha))
    if checker != 0:
        y_predRot = temp
        y_predRot[middle] = fixer(y_predRot, val1, 90, -30, middle, ymain, -alpha)


    # Finger 4 - Ring
    y_predRot = aligner(y_predRot, 0, 4, 15)
    val1 = 180 - getAngleFrom(0, 4, 15, y_predRot, indices)
    if y_predRot[15][1] < y_predRot[4][1]:
        val1 = -1 * val1
    ring = [4, 15, 16, 17]
    temp = y_predRot.copy()
    y_predRot[ring] = fixer(y_predRot, val1, 90, -40, ring, -zmain, alpha)
    val = 180 - getAngleFrom(0, 4, 15, y_predRot, indices)
    if y_predRot[15][1] < y_predRot[4][1]:
        val = -1 * val
    checker = np.round(boundcheck(y_predRot, val, 90, -40, ring, -zmain, alpha))
    if checker != 0:
        y_predRot = temp
        y_predRot[ring] = fixer(y_predRot, val1, 90, -40, ring, -zmain, -alpha)


    y_predRot = aligner(y_predRot, 5, 4, 15)
    val1 = 90 - getAngleFrom(5, 4, 15, y_predRot, indices)
    temp = y_predRot.copy()
    y_predRot[ring] = fixer(y_predRot, val1, 15, -15, ring, -zmain, alpha)
    val = 90 - getAngleFrom(5, 4, 15, y_predRot, indices)
    checker = np.round(boundcheck(y_predRot, val, 15, -15, ring, -zmain, alpha))
    if checker != 0:
        y_predRot = temp
        y_predRot[ring] = fixer(y_predRot, val1, 15, -15, ring, -zmain, -alpha)
        val = 90 - getAngleFrom(5, 4, 15, y_predRot, indices)


    indices = [0, 2]
    y_predRot = aligner(y_predRot, 4, 15, 16)
    val1 = 180 - getAngleFrom(4, 15, 16, y_predRot, indices)
    if y_predRot[16][1] < y_predRot[15][1]:
        val1 = -1 * val1
    ring = [15, 16, 17]
    temp = y_predRot.copy()
    print("The Current is " + str(val1))
    y_predRot[ring] = fixer(y_predRot, val1, 130, 0, ring, ymain, alpha)
    val = 180 - getAngleFrom(4, 15, 16, y_predRot, indices)
    if y_predRot[16][1] < y_predRot[15][1]:
        val = -1 * val
    print("The check is " + str(val))
    checker = np.round(boundcheck(y_predRot, val, 130, 0, ring, ymain, alpha))
    print("The check is " + str(checker))
    if checker != 0:
        y_predRot = temp
        y_predRot[ring] = fixer(y_predRot, val1, 130, 0, ring, ymain, -alpha)
        val = 180 - getAngleFrom(4, 15, 16, y_predRot, indices)
        if y_predRot[16][1] < y_predRot[15][1]:
            val = -1 * val
        print("Now " + str(np.round(boundcheck(y_predRot, val, 130, 0, ring, ymain, alpha))))


    indices = [0, 1]
    y_predRot = aligner(y_predRot, 15, 16, 17)
    val1 = 180 - getAngleFrom(15, 16, 17, y_predRot, indices)
    if y_predRot[17][1] > y_predRot[16][1]:  # greater
        val1 = -1 * val1
    ring = [16, 17]
    temp = y_predRot.copy()
    y_predRot[ring] = fixer(y_predRot, val1, 90, -30, ring, zmain, alpha)
    val = 180 - getAngleFrom(15, 16, 17, y_predRot, indices)
    if y_predRot[17][1] > y_predRot[16][1]:  # greater
        val = -1 * val
    checker = np.round(boundcheck(y_predRot, val, 90, -30, ring, zmain, alpha))
    if checker != 0:
        y_predRot = temp
        y_predRot[ring] = fixer(y_predRot, val1, 90, -30, ring, zmain, -alpha)


    # Finger 5 Pinky
    y_predRot = aligner(y_predRot, 0, 5, 18)
    val1 = 180 - getAngleFrom(0, 5, 18, y_predRot, indices)
    if y_predRot[18][1] < y_predRot[5][1]:
        val1 = -1 * val1
    pinky = [5, 18, 19, 20]
    temp = y_predRot.copy()
    y_predRot[pinky] = fixer(y_predRot, val1, 90, -40, pinky, -zmain, alpha)
    val = 180 - getAngleFrom(0, 5, 18, y_predRot, indices)
    if y_predRot[18][1] < y_predRot[5][1]:
        val = -1 * val
    checker = np.round(boundcheck(y_predRot, val, 90, -40, pinky, -zmain, alpha))
    if checker != 0:
        y_predRot = temp
        y_predRot[pinky] = fixer(y_predRot, val1, 90, -40, pinky, -zmain, -alpha)


    y_predRot = aligner(y_predRot, 4, 5, 18)
    val1 = getAngleFrom(4, 5, 18, y_predRot, indices) - 90
    temp = y_predRot.copy()
    y_predRot[pinky] = fixer(y_predRot, val1, 15, -15, pinky, zmain, alpha)
    val = getAngleFrom(4, 5, 18, y_predRot, indices) - 90
    checker = np.round(boundcheck(y_predRot, val, 15, -15, pinky, zmain, alpha))
    if checker != 0:
        y_predRot = temp
        y_predRot[pinky] = fixer(y_predRot, val1, 15, -15, pinky, zmain, -alpha)


    indices = [0, 1]
    y_predRot = aligner(y_predRot, 5, 18, 19)
    val1 = 180 - getAngleFrom(5, 18, 19, y_predRot, indices)
    if y_predRot[19][1] < y_predRot[18][1]:
        val1 = -1 * val1
    pinky = [18, 19, 20]
    temp = y_predRot.copy()
    y_predRot[pinky] = fixer(y_predRot, val1, 130, 0, pinky, ymain, alpha)
    val = 180 - getAngleFrom(5, 18, 19, y_predRot, indices)
    if y_predRot[19][1] < y_predRot[18][1]:
        val = -1 * val
    checker = np.round(boundcheck(y_predRot, val, 130, 0, pinky, ymain, alpha))
    if checker != 0:
        y_predRot = temp
        y_predRot[pinky] = fixer(y_predRot, val1, 130, 0, pinky, ymain, -alpha)


    indices = [0, 1]
    y_predRot = aligner(y_predRot, 18, 19, 20)
    val1 = 180 - getAngleFrom(18, 19, 20, y_predRot, indices)
    if y_predRot[20][1] > y_predRot[19][1]:
        val1 = -1 * val1
    pinky = [19, 20]
    temp = y_predRot.copy()
    y_predRot[pinky] = fixer(y_predRot, val1, 90, -30, pinky, zmain, alpha)
    val = 180 - getAngleFrom(18, 19, 20, y_predRot, indices)
    if y_predRot[20][1] > y_predRot[19][1]:
        val = -1 * val
    checker = np.round(boundcheck(y_predRot, val, 90, -30, pinky, zmain, alpha))
    if checker != 0:
        y_predRot = temp
        y_predRot[pinky] = fixer(y_predRot, val1, 90, -30, pinky, zmain, -alpha)


    # Match former alignment
    translate = y_values[0] - y_predRot[0]
    y_predRot = y_predRot + translate

    v1 = y_predRot[3] - y_predRot[0]
    v2 = y_values[3] - y_predRot[0]
    rad = angle_between(v1, v2)
    d = np.degrees(rad)
    axisPoint = np.cross(v1, v2)

    test = PointRotate3D(y_predRot[0],
                         y_predRot[0] + unit_vector(axisPoint),
                         y_predRot[3],
                         np.radians(d))
    values = np.zeros((21, 3))

    if np.linalg.norm(np.round(y_values[3] - test)) == 0:
        for j in range(0, 21):
            values[j, :] = PointRotate3D(y_predRot[0],
                                         y_predRot[0] + unit_vector(axisPoint),
                                         y_predRot[j],
                                         np.radians(d))
    else:
        for j in range(0, 21):
            deg = 360 - d
            values[j, :] = PointRotate3D(y_predRot[0],
                                         y_predRot[0] + unit_vector(axisPoint),
                                         y_predRot[j],
                                         np.radians(deg))

    y_predRot = values

    # Final Rotation
    def getProjpoint(point, orig, normal):
        v = point - orig
        dist = np.dot(v, normal)
        projected_point = point - dist * normal
        return projected_point

    norm = unit_vector(y_predRot[3] - y_predRot[0])
    a = getProjpoint(y_predRot[4], y_predRot[0], norm)
    b = getProjpoint(y_values[4], y_predRot[0], norm)
    v1 = a - y_predRot[0]
    v2 = b - y_predRot[0]
    rad1 = angle_between(v1, v2)
    d = np.degrees(rad1)

    test = PointRotate3D(y_predRot[0],
                         y_predRot[3],
                         y_predRot[4],
                         np.radians(d))

    values = np.zeros((21, 3))
    c1 = np.linalg.norm(np.round(y_values[4] - test))

    if c1 == 0:
        for j in range(0, 21):
            values[j, :] = PointRotate3D(y_predRot[0],
                                         y_predRot[3],
                                         y_predRot[j],
                                         np.radians(d))
    else:
        for j in range(0, 21):
            deg1 = 360 - d
            values[j, :] = PointRotate3D(y_predRot[0],
                                         y_predRot[3],
                                         y_predRot[j],
                                         np.radians(deg1))
    y_predRot = values
    return y_predRot


def correctionCheckCombinedDebug(y_values, alpha):
    y_predRot = coplanar(y_values, 1, 6, 8, alpha)
    y_predRot = coplanar(y_predRot, 2, 9, 11, alpha)
    y_predRot = coplanar(y_predRot, 3, 12, 14, alpha)
    y_predRot = coplanar(y_predRot, 4, 15, 17, alpha)
    y_predRot = coplanar(y_predRot, 5, 18, 20, alpha)
    realError = 0.0
    debug = False
    # if alpha == 1.0:
    #     print("Start " + str(alpha))
    #     debug = True

    # Thumb
    y_predRot = aligner(y_predRot, 1, 2, 3)
    indices = [0, 1]
    valCurrent = getAngleFrom(3, 1, 6, y_predRot, indices) - getAngleFrom(3, 1, 2, y_predRot, indices)
    thumb = [1, 6, 7, 8]
    temp = y_predRot.copy()
    y_predRot[thumb] = fixer(y_predRot, valCurrent, 45, -20, thumb, zmain, alpha)
    setFirst = y_predRot[thumb]
    val = getAngleFrom(3, 1, 6, y_predRot, indices) - getAngleFrom(3, 1, 2, y_predRot, indices)
    checkerA = np.round(boundcheck(y_predRot, val, 45, -20, thumb, zmain, alpha))
    y_predRot[thumb] = fixer(temp, valCurrent, 45, -20, thumb, zmain, -alpha)
    val = getAngleFrom(3, 1, 6, y_predRot, indices) - getAngleFrom(3, 1, 2, y_predRot, indices)
    checkerB = np.round(boundcheck(y_predRot, val, 45, -20, thumb, zmain, alpha))
    if checkerA > checkerB:
        realError += checkerB
    else:
        realError += checkerA
        y_predRot[thumb] = setFirst
    if debug:
        print(realError)


    indices = [0, 2]
    valCurrent = getAngleFrom(2, 1, 6, y_predRot, indices)
    if y_predRot[6][2] < y_predRot[1][2]:
        valCurrent = 360. - valCurrent
    temp = y_predRot.copy()
    y_predRot[thumb] = fixer(y_predRot, valCurrent, 45, 0, thumb, ymain, alpha)
    setFirst = y_predRot[thumb]
    val = getAngleFrom(2, 1, 6, y_predRot, indices)
    if y_predRot[6][2] < y_predRot[1][2]:
        val = 360. - val
    checkerA = np.round(boundcheck(y_predRot, val, 45, 0, thumb, ymain, alpha))
    y_predRot[thumb] = fixer(temp, valCurrent, 45, 0, thumb, ymain, -alpha)
    val = getAngleFrom(2, 1, 6, y_predRot, indices)
    if y_predRot[6][2] < y_predRot[1][2]:
        val = 360. - val
    checkerB = np.round(boundcheck(y_predRot, val, 45, 0, thumb, ymain, alpha))
    if checkerA > checkerB:
        realError += checkerB
    else:
        realError += checkerA
        y_predRot[thumb] = setFirst
    if debug:
        print(realError)


    y_predRot = aligner(y_predRot, 1, 6, 2)
    valCurrent = 180 - getAngleFrom(1, 6, 7, y_predRot, indices)
    if y_predRot[7][2] < y_predRot[6][2]:
        valCurrent = valCurrent * -1
    thumb = [6, 7, 8]
    temp = y_predRot.copy()
    y_predRot[thumb] = fixer(y_predRot, valCurrent, 80, 0, thumb, ymain, alpha)
    setFirst = y_predRot[thumb]
    val = 180 - getAngleFrom(1, 6, 7, y_predRot, indices)
    if y_predRot[7][2] < y_predRot[6][2]:
        val = val * -1
    checkerA = np.round(boundcheck(y_predRot, val, 80, 0, thumb, ymain, alpha))
    y_predRot[thumb] = fixer(temp, valCurrent, 80, 0, thumb, ymain, -alpha)
    val = 180 - getAngleFrom(1, 6, 7, y_predRot, indices)
    if y_predRot[7][2] < y_predRot[6][2]:
        val = val * -1
    checkerB = np.round(boundcheck(y_predRot, val, 80, 0, thumb, ymain, alpha))
    if checkerA > checkerB:
        realError += checkerB
    else:
        realError += checkerA
        y_predRot[thumb] = setFirst
    if debug:
        print(realError)


    indices = [0, 1]
    valCurrent = 90 - getAngleFrom(7, 6, 6, y_predRot, indices, tensor01)
    temp = y_predRot.copy()
    y_predRot[thumb] = fixer(y_predRot, valCurrent, 7, -12, thumb, zmain, alpha)
    setFirst = y_predRot[thumb]
    val = 90 - getAngleFrom(7, 6, 6, y_predRot, indices, tensor01)
    checkerA = np.round(boundcheck(y_predRot, val, 7, -12, thumb, zmain, alpha))
    y_predRot[thumb] = fixer(temp, valCurrent, 7, -12, thumb, zmain, -alpha)
    val = 90 - getAngleFrom(7, 6, 6, y_predRot, indices, tensor01)
    checkerB = np.round(boundcheck(y_predRot, val, 7, -12, thumb, zmain, alpha))
    if checkerA > checkerB:
        realError += checkerB
    else:
        realError += checkerA
        y_predRot[thumb] = setFirst
    if debug:
        print(realError)


    indices = [1, 2]
    valCurrent = getAngleFrom(8, 7, 7, y_predRot, indices, tensor01)
    thumb = [7, 8]
    temp = y_predRot.copy()
    y_predRot[thumb] = fixer(y_predRot, valCurrent, 0, 0, thumb, -xmain, alpha)
    setFirst = y_predRot[thumb]
    val = getAngleFrom(8, 7, 7, y_predRot, indices, tensor01)
    checkerA = np.round(boundcheck(y_predRot, val, 0, 0, thumb, -xmain, alpha))
    y_predRot[thumb] = fixer(temp, valCurrent, 0, 0, thumb, -xmain, -alpha)
    val = getAngleFrom(8, 7, 7, y_predRot, indices, tensor01)
    checkerB = np.round(boundcheck(y_predRot, val, 0, 0, thumb, -xmain, alpha))
    if checkerA > checkerB:
        realError += checkerB
    else:
        realError += checkerA
        y_predRot[thumb] = setFirst
    if debug:
        print(realError)


    indices = [0, 2]
    valCurrent = 180 - getAngleFrom(6, 7, 8, y_predRot, indices)
    if y_predRot[8][2] < y_predRot[7][2]:
        valCurrent = -1 * valCurrent
    thumb = [7, 8]
    temp = y_predRot.copy()
    y_predRot[thumb] = fixer(y_predRot, valCurrent, 90, -30, thumb, ymain, alpha)
    setFirst = y_predRot[thumb]
    val = 180 - getAngleFrom(6, 7, 8, y_predRot, indices)
    if y_predRot[8][2] < y_predRot[7][2]:
        val = -1 * val
    checkerA = np.round(boundcheck(y_predRot, val, 90, -30, thumb, ymain, alpha))
    y_predRot[thumb] = fixer(temp, valCurrent, 90, -30, thumb, ymain, -alpha)
    val = 180 - getAngleFrom(6, 7, 8, y_predRot, indices)
    if y_predRot[8][2] < y_predRot[7][2]:
        val = -1 * val
    checkerB = np.round(boundcheck(y_predRot, val, 90, -30, thumb, ymain, alpha))
    if checkerA > checkerB:
        realError += checkerB
    else:
        realError += checkerA
        y_predRot[thumb] = setFirst
    if debug:
        print(realError)


    # Finger 2 - Index
    y_predRot = aligner(y_predRot, 0, 2, 9)
    indices = [0, 1]
    valCurrent = 180 - getAngleFrom(0, 2, 9, y_predRot, indices)
    if y_predRot[9][1] < y_predRot[2][1]:
        valCurrent = -1 * valCurrent
    index = [2, 9, 10, 11]
    temp = y_predRot.copy()
    y_predRot[index] = fixer(y_predRot, valCurrent, 90, -40, index, -zmain, alpha)
    setFirst = y_predRot[index]
    val = 180 - getAngleFrom(0, 2, 9, y_predRot, indices)
    if y_predRot[9][1] < y_predRot[2][1]:
        val = -1 * val
    checkerA = np.round(boundcheck(y_predRot, val, 90, -40, index, -zmain, alpha))
    y_predRot[index] = fixer(temp, valCurrent, 90, -40, index, -zmain, -alpha)
    val = 180 - getAngleFrom(0, 2, 9, y_predRot, indices)
    if y_predRot[9][1] < y_predRot[2][1]:
        val = -1 * val
    checkerB = np.round(boundcheck(y_predRot, val, 90, -40, index, -zmain, alpha))
    if checkerA > checkerB:
        realError += checkerB
    else:
        realError += checkerA
        y_predRot[index] = setFirst
    if debug:
        print(realError)


    y_predRot = aligner(y_predRot, 3, 2, 9)
    valCurrent = 90 - getAngleFrom(3, 2, 9, y_predRot, indices)
    temp = y_predRot.copy()
    y_predRot[index] = fixer(y_predRot, valCurrent, 15, -15, index, -zmain, alpha)
    setFirst = y_predRot[index]
    val = 90 - getAngleFrom(3, 2, 9, y_predRot, indices)
    checkerA = np.round(boundcheck(y_predRot, val, 15, -15, index, -zmain, alpha))
    y_predRot[index] = fixer(y_predRot, valCurrent, 15, -15, index, -zmain, -alpha)
    val = 90 - getAngleFrom(3, 2, 9, y_predRot, indices)
    checkerB = np.round(boundcheck(y_predRot, val, 15, -15, index, -zmain, alpha))
    if checkerA > checkerB:
        realError += checkerB
    else:
        realError += checkerA
        y_predRot[index] = setFirst
    if debug:
        print(realError)


    indices = [0, 2]
    y_predRot = aligner(y_predRot, 2, 9, 10)
    valCurrent = 180 - getAngleFrom(2, 9, 10, y_predRot, indices)
    if y_predRot[10][1] < y_predRot[9][1]:
        valCurrent = -1 * valCurrent
    index = [9, 10, 11]
    temp = y_predRot.copy()
    y_predRot[index] = fixer(y_predRot, valCurrent, 130, 0, index, ymain, alpha)
    setFirst = y_predRot[index]
    val = 180 - getAngleFrom(2, 9, 10, y_predRot, indices)
    if y_predRot[10][1] < y_predRot[9][1]:
        val = -1 * val
    checkerA = np.round(boundcheck(y_predRot, val, 130, 0, index, ymain, alpha))
    y_predRot[index] = fixer(temp, valCurrent, 130, 0, index, ymain, -alpha)
    val = 180 - getAngleFrom(2, 9, 10, y_predRot, indices)
    if y_predRot[10][1] < y_predRot[9][1]:
        val = -1 * val
    checkerB = np.round(boundcheck(y_predRot, val, 130, 0, index, ymain, alpha))
    if checkerA > checkerB:
        realError += checkerB
    else:
        realError += checkerA
        y_predRot[index] = setFirst
    if debug:
        print(realError)


    indices = [0, 1]
    y_predRot = aligner(y_predRot, 9, 10, 11)
    valCurrent = 180 - getAngleFrom(9, 10, 11, y_predRot, indices)
    if y_predRot[11][1] > y_predRot[10][1]:
        valCurrent = -1 * valCurrent
    index = [10, 11]
    temp = y_predRot.copy()
    y_predRot[index] = fixer(y_predRot, valCurrent, 90, -30, index, -zmain, alpha)
    setFirst = y_predRot[index]
    val = 180 - getAngleFrom(9, 10, 11, y_predRot, indices)
    if y_predRot[11][1] > y_predRot[10][1]:
        val = -1 * val
    checkerA = np.round(boundcheck(y_predRot, val, 90, -30, index, -zmain, alpha))
    y_predRot[index] = fixer(temp, valCurrent, 90, -30, index, -zmain, -alpha)
    val = 180 - getAngleFrom(9, 10, 11, y_predRot, indices)
    if y_predRot[11][1] > y_predRot[10][1]:
        val = -1 * val
    checkerB = np.round(boundcheck(y_predRot, val, 90, -30, index, -zmain, alpha))
    if checkerA > checkerB:
        realError += checkerB
    else:
        realError += checkerA
        y_predRot[index] = setFirst
    if debug:
        print(realError)


    # Finger 3 - Middle
    y_predRot = aligner(y_predRot, 0, 3, 12)
    valCurrent = 180 - getAngleFrom(0, 3, 12, y_predRot, indices)
    if y_predRot[12][1] < y_predRot[3][1]:
        valCurrent = -1 * valCurrent
    middle = [3, 12, 13, 14]
    temp = y_predRot.copy()
    y_predRot[middle] = fixer(y_predRot, valCurrent, 90, -40, middle, -zmain, alpha)
    setFirst = y_predRot[middle]
    val = 180 - getAngleFrom(0, 3, 12, y_predRot, indices)
    if y_predRot[12][1] < y_predRot[3][1]:
        val = -1 * val
    checkerA = np.round(boundcheck(y_predRot, val, 90, -40, middle, -zmain, alpha))
    y_predRot[middle] = fixer(temp, valCurrent, 90, -40, middle, -zmain, -alpha)
    val = 180 - getAngleFrom(0, 3, 12, y_predRot, indices)
    if y_predRot[12][1] < y_predRot[3][1]:
        val = -1 * val
    checkerB = np.round(boundcheck(y_predRot, val, 90, -40, middle, -zmain, alpha))
    if checkerA > checkerB:
        realError += checkerB
    else:
        realError += checkerA
        y_predRot[middle] = setFirst
    if debug:
        print(realError)


    y_predRot = aligner(y_predRot, 4, 3, 12)
    valCurrent = 90 - getAngleFrom(4, 3, 12, y_predRot, indices)
    temp = y_predRot.copy()
    y_predRot[middle] = fixer(y_predRot, valCurrent, 15, -15, middle, -zmain, alpha)
    setFirst = y_predRot[middle]
    val = 90 - getAngleFrom(4, 3, 12, y_predRot, indices)
    checkerA = np.round(boundcheck(y_predRot, val, 15, -15, middle, -zmain, alpha))
    y_predRot[middle] = fixer(temp, valCurrent, 15, -15, middle, -zmain, -alpha)
    val = 90 - getAngleFrom(4, 3, 12, y_predRot, indices)
    checkerB = np.round(boundcheck(y_predRot, val, 15, -15, middle, -zmain, alpha))
    if checkerA > checkerB:
        realError += checkerB
    else:
        realError += checkerA
        y_predRot[middle] = setFirst
    if debug:
        print(realError)


    indices = [0, 2]
    y_predRot = aligner(y_predRot, 3, 12, 13)
    valCurrent = 180 - getAngleFrom(3, 12, 13, y_predRot, indices)
    if y_predRot[13][1] < y_predRot[12][1]:
        valCurrent = -1 * valCurrent
    middle = [12, 13, 14]
    temp = y_predRot.copy()
    y_predRot[middle] = fixer(y_predRot, valCurrent, 130, 0, middle, ymain, alpha)
    setFirst = y_predRot[middle]
    val = 180 - getAngleFrom(3, 12, 13, y_predRot, indices)
    if y_predRot[13][1] < y_predRot[12][1]:
        val = -1 * val
    checkerA = np.round(boundcheck(y_predRot, val, 130, 0, middle, ymain, alpha))
    y_predRot[middle] = fixer(temp, valCurrent, 130, 0, middle, ymain, -alpha)
    val = 180 - getAngleFrom(3, 12, 13, y_predRot, indices)
    if y_predRot[13][1] < y_predRot[12][1]:
        val = -1 * val
    checkerB = np.round(boundcheck(y_predRot, val, 130, 0, middle, ymain, alpha))
    if checkerA > checkerB:
        realError += checkerB
    else:
        realError += checkerA
        y_predRot[middle] = setFirst
    if debug:
        print(realError)


    indices = [0, 1]
    y_predRot = aligner(y_predRot, 12, 13, 14)
    valCurrent = 180 - getAngleFrom(12, 13, 14, y_predRot, indices)
    if y_predRot[14][1] < y_predRot[13][1]:
        valCurrent = -1 * valCurrent
    middle = [13, 14]
    temp = y_predRot.copy()
    y_predRot[middle] = fixer(y_predRot, valCurrent, 90, -30, middle, ymain, alpha)
    setFirst = y_predRot[middle]
    val = 180 - getAngleFrom(12, 13, 14, y_predRot, indices)
    if y_predRot[14][1] < y_predRot[13][1]:
        val = -1 * val
    checkerA = np.round(boundcheck(y_predRot, val, 90, -30, middle, ymain, alpha))
    y_predRot[middle] = fixer(temp, valCurrent, 90, -30, middle, ymain, -alpha)
    val = 180 - getAngleFrom(12, 13, 14, y_predRot, indices)
    if y_predRot[14][1] < y_predRot[13][1]:
        val = -1 * val
    checkerB = np.round(boundcheck(y_predRot, val, 90, -30, middle, ymain, alpha))
    if checkerA > checkerB:
        realError += checkerB
    else:
        realError += checkerA
        y_predRot[middle] = setFirst
    if debug:
        print(realError)


    # Finger 4 - Ring
    y_predRot = aligner(y_predRot, 0, 4, 15)
    valCurrent = 180 - getAngleFrom(0, 4, 15, y_predRot, indices)
    if y_predRot[15][1] < y_predRot[4][1]:
        valCurrent = -1 * valCurrent
    ring = [4, 15, 16, 17]
    temp = y_predRot.copy()
    y_predRot[ring] = fixer(y_predRot, valCurrent, 90, -40, ring, -zmain, alpha)
    setFirst = y_predRot[ring]
    val = 180 - getAngleFrom(0, 4, 15, y_predRot, indices)
    if y_predRot[15][1] < y_predRot[4][1]:
        val = -1 * val
    checkerA = np.round(boundcheck(y_predRot, val, 90, -40, ring, -zmain, alpha))
    y_predRot[ring] = fixer(temp, valCurrent, 90, -40, ring, -zmain, -alpha)
    val = 180 - getAngleFrom(0, 4, 15, y_predRot, indices)
    if y_predRot[15][1] < y_predRot[4][1]:
        val = -1 * val
    checkerB = np.round(boundcheck(y_predRot, val, 90, -40, ring, -zmain, alpha))
    if checkerA > checkerB:
        realError += checkerB
    else:
        realError += checkerA
        y_predRot[ring] = setFirst
    if debug:
        print(realError)


    y_predRot = aligner(y_predRot, 5, 4, 15)
    val1 = 90 - getAngleFrom(5, 4, 15, y_predRot, indices)
    temp = y_predRot.copy()
    y_predRot[ring] = fixer(y_predRot, val1, 15, -15, ring, -zmain, alpha)
    setFirst = y_predRot[ring]
    val = 90 - getAngleFrom(5, 4, 15, y_predRot, indices)
    checkerA = np.round(boundcheck(y_predRot, val, 15, -15, ring, -zmain, alpha))
    y_predRot[ring] = fixer(temp, val1, 15, -15, ring, -zmain, -alpha)
    val = 90 - getAngleFrom(5, 4, 15, y_predRot, indices)
    checkerB = np.round(boundcheck(y_predRot, val, 15, -15, ring, -zmain, alpha))
    if checkerA > checkerB:
        realError += checkerB
    else:
        realError += checkerA
        y_predRot[ring] = setFirst
    if debug:
        print(realError)


    indices = [0, 2]
    y_predRot = aligner(y_predRot, 4, 15, 16)
    val1 = 180 - getAngleFrom(4, 15, 16, y_predRot, indices)
    if y_predRot[16][1] < y_predRot[15][1]:
        val1 = -1 * val1
    ring = [15, 16, 17]
    temp = y_predRot.copy()
    y_predRot[ring] = fixer(y_predRot, val1, 130, 0, ring, ymain, alpha)
    setFirst = y_predRot[ring]
    val = 180 - getAngleFrom(4, 15, 16, y_predRot, indices)
    if y_predRot[16][1] < y_predRot[15][1]:
        val = -1 * val
    checkerA = np.round(boundcheck(y_predRot, val, 130, 0, ring, ymain, alpha))
    y_predRot[ring] = fixer(temp, val1, 130, 0, ring, ymain, -alpha)
    val = 180 - getAngleFrom(4, 15, 16, y_predRot, indices)
    if y_predRot[16][1] < y_predRot[15][1]:
        val = -1 * val
    checkerB = np.round(boundcheck(y_predRot, val, 130, 0, ring, ymain, alpha))
    if checkerA > checkerB:
        realError += checkerB
    else:
        realError += checkerA
        y_predRot[ring] = setFirst
    if debug:
        print(realError)


    indices = [0, 1]
    y_predRot = aligner(y_predRot, 15, 16, 17)
    val1 = 180 - getAngleFrom(15, 16, 17, y_predRot, indices)
    if y_predRot[17][1] > y_predRot[16][1]:  # greater
        val1 = -1 * val1
    ring = [16, 17]
    temp = y_predRot.copy()
    y_predRot[ring] = fixer(y_predRot, val1, 90, -30, ring, zmain, alpha)
    setFirst = y_predRot[ring]
    val = 180 - getAngleFrom(15, 16, 17, y_predRot, indices)
    if y_predRot[17][1] > y_predRot[16][1]:  # greater
        val = -1 * val
    checkerA = np.round(boundcheck(y_predRot, val, 90, -30, ring, zmain, alpha))
    y_predRot[ring] = fixer(temp, val1, 90, -30, ring, zmain, -alpha)
    val = 180 - getAngleFrom(15, 16, 17, y_predRot, indices)
    if y_predRot[17][1] > y_predRot[16][1]:  # greater
        val = -1 * val
    checkerB = np.round(boundcheck(y_predRot, val, 90, -30, ring, zmain, alpha))
    if checkerA > checkerB:
        realError += checkerB
    else:
        realError += checkerA
        y_predRot[ring] = setFirst
    if debug:
        print(realError)


    # Finger 5 Pinky
    y_predRot = aligner(y_predRot, 0, 5, 18)
    val1 = 180 - getAngleFrom(0, 5, 18, y_predRot, indices)
    if y_predRot[18][1] < y_predRot[5][1]:
        val1 = -1 * val1
    pinky = [5, 18, 19, 20]
    temp = y_predRot.copy()
    y_predRot[pinky] = fixer(y_predRot, val1, 90, -40, pinky, -zmain, alpha)
    setFirst = y_predRot[pinky]
    val = 180 - getAngleFrom(0, 5, 18, y_predRot, indices)
    if y_predRot[18][1] < y_predRot[5][1]:
        val = -1 * val
    checkerA = np.round(boundcheck(y_predRot, val, 90, -40, pinky, -zmain, alpha))
    y_predRot[pinky] = fixer(temp, val1, 90, -40, pinky, -zmain, -alpha)
    val = 180 - getAngleFrom(0, 5, 18, y_predRot, indices)
    if y_predRot[18][1] < y_predRot[5][1]:
        val = -1 * val
    checkerB = np.round(boundcheck(y_predRot, val, 90, -40, pinky, -zmain, alpha))
    if checkerA > checkerB:
        realError += checkerB
    else:
        realError += checkerA
        y_predRot[pinky] = setFirst
    if debug:
        print(realError)


    y_predRot = aligner(y_predRot, 4, 5, 18)
    val1 = getAngleFrom(4, 5, 18, y_predRot, indices) - 90
    temp = y_predRot.copy()
    y_predRot[pinky] = fixer(y_predRot, val1, 15, -15, pinky, zmain, alpha)
    setFirst = y_predRot[pinky]
    val = getAngleFrom(4, 5, 18, y_predRot, indices) - 90
    checkerA = np.round(boundcheck(y_predRot, val, 15, -15, pinky, zmain, alpha))
    y_predRot[pinky] = fixer(temp, val1, 15, -15, pinky, zmain, -alpha)
    val = getAngleFrom(4, 5, 18, y_predRot, indices) - 90
    checkerB = np.round(boundcheck(y_predRot, val, 15, -15, pinky, zmain, alpha))
    if checkerA > checkerB:
        realError += checkerB
    else:
        realError += checkerA
        y_predRot[pinky] = setFirst
    if debug:
        print(realError)


    indices = [0, 1]
    y_predRot = aligner(y_predRot, 5, 18, 19)
    val1 = 180 - getAngleFrom(5, 18, 19, y_predRot, indices)
    if y_predRot[19][1] < y_predRot[18][1]:
        val1 = -1 * val1
    pinky = [18, 19, 20]
    temp = y_predRot.copy()
    y_predRot[pinky] = fixer(y_predRot, val1, 130, 0, pinky, ymain, alpha)
    setFirst = y_predRot[pinky]
    val = 180 - getAngleFrom(5, 18, 19, y_predRot, indices)
    if y_predRot[19][1] < y_predRot[18][1]:
        val = -1 * val
    checkerA = np.round(boundcheck(y_predRot, val, 130, 0, pinky, ymain, alpha))
    y_predRot[pinky] = fixer(temp, val1, 130, 0, pinky, ymain, -alpha)
    val = 180 - getAngleFrom(5, 18, 19, y_predRot, indices)
    if y_predRot[19][1] < y_predRot[18][1]:
        val = -1 * val
    checkerB = np.round(boundcheck(y_predRot, val, 130, 0, pinky, ymain, alpha))
    if checkerA > checkerB:
        realError += checkerB
    else:
        realError += checkerA
        y_predRot[pinky] = setFirst
    if debug:
        print(realError)


    indices = [0, 1]
    y_predRot = aligner(y_predRot, 18, 19, 20)
    val1 = 180 - getAngleFrom(18, 19, 20, y_predRot, indices)
    if y_predRot[20][1] > y_predRot[19][1]:
        val1 = -1 * val1
    pinky = [19, 20]
    temp = y_predRot.copy()
    y_predRot[pinky] = fixer(y_predRot, val1, 90, -30, pinky, zmain, alpha)
    setFirst = y_predRot[pinky]
    val = 180 - getAngleFrom(18, 19, 20, y_predRot, indices)
    if y_predRot[20][1] > y_predRot[19][1]:
        val = -1 * val
    checkerA = np.round(boundcheck(y_predRot, val, 90, -30, pinky, zmain, alpha))
    y_predRot[pinky] = fixer(temp, val1, 90, -30, pinky, zmain, -alpha)
    val = 180 - getAngleFrom(18, 19, 20, y_predRot, indices)
    if y_predRot[20][1] > y_predRot[19][1]:
        val = -1 * val
    checkerB = np.round(boundcheck(y_predRot, val, 90, -30, pinky, zmain, alpha))
    if checkerA > checkerB:
        realError += checkerB
    else:
        realError += checkerA
        y_predRot[pinky] = setFirst
    if debug:
        print(realError)

    # Match former alignment
    translate = y_values[0] - y_predRot[0]
    y_predRot = y_predRot + translate

    v1 = y_predRot[3] - y_predRot[0]
    v2 = y_values[3] - y_predRot[0]
    rad = angle_between(v1, v2)
    d = np.degrees(rad)
    axisPoint = np.cross(v1, v2)

    test = PointRotate3D(y_predRot[0],
                         y_predRot[0] + unit_vector(axisPoint),
                         y_predRot[3],
                         np.radians(d))
    values = np.zeros((21, 3))

    if np.linalg.norm(np.round(y_values[3] - test)) == 0:
        for j in range(0, 21):
            values[j, :] = PointRotate3D(y_predRot[0],
                                         y_predRot[0] + unit_vector(axisPoint),
                                         y_predRot[j],
                                         np.radians(d))
    else:
        for j in range(0, 21):
            deg = 360 - d
            values[j, :] = PointRotate3D(y_predRot[0],
                                         y_predRot[0] + unit_vector(axisPoint),
                                         y_predRot[j],
                                         np.radians(deg))

    y_predRot = values

    # Final Rotation
    def getProjpoint(point, orig, normal):
        v = point - orig
        dist = np.dot(v, normal)
        projected_point = point - dist * normal
        return projected_point

    norm = unit_vector(y_predRot[3] - y_predRot[0])
    a = getProjpoint(y_predRot[4], y_predRot[0], norm)
    b = getProjpoint(y_values[4], y_predRot[0], norm)
    v1 = a - y_predRot[0]
    v2 = b - y_predRot[0]
    rad1 = angle_between(v1, v2)
    d = np.degrees(rad1)

    test = PointRotate3D(y_predRot[0],
                         y_predRot[3],
                         y_predRot[4],
                         np.radians(d))

    values = np.zeros((21, 3))
    c1 = np.linalg.norm(np.round(y_values[4] - test))

    if c1 == 0:
        for j in range(0, 21):
            values[j, :] = PointRotate3D(y_predRot[0],
                                         y_predRot[3],
                                         y_predRot[j],
                                         np.radians(d))
    else:
        for j in range(0, 21):
            deg1 = 360 - d
            values[j, :] = PointRotate3D(y_predRot[0],
                                         y_predRot[3],
                                         y_predRot[j],
                                         np.radians(deg1))
    y_predRot = values
    return y_predRot, realError


def world2pixel(x, fx, fy, ux, uy):
    x[:, 0] = x[:, 0] * fx / x[:, 2] + ux
    x[:, 1] = x[:, 1] * fy / x[:, 2] + uy
    return x


def pixel2world(x, fx, fy, ux, uy):
    x[:, 0] = (x[:, 0] - ux) * x[:, 2] / fx
    x[:, 1] = (x[:, 1] - uy) * x[:, 2] / fy
    return x