import math
import time
import sys
import string
import copy
import numpy as np
import random
from decimal import *
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import axes3d

# Implements slicing algorithm from Minetto, et al (2017)

class Point:
    def __init__(self, x_, y_, z_):
        self.x = x_
        self.y = y_
        self.z = z_

    def dotProduct(self, p):
        return self.x*p.x + self.y*p.y + self.z*p.z

    def normalize(self):
        return math.sqrt(self.x**2 + self.y**2 + self.z**2)

    def __eq__(self, other):
        return (self.x == other.x) and (self.y == other.y) and (self.z == other.z)
    def __hash__(self):
        return hash((self.x,self.y,self.z))
    def __len__(self):
        return 1
    def __str__(self):
        return "Point("+str(self.x)+","+str(self.y)+","+str(self.z)+")"
    def __repr__(self):
        return "Point("+str(self.x)+","+str(self.y)+","+str(self.z)+")"

class Line:
    def __init__(self, p0_, p1_):
        self.p0 = p0_
        self.p1 = p1_

    def __str__(self):
        return "Line("+repr(self.p0)+","+repr(self.p1)+")"
    def __repr__(self):
        return "Line("+repr(self.p0)+","+repr(self.p1)+")"

class Triangle:
    def __init__(self, p0_, p1_, p2_, norm_):
        self.p0 = p0_
        self.p1 = p1_
        self.p2 = p2_
        self.norm = norm_
        self.xmin = min(p0_.x, p1_.x, p2_.x)
        self.xmax = max(p0_.x, p1_.x, p2_.x)
        self.ymin = min(p0_.y, p1_.y, p2_.y)
        self.ymax = max(p0_.y, p1_.y, p2_.y)
        self.zmin = min(p0_.z, p1_.z, p2_.z)
        self.zmax = max(p0_.z, p1_.z, p2_.z)


    def __iter__(self):
        self.index = 3
        return self
    def __next__(self):
        if self.index == 0:
            raise StopIteration
        if self.index == 3:
            self.index = self.index - 1
            return self.p0
        if self.index == 2:
            self.index = self.index - 1
            return self.p1
        if self.index == 1:
            self.index = self.index - 1
            return self.p2
    def __eq__(self, other):
        if self.p0 == other.p0 and \
            self.p1 == other.p1 and \
            self.p2 == other.p2:
                return True
        else:
            return False
    def __str__(self):
        return "Triangle("+repr(self.p0)+","+repr(self.p1)+","+repr(self.p2)+")"
    def __repr__(self):
        return "Triangle("+repr(self.p0)+","+repr(self.p1)+","+repr(self.p2)+")"

def fileToTriangles(filename, base):
    with open(filename, 'r') as f:
        next(f)
        counter = 0
        triangles = list()
        points = list()
        for line in f:
            l_ = line.split(" ")
            l = [value for value in l_ if value != '']
            if counter == 6:
                counter = 0
                continue
            elif counter == 0:
                if l[0] == 'endsolid':
                    break
                points.insert(0, Point(Decimal(str(l[2])), Decimal(str(l[3])), Decimal(str(l[4][:-1]))))
            elif counter == 2:
                points.insert(0, Point(Decimal(str(l[1])), Decimal(str(l[2])), Decimal(str(l[3]))))
            elif counter == 3:
                points.insert(0, Point(Decimal(str(l[1])), Decimal(str(l[2])), Decimal(str(l[3]))))
            elif counter == 4:
                points.insert(0, Point(Decimal(str(l[1])), Decimal(str(l[2])), Decimal(str(l[3]))))
            counter += 1

        while points:
            triangles.insert(0, Triangle(round_point(points[2], base), \
                                        round_point(points[1], base), \
                                        round_point(points[0], base), \
                                        round_point(points[3], base)))
            points = points[4:]

        # Cleanup any triangles with coincident vertices
        for triangle in triangles:
            if triangle.p0 == triangle.p1 or triangle.p1 == triangle.p2 or triangle.p0 == triangle.p2:
                triangles.remove(triangle)

    return triangles

def round_point(point, base):
    return Point(round_value(point.x, base), round_value(point.y, base), round_value(point.z, base))

def round_value(x, base):
    return x.quantize(base)

def incremental_slicing(n, triangles, k, planes, base):
    # Split the triangle list
    triangle_list = build_triangle_lists(n, triangles, k, planes)

    # Perform a plane sweep
    active_triangles = list()
    segment_lists = list()

    #debug plotting
    fig = plt.figure()
    ax = fig.add_subplot(projection='3d')
    #ax.set_xlim([-10, 10])
    #ax.set_ylim([-10, 10])
    #ax.set_zlim([-10, 10])
    plt.ion()
    plt.show()
    for i in range(0,k):

        active_triangles_carry_forward = list()

        if len(triangle_list[i]) > 0:
            active_triangles.extend(triangle_list[i])
            #fig2 = plt.figure()
            #ax2 = fig2.add_subplot(projection='3d')
            ##ax2.set_xlim([-10, 10])
            ##ax2.set_ylim([-10, 10])
            ##ax2.set_zlim([-10, 10])
            #ax2.set_title('Active triangle list '+str(i))
            #for triangle in active_triangles:
            #    plot_trisurf(triangle, ax2)
            #plt.show()

        segment_lists.append(list())

        for triangle in active_triangles:
            if triangle.xmax > planes[i]:
                active_triangles_carry_forward.append(triangle)
                intersection = compute_intersection(triangle, planes[i], base)
                segment_lists[i].append(intersection)

                #plot_trisurf(triangle, ax)
                #fig.canvas.draw()
                #fig.canvas.flush_events()
                #plot_intersection(intersection, ax)
                #fig.canvas.draw()
                #fig.canvas.flush_events()
                #time.sleep(0.1)

        if len(active_triangles_carry_forward) > 0:
            active_triangles = copy.deepcopy(active_triangles_carry_forward)
        else:
            active_triangles = []

    return segment_lists

def build_triangle_lists(n, triangles, k, planes):
    triangle_list = list()
    for j in range(0,k+1):
        triangle_list.append(list())
    for triangle in triangles:
        i = binary_search(k,planes,triangle)
        triangle_list[i] += [triangle]
    return triangle_list

def print_triangle_list(triangle_list, planes):
    count = 0
    for t_list in triangle_list:
        if count < len(planes):
            print("======== List "+str(count)+" / Less than: "+str(planes[count])+" ============")
        else:
            print("======== List "+str(count)+" / More than: "+str(planes[count-1])+" ============")
        for triangle in t_list:
            print("Triangle xmin: "+str(triangle.xmin))
        count+=1

def binary_search(k, planes, triangle):
    if triangle.xmin > planes[-1]:
        return k
    left = 0 # lowest region
    right = k # highest region
    while left < right:
        mid = math.floor((left+right)/2)
        if triangle.xmin == planes[mid]:
            return mid
        elif triangle.xmin < planes[mid]:
            right = mid
        else:
            left = mid + 1
    return left

def compute_intersection(triangle, plane_x_coord, base):
    L0 = Line(triangle.p0, triangle.p1)
    L1 = Line(triangle.p1, triangle.p2)
    L2 = Line(triangle.p2, triangle.p0)

    lines = [L0, L1, L2]

    plane_normal = Point(1,0,0)
    plane_coord = Point(plane_x_coord, 0, 0)

    Q = list()

    for line in lines:
        u = Point(line.p1.x - line.p0.x, line.p1.y - line.p0.y, line.p1.z - line.p0.z)
        dot = plane_normal.dotProduct(u)

        if abs(dot) > 0.00001:
            w = Point(line.p0.x - plane_coord.x, line.p0.y - plane_coord.y, line.p0.z - plane_coord.z)
            fac = -(plane_normal.dotProduct(w))/ dot
            if (fac > 0.0) and (fac < 1.0):
                u = Point(u.x*fac, u.y*fac, u.z*fac)
                intersection = Point(line.p0.x + u.x, line.p0.y + u.y, line.p0.z + u.z)
                intersection = round_point(intersection, base)
                Q.append(intersection)
    return Q

def plot_triangle(triangle, ax, colour):
    x = []
    y = []
    z = []
    for point in triangle:
        x.append(float(point.x))
        y.append(float(point.y))
        z.append(float(point.z))
    x.append(x[0])
    y.append(y[0])
    z.append(z[0])
    ax.plot(x,y,z, colour, linewidth=0.2)
    ax.set_aspect('equal')

def plot_trisurf(triangle, ax):
    X = []
    Y = []
    Z = []
    for point in triangle:
        X.append(float(point.x))
        Y.append(float(point.y))
        Z.append(float(point.z))
    ax.plot_trisurf(X,Y,Z)
    ax.set_aspect('equal')

def plot_contour(contour, ax, colour):
    x = []
    y = []
    z = []
    for point in contour:
        x.append(float(point.x))
        y.append(float(point.y))
        z.append(float(point.z))
    #Close loop
    x.append(x[0])
    y.append(y[0])
    z.append(z[0])
    ax.plot(x,y,z, colour)

def plot_intersection(segment, ax):
    x = []
    y = []
    z = []
    if len(segment) == 1:
        point = segment
        x.append(float(point.x))
        y.append(float(point.y))
        z.append(float(point.z))
    else:
        for point in segment:
            x.append(float(point.x))
            y.append(float(point.y))
            z.append(float(point.z))
    ax.plot(x,y,z, 'r')

def contour_construction(segment_lists):
    # Insert segments into the hash table
    H = {}
    for plane_segments in segment_lists:
        if len(plane_segments) > 0:
            for segment in plane_segments:
                insert_hash(H, segment[0], segment[1])
                insert_hash(H, segment[1], segment[0])

    # Build the closed polygons
    C = []
    r = 0

    while len(H.keys()) > 0:
        p1 = choose_key(H)
        if len(H[p1]) == 2:
            p2, last = H.pop(p1)
            points = []
            points.append(p1)
            j = 1
            points.append(p2)
            while points[j] != last:
                u, v = H.pop(points[j])
                if u == points[j-1]:
                    points.append(v)
                else:
                    points.append(u)
                j += 1
            H.pop(last)
        else:
            p2 = H.pop(p1)
            points = []
            points.append(p1)
            points.append(p2)
            H.pop(p2)
        C.append(points)
        r += 1
    return C


def insert_hash(H, p1, p2):
    if p1 in H:
        temp = H[p1]
        H[p1] = [temp, p2]
    else:
        H[p1] = p2

def choose_key(H):
    random_key = random.choice(list(H.keys()))
    return random_key

def main():
    filename = sys.argv[1]
    number_of_planes = int(sys.argv[2])
    decimal_precision = int(sys.argv[3])

    # Set decimal precision
    base = Decimal("0E-"+str(decimal_precision))

    triangles = fileToTriangles(filename, base)

    fig = plt.figure()
    ax = fig.add_subplot(projection="3d")

    # Triangle mesh plotting
    for triangle in triangles:
        plot_triangle(triangle, ax, 'b')
    ax.set_title('Sphere mesh')
    plt.show()

    obj_min = triangles[0].xmin
    obj_max = triangles[0].xmax
    for t in triangles:
        if t.xmin < obj_min:
            obj_min = t.xmin
        if t.xmax > obj_max:
            obj_max = t.xmax

    n = len(triangles)
    k = number_of_planes
    offset = Decimal('0.01') # offset from ends of model object
    planes = np.linspace(obj_min+offset, obj_max-offset, num=k)

    segment_lists = incremental_slicing(n, triangles, k, planes, base)

    for seglist in segment_lists:
        for segment in seglist:
            plot_intersection(segment, ax)
    ax.set_aspect('equal')
    plt.show()

    contours = contour_construction(segment_lists)

    # Plotting
    fig = plt.figure()
    ax = fig.add_subplot(projection='3d')
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_zlabel('z')

    for contour in contours:
        plot_contour(contour, ax, 'r')

    ax.set_aspect('equal')
    plt.show()


if __name__ == "__main__":
    main()