import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

def generate_horizontal_surface(x_range, y_range, num_points):
    x = np.random.uniform(x_range[0], x_range[1], num_points)
    y = np.random.uniform(y_range[0], y_range[1], num_points)
    z = np.full(num_points, -10)
    return x, y, z

def generate_vertical_surface(x_range, z_range, num_points):
    x = np.random.uniform(x_range[0], x_range[1], num_points)
    z = np.random.uniform(z_range[0], z_range[1], num_points)
    y = np.full(num_points, -10)
    return x, y, z

def generate_cylindrical_surface(radius, height_range, num_points):
    theta = np.random.uniform(0, 2 * np.pi, num_points)
    z = np.random.uniform(height_range[0], height_range[1], num_points)
    x = radius * np.cos(theta)
    y = radius * np.sin(theta)
    return x, y, z

def save_to_xyz(filename, x, y, z):
    with open(filename, 'w') as f:
        for i in range(len(x)):
            f.write(f"{x[i]} {y[i]} {z[i]}\n")

def plot_combined_point_cloud(xyz_list, title):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    for x, y, z in xyz_list:
        ax.scatter(x, y, z, s=1)
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.set_title(title)
    plt.show()

x1, y1, z1 = generate_horizontal_surface((-10, 10), (-10, 10), 1000)
save_to_xyz("horizontal_surface.xyz", x1, y1, z1)

x2, y2, z2 = generate_vertical_surface((-10, 10), (-10, 10), 1000)
save_to_xyz("vertical_surface.xyz", x2, y2, z2)

x3, y3, z3 = generate_cylindrical_surface(5, (-10, 10), 1000)
save_to_xyz("cylindrical_surface.xyz", x3, y3, z3)

plot_combined_point_cloud([(x1, y1, z1), (x2, y2, z2), (x3, y3, z3)], "Połączone powierzchnie")
