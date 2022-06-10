import numpy as np

def build_transformation_matrix(alpha, beta, gamma, t_x, t_y, t_z):
    """Creates 3d transformation matrix 4x4

    Parameters
    ----------
    alpha : float
        Rotation around world z-axis (yaw) [degrees]
    beta : float
        Rotation around world y-axis (pitch) [degrees]
    gamma : float
        Rotation around world x-axis (roll) [degrees]
    t_x : float
        Translation in x-direction
    t_y : float
        Translation in y-direction
    t_z : float
        Translation in z-direction
    """

    transform = np.eye(4)
    alpha = np.radians(alpha)
    beta = np.radians(beta)
    gamma = np.radians(gamma)
    R_z = np.array([[np.cos(alpha), -np.sin(alpha), 0],\
                    [np.sin(alpha),  np.cos(alpha), 0],
                    [0,              0,             1]]) # Yaw

    R_y = np.array([[np.cos(beta),  0, np.sin(beta)],\
                    [0,             1, 0           ],
                    [-np.sin(beta), 0, np.cos(beta)]]) # Pitch

    R_x = np.array([[1, 0,              0,           ],\
                    [0, np.cos(gamma), -np.sin(gamma)],
                    [0, np.sin(gamma),  np.cos(gamma)]]) # Roll

    R = R_z @ R_y @ R_x # Rotation composition

    # Construct camera to world matrix with translation as well
    transform[0:3, 0:3] = R.T
    transform[0:3, 3] = [t_x, t_y, t_z]

    return transform

def gen_unit_sphere(num_pts):
    """ Generates random points on the surface of a unit sphere

    Parameters
    ----------
    num_pts : int
        Number of points to sample

    Returns
    -------
    numpy.ndarray
        Random homogeneous points on surface of unit sphere
    """

    vec = np.random.randn(4, num_pts)
    vec /= np.linalg.norm(vec, axis=0)
    vec[-1, :] /= vec[-1, :] # Rectify w component

    return vec

def gen_unit_cube(num_pts):
    """Generates random points on the surface of a unit cube

    Parameters
    ----------
    num_pts : int
        Number of points to sample (will be rounded to nearest int divisible by 6)

    Returns
    -------
    numpy.ndarray
        Random homogeneous points on surface of unit cube
    """

    # Build transformations (basis plane is xy plane)
    transformations = [
        build_transformation_matrix(0, 0, 0, 0, 0, 0.5),
        build_transformation_matrix(0, 0, 0, 0, 0, -0.5),
        build_transformation_matrix(0, 90, 0, -0.5, 0, 0),
        build_transformation_matrix(0, 90, 0, 0.5, 0, 0),
        build_transformation_matrix(0, 0, 90, 0, -0.5, 0),
        build_transformation_matrix(0, 0, 90, 0, 0.5, 0)
    ]
    num_points_rounded = get_closest_divisible(num_pts, 6) # Must be divided into sides
    untransformed_pts = np.ones((4, num_points_rounded))
    untransformed_pts[0:2, :] = np.random.rand(2, num_points_rounded) - 0.5
    untransformed_pts[2, :] *= 0

    # Rotate points to cube sides
    untransformed_side_pts = np.split(untransformed_pts, 6, axis=1)
    transformed_side_pts = []
    for side, transform in zip(untransformed_side_pts, transformations):
        transformed_side_pts.append(transform @ side)
    transformed_side_pts = np.concatenate(transformed_side_pts, axis=1)

    return transformed_side_pts

def get_closest_divisible(n, m):
    """ Calculates integer closest to n that is divisible by m

    Parameters
    ----------
    n : int
        Target integer
    m : int
        Dividend

    Returns
    -------
    int
        Integer closest to n that is divisible by m
    """

    q = int(n / m)
    n_1 = m * q
    if (n * m) > 0:
        n_2 = m*(q+1)
    else:
        n_2 = m*(q-1)

    if abs(n - n_1) < abs(n - n_2):
        return n_1
    else:
        return n_2

def set_axes_equal(ax):
    '''Make axes of 3D plot have equal scale so that spheres appear as spheres,
    cubes as cubes, etc..  This is one possible solution to Matplotlib's
    ax.set_aspect('equal') and ax.axis('equal') not working for 3D.

    Input
      ax: a matplotlib axis, e.g., as output from plt.gca().
    '''

    x_limits = ax.get_xlim3d()
    y_limits = ax.get_ylim3d()
    z_limits = ax.get_zlim3d()

    x_range = abs(x_limits[1] - x_limits[0])
    x_middle = np.mean(x_limits)
    y_range = abs(y_limits[1] - y_limits[0])
    y_middle = np.mean(y_limits)
    z_range = abs(z_limits[1] - z_limits[0])
    z_middle = np.mean(z_limits)

    # The plot bounding box is a sphere in the sense of the infinity
    # norm, hence I call half the max range the plot radius.
    plot_radius = 0.5*max([x_range, y_range, z_range])

    ax.set_xlim3d([x_middle - plot_radius, x_middle + plot_radius])
    ax.set_ylim3d([y_middle - plot_radius, y_middle + plot_radius])
    ax.set_zlim3d([z_middle - plot_radius, z_middle + plot_radius])

def label_points(points, label):
    """Labels a batch of points with label

    Parameters
    ----------
    points : numpy.ndarray
        Points to be labeled
    label : int
        Label

    Returns
    -------
    numpy.ndarray
        Points in form (x, y, z, w, label) [5xN]
    """

    label_array = label * np.ones(points.shape[1])
    points = np.vstack([points, label_array])

    return points

if __name__ == '__main__':
    import matplotlib.pyplot as plt
    cube_pts = gen_unit_cube(1000)
    cube_transform = build_transformation_matrix(0, 30, 0, 2, 0, 2)
    cube_label = np.zeros(cube_pts.shape[1])
    cube_pts = cube_transform @ cube_pts
    cube_pts = label_points(cube_pts, 0)
    sphere_pts = gen_unit_sphere(1000)
    sphere_pts = label_points(sphere_pts, 1)

    pt_cloud = np.concatenate((cube_pts, sphere_pts), axis=1)
    x, y, z, w, label = pt_cloud
    # fig = plt.figure()
    ax = plt.axes(projection='3d')
    ax.scatter(x, y, z, s=2, c=label, alpha=1.0, cmap='plasma')
    set_axes_equal(ax)
    plt.show()
