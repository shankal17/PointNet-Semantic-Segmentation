import numpy as np
from unit_shape_gen import *

if __name__ == '__main__':
    cube_pts = gen_unit_cube(1000)
    cube_transform = build_transformation_matrix(0, 0, 0, 2, 1, -1)
    cube_label = np.zeros(cube_pts.shape[1])
    cube_pts = cube_transform @ cube_pts
    cube_pts = label_points(cube_pts, 0)
    sphere_pts = gen_unit_sphere(1000)
    sphere_pts = label_points(sphere_pts, 2)

    pt_cloud = np.concatenate((cube_pts, sphere_pts), axis=1)
    np.save('data/sample_point_cloud_3.npy', pt_cloud)

