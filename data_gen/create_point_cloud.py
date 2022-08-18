import numpy as np
from unit_shape_gen import *

if __name__ == '__main__':
    cube_pts = gen_unit_cube(1000)
    cube_transform = build_transformation_matrix(0, 2, 0, 4, 2, -1)
    cube_label = np.zeros(cube_pts.shape[1])
    cube_pts = cube_transform @ cube_pts
    cube_pts = label_points(cube_pts, 0)
    sphere_pts = gen_unit_sphere(1000)
    sphere_pts = label_points(sphere_pts, 2)

    pt_cloud = np.concatenate((cube_pts, sphere_pts), axis=1)
    for i in range(4):
        name = 'data/sample_point_cloud_' + str(i) + '.npy'
        np.save(name, pt_cloud)
    # np.save('data/sample_point_cloud_15.npy', pt_cloud)

