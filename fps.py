import numpy as np
import laspy
from tqdm import tqdm

def read_las(path):
    """Opens 

    Parameters
    ----------
    x: numpy.array
        Independent dimension
    c: float
        Catenary parameter
    a: float
        x translation
    b: float
        y translation

    Returns
    -------
    numpy.array
        Dependent (y-dimension) data
    """

    # Open file
    las = laspy.read(path)
    point_records = las.points.copy()
    scale = las._points.scales[0]

    # Preprocess
    x = scale * np.array(point_records['X'])
    y = scale * np.array(point_records['Y'])
    z = scale * np.array(point_records['Z'])
    points = np.vstack((x, y, z))

    return points

def fps(points, n_samples):
    """
    points: [N, 3] array containing the whole point cloud
    n_samples: samples you want in the sampled point cloud typically << N 
    """
    points = np.array(points)
    
    # Represent the points by their indices in points
    points_left = np.arange(len(points)) # [P]

    # Initialise an array for the sampled indices
    sample_inds = np.zeros(n_samples, dtype='int') # [S]

    # Initialise distances to inf
    dists = np.ones_like(points_left) * float('inf') # [P]

    # Select a point from points by its index, save it
    selected = 0
    sample_inds[0] = points_left[selected]

    # Delete selected 
    points_left = np.delete(points_left, selected) # [P - 1]

    # Iteratively select points for a maximum of n_samples
    for i in tqdm(range(1, n_samples)):
        # Find the distance to the last added point in selected
        # and all the others
        last_added = sample_inds[i-1]
        
        dist_to_last_added_point = ((points[last_added] - points[points_left])**2).sum(-1) # [P - i]

        # If closer, updated distances
        dists[points_left] = np.minimum(dist_to_last_added_point, dists[points_left]) # [P - i]

        # We want to pick the one that has the largest nearest neighbour
        # distance to the sampled points
        selected = np.argmax(dists[points_left])
        sample_inds[i] = points_left[selected]

        # Update points_left
        points_left = np.delete(points_left, selected)

    return points[sample_inds]

if __name__ == '__main__':
    import matplotlib.pyplot as plt
    pc = np.loadtxt('bunnyData.txt')
    # print(pc.shape)
    num_pts = pc.shape[0]
    sampled = fps(pc, int(num_pts/10))

    fig = plt.figure()
    ax1 = fig.add_subplot(121, projection='3d')
    ax2 = fig.add_subplot(122, projection='3d')
    points = pc.T
    sampled_points = sampled.T
    ax1.scatter(points[0], points[1], points[2], s=1)
    ax2.scatter(sampled_points[0], sampled_points[1], sampled_points[2], s=1)
    plt.show()
    