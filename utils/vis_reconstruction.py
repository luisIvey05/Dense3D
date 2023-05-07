import open3d as o3d
import numpy as np
from PIL import Image
import glob
import copy
from open3d_example import draw_geometries_flip

flip_transform = [[1, 0, 0, 0], [0, -1, 0, 0], [0, 0, -1, 0], [0, 0, 0, 1]]


def draw_geometries_flip(pcds):
    pcds_transform = []
    for pcd in pcds:
        pcd_temp = copy.deepcopy(pcd)
        pcd_temp.transform(flip_transform)
        pcds_transform.append(pcd_temp)
    o3d.visualization.draw_geometries(pcds_transform)


# mesh = o3d.io.read_triangle_mesh("./fragment_013.ply")
# print(mesh)
# mesh.compute_vertex_normals()
# o3d.visualization.draw_geometries([mesh])
# pcd = mesh.sample_points_uniformly(number_of_points=500)
# o3d.visualization.draw_geometries([pcd])

# fragment_files = glob.glob("fragments/*.ply")
# print(fragment_files)
# for i in range(0, len(fragment_files)):
#     print(fragment_files[i])
#     pcd = o3d.io.read_point_cloud(fragment_files[i])
#     draw_geometries_flip([pcd])

fragment_file = "fragment_000.ply"
pcd = o3d.io.read_point_cloud(fragment_file)
point_cloud_np = np.asarray(pcd.points)

# Save the point cloud to a .npy file
np.save("point_cloud.npy", point_cloud_np)

#draw_geometries_flip([pcd])
o3d.visualization.draw_geometries([pcd])
