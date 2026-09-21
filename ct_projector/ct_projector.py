import numpy as np
import pyvista as pv
from itertools import chain
from pathlib import Path
import SimpleITK as sitk

from common import Line, Plane
import matplotlib.pyplot as plt
from functools import partial
import os
from timeit import timeit
import time
from numba import jit
from multiprocessing import Pool

def lines_parallel_planes_intersection(l_0:np.ndarray, l_dir:np.ndarray, l_length:float, p_0:np.ndarray, p_n:np.ndarray, offsets:np.ndarray):
    """
    Returns the "t" value of the parametric form of the intersections
    """
    dot_prod_down = np.vecdot(l_dir, p_n[None, ...])
    dot_prod_up = np.vecdot(p_0[None, ...] - l_0, p_n[None, ...])[..., None]

    # TODO : Handle special case of parallel lines or not-touching lines
    #if np.isclose(dot_prod_down, 0.0):
    #    if np.isclose(dot_prod_up, 0.0):
    #        return l_0
    #    return None

    with np.errstate(divide='ignore', invalid='ignore'):
        return (dot_prod_up + offsets[None, ...]) / (dot_prod_down[..., None] * l_length)


def line_plane_intersection(line:Line, plane:Plane):
    dot_prod_down = np.dot(line.direction, plane.normal)
    dot_prod_up = np.dot(plane.x0 - line.x0, plane.normal)

    if np.isclose(dot_prod_down, 0.0):
        if np.isclose(dot_prod_up, 0.0):
            # The whole line intersects with the plane
            return line.x0

        # The line is parallel to the plane
        return None
    d = dot_prod_up / dot_prod_down
    return line.x0 + line.direction * d

    
def matrix_transform(matrix:np.ndarray, coordinates:np.ndarray) -> np.ndarray:
    h_coords = np.hstack((coordinates, np.ones((len(coordinates), 1))))
    h_coords = h_coords @ matrix
    h_coords /= h_coords[:, 3]
    return h_coords[:, :3]

def ijk_to_xyz_matrix(img:sitk.Image) -> np.ndarray:
    return img.GetOrigin() + (img.GetSpacing() @ np.eye(4)) @ (img.GetDirection())


def define_intersection_planes(img:sitk.Image) -> list[Plane]:
    directions = np.reshape(img.GetDirection(), (3,3))

    first_point = np.array(img.TransformContinuousIndexToPhysicalPoint([-0.5,-0.5,-0.5]))
    planes = []
    for i in range(3):
        planes.append(Plane(first_point, directions[:, i]))

    return planes

def parametric_norm(v, line_length:float):
    """
    alpha 1 and 2 can be subtracted and multiplied by line length I guess
    """

def main():
    reader = sitk.ImageFileReader()
    reader.SetFileName(Path(__file__).parent / "CTPelvic1K_dataset6_data/dataset6_CLINIC_0001_data.nii.gz")
    img:sitk.Image = reader.Execute()

    size = img.GetSize()
    spacing = img.GetSpacing()
    origin = img.GetOrigin()

    directions = np.reshape(img.GetDirection(), (3,3))

    # Here we'll define the viewport coordinates frame
    projection_center = np.array(img.TransformContinuousIndexToPhysicalPoint([size[0]/2,0,size[2]/2])) + [0, -size[1], 0]
    projection_direction = directions[1]
    desired_pixel_size_mm = 0.7
    image_plane_position = projection_center.copy()
    image_plane_position[1] += 1000
    height = 256
    width = 256

    view_window:pv.PolyData = pv.Plane(center=image_plane_position, direction=projection_direction, i_size=512, j_size=512, i_resolution=height, j_resolution=width)
    
    planes = define_intersection_planes(img)    

    pixel_centers = view_window.cell_centers().points

    # Plot the whole thing
    # grid = pv.ImageData()  # Use pv.UniformGrid() if using an older PyVista version
    # grid.dimensions = size
    # grid.spacing = spacing
    # grid.origin = origin
    # grid.point_data["Scalars"] = sitk.GetArrayFromImage(img).flatten()

    # pl = pv.Plotter()
    # pl.add_mesh(view_window, show_edges=True)
    # pl.add_volume(grid, scalars="Scalars", cmap='bone', opacity=0.01, show_scalar_bar=False)

    # colors = ["red", "green", "blue"]
    # for i,p in enumerate(planes):
    #     pl.add_mesh(pv.Plane(center=p.x0, direction=p.normal, i_size=1000, j_size=1000), opacity=0.3, color=colors[i])

    # for i, plane in enumerate(planes):
    #     offsets2 = [plane.x0 + plane.normal * (img.GetSpacing()[i] * j) for j in range(img.GetSize()[i])]
    #     pl.add_points(np.array(offsets2), color=colors[i], point_size=6)
    # pl.add_axes(line_width=5)

    begin = time.time()

    offsets = [
        [img.GetSpacing()[i] * j for j in range(img.GetSize()[i] + 1)]
        for i in range(3)
    ]
    rays_count = height * width
    alpha_mins = np.zeros((rays_count), dtype=float)
    alpha_maxs = np.ones((rays_count), dtype=float)

    starting_indices = np.cumsum([0] + [len(o) for o in offsets])[:-1]
    ending_indices = starting_indices + [len(o) - 1 for o in offsets]
    all_intersects = None
    line_origins = projection_center[None, ...]
    line_directions = pixel_centers - line_origins#
    line_lengths = np.linalg.norm(line_directions, axis=1, keepdims=True)
    line_directions /= line_lengths
    for i, plane in enumerate(planes):
        plane_offsets = offsets[i]
        intersects_t = lines_parallel_planes_intersection(line_origins, line_directions, line_lengths, plane.x0, plane.normal, np.array(plane_offsets))
        all_intersects = np.hstack((all_intersects, intersects_t)) if all_intersects is not None else intersects_t

    alpha_mins = np.clip(np.max(np.minimum(all_intersects[:, starting_indices], all_intersects[:, ending_indices]), axis=1), a_min=0.0, a_max=None)[..., None]
    alpha_maxs = np.clip(np.min(np.maximum(all_intersects[:, starting_indices], all_intersects[:, ending_indices]), axis=1), a_min=None, a_max=1.0)[..., None]

    mask = np.logical_and(alpha_mins <= all_intersects, all_intersects <= alpha_maxs)
    valid_intersects = []
    image = np.zeros((width, height), dtype=float)
    np_img = sitk.GetArrayFromImage(img)
    np_img = (np_img - np_img.min()) / (np.ptp(np_img))
    
    for line_id in range(len(mask)):
        #line = projection_lines[line_id]
        #self.x0[None, ...] + self.vector * ts[..., None]
        alphas = np.sort(all_intersects[line_id, mask[line_id]])
        line_start_end = projection_center + line_directions[line_id] * line_lengths[line_id] * np.vstack((alphas[0], alphas[-1]))

        begin_ijk = np.array(img.TransformPhysicalPointToContinuousIndex(line_start_end[0] + line_directions[line_id]*1e-8))[None, ...]
        end_ijk = np.array(img.TransformPhysicalPointToContinuousIndex(line_start_end[1] + line_directions[line_id]*1e-8))[None, ...]
        spread_ijk = end_ijk - begin_ijk
        rhos = np.rint(begin_ijk + spread_ijk * alphas[..., None])[:-1].astype(int)
        rhos_z, rhos_y, rhos_x = rhos[:,0], rhos[:,1], rhos[:, 2]
        ls = np.abs(((alphas[1:] - alphas[:-1]) * line_lengths[line_id])[..., None])

        image[line_id % width, line_id // height] = np.sum(np_img[rhos_x, rhos_y, rhos_z] * ls)
    image /= image.max()
    plt.imshow(image, cmap="gray")
    plt.show()


    # pl.add_points(np.array(list(chain.from_iterable(valid_intersects))), color="purple", point_size=5)
    #for proj_line in projection_lines:
        #pl.add_lines(np.array([proj_line.x0, line_plane_intersection(proj_line, Plane(image_plane_position, projection_direction))]), color="green")
        
    #    pts = find_intersections(img, planes, proj_line)

        #pl.add_points(pts, color="purple", point_size=6)
        # intersection_per_line = []
        # for p_i in planes:
        #     for plane in p_i:
        #         intersection_per_line.append(line_plane_intersection(proj_line, plane))
        # pl.add_points(np.array(intersection_per_line), point_size=4, color="blue")
    
    #pl.add_points(projection_center[None, ...], point_size=20, color="red")
    #pl.add_points(pixel_centers, point_size=8, color="purple")
    #pl.show()


if __name__ == "__main__":
    main()
    #print(timeit(main, number=10) / 10)
    