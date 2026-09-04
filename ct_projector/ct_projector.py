import numpy as np
import pyvista as pv
from itertools import chain
from pathlib import Path
import SimpleITK as sitk

from common import Line, Plane
from functools import partial
import os
#from timeit import timeit
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

    return (dot_prod_up + offsets[None, ...]) / (dot_prod_down * l_length)[..., None]
    #return [((dot_prod_up + o) / dot_prod_down) / line.length for o in offsets]


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

def ijk_to_xyz(img:sitk.Image, ijk_coordinates:np.ndarray) -> np.ndarray:
    return matrix_transform(img.GetOrigin() + (img.GetSpacing() @ np.eye(4)) @ (img.GetDirection()), ijk_coordinates)

def define_intersection_planes(img:sitk.Image) -> list[Plane]:
    directions = np.reshape(img.GetDirection(), (3,3))

    first_point = np.array(img.TransformContinuousIndexToPhysicalPoint([-0.5,-0.5,-0.5]))
    planes = []
    for i in range(3):
        planes.append(Plane(first_point, directions[:, i]))

    return planes

def main():
    reader = sitk.ImageFileReader()
    reader.SetFileName(Path(__file__).parent / "CTPelvic1K_dataset6_data/dataset6_CLINIC_0001_data.nii.gz")
    #reader.LoadPrivateTagsOn()
    #reader.ReadImageInformation()
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
    view_window:pv.PolyData = pv.Plane(center=image_plane_position, direction=projection_direction, i_size=512, j_size=512, i_resolution=256, j_resolution=256)

    planes = define_intersection_planes(img)

    pixel_centers = view_window.cell_centers().points
    projection_lines = [Line(projection_center, p) for p in pixel_centers]

    # intersection_per_lines = []
    # for proj_line in projection_lines:
    #     intersection_per_line = []
    #     for p_i in planes:
    #         for plane in p_i:
    #             intersection_per_line.append(line_plane_intersection(proj_line, plane))
    #     intersection_per_lines.append(intersection_per_line)

    # Plot the whole thing
    grid = pv.ImageData()  # Use pv.UniformGrid() if using an older PyVista version
    grid.dimensions = size
    grid.spacing = spacing
    grid.origin = origin
    grid.point_data["Scalars"] = sitk.GetArrayFromImage(img).flatten()

    pl = pv.Plotter()
    pl.add_mesh(view_window, show_edges=True)
    pl.add_volume(grid, scalars="Scalars", cmap='bone', opacity=0.1, show_scalar_bar=False)

    colors = ["red", "green", "blue"]
    for i,p in enumerate(planes):
        pl.add_mesh(pv.Plane(center=p.x0, direction=p.normal, i_size=1000, j_size=1000), opacity=0.3, color=colors[i])

    for i, plane in enumerate(planes):
        offsets2 = [plane.x0 + plane.normal * (img.GetSpacing()[i] * j) for j in range(img.GetSize()[i])]
        pl.add_points(np.array(offsets2), color=colors[i], point_size=6)
    pl.add_axes(line_width=5)

    begin = time.time()

    # prepared_func = partial(find_intersections, planes=planes, img=img)
    # with Pool(4) as pool:
    #     pool.map(prepared_func, projection_lines)
    offsets = [
        [img.GetSpacing()[i] * j for j in range(img.GetSize()[i])]
        for i in range(3)
    ]
    line_intersections = find_intersections(projection_lines, planes, offsets)
    print(time.time() - begin)
    pl.add_points(np.array(list(chain.from_iterable(line_intersections)))[::1000], color="purple", size=3)
    #for proj_line in projection_lines:
        #pl.add_lines(np.array([proj_line.x0, line_plane_intersection(proj_line, Plane(image_plane_position, projection_direction))]), color="green")
        
    #    pts = find_intersections(img, planes, proj_line)

        #pl.add_points(pts, color="purple", point_size=6)
        # intersection_per_line = []
        # for p_i in planes:
        #     for plane in p_i:
        #         intersection_per_line.append(line_plane_intersection(proj_line, plane))
        # pl.add_points(np.array(intersection_per_line), point_size=4, color="blue")
    
    pl.add_points(projection_center[None, ...], point_size=20, color="red")
    #pl.add_points(pixel_centers, point_size=8, color="purple")
    pl.show()

def find_intersections(proj_lines:list[Line], planes:list[Plane], offsets:np.ndarray):
    alpha_mins = np.zeros((len(proj_lines)), dtype=float)
    alpha_maxs = np.ones((len(proj_lines)), dtype=float)

    starting_indices = np.cumsum([0] + [len(o) for o in offsets])[:-1]
    ending_indices = starting_indices + [len(o) - 1 for o in offsets]
    all_intersects = None#np.zeros((sum(proj_lines), sum(len(pl) for pl in planes)), dtype=float)
    for i, plane in enumerate(planes):
        plane_offsets = offsets[i]
        line_origins = np.array([l.x0 for l in proj_lines])
        line_directions = np.array([l.direction for l in proj_lines])
        line_lengths = np.array([l.length for l in proj_lines])
        intersects_t = lines_parallel_planes_intersection(line_origins, line_directions, line_lengths, plane.x0, plane.normal, np.array(plane_offsets))
        all_intersects = np.hstack((all_intersects, intersects_t)) if all_intersects is not None else intersects_t

        #all_interesects += intersects_t.tolist()
        # TODO : Could be improved by computing once at the end, using the image size to get first and last of each plane
        #alpha_mins = np.maximum(alpha_mins, np.minimum(intersects_t[:, 0], intersects_t[:, -1]))
        #alpha_maxs = np.minimum(alpha_maxs, np.maximum(intersects_t[:, 0], intersects_t[:, -1]))
        #alpha_min = max(alpha_min, min(intersects_t[0], intersects_t[-1]))
        #alpha_max = min(alpha_max, max(intersects_t[0], intersects_t[-1]))

    
    alpha_mins = np.clip(np.max(np.minimum(all_intersects[:, starting_indices], all_intersects[:, ending_indices]), axis=1), a_min=0.0, a_max=None)[..., None]
    alpha_maxs = np.clip(np.min(np.maximum(all_intersects[:, starting_indices], all_intersects[:, ending_indices]), axis=1), a_min=None, a_max=1.0)[..., None]

    mask = np.logical_and(alpha_mins <= all_intersects, all_intersects <= alpha_maxs)
    valid_intersects = []
    for line_id in range(len(mask)):
        alphas = np.sort(all_intersects[line_id, mask[line_id]])
        valid_intersects.append(proj_lines[line_id].middle_points(alphas))
    return valid_intersects
    #all_intersects = [
    #    sorted([a for a in all_intersects[i] if a <= alpha_maxs[i] and a >= alpha_mins[i]])
    #    for i in range(len(all_intersects))
    #]
    pass
    print(len(all_intersects))
    print(len(all_intersects[0]))
    #all_interesects = sorted([a for a in all_interesects if a <= alpha_max and a >= alpha_min])

    #return proj_line.middle_points(all_interesects)
    #start_idx = img.TransformContinuousIndexToPhysicalPoint((-0.5, -0.5, -0.5))


if __name__ == "__main__":
    main()
    