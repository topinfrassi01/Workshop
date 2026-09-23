import numpy as np
import pyvista as pv
from pathlib import Path
import SimpleITK as sitk

from common import Line, Plane
import matplotlib.pyplot as plt
import time


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
    h_coords = (matrix @ h_coords.T).T
    h_coords /= h_coords[:, 3:]
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


def phys_to_continuous_index_matrix(img:sitk.Image) -> np.ndarray:
    origin = np.array(img.GetOrigin())
    spacing = np.array(img.GetSpacing())
    directions = np.reshape(img.GetDirection(), (3,3)).T
    
    inv_scaling = np.diag(1.0 / spacing)
    R = inv_scaling @ directions

    # -R is just the inverse of R
    t = -R @ origin
    
    M = np.eye(4)
    M[:3, :3] = R
    M[:3, 3] = t
    
    return M


def main(axes):
    reader = sitk.ImageFileReader()
    reader.SetFileName(Path(__file__).parent / "CTPelvic1K_dataset6_data/dataset6_CLINIC_0001_data.nii.gz")
    img:sitk.Image = reader.Execute()

    size = img.GetSize()
    directions = np.reshape(img.GetDirection(), (3,3))
    phys_to_cont_idx_matrix = phys_to_continuous_index_matrix(img)

    # Here we'll define the viewport coordinates frame
    projection_center = np.array(img.TransformContinuousIndexToPhysicalPoint([size[0]/2,0,size[2]/2])) + [0, -size[1], 0]
    projection_direction = directions[1]
    desired_pixel_size_mm = 0.7
    image_plane_position = projection_center.copy()
    image_plane_position[1] += 1000
    height = 128
    width = 128

    view_window:pv.PolyData = pv.Plane(center=image_plane_position, direction=projection_direction, i_size=256, j_size=256, i_resolution=height, j_resolution=width)
    
    planes = define_intersection_planes(img)    

    pixel_centers = view_window.cell_centers().points

    # Plot the whole thing
    grid = pv.ImageData()
    grid.dimensions = size
    grid.spacing = img.GetSpacing()
    grid.origin = img.GetOrigin()
    grid.point_data["Scalars"] = sitk.GetArrayFromImage(img).flatten()

    pl = pv.Plotter()
    pl.add_mesh(view_window, show_edges=True)
    pl.add_volume(grid, scalars="Scalars", cmap='bone', opacity=0.01, show_scalar_bar=False)

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

    all_intersects:np.ndarray = np.clip(all_intersects, a_min=alpha_mins, a_max=alpha_maxs)
    all_intersects.sort(axis=1, kind="M")
    np_img = sitk.GetArrayFromImage(img)
    np_img = (np_img - np_img.min()) / (np.ptp(np_img))
    line_start_ends = np.vstack((projection_center[None, ...] + line_directions * line_lengths * all_intersects[None, :,0:1], projection_center[None, ...] + line_directions * line_lengths * all_intersects[None, :,-1:]))# * np.vstack((all_intersects[:,0], all_intersects[:,-1]))[None, ...]
    #pl.add_points(line_start_ends[0], color="blue")
    #pl.add_points(line_start_ends[1], color="red")
    #pl.show()
    
    begins_ijk = matrix_transform(phys_to_cont_idx_matrix, line_start_ends[0] + line_directions*1e-8)
    #begins_ijk = np.array([img.TransformPhysicalPointToContinuousIndex(line_start_ends[0, i] + line_directions[i]*1e-8) for i in range(len(line_start_ends[0]))])
    ends_ijk = matrix_transform(phys_to_cont_idx_matrix, line_start_ends[1] + line_directions*1e-8)

    # this works but should be unit tested
    # begins_xyz = np.array([img.TransformContinuousIndexToPhysicalPoint(p) for p in begins_ijk])
    # ends_xyz = np.array([img.TransformContinuousIndexToPhysicalPoint(p) for p in ends_ijk])
    # pl.add_points(begins_xyz, color="blue")
    # pl.add_points(ends_xyz, color="red")
    # pl.show()

    spread_ijk = ends_ijk - begins_ijk
    rhos = np.rint(begins_ijk[:, None, :] + spread_ijk[:, None, :] * all_intersects[..., None])[:,:-1,:].astype(int)
    ls = (np.abs(np.diff(all_intersects, axis=1)) * line_lengths).astype(np.float64)
    # image = np.zeros((width, height), dtype=float)
    # all_sums = []
    # all_ls = []
    # for line_id in range(len(all_intersects)):
    #     rhos_z, rhos_y, rhos_x = rhos[line_id,:,0], rhos[line_id,:,1], rhos[line_id,:, 2]
    #     ls_l = ls[line_id]
    #     nonzero_indices = np.nonzero(ls_l)
    #     all_ls += ls_l[nonzero_indices].flatten().tolist()
    #     all_sums += np_img[rhos_x[nonzero_indices], rhos_y[nonzero_indices], rhos_z[nonzero_indices]].flatten().tolist()

    #     image[line_id % width, line_id // height] = np.sum(np_img[rhos_x[nonzero_indices], rhos_y[nonzero_indices], rhos_z[nonzero_indices]] * ls_l[nonzero_indices], dtype=np.float64)
    
    #ax.hist(all_sums)   
    traversed_voxels = np_img[rhos[:,:,2], rhos[:,:,1], rhos[:,:,0]]
    image = np.sum(traversed_voxels * ls, axis=-1, dtype=np.float64).reshape(height, width).T
    
    #ax.hist(image.flatten())

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

def old_main(axes):
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
    height = 128
    width = 128

    view_window:pv.PolyData = pv.Plane(center=image_plane_position, direction=projection_direction, i_size=256, j_size=256, i_resolution=height, j_resolution=width)
    
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

    all_intersects:np.ndarray = np.clip(all_intersects, a_min=alpha_mins, a_max=alpha_maxs)
    all_intersects.sort(axis=1, kind="M")

    #mask = np.logical_and(alpha_mins <= all_intersects, all_intersects <= alpha_maxs)
    valid_intersects = []
    image = np.zeros((width, height), dtype=float)
    np_img = sitk.GetArrayFromImage(img)
    np_img = (np_img - np_img.min()) / (np.ptp(np_img))
    
    
    all_intersects:np.ndarray = np.clip(all_intersects, a_min=alpha_mins, a_max=alpha_maxs)

    all_intersects.sort(axis=1, kind="M")
    all_sums = []
    all_ls = []
    for line_id in range(len(all_intersects)):
        #line = projection_lines[line_id]
        #self.x0[None, ...] + self.vector * ts[..., None]
        alphas_start = int(np.searchsorted(all_intersects[line_id], alpha_mins[line_id], side="right")[0])
        alphas_end = int(np.searchsorted(all_intersects[line_id], alpha_maxs[line_id], side="left")[0])
        alphas = all_intersects[line_id, alphas_start:alphas_end]
        line_start_end = projection_center + line_directions[line_id] * line_lengths[line_id] * np.vstack((alphas[0], alphas[-1]))

        begin_ijk = np.array(img.TransformPhysicalPointToContinuousIndex(line_start_end[0] + line_directions[line_id]*1e-8))[None, ...]
        end_ijk = np.array(img.TransformPhysicalPointToContinuousIndex(line_start_end[1] + line_directions[line_id]*1e-8))[None, ...]
        spread_ijk = end_ijk - begin_ijk
        rhos = np.rint(begin_ijk + spread_ijk * alphas[..., None])[:-1].astype(int)
        rhos_z, rhos_y, rhos_x = rhos[:,0], rhos[:,1], rhos[:, 2]
        ls = np.abs(((alphas[1:] - alphas[:-1]) * line_lengths[line_id])[..., None]).astype(np.float64)
        all_sums += np_img[rhos_x, rhos_y, rhos_z].flatten().tolist()
        all_ls += ls.flatten().tolist()

        image[line_id % width, line_id // height] = np.sum(np_img[rhos_x, rhos_y, rhos_z] * ls, dtype=np.float64)

    axes[0].hist(all_ls)
    axes[1].hist(all_sums)
    axes[2].imshow(image, cmap="gray")
    #ax.hist(all_sums)
    #ax.hist(image.flatten())
    #ax.imshow(image, cmap="gray")


if __name__ == "__main__":
    fig, ax = plt.subplots(2,3)
    old_main(ax[1])
    main(ax[0])
    plt.show()
    #print(timeit(main, number=10) / 10)
