import cProfile
import pstats
import numpy as np
import pyvista as pv
from pathlib import Path
import SimpleITK as sitk

import matplotlib.pyplot as plt
import time

from common import Image
from algebra import *
from timeit import timeit

from math import pi
import trimesh


class CtProjector:
    def __init__(
        self,
        image:Image,
        *,
        use_float64 = True
    ):
        self._image = image
        self.dtype = np.float64 if use_float64 else np.float32
    
    def _define_uvn_frame(
        self,
        distance_to_detector:float,
        pitch:float = 0.0,
        yaw:float = 0.0,
        roll:float = 0.0
    ):
        image_center = np.eye(4)
        image_center[:3,3] = self._image.xyz_center

        world_frame = np.eye(4)
        world_frame[:3,:3] = self._image.directions

        world_to_uvn = rotation_matrix(pitch=-pi/2)

        translation_to_distance = np.eye(4)
        translation_to_distance[2, 3] = -distance_to_detector
        
        return image_center @ world_to_uvn @ rotation_matrix(pitch=pitch, yaw=yaw, roll=roll) @ translation_to_distance @ world_frame

    def _define_projection_lines(
        self,
        uvn_frame:np.ndarray,
        pixel_resolution:float,
        image_size:tuple[int,int]
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        half_w = image_size[0] // 2
        half_h = image_size[1] // 2
        arr_w = np.arange(-half_w, half_w)
        arr_h = np.arange(-half_h, half_h)

        x_values, y_values = np.meshgrid(arr_w, arr_h)
        pixels_3d = np.vstack([x_values.reshape(-1), y_values.reshape(-1), np.zeros(len(x_values)*len(y_values))]).T
        pixels_3d *= pixel_resolution

        converted = matrix_transform(np.linalg.inv(uvn_frame), self._image.xyz_corners)

        points = unit_vector_projection(np.array([0,0,1], dtype=float), converted)
        dist = np.max(points[:,2])
        
        pixels_3d = matrix_transform(uvn_frame, pixels_3d + [0,0,dist])
        
        line_directions = pixels_3d - uvn_frame[:3, 3]
        line_lengths = np.linalg.norm(line_directions, axis=1, keepdims=True)
        line_directions /= line_lengths

        return pixels_3d, line_directions, line_lengths
        

    def _identify_alphas(self, line_origins, line_directions, line_lengths):
        offsets = [[self._image.spacing[i] * j for j in range(self._image.ct_array.shape[i] + 1)] for i in range(3)]
        starting_indices = np.cumsum([0] + [len(o) for o in offsets])[:-1]
        ending_indices = starting_indices + [len(o) - 1 for o in offsets]
        all_alphas = None
        image_origin, image_planes = self._image.intersection_planes()
        for i, plane_normal in enumerate(image_planes):
            plane_offsets = np.array(offsets[i])
            alphas = lines_parallel_planes_intersection(line_origins, line_directions, line_lengths, image_origin, plane_normal, plane_offsets)
            all_alphas = np.hstack((all_alphas, alphas)) if all_alphas is not None else alphas

        alpha_mins = np.clip(np.max(np.minimum(all_alphas[:, starting_indices], all_alphas[:, ending_indices]), axis=1), a_min=0.0, a_max=None)[..., None]
        alpha_maxs = np.clip(np.min(np.maximum(all_alphas[:, starting_indices], all_alphas[:, ending_indices]), axis=1), a_min=None, a_max=1.0)[..., None]

        all_alphas = np.clip(all_alphas, a_min=alpha_mins, a_max=alpha_maxs)
        all_alphas.sort(axis=1, kind="M")
        return all_alphas

    def _identify_rhos(self, rays_origin, line_directions, line_lengths, all_alphas):
        line_start_ends = (rays_origin[None,...] + line_directions[None,...] * line_lengths[None,...] * all_alphas[:,[0,-1]].T[...,None])
        ijk = matrix_transform(self._image.xyz_to_ijk_matrix, line_start_ends + line_directions*1e-8)
        begins_ijk = ijk[0]
        ends_ijk = ijk[1]
        spread_ijk = ends_ijk - begins_ijk
        return np.rint(begins_ijk[:, None, :] + spread_ijk[:, None, :] * all_alphas[..., None])[:,:-1,:].astype(int)

    def render_xray(
        self,
        distance_to_detector:float,
        pitch:float,
        yaw:float,
        roll:float,
        pixel_resolution:float,
        image_size:tuple[int,int]
    ) -> np.ndarray:
        uvn_frame = self._define_uvn_frame(distance_to_detector, pitch, yaw, roll)
        rays_origin = uvn_frame[:3, 3][None, ...]
        _, line_directions, line_lengths = self._define_projection_lines(uvn_frame, pixel_resolution, image_size)
        
        all_alphas = self._identify_alphas(rays_origin, line_directions, line_lengths)
        ls = (np.abs(np.diff(all_alphas, axis=1)) * line_lengths)

        rhos = self._identify_rhos(rays_origin, line_directions, line_lengths, all_alphas)
        traversed_voxels = self._image.ct_array[rhos[:,:,0], rhos[:,:,1], rhos[:,:,2]]
        image = np.sum(traversed_voxels * ls, axis=-1, dtype=np.float64).reshape(image_size)
        return image
        
def main():
    reader = sitk.ImageFileReader()
    reader.SetFileName(Path(__file__).parent / "CTPelvic1K_dataset6_data/dataset6_CLINIC_0001_data.nii.gz")
    itk_img:sitk.Image = reader.Execute()

    size = itk_img.GetSize()
    directions = np.reshape(itk_img.GetDirection(), (3,3))

    #thresh = sitk.BinaryThreshold(itk_img, 2)
    #component_image = sitk.ConnectedComponent(thresh)
    #sorted_component_image = sitk.RelabelComponent(component_image, sortByObjectSize=True)
    #largest_component_binary_image = sorted_component_image == 1
    #itk_img = sitk.Mask(itk_img, largest_component_binary_image)
    image = Image(sitk.GetArrayFromImage(itk_img).transpose(2,1,0), np.array(itk_img.GetOrigin()), np.array(itk_img.GetSpacing()), directions)
    
    ct_projector = CtProjector(image)

    # with cProfile.Profile() as pr:
    b=time()
    ct_projector.render_xray(500.0, 0.0, 0.0, 0.0, 1.2, (256,256))
    print(time() - b)
    #plt.axis("off")
    #plt.tight_layout()
    #plt.show()
    #     pr.print_stats()
    # exit()
    
    uvn_frame = ct_projector._define_uvn_frame(500.0)
    frame_origin = uvn_frame[:3, 3]
    pixel_positions_3d, line_directions, line_lengths = ct_projector._define_projection_lines(uvn_frame, 3.0, (128,128))
    #alphas = ct_projector._identify_alphas(frame_origin, line_directions, line_lengths)
    #pos_xyz = (frame_origin[None, ...] + line_directions[:,None,:] * line_lengths[:,None,:] * alphas[...,None]).reshape(-1, 3)
    
    
    pl = pv.Plotter()
    #pl.add_points(pos_xyz[::1000], color="red", point_size=5)
    grid = pv.ImageData()
    grid.dimensions = size
    grid.spacing = itk_img.GetSpacing()
    grid.origin = itk_img.GetOrigin()
    grid.point_data["Scalars"] = sitk.GetArrayFromImage(itk_img).flatten()

    coordinate_system:pv.PolyData = pv.wrap(pv.from_trimesh(trimesh.creation.axis(transform=uvn_frame, axis_length=400.0, axis_radius=5.0)))
    pixel_grid = pv.PolyData(pixel_positions_3d)
    pl.add_volume(grid, scalars="Scalars", cmap='bone', opacity="sigmoid", show_scalar_bar=False)
    pl.add_points(pixel_grid, color="red")
    pl.add_mesh(coordinate_system)

    image_origin, image_planes = ct_projector._image.intersection_planes()
    offsets = [[float(ct_projector._image.spacing[i] * j) for j in range(ct_projector._image.ct_array.shape[i] + 1)] for i in range(3)]
    colors = ["red", "green", "blue"]
    for i in range(3):
       offset_direction = np.array([0,0,0])
       offset_direction[i] = 1.0
       for o in offsets[i][::100]:
           pl.add_mesh(pv.Plane(center=image_origin + (offset_direction*o), direction=image_planes[i], i_size=1000, j_size=1000), color=colors[i], opacity=0.2)

    #lines = []
    #for l in range(0,len(pixel_locations), 100):
    #    lines.append(rays_origin[0])
    #    lines.append(pixel_locations[l])
    #pl.add_lines(np.array(lines), color="purple")
    pl.show()
    # pl.open_gif('coordinate_systems.gif')

    # for angle in np.linspace(0, 2*pi, 100):
    #     uvn_frame = ct_projector._define_uvn_frame(500.0, yaw=angle)
    #     pixel_positions_3d = ct_projector._position_3d_pixels(uvn_frame, 32.0, (16,16))
    #     coordinate_system.points = trimesh.creation.axis(transform=uvn_frame, axis_length=400.0, axis_radius=5.0).vertices
    #     pixel_grid.points = pixel_positions_3d
    #     pl.write_frame()
    # pl.close()

    #pl.show()
    exit()
    #phys_to_cont_idx_matrix = phys_to_continuous_index_matrix(img)

    # Here we'll define the viewport coordinates frame
    #projection_center = np.array(img.TransformContinuousIndexToPhysicalPoint([size[0]/2,0,size[2]/2])) + [0, -size[1], 0]
    #projection_direction = directions[1]
    desired_pixel_size_mm = 0.7
    #image_plane_position = projection_center.copy()
    #image_plane_position[1] += 1000
    height = 128
    width = 128

    view_window:pv.PolyData = pv.Plane(center=image_plane_position, direction=projection_direction, i_size=256, j_size=256, i_resolution=height, j_resolution=width)
    
    planes = define_intersection_planes(itk_img)    

    pixel_centers = view_window.cell_centers().points

    # Plot the whole thing
    grid = pv.ImageData()
    grid.dimensions = size
    grid.spacing = itk_img.GetSpacing()
    grid.origin = itk_img.GetOrigin()
    grid.point_data["Scalars"] = sitk.GetArrayFromImage(itk_img).flatten()

    #pl = pv.Plotter()
    #pl.add_mesh(view_window, show_edges=True)
    #pl.add_volume(grid, scalars="Scalars", cmap='bone', opacity=0.01, show_scalar_bar=False)

    offsets = [
        [itk_img.GetSpacing()[i] * j for j in range(itk_img.GetSize()[i] + 1)]
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
    np_img = sitk.GetArrayFromImage(itk_img)
    np_img = (np_img - np_img.min()) / (np.ptp(np_img))
    line_start_ends = np.vstack((projection_center[None, ...] + line_directions * line_lengths * all_intersects[None, :,0:1], projection_center[None, ...] + line_directions * line_lengths * all_intersects[None, :,-1:]))# * np.vstack((all_intersects[:,0], all_intersects[:,-1]))[None, ...]
    #pl.add_points(line_start_ends[0], color="blue")
    #pl.add_points(line_start_ends[1], color="red")
    #pl.show()
    
    begins_ijk = matrix_transform(phys_to_cont_idx_matrix, line_start_ends[0] + line_directions*1e-8)
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
    traversed_voxels = np_img[rhos[:,:,2], rhos[:,:,1], rhos[:,:,0]]
    image = np.sum(traversed_voxels * ls, axis=-1, dtype=np.float64).reshape(height, width).T
    
    ax.hist(image.flatten())


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
    #fig, ax = plt.subplots(2,3)
    #old_main(ax[1])
    main()
    #plt.show()
    #print(timeit(main, number=10) / 10)
