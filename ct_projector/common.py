from operator import attrgetter
from functools import cached_property
from itertools import product
from typing import Sequence, Any
import numpy as np
from attrs import define, field
from typing import NamedTuple

from math import cos, sin
from algebra import *

def _validate_single_coordinates_vector(instance, attribute, value:np.ndarray):
    if value.shape != (3,):
        raise ValueError(f"Invalid value for {attribute}")

def _validate_orthonormal_matrix(instance, attribute, value:np.ndarray):
    if value.shape != (3,3):
        raise ValueError("Directions must be an orthonormal column matrix of shape (3,3)")
    
    for v in range(3):
        vector = value[:, v]
        if np.linalg.norm(vector) != 1.0:
            raise ValueError("Directions must be orthonormal")

    if not np.allclose((value @ value.T), np.eye(3)):
        raise ValueError("Directions must be orthonormal")

def _validate_3d_image(instance, attribute, value:np.ndarray):
    if value.ndim != 3:
        raise ValueError(f"{attribute} must have ndim 3, found {value.ndim}")

def _convert_hu_to_density(img:np.ndarray, bone_attenuation_multiplier=1.0):
    """
    TAKEN FROM https://github.com/eigenvivek/DiffDRR/blob/main/diffdrr/data.py#L214

    Convert Hounsfield units to a density-like quantity for rendering.

    Voxels are split at -800 and 350 HU into air, soft tissue, and bone. Soft
    tissue and bone keep their HU values (bone scaled by
    `bone_attenuation_multiplier`), every voxel at or below -800 HU is assigned
    the lowest soft-tissue value present in the volume, and the result is
    min-max normalized per volume.

    The -800 HU floor removes all structure below that threshold. This is
    immaterial where bone carries the signal, but on thoracic CT most aerated
    lung parenchyma (roughly -950 to -700 HU) is mapped onto the floor and
    becomes indistinguishable from the air outside the patient. See
    `nanodrr.data.preprocess.hu_to_mu` for a physically derived alternative.
    """
    # volume can be loaded as int16, need to convert to float32 to use float bone_attenuation_multiplier
    img = img.astype(np.float32)
    
    air = img <= -800
    bone = img > 350
    soft_tissue = np.logical_not(np.logical_or(air, bone))

    density = np.empty_like(img)
    min_soft_tissue = img[soft_tissue].min()
    density[air] = min_soft_tissue
    density[soft_tissue] = img[soft_tissue]
    density[bone] = img[bone] * bone_attenuation_multiplier
    density -= min_soft_tissue
    density /= density.max()
    return density

@define()
class Image:
    ct_array:np.ndarray = field(validator=[_validate_3d_image], converter=_convert_hu_to_density)
    origin:np.ndarray = field(validator=[_validate_single_coordinates_vector])
    spacing:np.ndarray = field(validator=[_validate_single_coordinates_vector])
    directions:np.ndarray = field(validator=[_validate_orthonormal_matrix])

    @cached_property
    def ijk_to_xyz_matrix(self):
        return np.linalg.inv(self.xyz_to_ijk_matrix)

    @cached_property
    def xyz_to_ijk_matrix(self):
        inv_scaling = np.diag(1.0 / self.spacing)
        R = inv_scaling @ self.directions

        # -R is just the inverse of R
        t = -R @ self.origin
        
        M = np.eye(4)
        M[:3, :3] = R
        M[:3, 3] = t
        
        return M
    
    @cached_property
    def xyz_corners(self) -> np.ndarray:
        corners_ijk = np.array(list(product([0,self.ct_array.shape[0]], [0,self.ct_array.shape[1]],[0,self.ct_array.shape[2]])), dtype=float)
        return matrix_transform(self.ijk_to_xyz_matrix, corners_ijk)


    @cached_property
    def xyz_center(self):
        size = np.array(self.ct_array.shape, dtype=float)
        size /= 2
        return matrix_transform(self.ijk_to_xyz_matrix, size)
    
    @cached_property
    def front_facing_vector(self) -> np.ndarray:
        return -self.directions[1]

    def intersection_planes(self) -> tuple[np.ndarray, np.ndarray]:
        first_point = matrix_transform(self.ijk_to_xyz_matrix, [-0.5,-0.5,-0.5])
        return first_point, self.directions.T

def attr_from_list(attrib: str, array:Sequence[Any]) -> list[Any]:
    return list(map(attrgetter(attrib), array))

def unit_vector(v: np.ndarray):
    if v.ndim == 2:
        return v / np.linalg.norm(v, axis=1)
    else:
        return v / np.linalg.norm(v)


# @attrs.frozen()
# class Line:
#     # TODO : Validate shape for all
#     x0:np.ndarray
#     x1:np.ndarray
    
#     @cached_property
#     def direction(self) -> np.ndarray:
#         return unit_vector(self.x1 - self.x0)

#     @cached_property
#     def length(self) -> float:
#         return np.linalg.norm(self.x1 - self.x0)
    
#     @cached_property
#     def vector(self) -> np.ndarray:
#         return self.x1 - self.x0

#     def middle_points(self, ts:np.ndarray) -> np.ndarray:
#         ts = np.asarray(ts, float)
#         return self.x0[None, ...] + self.vector * ts[..., None]

# @attrs.frozen()
# class Plane:
#     x0:np.ndarray
#     normal:np.ndarray  # validate is unit vector