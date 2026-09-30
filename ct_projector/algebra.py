import numpy as np
from math import cos, sin

def matrix_transform(matrix:np.ndarray, coordinates:np.ndarray) -> np.ndarray:
    coordinates = np.asarray(coordinates)
    if coordinates.ndim == 1:
        coordinates = coordinates[None, ...]

    h_coords = np.hstack((coordinates, np.ones((len(coordinates), 1))))
    h_coords = (matrix @ h_coords.T).T
    h_coords /= h_coords[:, 3:]
    return h_coords[:, :3]


def lines_parallel_planes_intersection(l_0:np.ndarray, l_dir:np.ndarray, l_length:float, p_0:np.ndarray, p_n:np.ndarray, offsets:np.ndarray):
    """
    Returns the "t" value of the parametric form of the intersections
    """
    dot_prod_down = np.vecdot(l_dir, p_n)
    dot_prod_up = np.vecdot(p_0 - l_0, p_n)[..., None]

    with np.errstate(divide='ignore', invalid='ignore'):
        return (dot_prod_up + offsets[None, ...]) / (dot_prod_down[..., None] * l_length)


# def line_plane_intersection(line:Line, plane:Plane):
#     dot_prod_down = np.dot(line.direction, plane.normal)
#     dot_prod_up = np.dot(plane.x0 - line.x0, plane.normal)

#     if np.isclose(dot_prod_down, 0.0):
#         return np.nan

#     d = dot_prod_up / dot_prod_down
#     return line.x0 + line.direction * d

def rotation_matrix(*, pitch:float = 0.0, yaw:float = 0.0, roll:float = 0.0) -> np.ndarray:
        pitch_matrix = np.eye(3)
        pitch_matrix[1,1] = cos(pitch)
        pitch_matrix[1,2] = -sin(pitch)
        pitch_matrix[2,1] = sin(pitch)
        pitch_matrix[2,2] = cos(pitch)

        yaw_matrix = np.eye(3)
        yaw_matrix[0,0] = cos(yaw)
        yaw_matrix[0,2] = sin(yaw)
        yaw_matrix[2,0] = -sin(yaw)
        yaw_matrix[2,2] = cos(yaw)

        roll_matrix = np.eye(3)
        roll_matrix[0,0] = cos(roll)
        roll_matrix[0,1] = -sin(roll)
        roll_matrix[1,0] = sin(roll)
        roll_matrix[1,1] = cos(roll)

        rotation_matrix = np.eye(4)
        rotation_matrix[:3,:3] = roll_matrix @ yaw_matrix @ pitch_matrix
        return rotation_matrix

def unit_vector_projection(v:np.ndarray, pts:np.ndarray) -> np.ndarray:
    dot_vv = np.dot(v,v)
    dots = np.vecdot(v[None, ...], pts, keepdims=True)
    return (v * dots) / dot_vv