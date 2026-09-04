from operator import attrgetter
from functools import cached_property
from typing import Sequence, Any
import numpy as np
import attrs

def attr_from_list(attrib: str, array:Sequence[Any]) -> list[Any]:
    return list(map(attrgetter(attrib), array))

def unit_vector(v: np.ndarray):
    if v.ndim == 2:
        return v / np.linalg.norm(v, axis=1)
    else:
        return v / np.linalg.norm(v)


@attrs.frozen()
class Line:
    # TODO : Validate shape for all
    x0:np.ndarray
    x1:np.ndarray
    
    @cached_property
    def direction(self) -> np.ndarray:
        return unit_vector(self.x1 - self.x0)

    @cached_property
    def length(self) -> float:
        return np.linalg.norm(self.x1 - self.x0)
    
    @cached_property
    def vector(self) -> np.ndarray:
        return self.x1 - self.x0

    def middle_points(self, ts:np.ndarray) -> np.ndarray:
        ts = np.asarray(ts, float)
        return self.x0[None, ...] + self.vector * ts[..., None]

@attrs.frozen()
class Plane:
    x0:np.ndarray
    normal:np.ndarray  # validate is unit vector