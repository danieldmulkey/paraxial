import math

import pytest
import numpy as np
import sympy as sm

from paraxial import sources, surfaces


def test_ray_numpy_use():
    heights = np.linspace(0, 1, 100)
    angles = np.linspace(1, 0, 100)
    rays = sources.Ray(heights, angles)
    identical_rays = sources.Ray(heights, angles)
    assert np.all(rays == identical_rays)
    
def test_ray_inplace_numpy():
    n1 = 1.5
    n2 = 2
    heights = np.linspace(0, 1, 10)
    angles = np.linspace(1, 0, 10)
    
    rays = sources.Ray(heights, angles)
    assert np.allclose(rays.u, angles)
    
    glass = surfaces.ABCD(n1=1, n2=n1)
    rays @= glass
    assert np.allclose(rays.u, angles/n1)
    
    air = surfaces.ABCD(n1=n1, n2=1)
    rays @= air
    assert np.allclose(rays.u, angles)

    glass = surfaces.ABCD(n1=1, n2=n2)
    rays @= glass
    assert np.allclose(rays.u, angles/n2)

