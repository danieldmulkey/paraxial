"""These tests focus on testing children of abcd.ABCD involving
multiple ABCDs in their creation"""

import pytest

from paraxial import surfaces


def test_thick_lens():
    thick = surfaces.ThickLens(25e-3, -30e-3, 2e-3, 1.5)
    assert thick.f1 == thick.f2  # lens is in air
    assert thick.P1 != thick.P2  # principal planes NOT superimposed
    assert thick.F1 != -thick.F2  # required by planes and fs
    assert thick.N1 != -thick.N2


def test_lensmaker_equation():
    # equivalent by lensmaker's equation
    thin = surfaces.ThinLens(10e-3)
    thick = surfaces.ThickLens(10e-3, -10e-3, 0, 1.5)
    assert thin == thick

