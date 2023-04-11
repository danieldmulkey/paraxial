import math
import random

import pytest

from paraxial import sources, surfaces


def test_types():
    mat = surfaces.ABCD()
    gb = sources.GaussianBeam(z=10e-3, zR=1e-3)
    assert isinstance(mat, surfaces.ABCD)
    assert isinstance(gb, sources.Ray)
    assert isinstance(gb, sources.GaussianBeam)
    assert isinstance(mat @ gb, sources.GaussianBeam)


def test_gb_attributes():
    λ0 = 500e-9
    z = 10e-3
    zR = 1e-3
    gb = sources.GaussianBeam(λ0=λ0, z=z, zR=zR)
    assert gb.n == 1
    assert gb.λ0 == λ0
    assert gb.y == 1
    assert gb.u == pytest.approx(1 / (z + 1j * zR))


def test_gb_equality():
    g1 = sources.GaussianBeam(R=100e-3, w=1e-3)
    g2 = sources.GaussianBeam(R=100e-3, w=1e-3)
    assert g1 == g2


def test_arg_error_catching():
    with pytest.raises(ValueError):
        sources.GaussianBeam(z=1, zR=0)
    with pytest.raises(ValueError):
        sources.GaussianBeam(R=1, w=0)


def test_gb_from_q():
    gb = sources.GaussianBeam(q=10e-3 + 1e-3j, n=1.5)
    assert gb.q == pytest.approx(10e-3 + 1e-3j)


def test_gb_from_R_w():
    gb = sources.GaussianBeam(R=100e-3, w=1e-3, n=1.5)
    assert gb.R == 100e-3
    assert gb.w == pytest.approx(1e-3)


def test_gb_from_z_zR():
    gb = sources.GaussianBeam(z=10e-3, zR=1e-3, n=1.5)
    assert gb.z == 10e-3
    assert gb.zR == pytest.approx(1e-3)


def test_gb_from_R_z():
    gb = sources.GaussianBeam(R=100e-3, z=10e-3, n=1.5)
    assert gb.R == pytest.approx(100e-3)
    assert gb.z == pytest.approx(10e-3)


def test_gb_from_R_zR():
    gb = sources.GaussianBeam(R=100e-3, zR=1e-3, n=1.5)
    assert gb.R == pytest.approx(100e-3)
    assert gb.zR == pytest.approx(1e-3)


def test_gb_from_w_z():
    gb = sources.GaussianBeam(w=1e-3, z=10e-3, n=1.5)
    assert gb.w == pytest.approx(1e-3)
    assert gb.z == pytest.approx(10e-3)

    gb = sources.GaussianBeam(w=1e-3, z=0)
    assert gb.w0 == pytest.approx(1e-3)
    # from Edmund Optics GB calculator:
    assert gb.θ == pytest.approx(0.16934e-3, abs=1e-7)
    assert gb.R == pytest.approx(math.inf)


def test_gb_from_w_zR():
    gb = sources.GaussianBeam(w=1e-3, zR=1e-3, n=1.5)
    assert gb.w == pytest.approx(1e-3)
    assert gb.zR == pytest.approx(1e-3)


def test_convenience():
    gb = sources.GaussianBeam(R=0, w=1e-3)
    assert gb.R == math.inf


def test_flat_interface():
    n1 = 1 + random.random() * 3
    n2 = 1 + random.random() * 3
    z = 10e-3
    zR = 1e-3

    gb1 = sources.GaussianBeam(z=z, zR=zR, n=n1)
    glass = surfaces.ABCD(n1=n1, n2=n2)
    gb2 = glass @ gb1

    # attributes:
    assert gb1.n == n1
    assert gb2.n == n2
    assert gb1.λ0 == gb2.λ0

    # properties:
    assert gb2.R == pytest.approx(n2 / n1 * gb1.R)
    assert gb2.w == pytest.approx(gb1.w)

    assert gb2.z == pytest.approx(n2 / n1 * gb1.z)
    assert gb2.zR == pytest.approx(n2 / n1 * gb1.zR)

    assert gb2.w0 == pytest.approx(gb1.w0)
    assert gb2.div == pytest.approx(
        n1 / n2 * gb1.div
    )


def test_free_space():
    # before & after free space
    n = 1 + random.random() * 3
    z = 10e-3
    zR = 1e-3
    L = 100e-3

    gb1 = sources.GaussianBeam(z=z, zR=zR, n=n)
    fs = surfaces.ABCD(B=L / n, n1=n, n2=n)
    gb2 = fs @ gb1

    # attributes:
    assert gb1.n == gb2.n == n
    assert gb1.λ0 == gb2.λ0

    # properties:
    assert gb2.z == pytest.approx(
        gb1.z + L
    )  # no 1/n because same media
    assert gb2.zR == pytest.approx(gb1.zR)

    assert gb2.w0 == pytest.approx(gb1.w0)
    assert gb2.div == pytest.approx(gb1.div)


def test_prop_to_focus_air():
    # verify distance to focus is physical, not reduced
    z = -10e-3
    zR = 1e-3
    L = -z

    gb1 = sources.GaussianBeam(z=z, zR=zR)
    glass = surfaces.ABCD(B=L)
    gb2 = glass @ gb1

    # attributes:
    assert gb1.λ0 == gb2.λ0
    assert gb1.n == gb2.n

    # properties:
    assert abs(gb2.R) >= 1e9
    assert gb2.w == pytest.approx(gb1.w0)

    assert gb2.z == pytest.approx(0)
    assert gb2.zR == pytest.approx(gb1.zR)


def test_prop_to_focus_glass():
    # verify distance to focus is physical, not reduced
    n2 = 1.5
    z = -10e-3
    zR = 1e-3
    L = -z * n2  # physical prop distance is 15 mm

    gb1 = sources.GaussianBeam(z=z, zR=zR)
    glass = surfaces.ABCD(B=L / n2, n2=n2)  # need L/n
    gb2 = glass @ gb1

    # attributes:
    assert gb2.λ0 == gb1.λ0

    # properties:
    assert abs(gb2.R) >= 1e12
    assert gb2.w == pytest.approx(gb1.w0)

    assert gb2.z == pytest.approx(0)
    assert gb2.zR == pytest.approx(n2 * gb1.zR)


def test_refraction():
    # before & after refraction (e.g. updating n)
    n = 1 + random.random() * 3
    z = 10e-3
    zR = 1e-3
    R = 100e-3

    gb1 = sources.GaussianBeam(z=z, zR=zR)
    refr = surfaces.ABCD(C=-(n - 1) / R, n2=n)
    gb2 = refr @ gb1

    # attributes:
    assert gb1.λ0 == gb2.λ0

    # properties:
    assert gb2.n / gb2.R == pytest.approx(gb1.n / gb1.R - (n - 1) / R)
    assert gb2.w == pytest.approx(gb1.w)


def test_init_in_material():
    # starting in-material vs out
    n1 = 1  # 1 + random.random() * 3
    n2 = 1 + random.random() * 3
    z = 10e-3
    zR = 1e-3

    gb1 = sources.GaussianBeam(z=z, zR=zR, n=n1)
    gb2 = sources.GaussianBeam(z=z, zR=zR, n=n2)

    assert gb1.n == n1
    assert gb2.n == n2

    assert gb2.R == pytest.approx(gb1.R)
    assert gb2.z == pytest.approx(gb1.z)
    assert gb2.zR == pytest.approx(gb1.zR)

    assert gb2.w0 == pytest.approx(gb1.w0 / n2 ** 0.5)


def test_in_place_update():
    z = 10e-3
    zR = 1e-3
    gb1 = sources.GaussianBeam(z=z, zR=zR)
    glass = surfaces.ABCD(B=10e-3)
    gb2 = glass @ gb1
    gb1 @= glass
    assert gb1 == gb2


def test_self_overwrite_update():
    z = 10e-3
    zR = 1e-3
    gb1 = sources.GaussianBeam(z=z, zR=zR)
    glass = surfaces.ABCD(B=10e-3)
    gb2 = glass @ gb1
    gb1 = glass @ gb1
    assert gb1 == gb2
