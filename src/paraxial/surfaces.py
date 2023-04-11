import math
import copy

import numpy as np

from . import sources

# TODO: check for non-unity determinant?
class ABCD:
    def __init__(
        self, A=1, B=0, C=0, D=1, n1=1, n2=1, disp=0, tilt=0, length=0, E=None, F=None
    ):
        self.n1 = n1
        self.n2 = n2
        self.A = A
        self.B = B
        self.C = C
        self.D = D
        self.E = (1 - A) * disp + (length - n1 * B) * tilt if E is None else E
        self.F = -C * disp + (n2 - n1 * D) * tilt if F is None else F

    def __repr__(self):
        return (
            f"[[{self.A:g} {self.B:g} {self.E:g}]\n"
            f" [{self.C:g} {self.D:g} {self.F:g}]\n"
            f" [0 0 1]]"
        )

    def __eq__(self, other):
        return (
            (self.A == other.A)
            and (self.B == other.B)
            and (self.C == other.C)
            and (self.D == other.D)
            and (self.E == other.E)
            and (self.F == other.F)
        )

    # to enable (self @ other) syntax
    def __matmul__(self, other):
        # self @ other
        if isinstance(other, ABCD):
            A = self.A * other.A + self.B * other.C
            B = self.A * other.B + self.B * other.D
            C = self.C * other.A + self.D * other.C
            D = self.C * other.B + self.D * other.D
            E = self.A * other.E + self.B * other.F + self.E
            F = self.C * other.E + self.D * other.F + self.F
            return ABCD(A=A, B=B, C=C, D=D, n1=other.n1, n2=self.n2, E=E, F=F)
        # self @ Ray
        elif isinstance(other, sources.Ray):
            y = self.A * other.y + self.B * other.n * other.u + self.E
            u = (
                self.C * other.y + self.D * other.n * other.u + self.F
            ) / self.n2

            ret = copy.deepcopy(other)  # so "other" unchanged
            ret.y = y
            ret.u = u
            ret.n = copy.deepcopy(self.n2)
            return ret

    # to enable (self @= other) syntax
    def __imatmul__(self, other):
        return other @ self

    def add_misalignments(self, disp, tilt, length):
        self.E = (1 - self.A) * disp + (length - self.n1 * self.B) * tilt 
        self.F = -self.C * disp + (self.n2 - self.n1 * self.D) * tilt

    @property
    def F1(self):
        try:
            return self.n1 * self.D / self.C
        except ZeroDivisionError:
            return -math.inf

    @property
    def P1(self):
        try:
            return self.n1 * (self.D - 1) / self.C
        except ZeroDivisionError:
            return -math.inf

    @property
    def N1(self):
        try:
            return (self.D * self.n1 - self.n2) / self.C
        except ZeroDivisionError:
            return -math.inf

    @property
    def f1(self):
        try:
            return -self.n1 / self.C
        except ZeroDivisionError:
            return math.inf

    @property
    def F2(self):
        try:
            return -self.n2 * self.A / self.C
        except ZeroDivisionError:
            return math.inf

    @property
    def P2(self):
        try:
            return self.n2 * (1 - self.A) / self.C
        except ZeroDivisionError:
            return math.inf

    @property
    def N2(self):
        try:
            return (self.n1 - self.A * self.n2) / self.C
        except ZeroDivisionError:
            return math.inf

    @property
    def f2(self):
        try:
            return -self.n2 / self.C
        except ZeroDivisionError:
            return math.inf


class Transfer(ABCD):
    def __init__(self, t, n=1, **kwargs):
        super().__init__(1, t / n, 0, 1, **kwargs)


class Refraction(ABCD):
    """ABCD matrix for refraction at dielectric interface.
    AOI is angle-of-incidence of axial ray in radians
    (e.g. coming in to lens at a non-normal angle) 
    and T_or_S specifies tangential or saggital. R is
    positive for a convex surface, typical of geometrical
    optics but counter to Siegman."""

    def __init__(self, R, n1, n2, AOI=0, T_or_S="T", **kwargs):
        AOE = math.asin(n1 * math.sin(AOI) / n2)
        C1 = math.cos(AOI)
        C2 = math.cos(AOE)

        if T_or_S.lower() == "t":
            denominator = C1 * C2
            A = C2 / C1
            D = C1 / C2
        elif T_or_S.lower() == "s":
            denominator = 1
            dne = n2 * C2 - n1 * C1
            A = D = 1
        else:
            raise ValueError(f"Unknown T_or_S: {T_or_S}")
        numerator = n2 * C2 - n1 * C1
        dne = numerator / denominator
        B = 0
        C = -dne / R
        super().__init__(A, B, C, D, n1, n2, **kwargs)


class Mirror(ABCD):
    """ABCD matrix for reflection at mirror.
    AOI is angle-of-incidence of axial ray in radians
    (e.g. coming in to mirror at a non-normal angle) 
    and T_or_S specifies tangential or saggital. R is
    positive for a convex surface, typical of geometrical
    optics but counter to Siegman."""

    def __init__(self, R, AOI=0, T_or_S="T", **kwargs):
        if T_or_S.lower() == "t":
            Re = R * math.cos(AOI)
        elif T_or_S.lower() == "s":
            Re = R / math.cos(AOI)
        else:
            raise ValueError(f"Unknown T_or_S: {T_or_S}")
        super().__init__(1, 0, 2 / Re, 1, **kwargs)


class Duct(ABCD):
    """ABCD matrix for media with radially varying index as
    n(y) = n0 - n2 * y**2 / 2. For example, a GRIN fiber or
    index variation caused by thermal lensing."""

    def __init__(self, t, n0, n2, **kwargs):
        g = (n2 / n0) ** 0.5
        A = math.cos(g * t)
        B = math.sin(g * t) / (n0 * g)
        C = -n0 * g * math.sin(g * t)
        D = math.cos(g * t)
        super().__init__(A, B, C, D, **kwargs)


class Grating(ABCD):
    """ABCD matrix for diffraction grating per Siegman. 
    Grating lines run in the x / saggital direction with 
    grating spacing d in the y / tangential direction. 
    Diffraction takes place in the YZ plane.
    Calling diffraction order m. Assumed to be reflective
    with positive R convex. sign == 1 is for the convention
    sin(θ1) - sin(θ2) whereas sign == -1 specifies
    sin(θ1) + sin(θ2)"""

    # TODO: check math for non-vacuum media
    def __init__(
        self,
        R,
        m=-1,
        d=1e-6,
        λ0=532e-9,
        AOI=0,
        T_or_S="T",
        sign=1,
        **kwargs,
    ):
        sign = 1 if sign >= 0 else -1
        AOE = math.asin(m * λ0 / d + sign * math.sin(AOI))
        C1 = math.cos(AOI)
        C2 = math.cos(AOE)

        if T_or_S.lower() == "t":
            # "Lasers" is missing the 2x here:
            Rt = 2 * R * C1 * C2 / (C1 + C2)
            M = C2 / C1
            A = M
            B = 0
            C = 2 / Rt
            D = 1 / M
        elif T_or_S.lower() == "s":
            Rs = 2 * R / (C1 + C2)
            A = D = 1
            B = 0
            C = 2 / Rs
        else:
            raise ValueError(f"Unknown T_or_S: {T_or_S}")
        super().__init__(A, B, C, D, **kwargs)


class ThinLens(ABCD):
    def __init__(self, f, **kwargs):
        super().__init__(1, 0, -1 / f, 1, **kwargs)


class ThickLens(ABCD):
    def __init__(self, R1, R2, t, n_glass, n_ambient=1, **kwargs):
        r1 = Refraction(R1, n_ambient, n_glass)
        t = Transfer(t, n_glass)
        r2 = Refraction(R2, n_glass, n_ambient)
        net = r2 @ t @ r1
        super().__init__(
            net.A, net.B, net.C, net.D, net.n1, net.n2, **kwargs
        )

