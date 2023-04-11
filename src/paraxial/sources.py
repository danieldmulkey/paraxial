import math
import copy

import numpy as np

from . import equations

class Ray:
    def __init__(self, y=0, u=0, n=1, λ0=532e-9):
        self.y = y
        self.u = u
        self.n = n
        self.λ0 = λ0

    def __repr__(self):
        return f"[[{self.y}]\n [{self.n * self.u}]]"

    def __eq__(self, other):
        return (
            np.all(self.y == other.y)
            and np.all(self.u == other.u)
            and np.all(self.n == other.n)
            and np.all(self.λ0 == other.λ0)
        )

    # to enable (self @= other) syntax
    def __imatmul__(self, other):
        # other assumed to be ABCD
        return other @ self

    @property
    def wvl(self):
        # In case you don't have Greek installed:
        return self.λ0

class GaussianBeam(Ray):
    def __init__(
        self,
        *,
        λ0=532e-9,
        n=1,
        sign=1,
        q=None,
        R=None,
        w=None,
        z=None,
        zR=None,
    ):
        """Creates a GaussianBeam class. Pick any two beam 
        parameters and provide them as keyword arguments. 
        Arguments are always physical values (e.g. z is 
        distance as measured, not air-equivalent thickness 
        and q is z + 1j * zR, not reduced q).

        The reduced q value is what can be used for ABCD 
        calculations. While this is used internally, the 
        properties of GaussianBeam always return physical 
        values (including the non-reduced q)."""

        # As a convenience:
        if R == 0:
            R = math.inf

        if w == 0 or zR == 0:
            raise ValueError("Invalid argument w or zR")

        # Conditionals structured to allow values to be zero
        if q is not None:
            qr = q / n
        elif R is not None and w is not None:
            qr = equations.q_from_R_w(R, w, λ0, n)
        elif z is not None and zR is not None:
            qr = equations.q_from_z_zR(z, zR, n)
        elif R is not None and z is not None:
            qr = equations.q_from_R_z(R, z, n)
        elif R is not None and zR is not None:
            qr = equations.q_from_R_zR(R, zR, n, sign)
        elif w is not None and z is not None:
            qr = equations.q_from_w_z(w, z, λ0, n, sign)
        elif w is not None and zR is not None:
            qr = equations.q_from_w_zR(w, zR, λ0, n, sign)
        else:
            raise ValueError("No valid configuration for beam found")
        super().__init__(1, 1 / (n * qr), n, λ0)

    def __repr__(self):
        return (
            f"R & w: {self.R * 1e3:g} mm & {self.w * 1e3:g} mm\n"
            f"z & zR: {self.z * 1e3:g} mm & {self.zR * 1e3:g} mm\n"
            f"w0 & θ: {self.w0 * 1e3:g} mm & {self.θ * 1e3:g} mrad\n"
            f"q: {self.q * 1e3:g} mm"
        )

    @property
    def q(self):
        return self.y / self.u

    @property
    def R(self):
        try:
            real = (1 / self.q).real
            return 1 / real
        except ZeroDivisionError:
            return math.inf

    @property
    def w(self):
        imag = (self.n / self.q).imag
        return (-math.pi / self.λ0 * imag) ** (-0.5)

    @property
    def z(self):
        return self.q.real

    @property
    def zR(self):
        return self.q.imag

    @property
    def w0(self):
        return (self.λ0 * self.zR / (self.n * math.pi)) ** 0.5

    @property
    def θ(self):
        return self.w0 / self.zR

    @property
    def div(self):
        # In case you don't have Greek installed:
        return self.θ

