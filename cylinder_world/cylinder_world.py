#   Cylinder World Renderer
#   Copyright (C) 2021  Andreas Stöckel
#
#   This program is free software: you can redistribute it and/or modify
#   it under the terms of the GNU General Public License as published by
#   the Free Software Foundation, either version 3 of the License, or
#   (at your option) any later version.
#
#   This program is distributed in the hope that it will be useful,
#   but WITHOUT ANY WARRANTY; without even the implied warranty of
#   MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
#   GNU General Public License for more details.
#
#   You should have received a copy of the GNU General Public License
#   along with this program.  If not, see <https://www.gnu.org/licenses/>.

import numpy as np


class CylinderWorld:
    """
    Implements a the renderer for a simple "cylinder world". Renders a one-
    dimensional panoramic image from a location (x, y, θ) of a randomly textured
    cylinder of radius r. Automatically filters the texture to prevent aliasing
    artefacts.
    """
    def __init__(self,
                 resolution=32,
                 radius=1.0,
                 n_coeffs=8,
                 coeff_pow=-0.5,
                 rng=np.random):
        """
        Creates a new CylinderWorld instance.

        resolution: Number of pixels in the final rendered panoramic image.

        radius: Radius of the cylinder.

        n_coeffs: Number of Fourier coefficients for the randomly generated 
            texture.

        coeff_pow: Used to dampen the individual Fourier coefficient. The i-th
            Fourier coefficient (with 1 <= i <= n_coeffs) is multiplied by
            i^coeff_pow. The default value of -0.5 causes a fall-off
            proprtional to 1 / sqrt(i).

        rng: np.random.RandomState instance to use when sampling the random
            Fourier coefficients.
        """

        # Copy the given arguments
        assert int(resolution) > 0
        assert radius > 0
        assert int(n_coeffs) > 0
        self._res = int(resolution)
        self._r = radius
        self._n_coeffs = int(n_coeffs)

        # Compute the image-space angles
        self._βs = np.linspace(0, 2.0 * np.pi,
                               self._res + 1) - (np.pi / self._res)
        self._βs_centre = np.linspace(0, 2.0 * np.pi, self._res + 1)[:-1]

        # Sample the texture Fourier coefficients, dampen higher frequencies
        scale = np.power(1 + np.arange(self._n_coeffs), coeff_pow)
        self.As = rng.normal(0, 1, self._n_coeffs) * scale
        self.Bs = rng.normal(0, 1, self._n_coeffs) * scale

        # Frequencies used when sampling
        self._fs = np.arange(1, self._n_coeffs + 1)

        # Texture Fourier coefficients for range-based sampling I(α0, α1)
        self.Asp = self.Bs / self._fs
        self.Bsp = -self.As / self._fs

        # Texture Fourier coefficients for the derivative dI(α) / dα
        self.Asd = -self.Bs * self._fs
        self.Bsd = self.As * self._fs

    def _sample_texture(self, αs, As, Bs):
        # Make sure the incoming angles are a numpy array; multiply all angles
        # by the frequencies
        αs = np.asarray(αs, dtype=np.float)
        smpls = np.outer(αs, self._fs)

        # Weigh the sines and cosine terms and compute the sum over all
        # frequencies. Reshape the output to have the same shape as the input
        res = As * np.sin(smpls) + Bs * np.cos(smpls)
        return np.sum(res, axis=1).reshape(αs.shape)

    def sample_texture(self, αs):
        """
        Samples the cylinder texture at the given angles (in radians).
        """
        return self._sample_texture(αs, self.As, self.Bs)

    def sample_texture_derivative(self, αs):
        """
        Returns the derivative of the texture at the given angles (in radians).
        """
        return self._sample_texture(αs, self.Asd, self.Bsd)

    @staticmethod
    def _inner_angle(αs0, αs1):
        """
        Computes the inner angle between pairs of angles α0, α1.
        """
        αs0, αs1 = np.asarray(αs0), np.asarray(αs1)
        αs0 = αs0 % (2.0 * np.pi)
        αs1 = αs1 % (2.0 * np.pi)

        d1 = np.abs(αs1 - αs0)
        d2 = 2.0 * np.pi - np.abs(αs1 - αs0)
        return np.sign(αs1 - αs0) * np.where(d1 < d2, d1, -d2)

    def _sample_texture_range(self, αs0, αs1, As, Bs):
        # Make sure that the inputs are arrays and make sure that both arrays
        # have the same shape
        αs0 = np.asarray(αs0, dtype=np.float)
        αs1 = np.asarray(αs1, dtype=np.float)
        assert αs0.shape == αs1.shape

        # Multiply all angles by the frequencies
        smpls0, smpls1 = np.outer(αs0, self._fs), np.outer(αs1, self._fs)

        # Compute the difference of the integrals. Reshape the output to have
        # the same shape as the input
        res = ((As * (np.sin(smpls1) - np.sin(smpls0))) +
               (Bs * (np.cos(smpls1) - np.cos(smpls0))))
        return np.sum(res, axis=1).reshape(
            αs0.shape) / CylinderWorld._inner_angle(αs0, αs1)

    def sample_texture_range(self, αs0, αs1):
        """
        Samples the cylinder texture by computing the mean intensity between
        pairs of angles αs0, αs1.
        """
        return self._sample_texture_range(αs0, αs1, self.Asp, self.Bsp)

    def _project(self, βs, xs, ys, θs):
        # Make sure all input arrays are numpy arrays
        βs = np.asarray(βs, dtype=np.float)
        xs = np.asarray(xs, dtype=np.float)
        ys = np.asarray(ys, dtype=np.float)
        θs = np.asarray(θs, dtype=np.float)

        # Make sure the input arrays have the right size
        assert xs.shape == ys.shape == θs.shape

        # Compute the sines and cosines of the input angles plus the
        # orientations thetas. Just create as many new axes as necessary
        ss = np.sin(βs[..., None] + θs[None])
        cs = np.cos(βs[..., None] + θs[None])

        # Compute the new radius rp
        xscs, ysss = xs * cs, ys * ss,
        r0p = np.sqrt(np.square(xscs + ysss) + self._r**2 - xs**2 - ys**2)
        rp = r0p - xscs - ysss

        # Compute the intersection points xsp, ysp
        xsp, ysp = xs + rp * cs, ys + rp * ss

        # Compute the texture angles
        αs = np.arctan2(ysp, xsp)

        # Return the texture angles as well as all intermediate values
        return αs, rp, r0p

    def project(self, βs, xs, ys, θs):
        """
        Projects the given image angles βs onto texture angles αs.
        """
        αs, _, _ = self._project(βs, xs, ys, θs)
        return αs

    def depth_at_angles(self, βs, xs, ys, θs):
        """
        Computes the depth for the given image-space angles βs and the given
        orientation and location.
        """
        _, rp, _ = self._project(βs, xs, ys, θs)
        return rp

    def depth(self, xs, ys, θs):
        """
        Returns a depth map for the given orientation and location.
        """
        return self.depth_at_angles(self._βs_centre, xs, ys, θs)

    def render_at_angles(self, βs, xs, ys, θs, anti_alias=False):
        """
        Renders the image for the given image angles. If anti_alias is True,
        the given list of angles must be n + 1 elements long, where n is the
        number of pixels. The angles will be treated as pairs of angles, i.e.,
            (βs[0], βs[1]), (βs[1], βs[2]), ... (βs[n - 1], βs[n])
        """
        if anti_alias:
            αs, _, _ = self._project(βs, xs, ys, θs)
            αs0, αs1 = αs[:-1], αs[1:]
            return self.sample_texture_range(αs0, αs1)
        else:
            αs, _, _ = self._project(βs, xs, ys, θs)
            return self.sample_texture(αs)

    def render_at_angles_with_gradient(self, βs, xs, ys, θs, anti_alias=False):
        if anti_alias:
            # Compute the texture-space angles
            αs, rp, r0p = self._project(βs, xs, ys, θs)

            # Split the texture-space angles and converted radii into two parts
            αs0, αs1 = αs[:-1], αs[1:]
            rp0, rp1 = rp[:-1], rp[1:]
            r0p0, r0p1 = r0p[:-1], r0p[1:]

            Is = self.sample_texture_range(αs0, αs1)
            Is0, Is1 = self.sample_texture(αs0), self.sample_texture(αs1)
            dα0β0, dα1β1 = rp0 / r0p0, rp1 / r0p1
            dIs = (Is1 * dα1β1 - Is0 * dα0β0 - Is *
                   (dα1β1 - dα0β0)) / CylinderWorld._inner_angle(αs0, αs1)
        else:
            αs, rp, r0p = self._project(βs, xs, ys, θs)
            Is = self.sample_texture(αs)
            dIs = (rp / r0p) * self.sample_texture_derivative(αs)
        return Is, dIs

    def render(self, xs, ys, θs, anti_alias=True):
        βs = self._βs if anti_alias else self._βs_centre
        return self.render_at_angles(βs, xs, ys, θs, anti_alias)

    def render_with_gradient(self, xs, ys, θs, anti_alias=True):
        βs = self._βs if anti_alias else self._βs_centre
        return self.render_at_angles_with_gradient(βs, xs, ys, θs, anti_alias)

    @property
    def resolution(self):
        return self._res

    @property
    def radius(self):
        return self._r

    @property
    def angles(self):
        return self._βs_centre

    @property
    def angle_ranges(self):
        return list(zip(self._βs[:-1], self._βs[1:]))

