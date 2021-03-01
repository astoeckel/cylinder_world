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
import os, sys
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))

from cylinder_world import CylinderWorld

SEED = 48281


def test_cylinder_world_image_space_angles():
    world = CylinderWorld(rng=np.random.RandomState(SEED))
    np.testing.assert_allclose(0.5 * (world._βs[:-1] + world._βs[1:]),
                               world._βs_centre)


def test_sample_texture_range():
    world = CylinderWorld(rng=np.random.RandomState(SEED))

    alpha0, alpha1 = 0.5, 0.75
    alphas = np.linspace(alpha0, alpha1, 100000)

    smpls = world.sample_texture(alphas)
    smpl_range = world.sample_texture_range(alpha0, alpha1)

    assert np.abs(np.mean(smpls) - smpl_range) < 1e-5


def test_sample_texture_derivative():
    world = CylinderWorld(rng=np.random.RandomState(SEED))

    eta = 1e-6
    alpha = 0.5
    smpl0 = world.sample_texture(alpha - 0.5 * eta)
    smpl1 = world.sample_texture(alpha + 0.5 * eta)
    dsmpl_num = (smpl1 - smpl0) / eta
    dsmpl_direct = world.sample_texture_derivative(alpha)

    assert np.abs(dsmpl_direct - dsmpl_num) < 1e-5


def test_depth():
    for radius in np.linspace(0.5, 2, 11):
        world = CylinderWorld(resolution=128,
                              n_coeffs=4,
                              rng=np.random.RandomState(SEED),
                              radius=radius)
        # At the origin, the distance to the cylinder must be equal to the
        # radius of the cylinder.
        ds = world.depth(0.0, 0.0, 0.0)
        np.testing.assert_allclose(np.ones_like(ds) * radius, ds)

        # Move in the x- direction and compute the depth
        d = world.depth_at_angles(0.0, 0.5 * radius, 0.0, 0.0)
        np.testing.assert_allclose(0.5 * radius, d)
        d = world.depth_at_angles(-np.pi, 0.5 * radius, 0.0, 0.0)
        np.testing.assert_allclose(1.5 * radius, d)

        # Move in the y- direction and compute the depth
        d = world.depth_at_angles(np.pi / 2, 0.0, 0.5 * radius, 0.0)
        np.testing.assert_allclose(0.5 * radius, d)
        d = world.depth_at_angles(-np.pi / 2, 0.0, 0.5 * radius, 0.0)
        np.testing.assert_allclose(1.5 * radius, d)


def test_radius():
    world_r1 = CylinderWorld(resolution=128,
                             n_coeffs=4,
                             rng=np.random.RandomState(SEED),
                             radius=1.0)
    for radius in np.linspace(0.5, 2, 11):
        world = CylinderWorld(resolution=128,
                              n_coeffs=4,
                              rng=np.random.RandomState(SEED),
                              radius=radius)
        for x in np.linspace(-0.9 * radius, 0.9 * radius, 11):
            for y in np.linspace(-0.9 * radius, 0.9 * radius, 11):
                if np.sqrt(x**2 + y**2) >= world.radius:
                    continue
                for theta in np.linspace(-np.pi, np.pi, 11):
                    I = world.render(x, y, theta)
                    I2 = world_r1.render(x / radius, y / radius, theta)
                    np.testing.assert_allclose(I, I2, atol=1e-6)


def test_render_with_gradient():
    world = CylinderWorld(resolution=1000,
                          n_coeffs=4,
                          rng=np.random.RandomState(SEED))

    for anti_alias in [False, True]:
        for x in np.linspace(-0.9, 0.9, 11):
            for y in np.linspace(-0.9, 0.9, 11):
                if x**2 + y**2 >= 0.9:
                    continue
                for theta in np.linspace(-np.pi, np.pi, 11):
                    I, dI_direct = world.render_with_gradient(
                        x, y, theta, anti_alias=anti_alias)
                    dI_num = (0.5 * (np.roll(I, -1) - np.roll(I, 1)) *
                              world.resolution / (2.0 * np.pi))
                    np.testing.assert_allclose(dI_direct, dI_num, atol=1e-2)


def test_render_at_angles_with_gradient():
    world = CylinderWorld(resolution=1000,
                          n_coeffs=4,
                          rng=np.random.RandomState(SEED))

    eta = 1e-6
    for anti_alias in [False, True]:
        for β in np.linspace(-np.pi, np.pi, 11):
            for x in np.linspace(-0.9, 0.9, 11):
                for y in np.linspace(-0.9, 0.9, 11):
                    if x**2 + y**2 >= 0.9:
                        continue
                    for theta in np.linspace(-np.pi, np.pi, 11):
                        I, dI_direct = world.render_at_angles_with_gradient(
                            β, x, y, theta, anti_alias=anti_alias)
                        I0 = world.render_at_angles(β - 0.5 * eta, x, y, theta,
                                                    anti_alias)
                        I1 = world.render_at_angles(β + 0.5 * eta, x, y, theta,
                                                    anti_alias)
                        dI_num = (I1 - I0) / eta
                        np.testing.assert_allclose(dI_direct,
                                                   dI_num,
                                                   atol=1e-5)


def test_inner_angle():
    assert CylinderWorld._inner_angle(1.0, 2.0) == 1.0
    assert CylinderWorld._inner_angle(2.0, 1.0) == -1.0
    assert CylinderWorld._inner_angle(-0.25, 0.25) == 0.5
    assert CylinderWorld._inner_angle(0.25, -0.25) == -0.5
    assert CylinderWorld._inner_angle(3.0 * np.pi, 2.0 * np.pi) == np.pi
    assert CylinderWorld._inner_angle(2.0 * np.pi, 3.0 * np.pi) == -np.pi
    assert CylinderWorld._inner_angle(3.0 * np.pi, 1.0 * np.pi) == 0.0
    assert CylinderWorld._inner_angle(1.0 * np.pi, 3.0 * np.pi) == 0.0
    assert CylinderWorld._inner_angle(-1.0 * np.pi, 4.0 * np.pi) == np.pi
    assert CylinderWorld._inner_angle(4.0 * np.pi, -1.0 * np.pi) == -np.pi

