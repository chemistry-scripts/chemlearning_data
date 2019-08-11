#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Copyright (c) 2019, E. Nicolas

"""Tests for molecule class"""

from chemlearning_data.molecule import Molecule


def test_xyz_geometry():
    """Testing xyz geometry generator"""
    elements_list = [1, 1]
    coordinates = [[0.0000, 0.0000, 0.0000], [0.0000, 0.0000, 0.7400]]
    ref_geometry = [
        "H                      0.000000                  0.000000                  0.000000",
        "H                      0.000000                  0.000000                  0.740000",
    ]

    molecule = Molecule(coordinates, elements_list)
    xyz_geometry = molecule.xyz_geometry()
    assert xyz_geometry == ref_geometry
