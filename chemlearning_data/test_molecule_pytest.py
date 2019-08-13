#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Copyright (c) 2019, E. Nicolas

"""Tests for molecule class"""

from chemlearning_data.molecule import Molecule
import pytest

@pytest.fixture
def molecule_1():
    elements_list = [1, 1]
    coordinates = [[0.0000, 0.0000, 0.0000], [0.0000, 0.0000, 0.7400]]
    return Molecule(coordinates, elements_list)

def test_xyz_geometry(molecule_1):
    """Testing xyz geometry generator"""
    ref_geometry = [
        "H                      0.000000                  0.000000                  0.000000",
        "H                      0.000000                  0.000000                  0.740000",
    ]

    xyz_geometry = molecule_1.xyz_geometry()
    assert xyz_geometry == ref_geometry
