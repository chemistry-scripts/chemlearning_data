#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Copyright (c) 2019, E. Nicolas

"""Tools to use data (especially from QM9) for machine learning applications."""

# Here comes your imports
import cclib
import tarfile
import os
import chemlearning_data.gaussian_job as gaussian_job


# Here comes your function definitions
def compute_dispersion_correction(xyz_geometry):
    pass


def extract_xyz_geometries(xyz_file):
    """Extract xyz geometries from files in QM9"""
    properties = []
    coordinates = []
    atoms = []

    # Open the file, then retrieve the first two lines for the properties included.
    # With natoms, read all useful lines to get the coordinates, splitting them between
    # atom nuclei on one list and xyz coordinates on the other. See qm9_readme for details
    with open(xyz_file) as file:
        n_atoms = int(file.readline())
        properties.append(file.readline().split("\t"))
        for i in range(0, n_atoms):
            line = file.readline().split("\t")
            atoms.append(line[0])
            coordinates.append(line[1:4])
    return coordinates, atoms, properties


def get_qm9files(data_location):
    with os.scandir(data_location) as it:
        for entry in it:
            # Ignore other files than .xyz
            if entry.name.endswith(".xyz"):
                # File is called dsgdb9nsd_012503.xyz
                # We retrieve only the part after _
                file_name = str(entry.name).split("_")[1]
                file_id = file_name.split(".")[0]
                extract_xyz_geometries(entry.name)


def main():
    """Launcher."""
    qm9_location = "qm9/qm9.tar.bz2"
    data_location = "data"
    qm9_tar = tarfile.open(name=qm9_location, mode="r:bz2")
    qm9_tar.extractall(path=data_location)
    qm9_tar.close()


if __name__ == "__main__":
    main()
