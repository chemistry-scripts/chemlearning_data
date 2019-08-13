#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Copyright (c) 2019, E. Nicolas

"""Tools to use data (especially from QM9) for machine learning applications."""

# Here comes your imports
import logging
import os
import tarfile
from concurrent.futures import ProcessPoolExecutor
from chemlearning_data.gaussian_job import GaussianJob


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
        for _ in range(0, n_atoms):
            line = file.readline().split("\t")
            atoms.append(line[0])
            coordinates.append(line[1:4])
    return coordinates, atoms, properties


def get_qm9files(data_location):
    """Get a list of all xyz files, returned as a dict of ids/file_name"""
    qm9files = dict()
    with os.scandir(data_location) as folder_content:
        for entry in folder_content:
            # Ignore other files than .xyz
            if entry.name.endswith(".xyz"):
                # File is called dsgdb9nsd_012503.xyz
                # We retrieve only the part after _
                file_name = str(entry.name).split("_")[1]
                file_id = file_name.split(".")[0]
                qm9files[file_id] = file_name
    return qm9files


def get_gaussian_arguments():
    """All arguments necessary for a Gaussian computation"""
    args = dict()
    args["functional"] = "B3LYP"
    args["dispersion"] = "GD3"
    args["basisset"] = "6-31G*"
    return args


def setup_logger():
    """Setup logging"""
    # Setup logging
    logger = logging.getLogger()
    logger.setLevel(logging.DEBUG)

    stream_handler = logging.StreamHandler()
    stream_handler.setLevel(logging.DEBUG)

    formatter = logging.Formatter("%(asctime)s :: %(levelname)s :: %(message)s")
    stream_handler.setFormatter(formatter)

    logger.addHandler(stream_handler)


def main():
    """Launcher."""
    # Setup all variables
    qm9_location = "qm9/qm9.tar.bz2"
    data_location = "data"
    computations_location = "computation"
    gaussian_arguments = get_gaussian_arguments()

    # Extract tar bz2 archive
    qm9_tar = tarfile.open(name=qm9_location, mode="r:bz2")
    qm9_tar.extractall(path=data_location)
    qm9_tar.close()
    # List all xyz files
    qm9files = get_qm9files(data_location=data_location)

    # Extract all useful data, retrieve a dict of Gaussian jobs
    gaussian_jobs = list()
    for file_id, file_name in zip(qm9files.keys(), qm9files.values()):
        molecules = extract_xyz_geometries(file_name)
        gaussian_job = GaussianJob(
            basedir=computations_location,
            name=file_name,
            molecule=molecules,
            job_id=file_id,
            gaussian_args=gaussian_arguments,
        )
        gaussian_jobs.append(gaussian_job)

    # Setup all computations
    for job in gaussian_jobs:
        job.setup_computation()

    # Run all computations in parallel
    with ProcessPoolExecutor() as executor:
        for job in gaussian_jobs:
            executor.submit(job.run)

    # Retrieve all useful energies
    energies = dict()
    for job in gaussian_jobs:
        energies[job.job_id] = job.get_energies()


if __name__ == "__main__":
    # Setup logger
    setup_logger()
    main()
