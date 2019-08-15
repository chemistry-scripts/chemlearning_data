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
from chemlearning_data.molecule import Molecule
from cclib.parser.utils import PeriodicTable


def extract_xyz_geometries(xyz_file):
    """Extract xyz geometries from files in QM9"""
    properties = list()
    coordinates = list()
    atoms = list()

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

    # Cleanup atoms list thanks to cclib PeriodicTable, convert them to atomic number
    periodic_table = PeriodicTable()
    elements_list = [periodic_table.number[atom] for atom in atoms]

    # Build the Molecule object and return it
    molecule = Molecule(coordinates, elements_list)
    return molecule


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


def compute_dispersion_correction(xyz_file, tar_file, locations, gaussian_args):
    """
    Wrapper around all operations:
        - Decompression of file
        - Parsing and retrieving useful data
        - Setting up computation
        - Running Gaussian computation
        - Retrieving computation results
    """
    logging.info("Computing dispersion correction for %s", str(xyz_file))
    # Extract proper file form tar, but only in RAM
    extracted_xyz = tar_file.extractfile(xyz_file)
    molecule = extract_xyz_geometries(extracted_xyz)

    # Get useful data for building the Gaussian job
    file_name = str(xyz_file).split("_")[1]
    file_id = file_name.split(".")[0]

    # Build the Gaussian job
    logging.debug("Setting up Gaussian job for %s", str(xyz_file))
    job = GaussianJob(
        basedir=locations["computations"],
        name=file_name,
        molecule=molecule,
        job_id=file_id,
        gaussian_args=gaussian_args,
    )
    job.setup_computation()

    # Run the job
    logging.debug("Starting Gaussian job for %s", str(xyz_file))
    job.run()

    # Retrieve results upon completion
    logging.debug("Parsing results for %s", str(xyz_file))
    energies = job.get_energies()
    # Retrieve all useful energies
    return energies


def main():
    """Launcher."""
    # Setup all variables
    qm9_location = "qm9/qm9.tar.bz2"
    data_location = "data"
    computations_location = "computation"
    folders = dict()
    folders["qm9"] = qm9_location
    folders["data"] = data_location
    folders["computations"] = computations_location

    # Set up local Gaussian arguments
    gaussian_arguments = get_gaussian_arguments()

    # Iterate over contents of tar file and submit every job to the executor
    results = list()
    with tarfile.open(name=qm9_location, mode="r:bz2") as qm9_tar:
        with ProcessPoolExecutor() as executor:
            for xyz_file in qm9_tar:
                future_result = executor.submit(
                    compute_dispersion_correction,
                    xyz_file=xyz_file,
                    tar_file=qm9_tar,
                    locations=folders,
                    gaussian_args=gaussian_arguments,
                )
                results.append((str(xyz_file), future_result))


if __name__ == "__main__":
    # Setup logger
    setup_logger()
    main()
