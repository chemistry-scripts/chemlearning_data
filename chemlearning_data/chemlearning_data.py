#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Copyright (c) 2019, E. Nicolas

"""Tools to use data (especially from QM9) for machine learning applications."""

# Here comes your imports
import logging
import logging.handlers
import os
import tarfile
from cclib.parser.utils import PeriodicTable
from chemlearning_data.gaussian_job import GaussianJob
from chemlearning_data.molecule import Molecule


def extract_xyz_geometries(xyz_file):
    """Extract xyz geometries from files in QM9"""
    properties = list()
    coordinates = list()
    atoms = list()

    # Open the file, then retrieve the first two lines for the properties included.
    # With natoms, read all useful lines to get the coordinates, splitting them between
    # atom nuclei on one list and xyz coordinates on the other. See qm9_readme for details
    n_atoms = int(xyz_file.readline())
    print(n_atoms)
    properties = xyz_file.readline().split(b"\t")
    properties = [prop.decode("utf-8") for prop in properties]

    for line in xyz_file.readlines()[0:n_atoms]:
        line = line.split(b"\t")
        line = [elem.decode("utf-8") for elem in line]
        atoms.append(str(line[0]))
        coordinates.append([float(xyz) for xyz in line[1:4]])

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
    file_name = xyz_file.name.split("_")[1] # Remove header dsgdb9nsd_
    file_id = file_name.split(".")[0] # Get file id. (Remove .xyz)

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
    qm9_location = "qm9/qm9_test.tar.bz2"
    data_location = "data"
    computations_location = "computation"
    output_file = "qm9/qm9_dispersion.data"
    folders = dict()
    folders["qm9"] = qm9_location
    folders["data"] = data_location
    folders["computations"] = computations_location

    # Set up local Gaussian arguments
    gaussian_arguments = get_gaussian_arguments()

    setup_logger()

    # Iterate over contents of tar file and submit every job to the executor
    results = list()
    with tarfile.open(name=qm9_location, mode="r:bz2") as qm9_tar:
        for xyz_file in qm9_tar:
            result = compute_dispersion_correction(
                xyz_file=xyz_file,
                tar_file=qm9_tar,
                locations=folders,
                gaussian_args=gaussian_arguments,
            )
            results.append((str(xyz_file), result))

    # Iterate over all results to build the final table
    with open(output_file, mode="w") as out_file:
        for result in results:
            out_file.write(result[0] + "\t" + "\t".join(result[1].result()) + "\n")


if __name__ == "__main__":
    main()
