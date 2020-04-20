#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Copyright (c) 2019, E. Nicolas

"""Tools to use data (especially from QM9) for machine learning applications."""

# Here comes your imports
import multiprocessing
import os
import re
import tarfile
import logging
from concurrent.futures.process import ProcessPoolExecutor
from pathlib import Path
from cclib.parser.utils import PeriodicTable
from chemlearning_data.gaussian_job import GaussianJob
from chemlearning_data.molecule import Molecule


def extract_xyz_geometries(xyz_file):
    """Extract xyz geometries from files in QM9"""
    coordinates = list()
    atoms = list()

    # Open the file, then retrieve the first two lines for the properties included.
    # With natoms, read all useful lines to get the coordinates, splitting them between
    # atom nuclei on one list and xyz coordinates on the other. See qm9_readme for details
    n_atoms = int(xyz_file.readline())
    xyz_file.readline()  # Property line. Do not use.

    for line in xyz_file.readlines()[0:n_atoms]:
        line = line.split(b"\t")
        line = [elem.decode("utf-8") for elem in line]
        atoms.append(str(line[0]))
        coords = line[1:4]
        for j, word in enumerate(coords):
            if "*^" in word:  # Some values are written as powers of 10: e.g. 1.999*^-6.
                values = re.split("\*\^", word)
                coords[j] = float(values[0]) * pow(10, int(values[1]))
            else:
                coords[j] = float(word)

        coordinates.append(coords)

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


def compute_dispersion_correction(
    molecule, file_id, file_name, locations, gaussian_args
):
    """
    Wrapper around all operations:
        - Decompression of file
        - Parsing and retrieving useful data
        - Setting up computation
        - Running Gaussian computation
        - Retrieving computation results
    """
    logging.info("Starting computation for %s", str(file_name))

    # Build the Gaussian job
    logging.debug("Setting up Gaussian job for %s", str(file_name))
    job = GaussianJob(
        basedir=locations["computations"],
        name=file_name,
        molecule=molecule,
        job_id=file_id,
        gaussian_args=gaussian_args,
    )
    job.setup_computation()

    # Run the job
    logging.debug("Starting Gaussian job for %s", str(file_name))
    job.run()

    # Retrieve results upon completion
    logging.debug("Parsing results for %s", str(file_name))
    # Retrieve all useful energies
    energies = job.get_energies()

    # Cleanup after job
    job.cleanup()

    return file_id, energies


def setup_logger():
    """Setup logging"""
    # Setup logging
    logging_level = logging.INFO

    logger_subprocess = multiprocessing.get_logger()
    logger_subprocess.setLevel(logging_level)

    logger_general = logging.getLogger()
    logger_general.setLevel(logging_level)

    stream_handler = logging.StreamHandler()
    stream_handler.setLevel(logging_level)

    formatter = logging.Formatter(
        "%(asctime)s :: %(levelname)s/%(processName)s :: %(message)s"
    )
    stream_handler.setFormatter(formatter)

    logger_subprocess.addHandler(stream_handler)
    logger_general.addHandler(stream_handler)
    return None


def main():
    """Launcher."""
    # Setup all variables
    qm9_location = "qm9"
    data_location = "data"
    computations_location = "computation"
    folders = dict()
    folders["basedir"] = os.getcwd()
    folders["qm9"] = os.path.join(os.getcwd(), qm9_location)
    folders["data"] = os.path.join(os.getcwd(), data_location)
    folders["computations"] = os.path.join(os.getcwd(), computations_location)

    qm9_location = os.path.join(folders["qm9"], "qm9_test.tar.bz2")
    # qm9_location = os.path.join(folders["qm9"], "qm9.tar.bz2")
    output_file = os.path.join(folders["data"], "qm9_dispersion.data")

    # Setup logging
    setup_logger()

    # Create all folders where necessary
    Path(folders["computations"]).mkdir(parents=True, exist_ok=True)
    Path(folders["data"]).mkdir(parents=True, exist_ok=True)

    # Set up local Gaussian arguments
    gaussian_arguments = get_gaussian_arguments()

    # Iterate over contents of tar file and submit every job to the executor
    results = list()
    with tarfile.open(name=qm9_location, mode="r:bz2") as qm9_tar:
        with ProcessPoolExecutor() as executor:
            for xyz_file in qm9_tar:
                # Extract proper file form tar, but only in RAM
                extracted_xyz = qm9_tar.extractfile(xyz_file)
                molecule = extract_xyz_geometries(extracted_xyz)

                # Get useful data for building the Gaussian job
                file_name = xyz_file.name.split("_")[1]  # Remove header dsgdb9nsd_
                file_id = file_name.split(".")[0]  # Get file id. (Remove .xyz)

                logging.info("Submitting %s", str(file_name))
                future_result = executor.submit(
                    compute_dispersion_correction,
                    molecule=molecule,
                    file_id=file_id,
                    file_name=file_name,
                    locations=folders,
                    gaussian_args=gaussian_arguments,
                )
                results.append(future_result)
                logging.info("Submitted %s", str(file_name))
            logging.info("All files submitted")
        logging.info("All subprocesses terminated")

    # Retrieve results
    os.chdir(folders["basedir"])
    # Iterate over all results to build the final table
    with open(output_file, mode="w") as out_file:
        out_file.write("File_ID\tSCF_Energy\tEnthalpy\tFree_Energy\n")
        for result in results:
            # Careful for actual value extracting, dict are not ordered. Use actual keys.
            file_id, energies = result.result()
            values = [
                energies["scfenergy"],
                energies["enthalpy"],
                energies["freeenergy"],
            ]
            values = [str(val) for val in values]
            out_file.write(str(file_id) + "\t" + "\t".join(values) + "\n")


if __name__ == "__main__":
    main()
