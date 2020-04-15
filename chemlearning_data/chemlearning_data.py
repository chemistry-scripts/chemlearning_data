#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Copyright (c) 2019, E. Nicolas

"""Tools to use data (especially from QM9) for machine learning applications."""

# Here comes your imports
import logging
import logging.handlers
import multiprocessing
import os
import tarfile
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
    print(n_atoms)
    _ = xyz_file.readline()  # Property line. Do not use.

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


def compute_dispersion_correction(queue, logger_configurer, xyz_file, tar_file, locations, gaussian_args):
    """
    Wrapper around all operations:
        - Decompression of file
        - Parsing and retrieving useful data
        - Setting up computation
        - Running Gaussian computation
        - Retrieving computation results
    """
    # Setup logging within queue
    logger_configurer(queue)

    logging.info("Computing dispersion correction for %s", str(xyz_file))
    # Extract proper file form tar, but only in RAM
    extracted_xyz = tar_file.extractfile(xyz_file)
    molecule = extract_xyz_geometries(extracted_xyz)

    # Get useful data for building the Gaussian job
    file_name = xyz_file.name.split("_")[1]  # Remove header dsgdb9nsd_
    file_id = file_name.split(".")[0]  # Get file id. (Remove .xyz)

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


# This is the listener process top-level loop: wait for logging events
# (LogRecords)on the queue and handle them, quit when you get a None for a
# LogRecord.
def listener_process(queue, configurer):
    configurer()
    while True:
        try:
            record = queue.get()
            if (
                record is None
            ):  # We send this as a sentinel to tell the listener to quit.
                break
            logger = logging.getLogger(record.name)
            logger.handle(record)  # No level or filter logic applied - just do it!
        except Exception:
            import sys
            import traceback

            print("Whoops! Problem:", file=sys.stderr)
            traceback.print_exc(file=sys.stderr)


def main():
    """Launcher."""
    # Setup all variables
    qm9_location = "qm9/qm9_test.tar.bz2"
    data_location = "data"
    computations_location = "computation"
    output_file = "qm9/qm9_dispersion.data"
    folders = dict()
    folders["basedir"] = os.getcwd()
    folders["qm9"] = os.path.join(os.getcwd(), qm9_location)
    folders["data"] = os.path.join(os.getcwd(), data_location)
    folders["computations"] = os.path.join(os.getcwd(), computations_location)

    # Create all folders where necessary
    for folder in folders:
        Path(folder).mkdir(parents=True, exist_ok=True)

    # Set up local Gaussian arguments
    gaussian_arguments = get_gaussian_arguments()

    # Set up listener for logging events
    queue = multiprocessing.Manager().Queue(-1)
    listener = multiprocessing.Process(
        target=listener_process, args=(queue, setup_logger)
    )

    listener.start()

    # Iterate over contents of tar file and submit every job to the executor
    results = list()
    with tarfile.open(name=qm9_location, mode="r:bz2") as qm9_tar:
        with ProcessPoolExecutor() as executor:
            for xyz_file in qm9_tar:
                future_result = executor.submit(
                    compute_dispersion_correction,
                    logger_configurer=setup_logger,
                    xyz_file=xyz_file,
                    tar_file=qm9_tar,
                    locations=folders,
                    gaussian_args=gaussian_arguments,
                )
                results.append((str(xyz_file), future_result))

    # Tell the listener it has to end by sending a None to the queue
    queue.put_nowait(None)
    listener.join()

    # Retrieve results
    os.chdir(folders["basedir"])
    logging.info("Writing data to output file: %s", output_file)
    # Iterate over all results to build the final table
    with open(output_file, mode="w") as out_file:
        for _, result in results:
            # Careful for actual value extracting, dict are not ordered. Use actual keys.
            values = [str(val) for val in result.result().values()]
            out_file.write("\t".join(values) + "\n")


if __name__ == "__main__":
    main()
