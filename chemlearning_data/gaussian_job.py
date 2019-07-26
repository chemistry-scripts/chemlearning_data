#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Copyright (c) 2019, E. Nicolas

"""Gaussian Job class to start job, run it and analyze it"""

import os
from cclib.io import ccread
import logging


class GaussianJob:
    """
    Class that can be used as a container for Gaussian jobs.

    Attributes:
        - input (input file, list of strings)
        - name (name of computation, string)
        - id (unique identifier, int)
        - natoms (number of atoms, int)
        - basedir (base directory, os.path object)
        - path (path in which to run current computation, os.path object)
        - input_filename (file_name.com, str)
        - output_filename (file_name.log, str)

    """

    # pylint: disable=too-many-instance-attributes

    def __init__(self, basedir, name, input_script, job_id, natoms):
        """Build  the GaussianJob class."""
        # pylint: disable=too-many-arguments
        # We need them all
        self.name = name
        self.input_script = input_script
        self.job_id = job_id
        self.natoms = natoms
        # base directory from which all computations are started
        self.basedir = basedir
        # Set path as: /base/directory/my_name.000xx/
        self.path = os.path.join(
            self.basedir, self.name.replace(" ", "_") + "." + str(self.job_id).zfill(4)
        )
        self.input_filename = self.name.replace(" ", "_") + ".com"
        self.output_filename = self.name.replace(" ", "_") + ".log"

    def run(self):
        """Start the job."""
        # Log computation start
        logger = logging.getLogger()
        logger.info("Starting computation %s", str(self.job_id))
        # Get into workdir, start gaussian, then back to basedir
        os.chdir(self.path)
        os.system("g16 < " + self.input_filename + " > " + self.output_filename)
        os.chdir(self.basedir)
        # Log end of computation
        logger.info("Finished computation %s", str(self.job_id))

    def extract_NBO_charges(self):
        """Extract NBO Charges parsing the output file."""
        # Log start
        logger = logging.getLogger()
        logger.info("Parsing results from computation %s", str(self.job_id))

        # Get into working directory
        os.chdir(self.path)

        # Initialize charges list
        charges = []

        with open(self.output_filename, mode="r") as out_file:
            line = "Foobar line"
            while line:
                line = out_file.readline()
                if "Summary of Natural Population Analysis:" in line:
                    logger.debug("ID %s: Found NPA table.", str(self.job_id))
                    # We have the table we want for the charges
                    # Read five lines to remove the header:
                    # Summary of Natural Population Analysis:
                    #
                    # Natural Population
                    # Natural    ---------------------------------------------
                    # Atom No    Charge        Core      Valence    Rydberg      Total
                    # ----------------------------------------------------------------
                    for _ in range(0, 5):
                        out_file.readline()
                    # Then we read the actual table:
                    for _ in range(0, self.natoms):
                        # Each line follow the header with the form:
                        # C  1    0.92349      1.99948     3.03282    0.04422     5.07651
                        line = out_file.readline()
                        line = line.split()
                        charges.append(line[2])
                    logger.debug(
                        "ID %s: Charges = %s",
                        str(self.job_id),
                        " ".join([str(i) for i in charges]),
                    )
                    # We have reached the end of the table, we can break the while loop
                    break
                # End of if 'Summary of Natural Population Analysis:'
        # Get back to the base directory
        os.chdir(self.basedir)
        return charges

    def get_coordinates(self):
        """Extract coordinates from output file."""
        # Log start
        logger = logging.getLogger()
        logger.info("Extracting coordinates for job %s", str(self.job_id))

        # Get into working directory
        os.chdir(self.path)

        # Parse file with cclib
        data = ccread(self.output_filename)

        #  Return the first coordinates, since it is a single point
        return data.atomcoords[0]

    def setup_computation(self):
        """
        Set computation up before running it.

        Create working directory, write input file
        """
        # Create working directory
        os.makedirs(self.path, mode=0o777, exist_ok=False)
        logging.info("Created directory %s", self.path)
        # Go into working directory
        os.chdir(self.path)
        # Write input file
        with open(self.input_filename, mode="w") as input_file:
            input_file.write("\n".join(self.input_script))
        # Get back to base directory
        os.chdir(self.basedir)

    def get_energies(self):
        """
        Retrieve HF energies plus thermochemical corrections

        :return:
        """
        # Log start
        logger = logging.getLogger()
        logger.info("Extracting energies for job %s", str(self.job_id))

        # Get into working directory
        os.chdir(self.path)

        # Parse file with cclib
        data = ccread(self.output_filename)

        #  Return the parsed energies as a dictionary
        energies = dict.fromkeys(["scfenergy", "enthalpy", "freeenergy"])
        energies["scfenergy"] = data.scfenergies[-1]
        energies["enthalpy"] = data.enthalpy
        energies["freeenergy"] = data.freeenergy

        return energies
