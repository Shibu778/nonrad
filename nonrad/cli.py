# -*- coding: utf-8 -*-
# Copyright (c) Chris G. Van de Walle
# Distributed under the terms of the MIT License.

# Author : Shibu Meher
# Date : 2021-06-01

import click
import os
import numpy as np
from pathlib import Path
from shutil import copyfile
from pymatgen.core import Structure
from nonrad.ccd import get_cc_structures


@click.group()
def nonrad():
    """Command Line Interface for nonrad."""
    pass


@click.command("pccd")
@click.argument("ground_path", type=click.Path(exists=True))
@click.argument("excited_path", type=click.Path(exists=True))
@click.argument("cc_dir", type=click.Path())
@click.option(
    "--displace",
    "-d",
    nargs=3,
    type=float,
    default=[-0.5, 0.5, 9],
    help="Displacement range and number of displacements.",
)
def prepare_ccd(ground_path, excited_path, cc_dir, displace=[-0.5, 0.5, 9]):
    """
    Prepare the input files for CCD calculation. From the ground and excited state calculations,
    the CONTCAR file is read. The displacements are generated and written to the ccd directory. \n

    ground_path : Path to the directory containing the ground state calculation files. \n
    excited_path : Path to the directory containing the excited state calculation files. \n
    cc_dir : Path to the directory where the CCD input files will be written. \n
    displace : List containing the minimum and maximum displacements as a percentage
        and the number of displacements to generate. Default is [-0.5, 0.5, 9]. \n
    """
    # equilibrium structures from your first-principles calculation
    ground_files = Path(ground_path)
    ground_struct = Structure.from_file(str(ground_files / "CONTCAR"))
    excited_files = Path(excited_path)
    excited_struct = Structure.from_file(str(excited_files / "CONTCAR"))

    # output directory that will contain the input files for the CC diagram
    cc_dir = Path(cc_dir)
    os.makedirs(str(cc_dir), exist_ok=True)
    os.makedirs(str(cc_dir / "ground"), exist_ok=True)
    os.makedirs(str(cc_dir / "excited"), exist_ok=True)

    # displacements as a percentage, this will generate the displacements
    # -50%, -37.5%, -25%, -12.5%, 0%, 12.5%, 25%, 37.5%, 50%
    displacements = np.linspace(displace[0], displace[1], int(displace[2]))

    # note: the returned structures won't include the 0% displacement, this is intended
    # it can be included by specifying remove_zero=False
    ground, excited = get_cc_structures(ground_struct, excited_struct, displacements)

    for i, struct in enumerate(ground):
        working_dir = cc_dir / "ground" / str(i)
        os.makedirs(str(working_dir), exist_ok=True)

        # write structure and copy necessary input files
        struct.to(filename=str(working_dir / "POSCAR"), fmt="poscar")
        for f in ["KPOINTS", "POTCAR", "INCAR", "job_script.sh"]:
            copyfile(str(ground_files / f), str(working_dir / f))

    for i, struct in enumerate(excited):
        working_dir = cc_dir / "excited" / str(i)
        os.makedirs(str(working_dir), exist_ok=True)

        # write structure and copy necessary input files
        struct.to(filename=str(working_dir / "POSCAR"), fmt="poscar")
        for f in ["KPOINTS", "POTCAR", "INCAR", "job_script.sh"]:
            copyfile(str(excited_files / f), str(working_dir / f))

    return 0


# Add subcommands to the main group
nonrad.add_command(prepare_ccd)

if __name__ == "__main__":
    nonrad()
