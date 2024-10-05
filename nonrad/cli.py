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
from nonrad.ccd import (
    get_cc_structures,
    get_dQ,
    get_PES_from_vaspruns,
    get_omega_from_PES,
)
from glob import glob


@click.group()
def nonrad():
    """Command Line Interface for nonrad."""
    pass


@click.command()
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
def prep_ccd(ground_path, excited_path, cc_dir, displace=[-0.5, 0.5, 9]):
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


@click.command()
@click.argument("cc_dir", type=click.Path())
@click.argument("ground_files", type=click.Path())
@click.argument("excited_files", type=click.Path())
@click.option("--plot", "-p", is_flag=True, help="Plot the potential energy surfaces.")
@click.option(
    "--plot_name",
    "-n",
    default="pes.png",
    help="Name of the plot file. Default is `pes.png`.",
)
def pes(cc_dir, ground_files, excited_files, plot=False, plot_name="pes.png"):
    """Extracting potential energy surfaces and relevant parameters from CCD calculations.

    Parmeters: \n
    ----------- \n
    cc_dir : str \n
        Path to the directory containing the CCD calculations. It should contain the `ground` and `excited` directories.
        Each of these directories should contain the directories corresponding to the displacements. \n
    ground_files : str \n
        Path to the directory containing the ground state calculation files. It should contain the `CONTCAR` and `vasprun.xml` files
        for 0% displacement. \n
    excited_files : str \n
        Path to the directory containing the excited state calculation files. It should contain the `CONTCAR` and `vasprun.xml` files
        for 0% displacement. \n
    plot : bool \n
        If True, the potential energy surfaces are plotted. Default is False. \n
    plot_name : str \n
        Name of the plot file. Default is `pes.png`.

    """
    ground_struct = Structure.from_file(Path(ground_files, "CONTCAR"))
    excited_struct = Structure.from_file(Path(excited_files, "CONTCAR"))

    # calculate dQ
    dQ = get_dQ(ground_struct, excited_struct)  # amu^{1/2} Angstrom
    print(f"dQ = {dQ:.2f} amu^{1/2} Angstrom")

    # this prepares a list of all vasprun.xml's from the CCD calculations
    ground_vaspruns = glob(Path(cc_dir, "ground", "*", "vasprun.xml"))
    excited_vaspruns = glob(Path(cc_dir, "excited", "*", "vasprun.xml"))

    # remember that the 0% displacement was removed before? we need to add that back in here
    ground_vaspruns = ground_vaspruns + [Path(ground_files, "vasprun.xml")]
    excited_vaspruns = excited_vaspruns + [Path(excited_files, "vasprun.xml")]

    # extract the potential energy surface
    Q_ground, E_ground = get_PES_from_vaspruns(
        ground_struct, excited_struct, ground_vaspruns
    )
    Q_excited, E_excited = get_PES_from_vaspruns(
        ground_struct, excited_struct, excited_vaspruns
    )

    # the energy surfaces are referenced to the minimums, so we need to add dE (defined before) to E_excited
    E_excited = dE + E_excited

    # calculate omega
    if plot:
        fig, ax = plt.subplots(figsize=(5, 5))
        ax.scatter(Q_ground, E_ground, s=10)
        ax.scatter(Q_excited, E_excited, s=10)
        # by passing in the axis object, it also plots the fitted curve
        q = np.linspace(-1.0, 3.5, 100)
        ground_omega = get_omega_from_PES(Q_ground, E_ground, ax=ax, q=q)
        excited_omega = get_omega_from_PES(Q_excited, E_excited, ax=ax, q=q)
        ax.set_xlabel("$Q$ [amu$^{1/2}$ $\AA$]")
        ax.set_ylabel("$E$ [eV]")
        plt.savefig(plot_name, dpi=300, format=plot_name.split(".")[-1])

    print(f"Ground state omega = {ground_omega:.2f} eV")
    print(f"Excited state omega = {excited_omega:.2f} eV")

    return 0


# Add subcommands to the main group
nonrad.add_command(prep_ccd)
nonrad.add_command(pes)

if __name__ == "__main__":
    nonrad()
