#!/usr/bin/env python
# coding: utf-8

# # Run main program

# In[ ]:


import sys
import ase.units

from esteem.trajectories import generate_md_trajectory, recalculate_trajectory
from esteem.trajectories import find_initial_geometry, get_trajectory_list


class QMDTrajTask:

    def __init__(self, **kwargs):
        self.wrapper = None
        self.script_settings = None
        self.task_command = 'qmd'
        self.train_params = {}
        args = self.make_parser().parse_args("")
        for arg in vars(args):
            setattr(self, arg, getattr(args, arg))

    # Main routine
    def run(self):
        """
        Sets up and runs ab initio molecular dynamics for a given molecule (whose name is
        provided by ``self.seed``) in the ground or excited state (specified by ``self.target``).

        One trajectory is generated for each label in ``self.which_trajs`` (default: all
        ``self.ntraj`` labels 'A', 'B', ...), starting from the geometry found by
        :func:`esteem.trajectories.find_initial_geometry` (no geometry optimisation is performed).
        The stored snapshots are written to ``{seed}_{targ}_{label}_{traj_suffix}.traj``, where
        ``targ`` is 'gs', 'es1' and so on, for use in future runs.

        The run is divided into equilibration (``self.nequil`` runs of ``self.md_steps`` MD steps
        each, with timestep ``self.md_timestep``), then snapshot generation (``self.nsnap`` runs of
        ``self.md_steps`` MD steps each, with timestep ``self.md_timestep``), with one snapshot
        stored at the end of each snapshot run.

        A thermostat (wrapper-dependent) at temperature ``self.temp`` means we stay in the NVT ensemble.

        Constraints can be applied using ``self.constraints`` - the meaning depends on the underlying wrapper.

        If ``self.input_suffix`` is set, no MD is run: instead the energies and forces of the frames
        of a pre-existing trajectory are recalculated (see
        :func:`esteem.trajectories.recalculate_trajectory`).

        The task settings are attributes of the task object (see below for a listing). Key settings
        include ``basis``, ``func``, ``md_timestep``, ``md_steps``, ``nsnap``, ``nequil``, ``temp``
        and ``constraints``.

        ``self.wrapper`` must be set to a wrapper object providing ``run_md`` (for the MD) and
        ``singlepoint`` (for the snapshots, and for recalculating an existing trajectory).
        """

        # Check input args are valid
        # validate_args(self)

        # Make sure trajectory choices are valid
        all_trajs = get_trajectory_list(self.ntraj)
        if self.which_trajs is None:
            which_trajs = all_trajs
        else:
            which_trajs = self.which_trajs
            for traj_label in which_trajs:
                if traj_label not in all_trajs:
                    raise Exception(f"Invalid trajectory name: {traj_label}")

        # Set up calculator parameters dict
        calc_params = {"basis": self.basis,
                       "func": self.func,
                       "target": self.target,
                       "disp": self.disp}

        if self.seed in self.charges:
            charge = self.charges[self.seed]
        else:
            charge = 0

        # Perform QM calculations on an existing trajectory
        if self.input_suffix is not None:
            for traj_label in which_trajs:
                input_target = None  # re-do if ever necessary to start from anything else
                input_traj_range = self.input_traj_range
                recalculate_trajectory(self.seed, self.target, traj_label, self.traj_suffix,
                                       input_target, self.input_suffix,
                                       self.wrapper, calc_params, charge=charge,
                                       solvent=self.solvent, input_traj_range=input_traj_range)
            return

        # Generate training data in ntraj labelled trajectories
        for traj_label in which_trajs:

            # Find (or relax) initial geometry
            if isinstance(self.target, list):
                targ = self.target[0]
            else:
                targ = self.target
            calc_params['target'] = targ
            model = find_initial_geometry(self.seed, geom_opt_func=None, calc_params={},  # self.wrapper.geom_opt,calc_params,
                                          which_traj=traj_label, ntraj=self.ntraj)
            if self.constraints is not None:
                if 'spring' in self.constraints:
                    spring = self.constraints['spring']
                    if 'bond' in spring:
                        # Extract atoms indices and bondlength from NWChem constraint
                        # NWChem expects 1-indexed atom numbers, ASE expects 0-indexed
                        # so subtract 1
                        atoms0 = int(spring.split()[1]) - 1
                        atoms1 = int(spring.split()[2]) - 1
                        bondlength = float(spring.split()[4]) * ase.units.Bohr
                        print('Setting distance: ', atoms0, atoms1, bondlength)
                        model.set_distance(atoms0, atoms1, bondlength, fix=0)

            # Pass in routine to actually run MD into generic Snapshot MD driver
            generate_md_trajectory(model, self.seed, self.target, traj_label, self.traj_suffix,
                                   traj_wrapper=self.wrapper, count_snaps=self.nsnap, count_equil=self.nequil,
                                   md_steps=self.md_steps, md_timestep=self.md_timestep, md_friction=None,
                                   temp=self.temp, traj_calc_params=calc_params,
                                   solvent=self.solvent, charge=charge, constraints=self.constraints,
                                   store_full_traj=True)

    # Create parser to read command line values

    def make_parser(self):

        import argparse

        main_help = ('''Generate trajectory files by running QMD or by recalculating
                        energies and forces at previously evaluated trajectory points.''')
        epi_help = ('')
        parser = argparse.ArgumentParser(description=main_help, epilog=epi_help)
        parser.add_argument('--seed', '-s', type=str, help="Name of the molecule or solute-solvent system (eg '{solute}_{solvent}'), used to find the input geometry '{seed}.xyz' and to name the trajectory files. Set automatically by the main driver.")
        parser.add_argument('--traj_suffix', '-S', default='training', type=str, help="Suffix appended to the trajectory filenames, named '{seed}_{targ}_{label}_{traj_suffix}.traj', where targ is 'gs', 'es1' and so on.")
        parser.add_argument('--geom_prefix', '-G', default='', type=str, help="Directory from which the initial geometry '{seed}.xyz' is copied into the '{seed}_qmd' working directory; '{solu}', '{solv}' and '{seed}' are substituted.")
        parser.add_argument('--basis', '-b', default='6-311++G**', type=str, help='Basis set passed to the QM wrapper (or other wrapper-dependent run parameter).')
        parser.add_argument('--func', '-f', default='PBE0', type=str, help='Exchange-correlation functional passed to the QM wrapper (or other wrapper-dependent run parameter).')
        parser.add_argument('--disp', '-d', default=True, type=bool, help='Whether to apply Grimme D3 dispersion corrections.')
        parser.add_argument('--target', '-t', default=None, type=int, help='Electronic state for the dynamics: None or 0 for the ground state, n for the nth excited state. A list of states can be given, in which case the first is used to choose the initial geometry.')
        parser.add_argument('--charges', '-C', default={}, nargs='?', type=dict, help='Dict mapping seed names to the total charge of that system; seeds not in the dict are given charge 0. Not for command-line use.')
        parser.add_argument('--restart', '-r', default=False, type=bool, help='Intended to make the calculator restart from previous results at each geometry when the trajectory has already been run for another state (currently not used by this task).')
        parser.add_argument('--md_timestep', '-q', default=0.5, type=float, help="Molecular dynamics timestep. Units depend on the wrapper: fs for NWChem, ASE time units (eg 0.5*ase.units.fs) for ORCA and ASE-based wrappers. Can be a dict with keys 'EQ' and 'MD' to give separate values for equilibration and snapshot runs.")
        parser.add_argument('--md_steps', '-Q', default=100, type=int, help='Number of molecular dynamics timesteps in each equilibration or snapshot run; one snapshot is stored at the end of each snapshot run.')
        parser.add_argument('--temp', '-T', default=300.0, type=float, help='Thermostat temperature for the NVT molecular dynamics, in K.')
        parser.add_argument('--ntraj', '-n', default=1, type=int, help="Total number of trajectories, labelled 'A', 'B', 'C' and so on (then 'AA', 'AB' after 'Z'). Run as a job array, each array task runs the trajectory with that index.")
        parser.add_argument('--nsnap', '-N', default=100, type=int, help='Number of snapshot runs in each trajectory, each of md_steps steps and contributing one snapshot.')
        parser.add_argument('--solvent', '-i', default=None, type=str, help="Name of the implicit solvent for the QM calculations, or None for gas phase; '{solu}', '{solv}' and '{seed}' are substituted.")
        parser.add_argument('--solvent_settings', default=None, type=str, help='Implicit solvent settings to use in place of the solvent name (currently not used by this task).')
        parser.add_argument('--input_suffix', '-I', default=None, type=str, help="Suffix of an existing ground-state trajectory '{seed}_gs_{label}_{input_suffix}.traj'. If set, no MD is run; instead the energies and forces of its frames are recalculated and written to a trajectory with traj_suffix.")
        parser.add_argument('--input_traj_range', '-R', default=None, type=int, help='Frames of the input trajectory to recalculate when input_suffix is set, as a range or list of frame indices. None means all frames.')
        parser.add_argument('--nequil', '-e', default=5, type=int, help='Number of equilibration runs, each of md_steps steps, performed before the snapshot runs.')
        parser.add_argument('--which-trajs', '-w', default=None, type=str, help="List of trajectory labels (eg ['A', 'B']) to generate or recalculate, from the ntraj available. None means all of them.")
        parser.add_argument('--constraints', '-c', default=None, type=str, help="Constraints passed to the wrapper's MD calculator (format depends on the wrapper). For a dict with a 'spring' entry containing an NWChem 'spring bond' string, the initial geometry is also set to the target bond length. None means no constraints.")
        parser.add_argument('--dynamics', '-X', default=None, type=str, help="ASE dynamics object or settings for the MD (currently not passed on by this task, which uses each wrapper's own dynamics).")

        return parser

        # Notes on defaults:
        # 1 a.u. = 0.02419 fs, 100 steps of 10 aut = 24fs between snapshots

    def validate_args(self, args):
        default_args = self.make_parser().parse_args("")
        for arg in vars(args):
            if arg not in default_args:
                raise Exception(f"Unrecognised argument '{arg}'")


# # Handle Inputs

# In[ ]:


def get_parser():
    qmdtraj = QMDTrajTask()
    return qmdtraj.make_parser()


if __name__ == '__main__':

    # Set up NWChem by default
    from esteem.wrappers import nwchem

    qm_wrapper = nwchem.NWChemWrapper()
    qm_wrapper.nwchem_setup()

    # Parse command line arguments
    parser = get_parser()
    args = parser.parse_args()
    print(args)

    # Run main program
    qmd = QMDTrajTask()
    for arg in vars(args):
        setattr(qmd, arg, getattr(args, arg))
    qmd.wrapper = qm_wrapper
    qmd.run()
