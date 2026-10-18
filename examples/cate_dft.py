from copy import deepcopy

# Import the drivers and wrappers we need
from esteem import drivers, parallel
from esteem.wrappers.orca import ORCAWrapper
from esteem.wrappers.amber import AmberWrapper

# Import the task classes we need
from esteem.tasks.solutes import SolutesTask
from esteem.tasks.solvate import SolvateTask
from esteem.tasks.clusters import ClustersTask
from esteem.tasks.spectra import SpectraTask

# Setup solute and solvents and target states
all_solutes = {'cate': 'catechol'}
all_solvents = {'cycl': 'cyclohexane', 'meth': 'methanol'}
all_solutes.update(all_solvents)
targets = {'gs':0,'es1':1,'es2':2}
orcacmd="/storage/nanosim/orca6/orca_6_0_0_shared_openmpi416_avx2/orca"

# Create tasks
solutes_task = SolutesTask()
solvate_task = SolvateTask()
clusters_task = ClustersTask()
spectra_task = SpectraTask()

# Set up Solutes task
all_solutes_tasks = {}
solutes_task.wrapper = ORCAWrapper()
solutes_task.wrapper.setup(nprocs=8)
solutes_task.script_settings = parallel.get_default_script_settings(solutes_task.wrapper)
from ase.calculators.orca import OrcaProfile
solutes_task.wrapper.orcaprofile=OrcaProfile(command=orcacmd)
# Setup different XC functionals as different tasks
funcs = ['PBE0','PBE']
basis_sets = ['def2-TZVP']
for basis in basis_sets:
    for func in funcs:
        for target in targets:  
            prefix = f'{target}_{func}'
            solutes_task.disp = True if 'D3BJ' not in func else False
            solutes_task.func = func
            solutes_task.basis = basis
            solutes_task.target = targets[target]
            solutes_task.directory = prefix
            all_solutes_tasks[prefix] = deepcopy(solutes_task)

# Set up solvate task
solvate_task.wrapper = AmberWrapper()
solvate_task.md_geom_prefix = f"gs_{solutes_task.func}"
solvate_task.nsteps = 2000
solvate_task.nsnaps = 500
solvate_task.script_settings = parallel.get_default_script_settings(solvate_task.wrapper)
solvate_task.boxsize = 15
solvate_task.ewaldcut = 9.0
all_solvate_tasks = {'md': solvate_task}

# Set up clusters task
clusters_task.wrapper = ORCAWrapper()
clusters_task.script_settings = parallel.get_default_script_settings(clusters_task.wrapper)
clusters_task.wrapper.setup(nprocs=8,maxcore=2000) # change to 32, 7200 for Sulis
clusters_task.script_settings['ntask'] = 8 # change to 64 for Sulis
clusters_task.wrapper.orcaprofile=OrcaProfile(command=orcacmd)
clusters_task.output = 'orca'
clusters_task.nroots = 2
clusters_task.target = [0,1,2]
clusters_task.func = solutes_task.func
clusters_task.basis = solutes_task.basis
clusters_task.ref_mol_dir = f"{{target}}_{solutes_task.func}"
all_clusters_tasks = {}

# specific radii for combinations of solvent and solute
# to ensure equal-sized clusters
solv_rad = {}
solus = list(all_solutes)[0:-len(list(all_solvents))]
solvs = list(all_solvents)
rads = [2.5,5.0]
for rad in rads:
    solv_rad[rad] = {}
    for solu in solus:
        for solv in solvs:
            solv_rad[rad][f'{solu}_{solv}'] = rad
    solv_rad[rad]['meth_meth'] = rad+1
    solv_rad[rad]['cycl_cycl'] = rad+0.5
    # Set up task as per size above
    for traj in ['A','B']:
        clusters_task.max_atoms = 60
        clusters_task.max_snapshots = 40 if traj=='A' else 50
        clusters_task.min_snapshots = 0 if traj=='A' else 40
        clusters_task.radius = solv_rad[rad]
        clusters_task.which_traj = traj
        suffix = f'solvR{rad}'
        clusters_task.exc_suffix = f"{suffix}"
        all_clusters_tasks[f"{suffix}_{traj}"] = deepcopy(clusters_task)

# Set up spectra task
spectra_task.exc_suffix    = 'solvR2.5'
spectra_task.broad         = 0.05 # eV
spectra_task.wavelength    = [300,800,1] # nm
spectra_task.warp_origin_prefix = 'gs_PBE/is_tddft'
spectra_task.warp_dest_prefix   = 'gs_PBE0/is_tddft'
all_spectra_tasks = {'default': spectra_task}

# Invoke main driver
drivers.main(all_solutes,all_solvents,
             all_solutes_tasks=all_solutes_tasks,
             all_solvate_tasks=all_solvate_tasks,
             all_clusters_tasks=all_clusters_tasks,
             all_spectra_tasks=all_spectra_tasks,
             make_script=parallel.make_sbatch)
# Quit - function defs for interactive use might follow
exit()

