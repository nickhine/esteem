from copy import deepcopy

# Import the drivers and wrappers we need
from esteem import drivers, parallel
from esteem.wrappers.orca import ORCAWrapper
from esteem.wrappers.amber import AmberWrapper
from esteem.wrappers.mace import MACEWrapper

# Import the task classes we need
from esteem.tasks.solutes import SolutesTask
from esteem.tasks.solvate import SolvateTask
from esteem.tasks.clusters import ClustersTask
from esteem.tasks.spectra import SpectraTask
from esteem.tasks.ml_training import MLTrainingTask
from esteem.tasks.ml_testing import MLTestingTask
from esteem.tasks.ml_trajectories import MLTrajTask
from esteem.active_learning import create_mltrain_tasks
from esteem.active_learning import create_mltraj_tasks
from esteem.active_learning import create_mltest_tasks
from esteem.active_learning import create_clusters_tasks
from esteem.active_learning import create_spectra_tasks

# Setup solute and solvents and target states
all_solutes = {'cate': 'catechol'}
all_solvents = {'cycl': 'cyclohexane', 'meth': 'methanol'}
all_solutes.update(all_solvents)
targets = {0:'gs',1:'es1',2:'es2'}
orcacmd="/storage/nanosim/orca6/orca_6_0_0_shared_openmpi416_avx2/orca"

# Create tasks
solutes_task = SolutesTask()
solvate_task = SolvateTask()
clusters_task = ClustersTask()
spectra_task = SpectraTask()
mltrain_task = MLTrainingTask()
mltest_task = MLTestingTask()
mltraj_task = MLTrajTask()

# Set up Solutes task
all_solutes_tasks = {}
solutes_task.wrapper = ORCAWrapper()
solutes_task.wrapper.setup(nprocs=8)
solutes_task.script_settings = parallel.get_default_script_settings(solutes_task.wrapper)
from ase.calculators.orca import OrcaProfile
solutes_task.wrapper.orcaprofile=OrcaProfile(command=orcacmd)
# Setup different XC functionals as different tasks
funcs = ['PBE','PBE0']
basis_sets = ['def2-TZVP']
for basis in basis_sets:
    for func in funcs:
        for target in targets:
            prefix = f'{targets[target]}_{func}'
            solutes_task.disp = True if 'D3BJ' not in func else False
            solutes_task.func = func
            solutes_task.basis = basis
            solutes_task.target = target
            solutes_task.directory = prefix
            all_solutes_tasks[prefix] = deepcopy(solutes_task)

# Set up solvate task
solvate_task.wrapper = AmberWrapper()
solvate_task.md_geom_prefix = f"gs_{solutes_task.func}"
solvate_task.nsteps = 2000
solvate_task.nsnaps = 500
solvate_task.script_settings = parallel.get_default_script_settings(solvate_task.wrapper)
solvate_task.boxsize = {'cycl': 18,'meth': 15}
solvate_task.ewaldcut = 9.0
all_solvate_tasks = {'md': solvate_task}

# Set up clusters task
clusters_task.wrapper = ORCAWrapper()
clusters_task.script_settings = parallel.get_default_script_settings(clusters_task.wrapper)
clusters_task.wrapper.setup(nprocs=8,maxcore=2500) # change to 32, 7200 for Sulis
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
    solv_rad[rad]['meth_meth'] = rad+1.0
    solv_rad[rad]['cycl_cycl'] = rad+0.5
    # Set up task as per size above
    clusters_task.max_atoms = 111
    clusters_task.subset_selection_method = "R"
    clusters_task.subset_selection_nmax = 100
    clusters_task.radius = solv_rad[rad]
    traj = 'A'
    clusters_task.which_traj = traj
    suffix = f'solvR{rad}'
    clusters_task.exc_suffix = f"{suffix}"
    all_clusters_tasks[f"{suffix}_{traj}"] = deepcopy(clusters_task)

# Set up tasks for clusters runs for each Active Learning iteration
meth=""
truth="orca"
train_calcs = ["MACEac0u","MACEac1u","MACEac2u"]
seed="{solu}_{solv}"
traj_suffix = 'mlclus'
md_suffix = "mldyn_recalc_carved"
md_dir_suffix = 'mldyn'
rand_seed = {'a':123,'b':456,'c':789} #,'d':101112,'e':131415}
clusters_task.repeat_without_solute = False
clusters_task.radius = None
clusters_task.subset_selection_nmax = 100
clusters_task.subset_selection_min_spacing = 20
clusters_task.subset_selection_bias_beta = 5000
active_clusters_tasks = create_clusters_tasks(clusters_task,train_calcs=train_calcs,
                                              seed=seed,traj_suffix=traj_suffix,
                                              md_suffix=md_suffix,md_dir_suffix=md_dir_suffix,
                                              targets=targets,rand_seed=rand_seed,
                                              meth="",truth="orca",separate_valid=False)
all_clusters_tasks.update(active_clusters_tasks)

# Set up tasks for ML Training
mltrain_task.wrapper = MACEWrapper()
seeds=[]
for solu in solus:
    seeds.append(f"{solu}_{{solv}}") # make list of all solutes, with solvent
seeds.append("{solv}_{solv}")        # add solvent in solvent
traj_suffixes = [truth] # what trajectory suffixes to train on
dir_suffixes = {truth: "solvR2.5"} # what directory suffixes to append to the seeds to find each trajectory suffix in
ntraj = {(targets[0],truth):1,(targets[1],truth):0,(targets[2],truth):0} # how many trajectories of each suffix to expect, labelled A, B, C etc
mltrain_task.wrapper.train_args['max_num_epochs'] = 500
mltrain_task.wrapper.train_args['swa'] = True
mltrain_task.wrapper.train_args['start_swa'] = 300
iter_dir_suffixes = ["mlclus"]
mltrain_task.ntraj=270
mltrain_task.geom_prefix = f'gs_{solutes_task.func}/is_opt_{{solv}}'
all_mltrain_tasks = create_mltrain_tasks(mltrain_task,train_calcs=train_calcs,
                                     seeds=seeds,targets=targets,rand_seed=rand_seed,
                                     meth="",truth=truth,traj_suffixes=traj_suffixes,
                                     dir_suffixes=dir_suffixes,ntraj=ntraj,
                                     iter_dir_suffixes=iter_dir_suffixes,
                                     delta_epochs=500,separate_valid=False)

# Set up tasks for Trajectories with ML calculators
mltraj_task.wrapper = MACEWrapper()
mltraj_task.snap_wrapper = MACEWrapper()
mltraj_task.geom_prefix = f'gs_{solutes_task.func}'
mltraj_task.calc_seed = "all_{solv}"
mltraj_task.md_init_traj_link = f"{{solu}}_{{solv}}_md/{{solu}}_{{solv}}_solv.traj"
mltraj_task.ntraj = len(rand_seed)
mltraj_task.md_steps = 5
mltraj_task.nequil = 200
mltraj_task.nsnap = 2000
from ase.units import fs
mltraj_task.md_timestep = {'MD': 0.5*fs, 'EQ': 0.5*fs}
mltraj_task.md_friction = {'MD': 0.002, 'EQ': 0.05}  # For Langevin dynamics
mltraj_task.store_full_traj = False
mltraj_task.carve_trajectory_radius = solv_rad[rads[0]]
mltraj_task.carve_trajectory_max_atoms = clusters_task.max_atoms
mltraj_task.recalculate_carved_traj = True
mltraj_task.continuation = True
mltraj_task.ref_mol_dir = f'{{target}}_{solutes_task.func}'
all_mltraj_tasks = create_mltraj_tasks(mltraj_task,train_calcs=train_calcs,targets=targets,
                    rand_seed=rand_seed,meth="",traj_suffix='mldyn',
                    md_wrapper=mltraj_task.wrapper,snap_wrapper=mltraj_task.snap_wrapper,
                    two_targets=False)
# Now add tasks for spectroscopy (more equilibration, longer runs, larger clusters, corrections)
mltraj_task.carve_trajectory_radius = solv_rad[rads[1]]
mltraj_task.carve_trajectory_max_atoms = 1000
mltraj_task.corr_traj = True
mltraj_task.md_steps = 2
mltraj_task.nequil = 500
mltraj_task.nsnap = 50000
spectra_mltraj_tasks = create_mltraj_tasks(mltraj_task,train_calcs=train_calcs,targets=targets,
                    rand_seed=rand_seed,meth="",traj_suffix='specdyn',
                    md_wrapper=mltraj_task.wrapper,snap_wrapper=mltraj_task.snap_wrapper,
                    two_targets=True)
all_mltraj_tasks.update(spectra_mltraj_tasks)

# Set up tasks for testing the ML calculators
mltest_task.wrapper = MACEWrapper()
mltest_task.script_settings = parallel.get_default_script_settings(mltest_task.wrapper)
mltest_task.ntraj = 300
mltest_task.ref_mol_dir = f'{{targ}}_{solutes_task.func}'
mltest_task.calc_seed = f"all_{{solv}}"
all_mltest_tasks = create_mltest_tasks(mltest_task,train_calcs=train_calcs,seeds=seeds,
                                       targets=targets,rand_seed=rand_seed,
                                       truth=truth,meth="",traj_suffixes=traj_suffixes,
                                       dir_suffixes=dir_suffixes,iter_dir_suffixes=iter_dir_suffixes,
                                       ntraj=ntraj,separate_valid=False)

# Set up tasks for plotting spectra
all_spectra_tasks = {}
func = solutes_task.func
spec_method = f'ImplicitSolvent_{func}'
spectra_task.broad = 0.05 # eV
spectra_task.inputformat   = 'orca'
spectra_task.wavelength = (300,700,1) # nm start stop step
spectra_task.warp_scheme = None
spectra_task.output = f'{{solu}}_{{solv}}_{spec_method}.png'
targ="gs"
spectra_task.files = [f'{targ}_{func}/is_tddft_{{solv}}/{{solu}}/{{solu}}_tddft.out']
spectra_task.exc_suffix = 'IS'
spectra_task.wrapper = ORCAWrapper()
all_spectra_tasks[f'{spec_method}_abs'] = deepcopy(spectra_task)
spectra_task.output = f'{{solu}}_{{solv}}_IS_emis.png'
targ="es1"
spectra_task.files = [f'{targ}_{func}/is_tddft_{{solv}}/{{solu}}_{targ}/{{solu}}_{targ}_tddft.out']
all_spectra_tasks[f'{spec_method}_emis'] = deepcopy(spectra_task)

spec_method = f'AMBERdynR{rads[0]}_{solutes_task.func}exc'
spectra_task.inputformat = "traj"
spectra_task.output = f'{{solu}}_{{solv}}_{spec_method}.png'
spectra_task.warp_scheme = None
spectra_task.wavelength = (300,700,1) # nm start stop step
spectra_task.wrapper = None
spectra_task.broad = 0.05
spectra_task.files = None
spectra_task.exc_suffix = f'solvR{rads[0]}'
spectra_task.trajectory = [[f"{{solu}}_{{solv}}_gs_A_orca.traj",f"{{solu}}_{{solv}}_es1_A_orca.traj"]]
all_spectra_tasks[spec_method] = deepcopy(spectra_task)

# Add active learning spectra tasks (vertical excitations)
all_spectra_tasks.update(create_spectra_tasks(spectra_task,train_calcs,targets, 
         rand_seed,meth,ntraj=len(rand_seed),traj_suffix='specdyn_recalc_carved',corr_traj=True))

# Invoke main driver
drivers.main(all_solutes,all_solvents,
             all_solutes_tasks=all_solutes_tasks,
             all_solvate_tasks=all_solvate_tasks,
             all_clusters_tasks=all_clusters_tasks,
             all_mltrain_tasks=all_mltrain_tasks,
             all_mltest_tasks=all_mltest_tasks,
             all_qmd_tasks=solutes_task,
             all_mltraj_tasks=all_mltraj_tasks,
             all_spectra_tasks=all_spectra_tasks,
             make_script=parallel.make_sbatch)
# Quit - function defs for interactive use might follow
exit()

