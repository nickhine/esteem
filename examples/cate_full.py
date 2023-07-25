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

# Setup solute and solvents
all_solutes = {'cate': 'catechol'}
all_solvents = {'cycl': 'cyclohexane', 'meth': 'methanol'}

# Set up Solutes task
solutes_task = SolutesTask()
solutes_task.wrapper = ORCAWrapper() 
solutes_task.basis = '6-311++G**'

# Setup different XC functionals as different tasks
all_solutes_tasks = {}
for func in ['PBE','PBE0']:
   all_solutes_tasks[func] = deepcopy(solutes_task)
   all_solutes_tasks[func].func = func
   all_solutes_tasks[func].directory = func

# Set up solvate task
solvate_task = SolvateTask()
solvate_task.wrapper = AmberWrapper()
solvate_task.boxsize = 15

# Set up clusters task
clusters_task = ClustersTask()
clusters_task.wrapper = ORCAWrapper()
clusters_task.output = 'orca'
all_clusters_tasks = {}
for rad in [0,2.5,3.5]:
   target = f'solvR{rad}'
   all_clusters_tasks[target] = deepcopy(clusters_task)
   all_clusters_tasks[target].radius = rad
   all_clusters_tasks[target].exc_suffix = target

# Set up spectra task
spectra_task = SpectraTask()
spectra_task.exc_suffix    = 'solvR0'
spectra_task.broad         = 0.05 # eV
spectra_task.wavelength    = [300,800,1] # nm
spectra_task.warp_origin_prefix = 'PBE/is_tddft'
spectra_task.warp_dest_prefix   = 'PBE0/is_tddft'

# Invoke main driver
drivers.main(all_solutes,all_solvents,
             all_solutes_tasks=all_solutes_tasks,
             all_solvate_tasks=solvate_task,
             all_clusters_tasks=all_clusters_tasks,
             all_spectra_tasks=spectra_task,
             make_script=parallel.make_sbatch)
# Quit - function defs for interactive use might follow
exit()

