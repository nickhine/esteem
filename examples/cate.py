from esteem import drivers, parallel
from esteem.wrappers import nwchem, amber, onetep

# List solutes and solvents and get default arguments
all_solutes = {'cate': 'catechol'}
all_solvents = {'cycl': 'cyclohexane', 'meth': 'methanol'}
from esteem.tasks.solutes import SolutesTask
from esteem.tasks.solvate import SolvateTask
from esteem.tasks.clusters import ClustersTask
from esteem.tasks.spectra import SpectraTask

solutes_task = SolutesTask()
solvate_task = SolvateTask()
clusters_task = ClustersTask()
spectra_task = SpectraTask()

# Some simple overrides for a quick job
solutes_task.basis = '6-31G'
solutes_task.func = 'PBE'
solutes_task.directory = 'PBE'
solvate_task.boxsize = 18
solvate_task.ewaldcut = 10
solvate_task.nsnaps = 20
clusters_task.radius = 3

# Setup parallel execution of tasks
solutes_task.wrapper = nwchem.NWChemWrapper()
solutes_task.script_settings = parallel.get_default_script_settings(solutes_task.wrapper)
solutes_task.wrapper.setup()

solvate_wrapper = amber.AmberWrapper()
solvate_task.script_settings = parallel.get_default_script_settings(solvate_task.wrapper)

clusters_wrapper = onetep.OnetepWrapper()
clusters_task.script_settings = parallel.get_default_script_settings(clusters_task.wrapper)

# Run main driver
drivers.main(all_solutes,all_solvents,
             solutes_task,solvate_task,clusters_task,spectra_task,
             make_script=parallel.make_sbatch)
exit()

