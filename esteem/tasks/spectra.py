#!/usr/bin/env python
# coding: utf-8

# In[ ]:


"""Task that generates and plots uv/vis Spectra for solutes in solvent.
Also contains routines to calculate spectral warp parameters and RGB colours from spectra"""


# # Main Routine

# In[ ]:


import numpy as np
wavelength_eV_conv=1239.84193

class SpectraTask:

    def __init__(self,**kwargs):
        self.wrapper = None
        self.script_settings = None
        self.task_command = 'spectra'
        args = self.make_parser().parse_args("")
        for arg in vars(args):
            setattr(self,arg,getattr(args,arg))
    #from amp import Ampc
    
    def read_excitations(self):

        from ase.calculators.onetep import Onetep;
        from ase.calculators.nwchem import NWChem;
        from ase.calculators.orca import ORCA;
        from esteem.wrappers import onetep
        from esteem.wrappers import nwchem
        from esteem.wrappers import orca
        from ase.io import Trajectory
        import os

        #validate_args(self)
        all_excitations = []
        all_transition_origins = []

        if self.inputformat is None and self.trajectory is not None:
            self.inputformat = 'traj'
        if self.inputformat.lower()=="nwchem":
            calc=NWChem(label="temp")
            wrapper=nwchem.NWChemWrapper()
            read_excitations = wrapper.read_excitations
        elif self.inputformat.lower()=="onetep":
            calc=Onetep(label="temp")
            wrapper=onetep.OnetepWrapper()
            read_excitations = wrapper.read_excitations
        elif self.inputformat.lower()=="orca":
            calc=ORCA(label="temp")
            wrapper=orca.ORCAWrapper()
            read_excitations = wrapper.read_excitations
        elif self.inputformat.lower()=="precalculated":
            from types import SimpleNamespace
            wrapper = None
            calc = SimpleNamespace()
            all_excitations = np.zeros((0,2))
            read_excitations = self.read_precalculated
        elif self.inputformat.lower()=="traj":
            if self.files != [] and self.files is not None:
                raise Exception("Cannot specify traj and set files simultaneously")
            if self.trajectory is None:
                raise Exception("Must specify trajectories if inputformat==traj")
            if type(self.trajectory)!=list:
                raise Exception("Must specify trajectories as list if inputformat==traj")
        else:
            raise Exception('Unrecognised input format. The value of inputformat was: {}'.format(self.inputformat))

        # Load pre-calculated vibronic transitions, if supplied 
        if self.vib_files is not None:
            vib_sticks = np.loadtxt(self.vib_files,usecols=(0,1))
        else:
            vib_sticks = None

        if self.trajectory is not None:
            if hasattr(self.wrapper,'xml_filename'):
                basename = self.wrapper.xml_filename
            for k,trajset in enumerate(self.trajectory):
                nroots = max(len(trajset)-1,1)
                trans_dip = 1.0
                traj = {}
                if self.correction_trajectory is not None:
                    corr_trajset = self.correction_trajectory[k]
                if self.vibration_trajectory is not None:
                    vib_trajset = self.vibration_trajectory[k]
                corr_traj = {}
                vib_traj = {}
                # Loop over trajectory sets, opening each trajecotory for reading
                # Also open correction and vibration trajectories if provided
                for i,trajfile in enumerate(trajset):
                    if self.verbosity!="low":
                        print(f'Opening trajectory {trajfile}')
                    traj[i] = Trajectory(trajfile)
                    if self.correction_trajectory is not None:
                        if self.verbosity!="low":
                            print(f'Opening correction trajectory {corr_trajset[i]}')
                        corr_traj[i] = Trajectory(corr_trajset[i])
                    else:
                        corr_traj[i] = None
                    if self.vibration_trajectory is not None:
                        if self.verbosity!="low":
                            print(f'Opening vibration trajectory {vib_trajset[i]}')
                        vib_traj[i] = Trajectory(vib_trajset[i])
                    else:
                        vib_traj[i] = None
                # Now run over the trajectory
                start_frame = max(self.start_frame,0)
                stride_frames = max(self.stride_frames,1)
                if self.max_frames is not None:
                    max_frames = min(self.max_frames,len(traj[0])-start_frame)
                else:
                    max_frames = len(traj[0])-start_frame
                # Keep going as long as there are at least max_frames left in this trajectory
                end_frame = start_frame + max_frames
                while end_frame <= len(traj[0]):
                    if hasattr(self.wrapper,'write_excitations'):
                        # reset for each trajectory, if using a wrapper that has a write_excitations function (eg SpecPyCode)
                        all_excitations = [] 
                    for j in range(start_frame,end_frame,stride_frames):
                        excitations = []
                        for i in range(nroots):
                            # Get energy for this excitation from first trajectory
                            e0 = traj[0][j].get_potential_energy()
                            e1 = None
                            if isinstance(e0,list) or isinstance(e0,np.ndarray):
                                if len(e0)>1:
                                    e1 = e0[i+1]
                                    e0 = e0[0]
                            if e1 is None:
                                # Try to get excited state energy for this excitation from next traj
                                try:
                                    assert len(traj[0][j])==len(traj[i+1][j])
                                    e1 = traj[i+1][j].get_potential_energy()
                                except:
                                    # If there is no corresponding excitation in one
                                    # of the roots, carry on anyway
                                    continue
                            # If we have supplied a wrapper for calculating vibronic transitions,
                            # use it now (or load its results if it has run already)
                            if hasattr(self.wrapper,'xml_filename'):
                                self.wrapper.xml_filename = basename.replace('{frame}',f'{j:04}')
                                if self.vibration_trajectory is None:
                                    vib_sticks = self.vib_wrapper(traj[0][j],traj[i+1][j])
                                else:
                                    try:
                                        assert len(vib_traj[0][j])==len(vib_traj[i+1][j])
                                    except IndexError:
                                        continue
                                    vib_sticks = self.vib_wrapper(vib_traj[0][j],vib_traj[i+1][j])
                            # If we have supplied a pair of "correction" trajectories,
                            # calculate the appropriate correction (as long as the contents of
                            # this frame of the correction trajectory is not just the solute,
                            # which means there was no solvent present in this frame)
                            if corr_traj[0] is not None and len(corr_traj[0][j])<len(traj[0][j]):
                                e0c = corr_traj[0][j].get_potential_energy()
                                e1c = None
                                if isinstance(e0c,list) or isinstance(e0c,np.ndarray):
                                    if len(e0c)>1:
                                        e1c = e0c[i+1]
                                        e0c = e0c[0]
                                if e1c is None:
                                    e1c = corr_traj[i+1][j].get_potential_energy()
                                    assert len(corr_traj[0][j])==len(corr_traj[i+1][j])
                            else:
                                e1c = 0.0; e0c = 0.0
                            # Account for calculators which return an array of values
                            if isinstance(e1,list) or isinstance(e1,np.ndarray):
                                e1 = e1[0]
                                e0 = e0[0]
                                if corr_traj[0] is not None and (isinstance(e1c,list) or isinstance(e1c,np.ndarray)):
                                    e1c = e1c[0]
                                    e0c = e0c[0]
                            ediff = e1 - e0 + e1c - e0c
                            # Swap sign of energy difference, if emission calculation is requested
                            if self.mode=='emission':
                                ediff = -ediff
                            # Print all results, if set to high verbosity
                            if self.verbosity=='high':
                                print(j,wavelength_eV_conv/ediff,ediff,e1,e0,e1c,e0c,e1c-e0c)
                            # append the excitations associated with this root to the full list
                            # unless waiting 
                            if vib_sticks is not None or self.wrapper is None or hasattr(self.wrapper,'write_excitations'):
                                excitations.append(np.array((i,ediff,trans_dip)))
                            #else:
                            #    print(f'skipped {j}')
                        # Now copy excitations associated with this frame to the full list, broadening
                        # with vibronic excitations if required
                        if len(excitations)>0:
                            if vib_sticks is None:
                                all_excitations.append(excitations)
                            else:
                                vibronic_excitations = []
                                s_elec = excitations
                                # Append each electronic transition broadened by vibronic lineshape (as sticks)
                                # DOES NOT WORK FOR MULTIPLE ELECTRONIC STATES  - ALL STATES WOULD USE SAME
                                # LINESHAPE (USUALLY NOT WANTED)
                                for e in s_elec:
                                    for v in vib_sticks:
                                        sign = -1 if self.mode=='emission' else 1
                                        s_vibronic = (e[0],e[1]+sign*v[0],e[2]*v[1])
                                        vibronic_excitations.append(s_vibronic)
                                all_excitations.append(vibronic_excitations)
                    # Advance to next "trajectory" for the purposes of splitting the trajectory
                    # into manageable chunks for the wrapper
                    start_frame = start_frame + max_frames
                    end_frame = end_frame + max_frames
                    # at end of each manageable chunk of trajectory, write excitations file
                    # for wrapper if required
                    if hasattr(self.wrapper,'write_excitations'):
                        print(f'# Writing {len(all_excitations)} excitations using spectra wrapper')
                        self.wrapper.write_excitations(np.array(all_excitations))

        if self.files is not None:
            for f in self.files:
                if not os.path.isfile(f):
                    raise OSError(f'# Could not read file {f}')
                #try:
                label = f.replace(".nwo","")
                label = label.replace(".out","")
                calc.label = label
                #calc.read(label)
                read_excitations(calc)
                if self.vib_files is not None:
                    vibronic_excitations = []
                    s_elec = calc.results['excitations']
                    for e in s_elec:
                        for v in vib_sticks:
                            sign = -1 if self.mode=='emission' else 1
                            s_vibronic = (e[0],e[1]+sign*v[0],e[2]*v[1])
                            vibronic_excitations.append(s_vibronic)
                    all_excitations.append(vibronic_excitations)
                else:
                    if type(all_excitations)==np.ndarray:
                        all_excitations = np.append(all_excitations,calc.results['excitations'],axis=0)
                    else:
                        all_excitations.append(calc.results['excitations'])
                if self.inputformat.lower()=="nwchem":
                    all_transition_origins.append(calc.results['transition_origins'])
                #calc.read(label)
                #except:
                #    print(f'Reading excitations failed for: {f}')

        stick_spectrum = []
        #print(all_excitations,np.array(all_excitations))
        all_excitations = np.array(all_excitations)
        if len(all_excitations.shape)==1:
            for i in all_excitations:
                for j in i:
                    if len(j)>0:
                        stick_spectrum.append(j)
        elif len(all_excitations.shape)==2:
            stick_spectrum = np.zeros((all_excitations.shape[0],3))
            stick_spectrum[:,1:3] = all_excitations[:,0:2]
        elif len(all_excitations.shape)==3:
            for i in range(all_excitations.shape[0]):
                for j in range(all_excitations.shape[1]):
                    stick_spectrum.append(all_excitations[i,j,:])
        stick_spectrum = np.array(stick_spectrum,ndmin=2)
        
        return stick_spectrum, all_transition_origins
    
    def read_precalculated(self,calc=None):
        calc.results = {'excitations': np.loadtxt(calc.label)}

    def vib_wrapper(self,model_init,model_targ):

        if not self.wrapper.results_exist():
            self.wrapper.data = []
            self.wrapper.add_model_data(model_init)
            self.wrapper.add_model_data(model_targ)
            self.wrapper.write_xml()
            self.wrapper.run()
        vib_sticks = self.wrapper.read()
        if vib_sticks is not None:
            if vib_sticks.shape == (0,):
                return vib_sticks
        else:
            return vib_sticks
        # Normalise
        #vib_sticks[:,1] = vib_sticks[:,1] / np.max(vib_sticks[:,1])
        return vib_sticks

    def plot(self,broad_spectrum,fig,ax,rgb,label,linestyle='solid',linewidth=1.5):

        # Set up and make plot
        if fig is None and ax is None:
            from matplotlib import pyplot
            fig, ax = pyplot.subplots()
            ax.set_xlabel('Wavelength (nm)')
            ax.set_ylabel('Absorbtion (arb)')
            #x_ticks = np.arange(self.wavelength[0], self.wavelength[1], 25)
            #pyplot.xticks(x_ticks)
        #spec_plot = plot_sticks(stick_spectrum,rgb,fig,ax,all_transition_origins)
        spec_plot = plot_spectrum(broad_spectrum,rgb,fig,ax,label,linestyle,linewidth)
        if self.output is not None:
            fig.savefig(self.output)
            
        return spec_plot,fig,ax

    def run(self,fig=None,ax=None,plotlabel=None,rgb=np.array((0.0,0.0,0.0)),linestyle='solid'):
        """
        Main routine for plotting a spectrum. Capable of applying Gaussian broadening to a stick spectrum to
        produce a broadened spectrum, and also of applying spectral warping to shift/scale the stick spectrum.

        *Arguments*:

        self: class
            All arguments for the job - see documentation below.
        fig, ax: matplotlib objects
            Figure and axis objects for matplotlib.pyplot. Initialised anew if they have value None on entry.
        plotlabel:
            Label to add the key for this dataset.
        rgb:
            Colour of the line/points for this dataset. If set to (-1.0,-1.0,-1.0), the RGB colour will be
            calculated from the spectrum.

        *Returns*:

            broad_spectrum,spec_plot,fig,ax,all_transition_origins

        *Output*:

            Plots the spectrum to 'self.output' as a png file if requested.

        """

        # Read all electronic excitations from self.files
        read_excit = True
        if hasattr(self.wrapper,'write_excitations'):
            if self.wrapper.trajs_exist():
                print('# Trajectories all exist, skipping conversion')
                read_excit = False
                stick_spectrum = np.array([[0,1,0]])
                all_transition_origins = []
            else:
                print('# Trajectories do not all exist, converting')
                self.wrapper.num_trajs = 0
        if read_excit:
            stick_spectrum, all_transition_origins = self.read_excitations()
        shape = stick_spectrum.shape
        if (shape==(1,0)):
            return None,None,fig,ax,None,None
        
        if self.warp_scheme is not None:
            for exc in stick_spectrum:
                try:
                    exc[1] = spectral_warp(exc[1],self)
                except Exception as e:
                    print(type(exc[1]),exc[1])
                    print(e)

        # Generate grid
        if self.energies is not None and self.wavelength is not None:
            raise Exception("Cannot specify both an energy grid and a wavelength grid")
        if self.energies is not None:
            en = np.arange(self.energies[0],self.energies[1],self.energies[2])
            wv = wavelength_eV_conv/en
        else:
            wv = np.arange(self.wavelength[0],self.wavelength[1],self.wavelength[2])
        
        # Calculate broadened spectrum
        if hasattr(self.wrapper,'write_excitations'):
            if not self.wrapper.results_exist():
                self.wrapper.write_input_file()
                self.wrapper.run()
            broad_spectrum = self.wrapper.read()
        
        else:
            # Calculate excitation contributions at each wavelength
            broad_spectrum = np.zeros((len(wv),3))
            for i,x in enumerate(wv):
                broad_spectrum[i,:] = np.array((x,spectral_value(x,stick_spectrum,self.broad),wavelength_eV_conv/x))

        # Renormalise so peak is at 1, if specified
        if self.renorm:
            if isinstance(self.renorm,float):
                broad_spectrum[:,1] = broad_spectrum[:,1] * self.renorm
            else: # renormalise so highest peak has maximum value 1
                broad_spectrum[:,1] = broad_spectrum[:,1] / np.amax(broad_spectrum[:,1])
                
        # Generate RGB colour from spectrum
        if (rgb == np.array((-1.0,-1.0,-1.0))).all():
            rgb = RGB_colour(stick_spectrum,self)
        
        # Plot the spectrum
        if True:
            spec_plot, fig, ax = self.plot(broad_spectrum,fig,ax,rgb,label=plotlabel,linestyle=linestyle)
        else:
            spec_plot = None

        return broad_spectrum,spec_plot,fig,ax,all_transition_origins,stick_spectrum

    def make_parser(self):
        # Parse command line values
        main_help = ('Spectra.py: Generates optical spectra from pre-existing excited state \n'+
                     'calculation output files')
        epi_help = ('')
        from argparse import ArgumentParser
        parser = ArgumentParser(description=main_help,epilog=epi_help)
        parser.add_argument('--files','-f',nargs='*',default=[],type=str,help='List of output files from which to generate spectrum')
        parser.add_argument('--vib_files','-v',nargs='*',default=None,type=str,help='List of output files from which to generate spectrum')
        parser.add_argument('--trajectory','-t',nargs='*',default=None,type=str,help='Trajectory file for input')
        parser.add_argument('--max_frames','-M',default=None,type=int,help='Maximum number of frames to process')
        parser.add_argument('--start_frame','-S',default=0,type=int,help='Starting point of frames to process')
        parser.add_argument('--stride_frames','-F',default=1,type=int,help='Stride for frames to process')
        parser.add_argument('--wavelength','-w',default=[200.0,800.0,1.0],nargs=3,type=float,help='List of minimum, maximum wavelength and step')
        parser.add_argument('--energies','-E',default=None,nargs=3,type=float,help='List of minimum, maximum energies and step')
        parser.add_argument('--warp_params','-s',default=[0.0],nargs='?',type=float,help='Parameters for spectral warp - meaning differs depending on warp_scheme argument')
        parser.add_argument('--warp_scheme',default="beta",nargs='?',type=str,help='Scheme for spectral warp (beta,alphabeta or betabeta)')
        parser.add_argument('--broad','-b',default=0.05,type=float,help='Broadening for spectrum, in eV')
        parser.add_argument('--mode','-m',default="absorption")
        parser.add_argument('--renorm','-R',default=True,type=bool,help='Renormalise spectrum so that highest peak is at 1.0')
        parser.add_argument('--inputformat','-i',default=None,type=str,help='Expected format of output files')
        parser.add_argument('--output','-o',default=None,type=str,help='File to write final plots to')
        parser.add_argument('--verbosity','-V',default='normal',type=str,help='Level of output')
        parser.add_argument('--illuminant','-I',default='D65_illuminant.txt',type=str,help='Spectrum of illuminant for calculating RGB colour')
        parser.add_argument('--XYZresponse','-X',default='XYZ_response.txt',type=str,help='Response spectrum X, Y and Z functions for calculating RGB colour')

        # For use in Drivers only, for setting up spectral warping and colouring plots
        parser.add_argument('--exc_suffix','-e',default="exc",nargs='?',type=str,help='Suffix of excitation calculation directories')
        parser.add_argument('--warp_broad',default=None,nargs='?',type=float,help='Broadening to apply when calculating warp parameters')
        parser.add_argument('--warp_origin_ref_peak_range',default=None,nargs='?',type=str,help='Wavelength range in origin spectrum to be included in finding peaks for spectral warping')
        parser.add_argument('--warp_dest_ref_peak_range',default=None,nargs='?',type=str,help='Wavelength range in destination spectrum to be included in finding peaks for spectral warping')
        parser.add_argument('--warp_inputformat',default="nwchem",nargs='?',type=str,help='Expected format of output files for calculating spectral warp parameters')
        parser.add_argument('--warp_dest_files',default="PBE0/is_tddft_{solv}/{solu}/tddft.nwo",type=str,help='Files for calculating spectral warp parameters')
        parser.add_argument('--warp_origin_files',default="PBE/is_tddft_{solv}/{solu}/tddft.nwo",type=str,help='Files for calculating spectral warp parameters')
        parser.add_argument('--line_colours',default=None,nargs='?',type=str,help='Line colours for final plot')
        parser.add_argument('--merge_solutes',default={},type=dict,help='Dictionary of solutes that should be merged into each key')

        return parser

    def validate_args(self):
        default_args = make_parser().parse_args("")
        for arg in vars(self):
            if arg not in default_args:
                raise Exception(f"Unrecognised argument '{arg}'")


# # Helper Routines

# In[ ]:



def find_spectral_warp_params(args,dest_spectrum,origin_spectrum,arrow1_pos=None,arrow2_pos=None):
        """
        Finds spectral warping parameters via a range of schemes. See spectra documentation page
        for more detail.

        *Arguments*

        dest_spectrum: numpy array of floats
            Contains the spectral warp 'destination' spectrum (usually a high level of theory that can only be afforded for the solute molecule)
        origin_spectrum: numpy array of floats
            Contains the spectral warp 'origin' spectrum (usually a cheap level of theory, same as in the Clusters job)
        args.warp_scheme: str
            The scheme used for the warping. Allowed values: 'beta', 'alphabeta', 'betabeta'
        args.warp_origin_ref_peak_range: list of 2 floats
            Peak range searched when looking for 'reference' peaks in the origin spectrum for spectral warping.
        args.warp_dest_ref_peak_range: list of 2 floats
            Peak range searched when looking for 'reference' peaks in the destination spectrum for spectral warping.
        args.broad:
            Energy gaussian broadening applied to the stick spectra, useful to merge peaks that you want to treat
            as one peak in spectral warping.

        *Returns*

            [beta], [alpha, beta], [beta1, beta2, omega1o, omega2o]: floats describing spectral warp parameters

            Also sets arrow1_pos, arrow2_pos for use in spectral warp plots.
        """

        if args.warp_scheme == None:
            return

        hc = wavelength_eV_conv

        # Mask anything outside the allowed range
        dest_spectrum_masked = mask_spectrum(dest_spectrum,
            args.warp_dest_ref_peak_range[0],args.warp_dest_ref_peak_range[1])
        origin_spectrum_masked = mask_spectrum(origin_spectrum,
            args.warp_origin_ref_peak_range[0],args.warp_origin_ref_peak_range[1])

        # Find maximum value of remaining spectra
        dest_maxloc = np.argmax(dest_spectrum_masked[:,1])
        dest_lambdamax = dest_spectrum[dest_maxloc,0]
        origin_maxloc = np.argmax(origin_spectrum_masked[:,1])
        origin_lambdamax = origin_spectrum[origin_maxloc,0]

        # Set up position of (first) arrow on plot
        if arrow1_pos is not None:
            arrow1_pos[0] = origin_lambdamax
            arrow1_pos[1] = origin_spectrum[origin_maxloc,1]
            arrow1_pos[2] = dest_lambdamax - origin_lambdamax
            arrow1_pos[3] = 0

        # Convert to energy
        omega1d = hc / dest_lambdamax
        omega1o = hc / origin_lambdamax
        print(f'lambda(max) of origin spectrum = {origin_lambdamax}')
        print(f'lambda(max) of dest   spectrum = {dest_lambdamax}')

        # Set acceptable ranges of result
        betamin = -2.5
        betamax = 2.5
        alphamin = 0.3
        alphamax = 1.8

        if args.warp_scheme == 'beta':

            beta = omega1d - omega1o

            # Report and check results
            print(f'Spectral warp beta param from origin to dest = {beta} eV')

            if beta<betamin or beta>betamax:
                raise Exception(f'beta parameter {beta} out of allowed range ({betamin} to {betamax}). Stopping.')

            return [beta,origin_lambdamax,dest_lambdamax]

        # Get spectrum masked again
        dest_spectrum_tmp = mask_spectrum(dest_spectrum,
            args.warp_dest_ref_peak_range[0],args.warp_dest_ref_peak_range[1])
        origin_spectrum_tmp = mask_spectrum(origin_spectrum,
            args.warp_origin_ref_peak_range[0],args.warp_origin_ref_peak_range[1])

        # Mask section +- 4*broad from omega1d of dest spectrum
        lower = hc / (omega1d+4*args.broad)
        upper = hc / (omega1d-4*args.broad)
        dest_spectrum_masked = mask_spectrum(dest_spectrum_tmp,lower,upper,swap=True)

        # Mask section +- 4*broad from omega1o of origin spectrum
        lower = hc / (omega1o+4*args.broad)
        upper = hc / (omega1o-4*args.broad)
        origin_spectrum_masked = mask_spectrum(origin_spectrum_tmp,lower,upper,swap=True)

        # Find maximum value of remaining spectra
        dest_maxloc = np.argmax(dest_spectrum_masked[:,1])
        dest_lambdamax = dest_spectrum[dest_maxloc,0]
        origin_maxloc = np.argmax(origin_spectrum_masked[:,1])
        origin_lambdamax = origin_spectrum[origin_maxloc,0]

        # Set up position of second arrow on plot
        if arrow2_pos is not None:
            arrow2_pos[0] = origin_lambdamax
            arrow2_pos[1] = origin_spectrum[origin_maxloc,1]
            arrow2_pos[2] = dest_lambdamax - origin_lambdamax
            arrow2_pos[3] = 0
        omega2d = hc / dest_lambdamax
        omega2o = hc / origin_lambdamax
        print(f'second peak lambda(max) of origin spectrum = {origin_lambdamax}')
        print(f'second peak lambda(max) of dest   spectrum = {dest_lambdamax}')

        if args.warp_scheme == 'avgebeta':

            beta = ((omega1d - omega1o) + (omega2d - omega2o))*0.5
            print(f'Spectral warp beta param from origin to dest = {beta} eV')

            if beta<betamin or beta>betamax:
                raise Exception(f'beta parameter {beta} out of allowed range ({betamin} to {betamax}). Stopping.')

            return [beta]

        if args.warp_scheme == 'alphabeta':

            alpha = (omega1d - omega2d) / (omega1o - omega2o)
            beta = omega2d - alpha*omega2o
            print(f'Spectral warp alpha, beta params from origin to dest = {alpha} {beta} eV')

            if beta<betamin or beta>betamax:
                raise Exception(f'beta parameter {beta} out of allowed range ({betamin} to {betamax}). Stopping.')
            if alpha<alphamin or alpha>alphamax:
                raise Exception(f'alpha parameter {alpha} out of allowed range ({alphamin} to {alphamax}). Stopping.')

            return [alpha, beta]

        if args.warp_scheme == 'betabeta':

            beta1 = omega1d - omega1o
            beta2 = omega2d - omega2o

            # Ensure omega1o < omega2o, swap if not.
            if (omega1o > omega2o):
                tmp = omega2o; omega2o = omega1o; omega1o = tmp
                tmp = beta2; beta2 = beta1; beta1 = tmp

            print(f'Spectral warp beta1, beta2, omega1, omega2 params from origin to dest = {beta1} {beta2} {omega1o} {omega2o} eV')

            if beta1<betamin or beta1>betamax:
                raise Exception(f'beta1 parameter {beta1} out of allowed range ({betamin} to {betamax}). Stopping.')
            if beta2<betamin or beta2>betamax:
                raise Exception(f'beta2 parameter {beta2} out of allowed range ({betamin} to {betamax}). Stopping.')

            return [beta1, beta2, omega1o, omega2o]

        raise Exception(f'Unrecognised warp scheme: {args.warp_scheme}')

def spectral_warp(exc,args):
    exc_w = exc
    if args.warp_scheme == 'beta' or args.warp_scheme == 'avgebeta':
        exc_w = exc + args.warp_params[0]
    if args.warp_scheme == 'alphabeta':
        exc_w = exc*args.warp_params[0] + args.warp_params[1]
    if args.warp_scheme == 'betabeta':
        beta1, beta2, omega1, omega2 = args.warp_params
        if (exc<omega1):
            beta = beta1
        elif (exc>omega2):
            beta = beta2
        elif (abs(exc-omega2)<abs(exc-omega1)):
            beta = beta2
        else:
            beta = beta1
        exc_w = exc + beta
    return exc_w


def mask_spectrum(spectrum,lower,upper,swap=False):
    spectrum_masked = spectrum.copy()
    mask = (spectrum_masked[:,0]>lower)&(spectrum_masked[:,0]<upper)
    if swap:
        mask = np.logical_not(mask)
    spectrum_masked[:,1] = spectrum_masked[:,1]*mask
    return spectrum_masked

def spectral_value(wavelength,spectrum,broad):
    wav_in_eV=wavelength_eV_conv/wavelength
    p = 1/(broad*np.sqrt(2*np.pi))
    if len(spectrum)>0:
        abs_value=p*np.sum(spectrum[:,2]*np.exp(-0.5 * ((wav_in_eV-spectrum[:,1])/broad)**2))
    return abs_value

def total_source_val(wavelength,intensity,spectrum,args):
    spectral_contribution=spectral_value(wavelength,spectrum,args)
    path_length=10
    exponential_term=np.exp(-spectral_contribution*path_length)
    source_val=exponential_term*intensity

    return source_val

def RGB_colour(spectrum,args):
    """
    Finds the RGB colour corresponding to a given absorption spectrum and illumination.

    spectrum: numpy array
        Spectrum for which to find the colour
    args: namespace or class
        Full set of arguments for the spectra task. Relevant to this routine are:

        ``args.XYZresponse`` and ``args.illuminant`` which supply the Color Space XYZ response
        spectra and the illuminant spectrum, respectively.

        These must be on the same wavelength scale as the spectrum.
    """

    xyz_funcs=np.genfromtxt(args.XYZresponse)
    illu=np.genfromtxt(args.illuminant)
    x_int=0.0
    y_int=0.0
    z_int=0.0
    step_length=(illu[1,0]-illu[0,0])
    h=step_length/2.0
    counter = 0

    # Numerical integration in a totally unpythonic way (to be fixed - must be a SciPy routine for this)
    for x in illu:
        wavelength=x[0]
        temp_val=total_source_val(wavelength,x[1],spectrum,args)
        if counter==0:
            x_int = x_int+temp_val*xyz_funcs[counter,1]
            y_int = y_int+temp_val*xyz_funcs[counter,2]
            z_int = z_int+temp_val*xyz_funcs[counter,3]
        else:
            x_int = x_int+2.0*temp_val*xyz_funcs[counter,1]
            y_int = y_int+2.0*temp_val*xyz_funcs[counter,2]
            z_int = z_int+2.0*temp_val*xyz_funcs[counter,3]
        counter=counter+1

    counter=counter-1
    wavelength=illu[counter,0]
    temp_val=total_source_val(wavelength,illu[counter,1],spectrum,args)
    x_int = (x_int-1.0*temp_val*xyz_funcs[counter,1])*h
    y_int = (y_int-1.0*temp_val*xyz_funcs[counter,2])*h
    z_int = (z_int-1.0*temp_val*xyz_funcs[counter,3])*h

    # Conversion (linear transformation) from Color Space X, Y, Z values to R, G, B
    r_int=0.41847*x_int-0.15866*y_int-0.082835*z_int
    g_int=-0.091169*x_int+0.25243*y_int+0.015708*z_int
    b_int=0.00092090*x_int-0.0025498*y_int+0.17860*z_int

    # Black magic (need to check ref for why 384.0)
    normalisation=384.0/(r_int+g_int+b_int)
    if normalisation*r_int>255.0:
        normalisation=255.0/r_int
    if normalisation*g_int>255.0:
        normalisation=255.0/g_int
    if normalisation*b_int>255.0:
        normalisation=255.0/b_int

    rgb = np.array((r_int,g_int,b_int))*normalisation

    print('R value:')
    print(rgb[0])
    print('G value:')
    print(rgb[1])
    print('B value:')
    print(rgb[2])

    return rgb

def plot_spectrum(broad_spectrum,rgb,fig,ax,label,linestyle='solid',linewidth=1.5):

    # Plot data
    spec = ax.plot(broad_spectrum[:,0], broad_spectrum[:,1],color=rgb*(1.0/256.0),
                   label=label,linestyle=linestyle,linewidth=linewidth)
    #pyplot.setp(spec,color=rgb*(1.0/256.0))
    #if label is not None:
    #    ax.legend()
    return spec

def plot_sticks(stick_spectrum,rgb,fig,ax,labels):

    # Plot data
    #print(stick_spectrum[:,0])
    #print(stick_spectrum[:,1])
    spec = ax.stem(stick_spectrum[:,0], stick_spectrum[:,1],color=rgb*(1.0/256.0))
    #spec = ax.stem([0,1,2],[12,51,1])
    #pyplot.setp(spec,color=rgb*(1.0/256.0))
    return spec


# # Command-line driver

# In[ ]:


def get_parser():
    return SpectraTask().make_parser()

if __name__ == '__main__':

    spec = SpectraTask()

    # Parse command line values
    args = spec.make_parser.parse_args()
    for arg in vars(args):
        setattr(spec,arg,getattr(args,arg))
    print('#',args)

    # Run main program
    spec.run()

