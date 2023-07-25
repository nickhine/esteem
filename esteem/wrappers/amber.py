#!/usr/bin/env python
# coding: utf-8

# In[ ]:


"""Defines the AmberWrapper Class"""

import subprocess
from os import path, makedirs
from shutil import copyfile, which
from ase import Atoms
from ase.calculators.amber import Amber
from ase.io import read, write
import numpy as np
from ase.io.trajectory import Trajectory


# See http://ambermd.org/tutorials/basic/tutorial1/section3.htm 
# and http://ambermd.org/tutorials/advanced/tutorial3/section1.htm for details

# NB: Hack to amber.py required here to fix reading of trajectories
# (issue with dimensions of coordinates and cell_lengths)


# From J. Chem. Phys. 148, 024110 (2018)
# The dyes are placed in large solvent boxes (see Table I for
# the number of atoms in the MD box for each system) and a
# two-step equilibration is carried out. First, a 20 ps temperature
# equilibration in the NVT ensemble is performed to raise the
# temperature of the system from 0 K to 300 K. This is followed
# by a 400 ps volume equilibration in the NPT ensemble. Since
# we are interested in generating uncorrelated snapshots rather
# than accurate short time scale dynamics, we run all produc-
# tion calculations in the NVT ensemble to guarantee a constant
# temperature. For the production trajectory of 8 ns in length,
# solute-solvent snapshots are extracted every 4 ps, producing a
# total of 2000 uncorrelated snapshots. All MD calculations are
# performed using a 2 fs time-step and a Langevin thermostat
# with a collision frequency of 1 ps^-1



class AmberWrapper():
    
    """Sets up the AMBER Calculator (via ASE) for Molecular Dynamics runs"""
    
    def __init__(self,nprocs=None):
        """Sets up instance attributes for AmberWrapper """

        # common Amber parameters
        self.known_solvents = []
        self.counterions_lines = ""
        self.cut = 12.0     # 12Ang cutoff for Ewald
        self.temp0 = 300.0  # 300 K
        self.ntt = 3        # Langevin thermostat
        self.gamma_ln = 1   # Collision frequency 1ps^-1
        self.dt = 0.002     # 2 fs timestep
        self.restraint_str = '' # Line to add to Sander inputs for restraints
        self.ntb = 0        # Periodic box
        self.ntp = 0        # Periodic box
        self.ntpr = 100     # Output frequency
        self.qmmm = False
        
        self.acpype_charge_type = 'bcc'
        self.acpype_atom_type = 'amber'

        self.amber_exe_serial = 'sander -O '
        if which('sander.MPI'):
            nprocs_str = ''
            if nprocs is not None:
                nprocs_str = f'-n {nprocs}'
            self.amber_exe_parallel = f'mpirun {nprocs_str} sander.MPI -O '
        else:
            self.amber_exe_parallel = self.amber_exe_serial

    def prepare_input_acpype(self,seed,netcharge=0,offset=0):
        """Prepares input parameters and topologies for Amber calculations"""
        
        import sys
        import signal
        from acpype.cli import init_main
        
        store_argv = sys.argv

        # Load molecular structure from .xyz file
        mol = read(f'{seed}.xyz')

        # Convert to pdb format
        write(f'{seed}.pdb',mol)
        
        acpype_atom_type = self.acpype_atom_type
        acpype_charge_type = self.acpype_charge_type

        sys.argv = f'acpype -i {seed}.pdb -n {netcharge} -c {acpype_charge_type} -a {acpype_atom_type}'.split()

        # Run ACPYPE
        init_main()
        # Cancel the timeout alarm that ACPYPE starts
        try:
            signal.alarm(0)
        except:
            print('# Unable to cancel timeout - next process may be killed early.')

        exts = {'_AC.frcmod':'.frcmod','_AC.prmtop':'.prmtop','_AC.lib':'.lib',
                '_AC.inpcrd':'.inpcrd',f'_{acpype_charge_type}_{acpype_atom_type}.mol2':'.mol2'}
        for ext1,ext2 in exts.items():
            copyfile(f'{seed}.acpype/{seed}{ext1}',f'./{seed}{ext2}')

        # Edit the resulting mol2 file to enforce charge conservation
        # and optionally shift atom names
        self.fix_mol2(seed,netcharge,offset)

        sys.argv = store_argv
            
    def prepare_input(self,seed,netcharge=0,offset=0):
        """Prepares input parameters and topologies for Amber calculations"""

        # Load molecular structure from .xyz file
        mol = read(f'{seed}.xyz')

        # Convert to pdb format
        write(f'{seed}.pdb',mol)

        # Make directory for antechamber outputs
        out_path = f'{seed}_antechamber'
        if not path.exists(out_path):
            makedirs(out_path)

        # run antechamber
        antechamber_command = (f"antechamber -i {seed}.pdb -fi pdb " +
                                           f"-o {seed}.mol2 -fo mol2 " +
                                           f"-rn {seed[:3]} -c bcc -du y " +
                                           f"-s 2 -nc {netcharge} > {seed}.ac_out")
        print(antechamber_command)
        errorcode = subprocess.call(antechamber_command, shell=True)
        if errorcode:
            raise RuntimeError('{} returned an error: {}'
                                .format('antechamber', errorcode))

        # Edit the resulting mol2 file to enforce charge conservation
        # and optionally shift atom names
        self.fix_mol2(seed,netcharge,offset)

        # save sqm outputs with seednames for later checking
        sqm_cleanup_command = (f"mv sqm.out {seed}_sqm.out;" +
                               f"mv sqm.in {seed}_sqm.in;" +
                               f"mv sqm.pdb {seed}_sqm.pdb;" +
                               f"mv ANTECHAMBER* ATOMTYPE.INF {out_path}") 
        errorcode = subprocess.call(sqm_cleanup_command, shell=True)
        if errorcode:
            raise RuntimeError('{} returned an error: {}'
                                .format('mv of sqm files', errorcode))

        # run parmchk to produce frcmod file
        parmchk_exe = "parmchk"
        if which("parmchk2"):
            parmchk_exe = "parmchk2"
        parmchk_command = f"{parmchk_exe} -i {seed}.mol2 -f mol2 -o {seed}.frcmod"
        errorcode = subprocess.call(parmchk_command, shell=True)
        if errorcode:
            raise RuntimeError('{} returned an error: {}'
                                .format('parmchk', errorcode))

        # run tleap to set up prmtop
        f = open(f'{seed}_tleap.in', 'w')
        tleap_str =  ("source leaprc.protein.ff14SB\n" +
                      "source leaprc.gaff\n" +
                      f"mol = loadmol2 {seed}.mol2\n" +
                      f"saveamberparm mol {seed}.prmtop {seed}.inpcrd\n" +
                      "quit\n")
        f.write(tleap_str)
        f.close()
        tleap_command = f"tleap -f {seed}_tleap.in > {seed}_tleap.out"
        errorcode = subprocess.call(tleap_command, shell=True)
        if errorcode:
            raise RuntimeError('{} returned an error: {}'
                                .format('tleap', errorcode))

    # Fix inaccurate total charge on mol2 files, and optionally offset atom labels
    def fix_mol2(self,seed,charge,offset):
        infile = f"{seed}.mol2_orig"
        outfile = f"{seed}.mol2"
        copyfile(outfile,infile)
        try:
            f_in = open(infile, 'r')
        except IOError:
            raise ReadError(f'Could not open mol2 file {infile}')
        try:
            f_out = open(outfile, 'w')
        except IOError:
            raise ReadError(f'Could not open mol2 file {outfile}')

        line = next(f_in)
        while line:
            if "@<TRIPOS>MOLECULE" in line:
                f_out.write(line)
                try:
                    line = next(f_in)
                except:
                    line = False
                f_out.write(line)
                try:
                    line = next(f_in)
                except:
                    line = False
                nat = int(line.split()[0])
            if "@<TRIPOS>ATOM" in line:
                f_out.write(line)
                lines = [next(f_in) for x in range(nat)]
                charges = [float(line.split()[-1]) for line in lines]
                lines = [line.split() for line in lines]
                # Calculate and apply charge correction:
                dc = (charge-sum(charges))/float(nat)
                charges = [c+dc for c in charges]
                for i in range(nat):
                    l = lines[i]
                    # add offset to atom numbers
                    num = ''.join(filter(lambda i: i.isdigit(), l[1]))
                    sym = ''.join(filter(lambda i: not i.isdigit(), l[1]))
                    try:
                        num = int(num)
                    except:
                        num = 0
                    if num+offset>0:
                        l[1] = sym + str(num+offset)
                    else:
                        l[1] = sym
                    l[-1] = charges[i]
                    # reproduce formatting of .mol2 file:
                    f_out.write('%7s %-8s %10.4f %10.4f %10.4f %-8s %3d %-8s %9.6f\n' 
                                % tuple([l[0],l[1],float(l[2]),float(l[3]),
                                        float(l[4]),l[5],int(l[6]),l[7],
                                        float(l[8])]))
                try:
                    line = next(f_in)
                except:
                    line = False

            f_out.write(line)
            try:
                line = next(f_in)
            except:
                line = False

    def ion_chg(self,atsym):
        if atsym in ['H','Na','Li','K','Rb','Cs']:
            chg = '+'
        if atsym in ['Be','Mg','Ca','Sr','Ba']:
            chg = '2+'
        if atsym in ['F','Cl','Br','I']:
            chg = '-'
        return chg

    def find_solu_center_at(self,solute):
        try:
            at = read(f"{solute}.xyz")
        except:
            raise Exception(f"Failed to read xyz file {solute}.xyz")
        com=at.get_center_of_mass()
        dpos=at.positions-com
        iat = np.argmin(np.sqrt((dpos*dpos).sum(axis=1)))
        sym = at[iat].symbol

        try:
            with open(f"{solute}.mol2") as f:
                for line in f:
                    if line.startswith("@<TRIPOS>ATOM"):
                        for i in range(iat+1):
                            sym2 = f.readline().split()[1]
        except:
            raise Exception(f"Failed to read mol2 file {solute}.mol2")
        if sym2[0] == sym[0]:
            return sym2
        else:
            raise Exception("Symbols did not match when finding center atom")

    def add_solvent_box(self,solute,solvent,counterions,solvatedseed,box_size):
        """Loads an Amber mol2 file for a solute and solvent, and creates a solvated box"""

        if solute in counterions:
            sym = Atoms(counterions[solute]).get_chemical_symbols()
            for at in set(sym):
                self.counterions_lines = self.counterions_lines + f"addIonsRand solute {at}{self.ion_chg(at)} {sym.count(at)}\n"
            print('Counterions lines: \n'+self.counterions_lines)

        # run tleap to make solvent box
        f = open('tleap_solvate.in', 'w')
        tleap_str = ("source leaprc.protein.ff14SB\n" +
                     "source leaprc.gaff\n" +
                     "source leaprc.water.tip3p\n"
                     "solute = loadmol2 "+solute+".mol2\n" +
                     "solvent = loadmol2 "+solvent+".mol2\n" +
                     #"check solvent\n" +
                     #"check solute\n" +
                     "loadamberparams "+solute+".frcmod\n" +
                     "loadamberparams "+solvent+".frcmod\n" +
                     "check solvent\n" +
                     "check solute\n" +
                     "charge solute\n" +
                     "solvatebox solute solvent "+str(box_size)+"\n" +
                     self.counterions_lines +
                     "saveamberparm solute "+solvatedseed+".prmtop "+solvatedseed+".crd\n" +
                     "savepdb solute "+solvatedseed+".pdb\n" +
                     "quit\n")
        f.write(tleap_str)
        f.close()
        tleap_command = "tleap -f tleap_solvate.in > tleap_solvate.out"
        errorcode = subprocess.call(tleap_command, shell=True)
        if errorcode:
            raise RuntimeError('{} returned an error: {}'
                                .format('tleap', errorcode))

        # no restraints unless we set one now
        self.restraint_str = ''

        # Find atom closest to solute center of mass
        solu_centre_at = self.find_solu_center_at(solute)

        # Add restraints to ensure ion is not within 15A of solute
        if solute in counterions:
            rest_atoms = []
            for i,at in enumerate(Atoms(counterions[solute])):
                symchg = at.symbol + self.ion_chg(at.symbol)
                solu_unit = solute[0:3]
                rest_at = [1,solu_unit,solu_centre_at,i+2,symchg,symchg] # i+2 since solute is unit 1, ions afterwards
                rest_atoms.append(rest_at)
            rest_r = [20,20,15,50,13,52]
            print('Distance Restraints: ',rest_atoms,rest_r)
            self.add_dist_restraint(solvatedseed,rest_atoms,rest_r)

    def make_frame_prmtop(self,frameseed,frame,solute,solvent,nat_solu,nat_solv,nmol_solv):

        write(frameseed+".pdb",frame)
        if solute==solvent:
            offset = 0
        else:
            offset = 99
        self.label_pdb_units(frameseed+".pdb",solute,solvent,nat_solu,nat_solv,nmol_solv,solv_offset=offset)
        # run tleap to make prmtop for the frame
        f = open('tleap_frame.in', 'w')
        solu_unit = solute[0:3]
        solv_unit = solvent[0:3]
        # "source leaprc.protein.ff14SB\n" + # NOT REQUIRED
        tleap_str = ("source leaprc.gaff\n" +
                     solu_unit+" = loadmol2 "+solute+".mol2\n" +
                     solv_unit+" = loadmol2 "+solvent+".mol2\n" +
                     "check "+solu_unit+"\n" +
                     "check "+solv_unit+"\n" +
                     "loadamberparams "+solute+".frcmod\n" +
                     "loadamberparams "+solvent+".frcmod\n" +
                     "check "+solu_unit+"\n" +
                     "check "+solv_unit+"\n" +
                     "frame = loadpdb "+frameseed+".pdb\n" +
                     "check frame\n" +
                     "saveamberparm frame "+frameseed+".prmtop "+frameseed+".crd\n" +
                     "quit\n")
        f.write(tleap_str)
        f.close()
        tleap_command = "tleap -f tleap_frame.in > tleap_frame.out"
        errorcode = subprocess.call(tleap_command, shell=True)
        if errorcode:
            raise RuntimeError('{} returned an error: {}'
                                .format('tleap', errorcode))

    # Reads in one unit from a pdb file and re-exports it with
    # edited unit numbers/names/offsets etc
    def fix_pdb_unit(self,f_in,f_out,unit,nat,unitname,atomnum_offset):
        format = ('%-6s%5d %4s %3s%6d    %8.3f%8.3f%8.3f%6.2f%6.2f'
                  '          %-2s  \n')
        lines = [next(f_in) for x in range(nat)]
        lines = [line.split() for line in lines]
        atomnum = {}
        for i in range(nat):
            l = lines[i]
            spec = l[2]
            l[3] = unitname[:3]
            if spec not in atomnum:
                atomnum[spec] = -1
            atomnum[spec] = atomnum[spec] + 1
            j = atomnum[spec] + atomnum_offset
            if j>0:
                l[2] = spec + str(j)
            else:
                l[2] = spec
            l[4] = unit
            # reproduce formatting of .pdb file:
            f_out.write(format % tuple([l[0],int(l[1]),l[2],l[3],int(l[4]),
                                float(l[5]),float(l[6]),float(l[7]),float(l[8]),
                                float(l[9]),l[10]]))

    def label_pdb_units(self,pdbfile,solute,solvent,nat_solu,nat_solv,nmol_solv,solv_offset):

        infile = pdbfile+"_orig"
        outfile = pdbfile
        copyfile(outfile,infile)
        try:
            f_in = open(infile, 'r')
        except IOError:
            raise ReadError('Could not open pdb file "%s"' % infile)
        try:
            f_out = open(outfile, 'w')
        except IOError:
            raise ReadError('Could not open pdb file "%s"' % outfile)

        line = next(f_in)
        f_out.write(line)
        #line = next(f_in)
        #f_out.write(line)
        format = ('%6s%5d %4s %3s%6d    %8.3f%8.3f%8.3f%6.2f%6.2f'
                  '          %2s  \n')

        unit = 1
        self.fix_pdb_unit(f_in,f_out,unit,nat_solu,solute,0)
        f_out.write("TER\n")
        for unit in range(2,nmol_solv+2):
            self.fix_pdb_unit(f_in,f_out,unit,nat_solv,solvent,solv_offset)
            f_out.write("TER\n")
        f_out.write("END\n")

    def add_dist_restraint(self,solvatedseed,rest_atoms,rest_r):

        # unpack restraint parameter list
        rk2,rk3,r1,r2,r3,r4 = rest_r

        # write map file
        restmap_file = solvatedseed+'_rest.map'
        f = open(restmap_file, 'w')
        restmap_str = ''
        mappings_needed = {}
        for i in range(len(rest_atoms)):
            i1,res1,at1,i2,res2,at2 = rest_atoms[i]
            if res1 not in mappings_needed:
                mappings_needed[res1] = [at1]
            else:
                if at1 not in mappings_needed[res1]:
                    mappings_needed[res1].append(at1)
            if res2 not in mappings_needed:
                mappings_needed[res2] = [at2]
            else:
                if at2 not in mappings_needed[res2]:
                    mappings_needed[res2].append(at2)
        for res in mappings_needed:
            restmap_str = restmap_str + f'RESIDUE {res}\n'
            for at in mappings_needed[res]:
                restmap_str = restmap_str + f'MAPPING {at} = {at}\n'
        f.write(restmap_str)
        f.close()

        # write seed_rest.dist file
        restdist_file = solvatedseed+'_rest.dist'
        f = open(restdist_file, 'w')
        restdist_str = ''
        for i in range(len(rest_atoms)):
            i1,res1,at1,i2,res2,at2 = rest_atoms[i]
            restdist_str = restdist_str + f'{i1} {res1} {at1} {i2} {res2} {at2} {r2} {r3}\n'
        f.write(restdist_str)
        f.close()
        RST_file = solvatedseed+".RST"
        makeDIST_command = f"makeDIST_RST -pdb {solvatedseed}.pdb -ual {restdist_file} -map {restmap_file} -rst {RST_file}"
        self.restraint_str = "DISANG="+RST_file+"\n"
        errorcode = subprocess.call(makeDIST_command, shell=True)
        if errorcode:
            raise RuntimeError('{} returned an error: {}'
                                .format('makeDIST_RST', errorcode))
        #fix_RST_rk(RST_file,rk2,rk3,r1,r2,r3,r4)

    def add_tors_restraint(self,solvatedseed,unit_name,torsion_atoms,rest_r):
        # unpack restraint parameter list
        rk2,rk3,r1,r2,r3,r4 = rest_r
        # write torsion.lib file
        torslib_file = solvatedseed+'_torsion.lib'
        f = open(torslib_file, 'w')
        torslib_str = ''
        for i in range(len(torsion_atoms)):
            torslib_str = (torslib_str + unit_name+" PH"+str(i) + 
                          ' '+' '.join(torsion_atoms[i])+" \n")
        f.write(torslib_str)
        f.close()
        torsdist_file = solvatedseed+'_torsion.dist'
        f = open(torsdist_file, 'w')
        torsdist_str = ''
        for i in range(len(torsion_atoms)):
            torsdist_str = (torsdist_str + "1 "+unit_name+" PH"+str(i) + 
                                           " "+str(r2)+" "+str(r3)+"\n")
        f.write(torsdist_str)
        f.close()

        RST_file = solvatedseed+".RST"
        makeANG_command = ("makeANG_RST -pdb " + solvatedseed + ".pdb " +
                           "-con " + torsdist_file + " -lib " + torslib_file + 
                           " > "+ RST_file)
        self.restraint_str = "DISANG="+RST_file+"\n"
        errorcode = subprocess.call(makeANG_command, shell=True)
        if errorcode:
            raise RuntimeError('{} returned an error: {}'
                                .format('makeANG_RST', errorcode))
        fix_RST_rk(RST_file,rk2,rk3,r1,r2,r3,r4)

    def fix_RST_rk(self,RST_file,rk2,rk3,r1,r2,r3,r4):
        outfile = RST_file
        infile = RST_file+"_orig"
        copyfile(outfile,infile)
        try:
            f_in = open(infile, 'r')
        except IOError:
            raise ReadError('Could not open RST file "%s"' % infile)
        try:
            f_out = open(outfile, 'w')
        except IOError:
            raise ReadError('Could not open RST file "%s"' % outfile)

        for line in f_in:
            fields = line.strip().split()
            written = False
            if len(fields)==7:
                if fields[0]=='rk2':
                    # reproduce formatting of .RST file:
                    f_out.write('	  rk2 =  %4.1f, rk3 =   %4.1f,				/\n' 
                                % (rk2,rk3))
                    written = True
            if len(fields)==12:
                if fields[0]=='r1':
                    # reproduce formatting of .RST file:
                    f_out.write('          r1 = %5.1f, r2 = %5.1f, r3 =  %5.1f, r4 =  %5.1f,\n'
                                % (r1,r2,r3,r4))
                    written = True
            if not written:
                f_out.write(line)


    def fix_RST_rk(self,RST_file,rk2,rk3,r1,r2,r3,r4):
        outfile = RST_file
        infile = RST_file+"_orig"
        copyfile(outfile,infile)
        try:
            f_in = open(infile, 'r')
        except IOError:
            raise ReadError('Could not open RST file "%s"' % infile)
        try:
            f_out = open(outfile, 'w')
        except IOError:
            raise ReadError('Could not open RST file "%s"' % outfile)

        for line in f_in:
            fields = line.strip().split()
            written = False
            if len(fields)==7:
                if fields[0]=='rk2':
                    # reproduce formatting of .RST file:
                    f_out.write('	  rk2 =  %4.1f, rk3 =   %4.1f,				/\n' 
                                % (rk2,rk3))
                    written = True
            if len(fields)==12:
                if fields[0]=='r1':
                    # reproduce formatting of .RST file:
                    f_out.write('          r1 = %5.1f, r2 = %5.1f, r3 =  %5.1f, r4 =  %5.1f,\n'
                                % (r1,r2,r3,r4))
                    written = True
            if not written:
                f_out.write(line)

    def crd_to_crdnc(self,seed,crdfile):
        f = open('ptraj.in', 'w')
        ptraj_str = ("trajin "+crdfile+".crd\n" +
                     "trajout "+crdfile+".crd.nc\n" +
                     "go\n")
        f.write(ptraj_str)
        f.close()
        cpptraj_command = "cpptraj "+seed+".prmtop ptraj.in > "+seed+".ptrajout"
        errorcode = subprocess.call(cpptraj_command, shell=True)
        if errorcode:
            raise RuntimeError('{} returned an error: {}'
                                .format('cpptraj', errorcode))

    def reimage(self,seed,infile,outfile):
        """Runs the cpptraj program with a simple script to center the frame
           on residue 1 (the solute) and translate molecules back into the home cell"""
        f = open('ptraj.in', 'w')
        ptraj_str = (f"trajin {infile}\n" +
                     f"trajout {outfile}\n" +
                     f"center :1-1\n" +
                     f"image familiar\n" +
                     f"go\n")
        f.write(ptraj_str)
        f.close()

        cpptraj_command = f"cpptraj {seed}.prmtop ptraj.in > {seed}.ptraj_reimage.out"
        errorcode = subprocess.call(cpptraj_command, shell=True)
        if errorcode:
            raise RuntimeError('{} returned an error: {}'
                                .format('cpptraj', errorcode))

    def dipole(self,seed,crdfile):
        """Runs the cpptraj program with a simple script to get the dipole
        moment from the final trajectory"""
        f = open('ptraj.in', 'w')
        if False: # just first molecule
            ptraj_str = (f"trajin {crdfile}\n" +
                         f"vector v1 :1-1 dipole out {seed}.dipole\n" +
                         "go\n")
        else: # everything
            ptraj_str = (f"trajin {crdfile}\n" +
                         f"vector v1 dipole out {seed}.dipole\n" +
                         "go\n")
        f.write(ptraj_str)
        f.close()

        cpptraj_command = f"cpptraj {seed}.prmtop ptraj.in > {seed}.ptraj_dipole.out"
        errorcode = subprocess.call(cpptraj_command, shell=True)
        if errorcode:
            raise RuntimeError('{} returned an error: {}'
                                .format('cpptraj', errorcode))
        dip = np.loadtxt(f'{seed}.dipole')[1:4]
        return dip

    def fix_amber_pdb(self,seed):
        """An awk-based hack to fix AMBER pdb files to make them work with ASE"""

        # Amber does not write atom symbols in columns 77-78 of its pdb files, fix this
        awkstr = ('awk \'/ATOM/ {a=$3;gsub(/[0-9]/,"",a); print $0,"        ",a; next};'+
             '{print}\' '+seed+'.pdb  > '+seed+'2.pdb; mv '+seed+'2.pdb '+seed+'.pdb')

        errorcode = subprocess.call(awkstr, shell=True)
        if errorcode:
            raise RuntimeError('{} returned an error: {}'
                                .format('awk', errorcode))

    # Set up Amber calculator
    def singlepoint(self,model,seed,calc_params={},forces=False,solvent=None,
                    readonly=False,continuation=False):
        """Runs a singlepoint calculation with the Amber ASE calculator"""
        f = open(seed+'.in', 'w')
        mmin_str =  f'''
zero step md to get energy and force
&cntrl
imin=0, nstlim=0,  ntx=1 !0 step md
cut={self.cut}, ntb={self.ntb},          !non-periodic
ntpr=1,ntwf=1,ntwe=1,ntwx=1 ! (output frequencies)
&end
'''
        f.write(mmin_str)
        f.close()

        calc_am = Amber(amber_exe=self.amber_exe_serial,
                     infile=seed+'.in',
                     outfile=seed+'.out',
                     topologyfile=seed+'.prmtop',
                     incoordfile=seed+'.crd')
        calc_am.write_coordinates(model,f'{seed}.crd')
        model.set_calculator(calc_am)

        if forces:
            e = model.get_potential_energy()
            f = model.get_forces()        
            calc_am.results['dipole'] = self.dipole(seed,calc_am.incoordfile)
            print(calc_am.results['dipole'])
            return e,f,calc_am
        else:
            return model.get_potential_energy()

    def geom_opt(self,model,seed,calc_params={},solvent=None):
        """Runs a geometry optimisation calculation with the Amber ASE calculator"""
        minin_str = (f'''
    Initial minimisation
    &cntrl
      imin=1, maxcyc=500, ncyc=50,
      cut={self.cut}
      ntb={self.ntb}
      ntp={self.ntp}, igb=0
      ntpr=1,ntwf=1,ntwe=100,ntwx=1000
    &end
    ''' + self.restraint_str)
        with open('min.in', 'w') as f:
            f.write(minin_str)
            f.close()
        calc_min = Amber(amber_exe=self.amber_exe_serial,
                          infile='min.in',
                          outfile='min.out',
                          topologyfile=f'{seed}.prmtop',
                          incoordfile=f'{seed}.crd',
                          outcoordfile='min.rst')
        model.set_calculator(calc_min)
        calc_min.write_coordinates(model,f'{seed}.crd')
        print("Pot, Kin Energy after minimisation: ", model.get_potential_energy(), model.get_kinetic_energy())
        calc_min.read_coordinates(model,f'{seed}.crd.nc')

    def heatup(self,model,seed,calc_params={},nsteps=100):
        """Runs a heatup temperature-ramp calculation with the Amber ASE calculator"""
        # NTC = 2, NTF = 2: hydrogens constrained at this stage
        # irest = 0 (new simulation)
        heatin_str =  (f'''
    Heating
    &cntrl
      imin=0,irest=0,ntx=1,
      nstlim={nsteps},dt={self.dt},
      ntc=2,ntf=2,nmropt=1,
      cut={self.cut}, ntb=1,
      ntt={self.ntt}, gamma_ln={self.gamma_ln*2},
      tempi=0.0, temp0={self.temp0}, ig=-1,
      ntpr={self.ntpr},ntwf={self.ntpr},ntwe={self.ntpr},ntwx=1000
    /
    &wt TYPE='TEMP0', istep1=0, istep2={nsteps},
      value1=0.1, value2={self.temp0}, /
    &wt TYPE='END' /
    ''' + self.restraint_str)
        with open('heat.in', 'w') as f:
            f.write(heatin_str)
            f.close()
        calc_heat = Amber(amber_exe=self.amber_exe_parallel,
                          infile='heat.in',
                          outfile='heat.out',
                          topologyfile=f'{seed}.prmtop',
                          incoordfile='min.rst',
                          outcoordfile='heat.rst')
                          #mdcoordfile='heat.mdcrd.nc')
        model.set_calculator(calc_heat)

        calc_heat.write_coordinates(model,'min.rst')
        new_pe = model.get_potential_energy()
        calc_heat.read_coordinates(model,calc_heat.outcoordfile)
        calc_heat.write_coordinates(model,'heat.rst')
        new_ke = model.get_kinetic_energy()
        print("Pot, Kin Energy after heating: ", new_pe, new_ke)

    def densityequil(self,model,seed,calc_params={},nsteps=100):
        """Runs a density equilibration calculation with fixed hydrogens with the Amber ASE calculator"""

        # NTB = 2, NTP = 1, TAUP = 1.0: Use constant pressure periodic boundary. Isotropic position scaling
        # should be used to maintain the pressure (NTP=1) and a relaxation time of 1 ps should be used (TAUP=1.0).
        # NTC = 2, NTF = 2: hydrogens constrained at this stage
        # irest = 1 (restart from previous simulation)
        densityin_str = (f'''
    Density equilibration
     &cntrl
      imin=0, irest=1, ntx=5,
      nstlim={nsteps},dt={self.dt},
      ntc=2,ntf=2,nmropt=1,
      cut={self.cut}, ntb=2, ntp=1, taup=1.0,
      ntt={self.ntt}, gamma_ln={self.gamma_ln*2},
      temp0={self.temp0}, ig=-1,
      ntpr={self.ntpr},ntwf={self.ntpr},ntwe={self.ntpr},ntwx=1000 /
     &wt TYPE='END' /
    ''' + self.restraint_str)

        with open('density.in', 'w') as f:
            f.write(densityin_str)

        calc_dens = Amber(amber_exe=self.amber_exe_parallel,
                          infile='density.in',
                          outfile='density.out',
                          topologyfile=f'{seed}.prmtop',
                          incoordfile='heat.rst',
                          outcoordfile='density.rst')
                          #mdcoordfile='density.mdcrd.nc')
        model.set_calculator(calc_dens)

        new_pe = model.get_potential_energy()
        calc_dens.read_coordinates(model,calc_dens.outcoordfile)
        new_ke = model.get_kinetic_energy()
        print("Pot, Kin Energy after density equilibration: ", new_pe, new_ke)

    def equil(self,model,seed,calc_params={},nsteps=100):
        """Runs an equilibration calculation at constant volume with flexible hydrogens with the Amber ASE calculator"""
        # NTP = 0: No pressure scaling (constant volume)
        # NTC = 1: SHAKE not used - no constraints
        # Energies at this stage no longer comparable to previous steps
        # due to extra DOFs
        eqin_str = (f'''
    Equilibration
    &cntrl
      imin=0, irest=1, ntx=5,
      nstlim={nsteps},
      dt={self.dt},
      ntc=1,nmropt=1,
      cut={self.cut}, ntb=2, ntp=1,
      ntt={self.ntt}, gamma_ln={self.gamma_ln},
      tempi={self.temp0}, temp0={self.temp0},
      ntpr={self.ntpr},ntwf={self.ntpr},ntwe={self.ntpr},ntwx=1000 /
     &wt TYPE='END' /
    ''' + self.restraint_str)
        with open('equil.in', 'w') as f:
            f.write(eqin_str)
            f.close()
        calc_equil = Amber(amber_exe=self.amber_exe_parallel,
                           infile='equil.in',
                           outfile='equil.out',
                           topologyfile=seed+'.prmtop',
                           incoordfile='density.rst',
                           outcoordfile='equil.rst')
                           #mdcoordfile='equil.mdcrd.nc')
        model.set_calculator(calc_equil)

        new_pe = model.get_potential_energy()
        calc_equil.read_coordinates(model,calc_equil.outcoordfile)
        new_ke = model.get_kinetic_energy()
        print("Pot, Kin Energy after equilibration: ", new_pe, new_ke)
        
    def snapshots(self,model,seed,calc_params={},nsnaps=1,nsteps=100,start=0):
        """Runs a long MD trajectory for snapshot generation with the Amber ASE calculator"""
        # NTC = 1: SHAKE not used - no constraints
        snap_str = (f'''
    Snapshot Generation
    &cntrl
      imin=0, irest=1, ntx=5,
      nstlim={nsteps},dt={self.dt}, ntc=1,
      cut={self.cut}, ntb=2, ntp=1, nmropt=1,
      ntt={self.ntt}, gamma_ln={self.gamma_ln},
      tempi={self.temp0}, temp0={self.temp0},
      ntpr={self.ntpr},ntwf={self.ntpr},ntwe={self.ntpr},ntwx=0 /
     &wt TYPE='END' /''' + "\n" + self.restraint_str)
        with open('snap.in', 'w') as f:
            f.write(snap_str)
            f.close()
        step = 0
        trajname = seed+'.traj'
        if start==0:
            copyfile('equil.rst',f'snap{-1:04}.rst')
            traj = Trajectory(trajname, 'w')
        else:
            traj = Trajectory(trajname)
            prevframe = len(traj)
            print(f'Found trajectory file {trajname} containing {prevframe} frames.')
            if prevframe != start+1:
                raise Exception(f'Error: This does not agree with the resumption point {start}.')
            traj.close()
            traj = Trajectory(trajname, 'a')
        snapout = model.copy()
        for step in range(start,nsnaps):
            calc_snap = Amber(amber_exe=self.amber_exe_parallel,
                               infile='snap.in',
                               outfile=f'snap{step:04}.out',
                               topologyfile=seed+'.prmtop',
                               incoordfile=f'snap{step-1:04}.rst',
                               outcoordfile=f'snap{step:04}.rst',
                               mdcoordfile=f'snap{step:04}.mdcrd.nc')
            model.set_calculator(calc_snap)
            new_pe = model.get_potential_energy()
            
            self.reimage(seed,calc_snap.outcoordfile,f'{seed}.rst_ri.nc')
            calc_snap.results['dipole'] = self.dipole(seed,calc_snap.outcoordfile)
            calc_snap.read_coordinates(snapout,f'{seed}.rst_ri.nc')
            traj.write(snapout,**calc_snap.results)
            #write(f'{seed}_snap{step:04}.xyz',snapout)

            calc_snap.read_coordinates(model,calc_snap.outcoordfile)
            new_ke = model.get_kinetic_energy()

            print("Pot, Kin Energy after snapshot",str(step),":",new_pe, new_ke)

    
    def traj_write(self,atoms,traj):
        kw = { #'dipole': self.dipole(seed,calc_am.incoordfile),
              #'charges': atoms.get_charges(),
              'energy': atoms.get_potential_energy(),
              'forces': atoms.get_forces()}
        traj.write(atoms,**kw)

    def run_mlmd(self,model,mdseed,calc_params,md_steps,md_timestep,superstep,temp,
                 solvent=None,restart=False,readonly=False,constraints=None,dynamics=None,
                 continuation=None):
        
        from ase.md.velocitydistribution import MaxwellBoltzmannDistribution, Stationary, ZeroRotation
        from ase.md import Langevin, npt
        from ase.io import Trajectory
        from ase import units
        
        """Runs a singlepoint calculation with the Amber ASE calculator"""
        f = open(mdseed+'.in', 'w')
        mmin_str =  f'''
zero step md to get energy and force
&cntrl
      imin=0, irest=1, ntx=5,
      nstlim=0,ntc=1,
      cut={self.cut}, ntb=2, ntp=1, nmropt=1,
      ntpr={self.ntpr},ntwf=1,ntwe={self.ntpr},ntwx=0 /
     &wt TYPE='END' /
'''
        f.write(mmin_str)
        f.close()

        calc_seed = calc_params['calc_seed']

        calc_am = Amber(amber_exe=self.amber_exe_parallel,
                     infile=f'{mdseed}.in',
                     outfile=f'{mdseed}.out',
                     topologyfile=f'../{calc_seed}_solv.prmtop',
                     incoordfile=f'{mdseed}.crd',
                     energyfile=f'{mdseed}.mden',
                     forcefile=f'{mdseed}.mdfrc')
        calc_am.write_coordinates(model,f'{mdseed}.crd')
        model.calc = calc_am

        # Initialise velocities if this is first step, otherwise inherit from model
        if np.all(model.get_momenta() == 0.0):
            MaxwellBoltzmannDistribution(model,temperature_K=temp)

        # For each ML superstep, remove C.O.M. translation and rotation    
        #Stationary(model)
        #ZeroRotation(model)
        #print(f'constraints: {model.constraints}')

        if readonly:
            model = read(mdseed+".xyz") # Read final image
            model.calc = calc_am
            model.get_potential_energy() # Recalculate energy for final image
            return None
        else:
            new_traj = False
            if dynamics is None or dynamics=="LANG":
                dynamics = Langevin(model, timestep=md_timestep, temperature_K=temp, friction=0.002)
                new_traj = True
            if dynamics=="NPT":
                ttime=25*units.fs
                # Bulk modulus for ethanol
                pfactor = 100*1.06e9*(units.J/units.m**3)*ttime**2
                print(f'pfactor = {pfactor}, ttime={ttime}')
                dynamics = npt.NPT(model, timestep=md_timestep, temperature_K=temp, externalstress=0,
                                   pfactor=pfactor,ttime=ttime)
                new_traj = True
            if new_traj:
                traj = Trajectory(mdseed+".traj", 'w', model)
                dynamics.attach(self.traj_write, interval=1, atoms=model, traj=traj)
            dynamics.run(md_steps)
            #model.calc.results['stress'] = model.get_stress()
            return dynamics


# In[ ]:




