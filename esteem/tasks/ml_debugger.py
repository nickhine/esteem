#!/usr/bin/env python
# coding: utf-8

# In[ ]:


"""Defines an interactive tool, MLDebugger, for finding where a Machine Learning
Molecular Dynamics trajectory goes wrong, by locating the first frame containing a
'bad' solvent molecule and extracting the frames around it"""


# # Main Routine

# In[ ]:


# Load essential modules
import sys
import os
import string
from ase.io.trajectory import Trajectory
from esteem.trajectories import generate_md_trajectory, find_initial_geometry, get_trajectory_list, targstr


class MLDebugger:
    """
    Finds where an ML MD trajectory of a solute in explicit solvent goes wrong.

    seed: str
        Name of the system
    nat_solu: int
        Number of atoms in the solute
    nat_solv: int
        Number of atoms in each solvent molecule
    """

    def __init__(self, seed, nat_solu=None, nat_solv=None):
        self.seed = seed
        self.nat_solu = nat_solu
        self.nat_solv = nat_solv
        self.keep_solute = False
        self.recenter = False

    def setup_from_seed(self, seed):
        'Not yet implemented'

    def reorder_frame(self, t, i, nat, nmol, rad, keep_idx=None, keep_solute=False, recenter=False):
        """
        Reorders frame ``t`` so that solvent molecule ``i`` comes first and the solute is removed,
        then keeps only the solvent molecules within ``rad`` of molecule ``i`` (or, if ``keep_idx``
        is given, only the atoms with those indices). The solute is appended at the end if
        ``self.keep_solute`` is True, and molecule ``i`` is centred in the cell if ``self.recenter``
        is True.

        Returns the reordered Atoms and the list of kept atom indices, to pass as ``keep_idx``
        for other frames.
        """

        from esteem.tasks.clusters import delete_distant_molecules
        nat_solu = self.nat_solu
        nat_solv = self.nat_solv
        solu = t[0:self.nat_solu].copy()
        solv = t[nat_solu + i * nat_solv:nat_solu + (i + 1) * nat_solv].copy()
        del t[list(range(0, nat_solu))]
        del t[list(range(i * nat_solv, (i + 1) * nat_solv))]
        t = solv + t
        if keep_idx is None:
            keep_idx = []
            if rad is not None:
                t = delete_distant_molecules(t, rad, nat - nat_solu, nat_solv, nmol - 1, nat_solv, keep_idx=keep_idx)
        else:
            t = t[keep_idx]
        if self.keep_solute:
            t = t + solu
        if self.recenter:
            com = solv.get_center_of_mass()
            cen = 0.5 * (t.cell[0] + t.cell[1] + t.cell[2])
            t.translate(cen - com)
            t.wrap()
        return t, keep_idx

    def debug_cluster_trajectory(self, traj, nat_solu, nat_solv, start=0, end=-1,
                                 before=20, after=20, stride=1, badness_test_func=None,
                                 radius=None):
        """
        Searches ``traj[start:end]`` for the first frame in which ``badness_test_func`` returns True
        for a solvent molecule (eg :func:`esteem.tasks.clusters.dist_test`), printing the energies
        of each frame checked.

        Returns a list of the frames from ``before`` frames before to ``after`` frames after the bad
        one (every ``stride`` frames). If ``radius`` is None (or 0), each contains just the bad
        molecule; if positive, the frames are reordered by :meth:`reorder_frame` to keep the
        solvent molecules within ``radius`` of it; if negative, the whole frames are returned.
        The list is empty if no bad molecule is found.
        """
        found_one = False
        recent_frames = []
        radius = None if radius == 0.0 else radius
        for j, a in enumerate(traj[start:end]):
            try:
                EP = a.get_potential_energy()
                EP = EP[0] if isinstance(EP, list) else EP
                EK = a.get_kinetic_energy()
                TT = a.get_temperature()
                print(start + j, EP, EK, EP + EK, TT)
            except:
                print(start + j)
            nat = len(a)
            solu = a[0:nat_solu]
            nmol = int((nat - nat_solu) / nat_solv)
            for i in range(nmol):
                solv = a[nat_solu + i * nat_solv:nat_solu + (i + 1) * nat_solv]
                if badness_test_func(solv):
                    print(f'Bad Molecule found at frame {start + j} molecule {i}')
                    t = traj[start + j].copy()
                    keep_idx = None
                    t, keep_idx = self.reorder_frame(t, i, nat, nmol, rad=radius, keep_idx=keep_idx)
                    recent_frames = []
                    for p in range(-before, after, stride):
                        if start + j + p < 0 or start + j + p > len(traj) - 1:
                            continue
                        if radius is not None:
                            t = traj[start + j + p].copy()
                            if radius >= 0:  # -ve radius to retain whole frame with no reordering
                                t, _ = self.reorder_frame(t, i, nat, nmol, rad=radius, keep_idx=keep_idx)
                            recent_frames.append(t)
                        else:  # Faster way to extract just the molecule
                            recent_frames.append(traj[start + j + p][nat_solu + i * nat_solv:nat_solu + (i + 1) * nat_solv])
                    found_one = True
                    break
            if found_one:
                break

        return recent_frames
