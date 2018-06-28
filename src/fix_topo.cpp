/* ----------------------------------------------------------------------
   LAMMPS - Large-scale Atomic/Molecular Massively Parallel Simulator
   http://lammps.sandia.gov, Sandia National Laboratories
   Steve Plimpton, sjplimp@sandia.gov

   Copyright (2003) Sandia Corporation.  Under the terms of Contract
   DE-AC04-94AL85000 with Sandia Corporation, the U.S. Government retains
   certain rights in this software.  This software is distributed under
   the GNU General Public License.

   See the README file in the top-level LAMMPS directory.
------------------------------------------------------------------------- */

/* ----------------------------------------------------------------------
   Contributing author: Craig Tenney (University of Notre Dame)
     support for bond and angle restraints by Andres Jaramillo-Botero (Caltech)
------------------------------------------------------------------------- */

#include <cmath>
#include <cstring>
#include <cstdlib>
#include "fix_topo.h"
#include "atom.h"
#include "force.h"
#include "bond.h"
#include "angle.h"
#include "dihedral.h"
#include "improper.h"
#include "modify.h"
#include "pair.h"
#include "comm.h"
#include "kspace.h"
#include "update.h"
#include "group.h"
#include "domain.h"
#include "compute.h"
#include "comm.h"
#include "respa.h"
#include "input.h"
#include "neighbor.h"
#include "math_const.h"
#include "memory.h"
#include "error.h"

using namespace LAMMPS_NS;
using namespace FixConst;
using namespace MathConst;

enum{BOND,ANGLE,DIHEDRAL,IMPROPER,VDW,COUL};

#define TOLERANCE 0.05
#define SMALL 0.001
#define DELTA 1

/* ---------------------------------------------------------------------- */

FixTopo::FixTopo(LAMMPS *lmp, int narg, char **arg) :
  Fix(lmp, narg, arg),
  rstyle(NULL), ids(NULL), type(NULL), q(NULL), f(NULL)
{
  if (narg < 4) error->all(FLERR,"Illegal fix topo command");

  global_freq = 1;

  // parse args

  nrestrain = maxrestrain = 0;
  anyvdwl = anyq = anybond = anyangle = anydihed = anyimpro = 0;

  int iarg = 3;
  while (iarg < narg) {
    if (nrestrain == maxrestrain) {
      maxrestrain += DELTA;
      memory->grow(rstyle,maxrestrain,"topo:rstyle");
      memory->grow(ids,maxrestrain,4,"topo:ids");
      memory->grow(type,maxrestrain,"topo:type");
      memory->grow(q,maxrestrain,"topo:q");
    }
    if (strcmp(arg[iarg],"bond") == 0) {
      if (iarg+4 > narg) error->all(FLERR,"Illegal fix topo command");
      rstyle[nrestrain] = BOND;
      ids[nrestrain][0] = force->inumeric(FLERR,arg[iarg+1]);
      ids[nrestrain][1] = force->inumeric(FLERR,arg[iarg+2]);
      type[nrestrain] = force->inumeric(FLERR,arg[iarg+3]);
      anybond = 1;
      iarg += 4;
    } else if (strcmp(arg[iarg],"vdw") == 0) {
      if (iarg+3 > narg) error->all(FLERR,"Illegal fix topo command");
      rstyle[nrestrain] = VDW;
      ids[nrestrain][0] = force->inumeric(FLERR,arg[iarg+1]);
      type[nrestrain] = force->inumeric(FLERR,arg[iarg+2]);
      anyvdwl = 1;
      iarg += 3;
    } else if (strcmp(arg[iarg],"coul") == 0) {
      if (iarg+3 > narg) error->all(FLERR,"Illegal fix topo command");
      rstyle[nrestrain] = COUL;
      ids[nrestrain][0] = force->inumeric(FLERR,arg[iarg+1]);
      q[nrestrain] = force->numeric(FLERR,arg[iarg+2]);
      anyq = 1;
      iarg += 3;
    } else error->all(FLERR,"Illegal fix topo command");

    nrestrain++;
  }

  // require atom map to lookup atom IDs
  if (atom->map_style == 0)
    error->all(FLERR,"fix topo requires an atom map, see atom_modify");

  // create array to store forces on all atoms
  memory->create(f,atom->natoms,3,"topo:f");
}

/* ---------------------------------------------------------------------- */

FixTopo::~FixTopo()
{
  memory->destroy(rstyle);
  memory->destroy(ids);
  memory->destroy(type);
  memory->destroy(q);
  memory->destroy(f);
}

/* ---------------------------------------------------------------------- */

int FixTopo::setmask()
{
  int mask = 0;
  mask |= PRE_FORCE;
  mask |= POST_FORCE;
  return mask;
}

/* ---------------------------------------------------------------------- */

void FixTopo::init()
{
  char *id_pe = (char *) "thermo_pe";
  int ipe = modify->find_compute(id_pe);
  c_pe = modify->compute[ipe];
}

/* ---------------------------------------------------------------------- */

void FixTopo::setup(int vflag)
{
  if (strcmp(update->integrate_style,"verlet") == 0)
    post_force(vflag);
}

/* ---------------------------------------------------------------------- */

void FixTopo::pre_force(int vflag)
{
  int i,j;
  int i1,i2,i3,i4;
  int itmp;
  int eflag = 0;
  vflag = 1;
  double dtmp;

  int nlocal = atom->nlocal;
  int natoms = atom->natoms;

  size_t nbytes = sizeof(double) * (atom->nlocal + atom->nghost);

  // apply restraints
  for (int m = 0; m < nrestrain; m++) {
    i1 = atom->map(ids[m][0]);
    if (i1 == -1) return;

    if (rstyle[m] == BOND) printf("  ... bonds are not implemented yet!\n");
    else if (rstyle[m] == ANGLE) printf("  ... angles are not implemented yet!\n");
    else if (rstyle[m] == DIHEDRAL) printf("  ... dihedrals are not implemented yet!\n");
    else if (rstyle[m] == IMPROPER) printf("  ... impropers are not implemented yet!\n");
    else if (rstyle[m] == VDW) {
      itmp = atom->type[i1];
      atom->type[i1] = type[m];
      type[m] = itmp;
    } else if (rstyle[m] == COUL) {
      dtmp = atom->q[i1];
      atom->q[i1] = q[m];
      q[m] = dtmp;
    }
  }

  // reinitialize interactions and neighborlist after restraints have been applied
  if (anyvdwl) force->pair->reinit();
  if (anybond) force->bond->reinit();
  if (anyangle) force->angle->init_style();
  if (anydihed) force->dihedral->init_style();
  if (anyimpro) force->improper->init_style();
  if (anyq && force->kspace) force->kspace->qsum_qsq();
  if (domain->triclinic) domain->x2lamda(atom->nlocal);
  domain->pbc();
  comm->exchange();
  atom->nghost = 0;
  comm->borders();
  if (domain->triclinic) domain->lamda2x(atom->nlocal+atom->nghost);
  if (modify->n_pre_neighbor) modify->pre_neighbor();
  neighbor->build(1);

  // recalculate interactions with new topology
  // first: store old forces to restore them afterwards
  double **f_old;
  memory->create(f_old,atom->natoms,3,"topo:f_old");
  for (i = 0; i < nlocal; i++) {
    f_old[i][0] = atom->f[i][0];
    f_old[i][1] = atom->f[i][1];
    f_old[i][2] = atom->f[i][2];
  }
  // second: set all forces on atoms to zero
  if (nbytes) memset(&atom->f[0][0],0,3*nbytes);
  // third: calculate forces with new topology
  if (force->pair) force->pair->compute(eflag,vflag);
  if (atom->molecular) {
    if (force->bond) force->bond->compute(eflag,vflag);
    if (force->angle) force->angle->compute(eflag,vflag);
    if (force->dihedral) force->dihedral->compute(eflag,vflag);
    if (force->improper) force->improper->compute(eflag,vflag);
  }
  if (force->kspace) force->kspace->compute(eflag,vflag);
  // store forces on atoms from new topology and restore initial forces
  for (i = 0; i < nlocal; i++) {
    f[i][0] = atom->f[i][0];
    f[i][1] = atom->f[i][1];
    f[i][2] = atom->f[i][2];
    atom->f[i][0] = f_old[i][0];
    atom->f[i][1] = f_old[i][1];
    atom->f[i][2] = f_old[i][2];
  }
  memory->destroy(f_old);

  // reverse restraints so long we've not reached end of sim, keep them afterwards
  if (update->ntimestep != update->endstep) {
    for (int m = 0; m < nrestrain; m++) {
      i1 = atom->map(ids[m][0]);
      if (i1 == -1) return;
      if (rstyle[m] == BOND) printf("  ... bonds are not implemented yet!\n");
      else if (rstyle[m] == ANGLE) printf("  ... angles are not implemented yet!\n");
      else if (rstyle[m] == DIHEDRAL) printf("  ... dihedrals are not implemented yet!\n");
      else if (rstyle[m] == IMPROPER) printf("  ... impropers are not implemented yet!\n");
      else if (rstyle[m] == VDW) {
        itmp = atom->type[i1];
        atom->type[i1] = type[m];
        type[m] = itmp;
      } else if (rstyle[m] == COUL) {
        dtmp = atom->q[i1];
        atom->q[i1] = q[m];
        q[m] = dtmp;
      }
    }

    // reinitialize interactions and neighborlist after restraints have been reversed
    if (anyvdwl) force->pair->reinit();
    if (anybond) force->bond->reinit();
    if (anyangle) force->angle->init_style();
    if (anydihed) force->dihedral->init_style();
    if (anyimpro) force->improper->init_style();
    if (anyq && force->kspace) force->kspace->qsum_qsq();
    if (domain->triclinic) domain->x2lamda(atom->nlocal);
    domain->pbc();
    comm->exchange();
    atom->nghost = 0;
    comm->borders();
    if (domain->triclinic) domain->lamda2x(atom->nlocal+atom->nghost);
    if (modify->n_pre_neighbor) modify->pre_neighbor();
    neighbor->build(1);
  }
}

/* ---------------------------------------------------------------------- */

void FixTopo::post_force(int vflag)
{
  // mix old and new forces
  double delta = update->ntimestep - update->beginstep;
  if (delta != 0.0) delta /= update->endstep - update->beginstep;

  double f_new[3];

  for (int i = 0; i < atom->nlocal; i++) {
    atom->f[i][0] = f[i][0] * delta + atom->f[i][0] * (1.0 - delta);
    atom->f[i][1] = f[i][1] * delta + atom->f[i][1] * (1.0 - delta);
    atom->f[i][2] = f[i][2] * delta + atom->f[i][2] * (1.0 - delta);
  }
}

/* ----------------------------------------------------------------------
   apply harmonic bond restraints
---------------------------------------------------------------------- */

void FixTopo::restrain_bond(int m)
{

}

/* ----------------------------------------------------------------------
   apply harmonic angle restraints
---------------------------------------------------------------------- */

void FixTopo::restrain_angle(int m)
{

}

/* ----------------------------------------------------------------------
   apply dihedral restraints
   adopted from dihedral_charmm
---------------------------------------------------------------------- */

void FixTopo::restrain_dihedral(int m)
{

}

/* ----------------------------------------------------------------------
   calculate difference of forces and energies of both topologies
---------------------------------------------------------------------- */

double FixTopo::topo_eval(double **f)
{
  if (domain->triclinic) domain->x2lamda(atom->nlocal);
  domain->pbc();
  comm->exchange();
  atom->nghost = 0;
  comm->borders();
  if (domain->triclinic) domain->lamda2x(atom->nlocal+atom->nghost);
  if (modify->n_pre_neighbor) modify->pre_neighbor();
  neighbor->build(1);
  int eflag = 1;
  int vflag = 0;

  // store old forces to restore them afterwards

  double **f_old;
  memory->create(f_old,atom->natoms,3,"topo:f_old");
  for (i = 0; i < nlocal; i++) {
    f_old[i][0] = atom->f[i][0];
    f_old[i][1] = atom->f[i][1];
    f_old[i][2] = atom->f[i][2];
  }

  // clear forces so we have a fresh array to calculate the forces

  size_t nbytes = sizeof(double) * (atom->nlocal + atom->nghost);
  if (nbytes) memset(&atom->f[0][0],0,3*nbytes);

  // if (modify->n_pre_force) modify->pre_force(vflag);

  if (force->pair) force->pair->compute(eflag,vflag);

  if (atom->molecular) {
    if (force->bond) force->bond->compute(eflag,vflag);
    if (force->angle) force->angle->compute(eflag,vflag);
    if (force->dihedral) force->dihedral->compute(eflag,vflag);
    if (force->improper) force->improper->compute(eflag,vflag);
  }

  if (force->kspace) force->kspace->compute(eflag,vflag);

  // perform a reverse_comm() of forces

  if (force->newton) comm->reverse_comm();

  // store forces on atoms from new topology and restore initial forces

  for (int i = 0; i < atom->nlocal; i++) {
    f[i][0] = atom->f[i][0];
    f[i][1] = atom->f[i][1];
    f[i][2] = atom->f[i][2];
  }

  // if (modify->n_post_force) modify->post_force(vflag);
  // if (modify->n_end_of_step) modify->end_of_step();

  // restore forces to initial values

  for (int i = 0; i < atom->nlocal; i++) {
    atom->f[i][0] = f_old[i][0];
    atom->f[i][1] = f_old[i][1];
    atom->f[i][2] = f_old[i][2];
  }

  update->eflag_global = update->ntimestep;
  double total_energy = c_pe->compute_scalar();

  // cleanup
  memory->destroy(f_old);

  return total_energy;
}
