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

enum{BOND,ANGLE,DIHEDRAL,VDW,COUL};

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
  double dtmp;

  int nlocal = atom->nlocal;
  int natoms = atom->natoms;
  int **nspecial = atom->nspecial;
  tagint **special = atom->special;

  // apply restraints
  for (int m = 0; m < nrestrain; m++) {
    i1 = atom->map(ids[m][0]);
    if (i1 == -1) return;

    if (rstyle[m] == BOND) printf("  ... bonds are not implemented yet!\n");
    else if (rstyle[m] == ANGLE) printf("  ... angles are not implemented yet!\n");
    else if (rstyle[m] == DIHEDRAL) printf("  ... dihedrals are not implemented yet!\n");
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

  // reinitialize things need to be reinitialized when restrains are applied
  if (anyvdwl) force->pair->reinit();
  if (anybond) force->bond->reinit();
  if (anyangle) force->angle->init_style();
  if (anydihed) force->dihedral->init_style();
  if (anyimpro) force->improper->init_style();
  if (anyq && force->kspace) force->kspace->qsum_qsq();

  // recalculate pair interactions with new info
  if (force->pair) force->pair->compute(eflag,vflag);
  if (atom->molecular) {
    if (force->bond) force->bond->compute(eflag,vflag);
    if (force->angle) force->angle->compute(eflag,vflag);
    if (force->dihedral) force->dihedral->compute(eflag,vflag);
    if (force->improper) force->improper->compute(eflag,vflag);
  }
  if (force->kspace) force->kspace->compute(eflag,vflag);

  // store forces on atoms from new topology
  for (i = 0; i < nlocal; i++) {
    f[i][0] = atom->f[i][0];
    f[i][1] = atom->f[i][1];
    f[i][2] = atom->f[i][2];
  }

  // reverse restraints
  for (int m = 0; m < nrestrain; m++) {
    i1 = atom->map(ids[m][0]);
    if (i1 == -1) return;
    if (rstyle[m] == BOND) printf("  ... bonds are not implemented yet!\n");
    else if (rstyle[m] == ANGLE) printf("  ... angles are not implemented yet!\n");
    else if (rstyle[m] == DIHEDRAL) printf("  ... dihedrals are not implemented yet!\n");
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

  // reinitialize things need to be reinitialized when restrains are applied
  if (anyvdwl) force->pair->init_style();
  if (anybond) force->bond->init_style();
  if (anyangle) force->angle->init_style();
  if (anydihed) force->dihedral->init_style();
  if (anyimpro) force->improper->init_style();
  if (anyq && force->kspace) force->kspace->qsum_qsq();
}

/* ---------------------------------------------------------------------- */

void FixTopo::post_force(int vflag)
{
  // mix old and new forces
  double delta = update->ntimestep - update->beginstep;
  if (delta != 0.0) delta /= update->endstep - update->beginstep;

  double f_new[3];

  for (int i = 0; i < atom->nlocal; i++) {
    f_new[0] = f[i][0] * delta + atom->f[i][0] * (1.0 - delta);
    f_new[1] = f[i][1] * delta + atom->f[i][1] * (1.0 - delta);
    f_new[2] = f[i][2] * delta + atom->f[i][2] * (1.0 - delta);
    printf("forces on atoms %d\n", atom->tag[i]);
    printf("  fx: %f %f %f\n",f[i][0],atom->f[i][0],f_new[0]);
    printf("  fy: %f %f %f\n",f[i][1],atom->f[i][1],f_new[1]);
    printf("  fz: %f %f %f\n",f[i][2],atom->f[i][2],f_new[2]);
  }
  printf("\n");
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
