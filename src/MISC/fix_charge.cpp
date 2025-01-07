// clang-format off
/* ----------------------------------------------------------------------
 LAMMPS - Large-scale Atomic/Molecular Massively Parallel Simulator
 https://www.lammps.org/, Sandia National Laboratories
 LAMMPS development team: developers@lammps.org

 Copyright (2003) Sandia Corporation.  Under the terms of Contract
 DE-AC04-94AL85000 with Sandia Corporation, the U.S. Government retains
 certain rights in this software.  This software is distributed under
 the GNU General Public License.

 See the README file in the top-level LAMMPS directory.
 ------------------------------------------------------------------------- */

/* ----------------------------------------------------------------------
   Contributing author: Robert Meißner (TU Hamburg, Germany), robert.meissner@tuhh.de
------------------------------------------------------------------------- */

// TODO: Check if cut+delta is smaller than pair cutoff.

#include "fix_charge.h"

#include "atom.h"
#include "error.h"
#include "neigh_list.h"
#include "neighbor.h"

#include <cmath>

using namespace LAMMPS_NS;
using namespace FixConst;

/* ---------------------------------------------------------------------- */

FixCharge::FixCharge(LAMMPS *lmp, int narg, char **arg) :
  Fix(lmp, narg, arg), list(nullptr)
{
  if (narg < 7) error->all(FLERR, "Illegal fix charge command");
  ntype = utils::inumeric(FLERR, arg[3],false,lmp); // desired neighbor type
  q0 = utils::numeric(FLERR, arg[4],false,lmp);
  delta = utils::numeric(FLERR, arg[5],false,lmp);
  cut = utils::numeric(FLERR, arg[6],false,lmp);
}

/* ---------------------------------------------------------------------- */

int FixCharge::setmask() {
  int mask = 0;
  mask |= PRE_FORCE;
  return mask;
}

/* ---------------------------------------------------------------------- */

void FixCharge::setup(int /*vflag*/)
{
  cutsq = cut*cut;
}

/* ---------------------------------------------------------------------- */

void FixCharge::init_list(int /*id*/, NeighList *ptr)
{
  list = ptr;
}


/* ---------------------------------------------------------------------- */

void FixCharge::pre_force()
{
  int i, j, jj, jnum, jtype;
  double xtmp, ytmp, ztmp, delx, dely, delz, rsq;
  int *jlist, *numneigh, **firstneigh;

  double **x = atom->x;
  double *q = atom->q;
  int *type = atom->type;
  int *mask = atom->mask;
  int nlocal = atom->nlocal;
  
  numneigh = list->numneigh;
  firstneigh = list->firstneigh;

  // loop over neighbors of my atoms
  for (i = 0; i < nlocal; i++) {
    if (mask[i] & groupbit) {
      q[i] = 0.0;
      jlist = firstneigh[i];
      jnum = numneigh[i];

      for (jj = 0; jj < jnum; jj++) {
        j = jlist[jj];
        j &= NEIGHMASK; // was macht das? ist irgendwie in lmptypes.h definiert
        jtype = type[j];

        if (jtype == ntype) {

          delx = x[i][0] - x[j][0];
          dely = x[i][1] - x[j][1];
          delz = x[i][2] - x[j][2];
          rsq = delx * delx + dely * dely + delz * delz;

          if (rsq < cutsq) {
            q[i] += q0;
          }
        }
      }
    }
  }
}
