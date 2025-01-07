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
#include "group.h"
#include "pair.h"
#include "math_const.h"

#include <cmath>

using namespace LAMMPS_NS;
using namespace FixConst;
using MathConst::MY_PI;
using MathConst::MY_2PI;

/* ---------------------------------------------------------------------- */

FixCharge::FixCharge(LAMMPS *lmp, int narg, char **arg) :
  Fix(lmp, narg, arg), list(nullptr)
{
  if (narg < 7) error->all(FLERR, "Illegal fix {} command", style);
  ntype = utils::inumeric(FLERR, arg[3],false,lmp); // desired neighbor type
  q0 = utils::numeric(FLERR, arg[4],false,lmp);
  del = utils::numeric(FLERR, arg[5],false,lmp);
  cut = utils::numeric(FLERR, arg[6],false,lmp);
  
  cuthi = cut+del;
  cutlo = cut-del;
  twodel = 2.0*del;
  cuthisq = cuthi*cuthi;
  cutlosq = cutlo*cutlo;
  delpi = MY_PI/del;
}

/* ---------------------------------------------------------------------- */

int FixCharge::setmask() {
  int mask = 0;
  mask |= PRE_FORCE;
  return mask;
}

/* ---------------------------------------------------------------------- */

void FixCharge::init()
{
  if (!atom->q_flag)
    error->all(FLERR,"Fix {} requires atom attribute q", style);

  ngroup = group->count(igroup);
  if (ngroup == 0) error->all(FLERR,"Fix {} group has no atoms", style);
  
  neighbor->add_request(this, NeighConst::REQ_FULL);
}

/* ---------------------------------------------------------------------- */

void FixCharge::init_list(int /*id*/, NeighList *ptr)
{
  list = ptr;
}

/* --------------------------------------------------------------------- */

void FixCharge::setup_pre_force(int /* vflag */)
{
  update_charges();
}

/* ---------------------------------------------------------------------- */

void FixCharge::pre_force(int /*vflag*/)
{
  update_charges();
}

/* ---------------------------------------------------------------------- */

void FixCharge::update_charges()
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

          if (rsq < cuthisq) {
            if (rsq < cutlosq) {
              q[i] += q0;
            } else {
              q[i] += q0 * fc(sqrt(rsq));
            }
          }
        }
      }
    }
  }
}

/* ---------------------------------------------------------------------- */

double FixCharge::fc(double r)
{ 
  double rcutlo = r-cutlo; // (r-R+Delta) = (r-(R-Delta)) hence r-cutlo and not r-cuthi
  return 1.0 - rcutlo / twodel + sin( delpi * rcutlo ) / MY_2PI;
}
