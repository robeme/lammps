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

enum{BOND,ANGLE,DIHEDRAL,IMPROPER,VDWL,COUL};

#define TOLERANCE 0.05
#define SMALL 0.001
#define DELTA 1

/* ---------------------------------------------------------------------- */

FixTopo::FixTopo(LAMMPS *lmp, int narg, char **arg) :
  Fix(lmp, narg, arg),
  rstyle(NULL), ids(NULL), type(NULL), q(NULL), f(NULL),  copy(NULL)
{
  if (narg < 4) error->all(FLERR,"Illegal fix topo command");

  scalar_flag = 1;
  global_freq = 1;
  extscalar = 1;
  vector_flag = 1;
  size_vector = 2;
  extvector = 1;
  force_reneighbor = 1;
  next_reneighbor = -1;
  resetflag = 0;

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
        type[nrestrain] = force->inumeric(FLERR,arg[iarg+1]);
        ids[nrestrain][0] = force->inumeric(FLERR,arg[iarg+2]);
        ids[nrestrain][1] = force->inumeric(FLERR,arg[iarg+3]);
        anybond = 1;
        iarg += 4;
      } else if (strcmp(arg[iarg],"angle") == 0) {
        if (iarg+5 > narg) error->all(FLERR,"Illegal fix topo command");
        rstyle[nrestrain] = ANGLE;
        type[nrestrain] = force->inumeric(FLERR,arg[iarg+1]);
        ids[nrestrain][0] = force->inumeric(FLERR,arg[iarg+2]);
        ids[nrestrain][1] = force->inumeric(FLERR,arg[iarg+3]);
        ids[nrestrain][2] = force->inumeric(FLERR,arg[iarg+4]);
        anyangle = 1;
        iarg += 5;
      } else if (strcmp(arg[iarg],"dihedral") == 0) {
        if (iarg+6 > narg) error->all(FLERR,"Illegal fix topo command");
        rstyle[nrestrain] = DIHEDRAL;
        type[nrestrain] = force->inumeric(FLERR,arg[iarg+1]);
        ids[nrestrain][0] = force->inumeric(FLERR,arg[iarg+2]);
        ids[nrestrain][1] = force->inumeric(FLERR,arg[iarg+3]);
        ids[nrestrain][2] = force->inumeric(FLERR,arg[iarg+4]);
        ids[nrestrain][3] = force->inumeric(FLERR,arg[iarg+5]);
        anydihed = 1;
        iarg += 6;
      } else if (strcmp(arg[iarg],"improper") == 0) {
        if (iarg+6 > narg) error->all(FLERR,"Illegal fix topo command");
        rstyle[nrestrain] = IMPROPER;
        type[nrestrain] = force->inumeric(FLERR,arg[iarg+1]);
        ids[nrestrain][0] = force->inumeric(FLERR,arg[iarg+2]);
        ids[nrestrain][1] = force->inumeric(FLERR,arg[iarg+3]);
        ids[nrestrain][2] = force->inumeric(FLERR,arg[iarg+4]);
        ids[nrestrain][3] = force->inumeric(FLERR,arg[iarg+5]);
        anyimpro = 1;
        iarg += 6;
      } else if (strcmp(arg[iarg],"atom") == 0) {
        if (iarg+4 > narg) error->all(FLERR,"Illegal fix topo command");
        ids[nrestrain][0] = force->inumeric(FLERR,arg[iarg+1]);
        if (strcmp(arg[iarg+2],"type") == 0) {
          rstyle[nrestrain] = VDWL;
          type[nrestrain] = force->inumeric(FLERR,arg[iarg+3]);
          anyvdwl = 1;
          iarg += 4;
        } else if (strcmp(arg[iarg+2],"charge") == 0) {
          rstyle[nrestrain] = COUL;
          q[nrestrain] = force->numeric(FLERR,arg[iarg+3]);
          anyq = 1;
          iarg += 4;
        } else error->all(FLERR,"Illegal fix topo command");
      } else error->all(FLERR,"Illegal fix topo command");

      nrestrain++;
  }

  // require atom map to lookup atom IDs
  if (atom->map_style == 0)
    error->all(FLERR,"fix topo requires an atom map, see atom_modify");

  memory->create(f,atom->natoms,3,"topo:f");

  // copy = special list for one atom
  // size = ms^2 + ms is sufficient
  // b/c in rebuild_special_one() neighs of all 1-2s are added,
  //   then a dedup(), then neighs of all 1-3s are added, then final dedup()
  // this means intermediate size cannot exceed ms^2 + ms

  int maxspecial = atom->maxspecial;
  copy = new tagint[maxspecial*maxspecial + maxspecial];
}

/* ---------------------------------------------------------------------- */

FixTopo::~FixTopo()
{
  memory->destroy(rstyle);
  memory->destroy(ids);
  memory->destroy(type);
  memory->destroy(q);
  memory->destroy(f);
  delete [] copy;
}

/* ---------------------------------------------------------------------- */

int FixTopo::setmask()
{
  int mask = 0;
  mask |= POST_FORCE;
  mask |= POST_RUN;
  mask |= THERMO_ENERGY;
  // NOTE: I don't know how to exclude this fix from pe calculation, thus if
  //       this is set to yes relative energies are OK, but energies of
  //       individual topologies are not correct
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

void FixTopo::post_force(int vflag)
{
  int i1,i2,i3,i4,itmp;
  double **f_new;
  double **f_old;

  // create temporary array to store forces on atoms
  memory->create(f_new,atom->natoms,3,"topo:f_new");
  memory->create(f_old,atom->natoms,3,"topo:f_old");

  // apply restraints and store old values via triangular exchange in topo_create()
  topo_create();
  energy_new = topo_eval(vflag);
  for (int i = 0; i < atom->nlocal; i++) {
    f_new[i][0] = f[i][0];
    f_new[i][1] = f[i][1];
    f_new[i][2] = f[i][2];
  }

  // reverse restraints and recalculate topology via triangular exchange in topo_create()
  topo_create();
  energy_old = topo_eval(vflag);
  for (int i = 0; i < atom->nlocal; i++) {
    f_old[i][0] = f[i][0];
    f_old[i][1] = f[i][1];
    f_old[i][2] = f[i][2];
  }

  // mix old and new forces with delta to smoothly turn on/off interactions
  // double delta = update->ntimestep - update->beginstep;
  // if (delta != 0.0) delta /= update->endstep - update->beginstep;

  double tf2 = update->endstep*update->endstep;
  double ti2 = update->beginstep*update->beginstep;
  double t2 = update->ntimestep*update->ntimestep;
  double tf2t2 = tf2-t2;
  double tf2ti2 = tf2-ti2;
  double S = 1.0 - tf2t2*tf2t2*(tf2+2.0*t2-3.0*ti2) / (tf2ti2*tf2ti2*tf2ti2);
  //printf("timestep: %d %d %d, S: %f\n",update->beginstep,update->endstep,update->ntimestep,S);

  energy = S * (energy_new - energy_old);

  for (int i = 0; i < atom->nlocal; i++) {
    atom->f[i][0] += S * ( f_new[i][0] - f_old[i][0] );
    atom->f[i][1] += S * ( f_new[i][1] - f_old[i][1] );
    atom->f[i][2] += S * ( f_new[i][2] - f_old[i][2] );
  }

  // cleanup
  memory->destroy(f_new);
  memory->destroy(f_old);
}

/* ---------------------------------------------------------------------- */

void FixTopo::post_run()
{
  if (!resetflag) topo_create();
}

/* ----------------------------------------------------------------------
   apply or remove new topology
---------------------------------------------------------------------- */

void FixTopo::topo_create()
{
  int i,ival;
  double dval;

  // apply restraints and store old values via triangular exchange for nonboned
  // interactions and by inverting the sign of the type for bonded interactions
  // so the next call of topo_create() will restore the topology

  for (int m = 0; m < nrestrain; m++) {
    i = atom->map(ids[m][0]);

    if (rstyle[m] == BOND) {
      if (type[m] > 0)      { create_bond(m); type[m] = -type[m];}
      else if (type[m] < 0) { break_bond(m);  type[m] = -type[m];}
    } else if (rstyle[m] == ANGLE) {
      if (type[m] > 0)      { create_angle(m); type[m] = -type[m];}
      else if (type[m] < 0) { break_angle(m);  type[m] = -type[m];}
    } else if (rstyle[m] == DIHEDRAL) {
      if (type[m] > 0)      { create_dihedral(m); type[m] = -type[m];}
      else if (type[m] < 0) { break_dihedral(m);  type[m] = -type[m];}
    } else if (rstyle[m] == IMPROPER) {
      if (type[m] > 0)      { create_improper(m); type[m] = -type[m];}
      else if (type[m] < 0) { break_improper(m);  type[m] = -type[m];}
    } else if (rstyle[m] == VDWL) {
      // if atom is not owned py proc skip
      if (i < 0) continue;
      ival = atom->type[i];
      atom->type[i] = type[m];
      type[m] = ival;
    } else if (rstyle[m] == COUL) {
      // if atom is not owned py proc skip
      if (i < 0) continue;
      dval = atom->q[i];
      atom->q[i] = q[m];
      q[m] = dval;
    }
  }

  // update topology AFTER all bonds have been applied
  if (anybond) topo_update();

  // NOTE: do we need to communicate the changes in topology to other procs?

  // re-initialize pair styles if any PAIR settings were changed
  // ditto for bond styles if any BOND setitings were changes
  // this resets other coeffs that may depend on changed values,
  //   and also offset and tail corrections

  if (anyvdwl) force->pair->reinit();
  if (anybond) force->bond->reinit();
  if (anyangle) force->angle->init_style();
  if (anydihed) force->dihedral->init_style();
  if (anyimpro) force->improper->init_style();
  if (anyq && force->kspace) force->kspace->qsum_qsq();
}

/* ----------------------------------------------------------------------
   create bond
---------------------------------------------------------------------- */

void FixTopo::create_bond(int nrestrain)
{
  int i,j,k,m,n,n1,n2,n3;
  tagint *slist;
  tagint *partner;

  memory->create(partner,atom->nmax,"topo:partner");

  tagint *tag = atom->tag;
  int nlocal = atom->nlocal;

  for (i = 0; i < nlocal; i++)
    partner[i] = 0;
  partner[atom->map(ids[nrestrain][0])] = ids[nrestrain][1];
  partner[atom->map(ids[nrestrain][1])] = ids[nrestrain][0];

  tagint **bond_atom = atom->bond_atom;
  int *num_bond = atom->num_bond;
  int **nspecial = atom->nspecial;
  tagint **special = atom->special;

  // create bonds for atoms I own
  // only if both atoms list each other as winning bond partner
  //   and probability constraint is satisfied
  // if other atom is owned by another proc, it should do same thing

  int **bond_type = atom->bond_type;
  int newton_bond = force->newton_bond;

  for (i = 0; i < nlocal; i++) {
    if (partner[i] == 0) continue;
    j = atom->map(partner[i]);
    if (partner[j] != tag[i]) continue;

    // if newton_bond is set, only store with I or J
    // if not newton_bond, store bond with both I and J
    // atom J will also do this consistently, whatever proc it is on

    if (!newton_bond || tag[i] < tag[j]) {
      if (num_bond[i] == atom->bond_per_atom)
        error->one(FLERR,"New bond exceeded bonds per atom in fix topo");
      bond_type[i][num_bond[i]] = abs(type[nrestrain]);
      bond_atom[i][num_bond[i]] = tag[j];
      num_bond[i]++;
    }

    // add a 1-2 neighbor to special bond list for atom I
    // atom J will also do this, whatever proc it is on
    // need to first remove tag[j] from later in list if it appears
    // prevents list from overflowing, will be rebuilt in rebuild_special_one()

    slist = special[i];
    n1 = nspecial[i][0];
    n2 = nspecial[i][1];
    n3 = nspecial[i][2];
    for (m = n1; m < n3; m++)
      if (slist[m] == tag[j]) break;
    if (m < n3) {
      for (n = m; n < n3-1; n++) slist[n] = slist[n+1];
      n3--;
      if (m < n2) n2--;
    }
    if (n3 == atom->maxspecial)
      error->one(FLERR,
                 "New bond exceeded special list size in fix topo");
    for (m = n3; m > n1; m--) slist[m] = slist[m-1];
    slist[n1] = tag[j];
    nspecial[i][0] = n1+1;
    nspecial[i][1] = n2+1;
    nspecial[i][2] = n3+1;

    if (tag[i] < tag[j]) atom->nbonds++;
  }

  memory->destroy(partner);
}

/* ----------------------------------------------------------------------
   remove bond
---------------------------------------------------------------------- */

void FixTopo::break_bond(int nrestrain)
{
  int i,j,k,m,n,n1,n3;
  tagint *slist;
  tagint *partner;

  memory->create(partner,atom->nmax,"topo:partner");

  tagint *tag = atom->tag;
  int nlocal = atom->nlocal;

  for (i = 0; i < nlocal; i++)
    partner[i] = 0;
  partner[atom->map(ids[nrestrain][0])] = ids[nrestrain][1];
  partner[atom->map(ids[nrestrain][1])] = ids[nrestrain][0];

  int **bond_type = atom->bond_type;
  tagint **bond_atom = atom->bond_atom;
  int *num_bond = atom->num_bond;
  int **nspecial = atom->nspecial;
  tagint **special = atom->special;

  int nbreak = 0;
  for (i = 0; i < nlocal; i++) {
    if (partner[i] == 0) continue;
    j = atom->map(partner[i]);
    if (partner[j] != tag[i]) continue;

    // delete bond from atom I if I stores it
    // atom J will also do this

    for (m = 0; m < num_bond[i]; m++) {
      if (bond_atom[i][m] == partner[i] &&
          bond_type[i][m] == abs(type[nrestrain])) {
        for (k = m; k < num_bond[i]-1; k++) {
          bond_atom[i][k] = bond_atom[i][k+1];
          bond_type[i][k] = bond_type[i][k+1];
        }
        num_bond[i]--;
        nbreak++;
        break;
      }
    }

    // remove J from special bond list for atom I
    // atom J will also do this, whatever proc it is on

    slist = special[i];
    n1 = nspecial[i][0];
    for (m = 0; m < n1; m++)
      if (slist[m] == partner[i]) break;
    n3 = nspecial[i][2];
    for (; m < n3-1; m++) slist[m] = slist[m+1];
    nspecial[i][0]--;
    nspecial[i][1]--;
    nspecial[i][2]--;

    // store final broken bond partners and count the broken bond once
    if (tag[i] < tag[j]) atom->nbonds--;
  }

  //if (nbreak < 1) error->all(FLERR,"bond has not been previously defined in fix topo");

  memory->destroy(partner);
}

/* ----------------------------------------------------------------------
   double loop over my atoms and topo bonds
   influenced = 1 if atom's topology is affected by topo bond
     yes if is one of 2 atoms in bond
     yes if both atom IDs appear in atom's special list
     else no
   if influenced:
     check for angles/dihedrals/impropers to break due to specific broken bonds
     rebuild the atom's special list of 1-2,1-3,1-4 neighs
------------------------------------------------------------------------- */

void FixTopo::topo_update()
{
  int i,j,k,n,influence,influenced,found;
  tagint id1,id2;
  tagint *slist;

  tagint *tag = atom->tag;
  int **nspecial = atom->nspecial;
  tagint **special = atom->special;
  int nlocal = atom->nlocal;

  for (i = 0; i < nlocal; i++) {
    influenced = 0;
    slist = special[i];

    for (j = 0; j < nrestrain; j++) {
      if (rstyle[j] != BOND) continue;
      id1 = ids[j][0];
      id2 = ids[j][1];

      influence = 0;
      if (tag[i] == id1 || tag[i] == id2) influence = 1;
      else {
        n = nspecial[i][2];
        found = 0;
        for (k = 0; k < n; k++)
          if (slist[k] == id1 || slist[k] == id2) found++;
        if (found == 2) influence = 1;
      }
      if (!influence) continue;
      influenced = 1;
    }

    if (influenced) rebuild_special_one(i);
  }
}

/* ----------------------------------------------------------------------
   apply harmonic angle restraints
---------------------------------------------------------------------- */

void FixTopo::create_angle(int restrain)
{
  /* create angle from restrain once or three times if newton is off */
  int m;

  int *num_angle = atom->num_angle;
  int **angle_type = atom->angle_type;
  tagint **angle_atom1 = atom->angle_atom1;
  tagint **angle_atom2 = atom->angle_atom2;
  tagint **angle_atom3 = atom->angle_atom3;

  if ((m = atom->map(ids[restrain][1])) >= 0) {
    if (num_angle[m] == atom->angle_per_atom)
      error->one(FLERR,"New angle exceeded angles per atom in create_bonds");
    angle_type[m][num_angle[m]] = type[restrain];
    angle_atom1[m][num_angle[m]] = ids[restrain][0];
    angle_atom2[m][num_angle[m]] = ids[restrain][1];
    angle_atom3[m][num_angle[m]] = ids[restrain][2];
    num_angle[m]++;
  }
  atom->nangles++;

  if (force->newton_bond) return;

  if ((m = atom->map(ids[restrain][0])) >= 0) {
    if (num_angle[m] == atom->angle_per_atom)
      error->one(FLERR,"New angle exceeded angles per atom in create_bonds");
    angle_type[m][num_angle[m]] = type[restrain];
    angle_atom1[m][num_angle[m]] = ids[restrain][0];
    angle_atom2[m][num_angle[m]] = ids[restrain][1];
    angle_atom3[m][num_angle[m]] = ids[restrain][2];
    num_angle[m]++;
  }

  if ((m = atom->map(ids[restrain][2])) >= 0) {
    if (num_angle[m] == atom->angle_per_atom)
      error->one(FLERR,"New angle exceeded angles per atom in create_bonds");
    angle_type[m][num_angle[m]] = type[restrain];
    angle_atom1[m][num_angle[m]] = ids[restrain][0];
    angle_atom2[m][num_angle[m]] = ids[restrain][1];
    angle_atom3[m][num_angle[m]] = ids[restrain][2];
    num_angle[m]++;
  }
}

/* ----------------------------------------------------------------------
   remove harmonic angle restraints
---------------------------------------------------------------------- */

void FixTopo::break_angle(int restrain)
{
  /* break an angle from the restrain. this is done by looking for an
  angle in which all three atoms of the restrain are in an arbitrary order
  defined and which has the desired restrain type.
  NOTE: epoxy rings are problematic since O-C1-C2 and O-C2-C1 are both
  defined, have the same atoms in the angle and have the same angle type.
  if both angles are defined in the restrain, the method won't find both
  and would normally produce an error ... (deactived for now) */

  int j,m,n,found;

  int *num_angle = atom->num_angle;
  int **angle_type = atom->angle_type;
  tagint **angle_atom1 = atom->angle_atom1;
  tagint **angle_atom2 = atom->angle_atom2;
  tagint **angle_atom3 = atom->angle_atom3;

  int nbreak = 0;
  for (int k = 0; k < 3; k++){
    if ((m = atom->map(ids[restrain][k])) >= 0) {
      j = 0;
      while (j < num_angle[m]) {
        found = 0;
        for (int i = 0; i < 3; i++) {
          if ((angle_atom1[m][j] == ids[restrain][i]) ||
              (angle_atom2[m][j] == ids[restrain][i]) ||
              (angle_atom3[m][j] == ids[restrain][i])) found++;
        }
        if ( (found == 3) && (angle_type[m][j] == abs(type[restrain])) ) {
          n = num_angle[m];
          angle_type[m][j] = angle_type[m][n-1];
          angle_atom1[m][j] = angle_atom1[m][n-1];
          angle_atom2[m][j] = angle_atom2[m][n-1];
          angle_atom3[m][j] = angle_atom3[m][n-1];
          num_angle[m]--;
          nbreak++;
        } else j++;
      }
    }
  }

  int breakcount;
  MPI_Allreduce(&nbreak,&breakcount,1,MPI_INT,MPI_SUM,world);

  // if (force->newton_bond && (allcount < 1))
  //   error->one(FLERR,"Angle has not been defined previously in fix topo");
  // if (!force->newton_bond && (allcount < 3))
  //   error->one(FLERR,"Angle has not been defined previously in fix topo");

  if (force->newton_bond)
    atom->nangles -= breakcount;
  else
    atom->nangles -= breakcount/3;
}

/* ----------------------------------------------------------------------
   apply dihedral restraints
---------------------------------------------------------------------- */

void FixTopo::create_dihedral(int restrain)
{
  /* create dihedral from restrain once or four times if newton is off.
  copied from delete_bonds.cpp and in contrast to delete_bonds atoms must
  not be necesseraly only on one processor. */

  int m;

  int *num_dihedral = atom->num_dihedral;
  int **dihedral_type = atom->dihedral_type;
  tagint **dihedral_atom1 = atom->dihedral_atom1;
  tagint **dihedral_atom2 = atom->dihedral_atom2;
  tagint **dihedral_atom3 = atom->dihedral_atom3;
  tagint **dihedral_atom4 = atom->dihedral_atom4;

  if ((m = atom->map(ids[restrain][1])) >= 0) {
    if (num_dihedral[m] == atom->dihedral_per_atom)
      error->one(FLERR,
                 "New dihedral exceeded dihedrals per atom in fix topo");
    dihedral_type[m][num_dihedral[m]] = type[restrain];
    dihedral_atom1[m][num_dihedral[m]] = ids[restrain][0];
    dihedral_atom2[m][num_dihedral[m]] = ids[restrain][1];
    dihedral_atom3[m][num_dihedral[m]] = ids[restrain][2];
    dihedral_atom4[m][num_dihedral[m]] = ids[restrain][3];
    num_dihedral[m]++;
  }
  atom->ndihedrals++;

  if (force->newton_bond) return;

  if ((m = atom->map(ids[restrain][0])) >= 0) {
    if (num_dihedral[m] == atom->dihedral_per_atom)
      error->one(FLERR,
                 "New dihedral exceeded dihedrals per atom in fix topo");
    dihedral_type[m][num_dihedral[m]] = type[restrain];
    dihedral_atom1[m][num_dihedral[m]] = ids[restrain][0];
    dihedral_atom2[m][num_dihedral[m]] = ids[restrain][1];
    dihedral_atom3[m][num_dihedral[m]] = ids[restrain][2];
    dihedral_atom4[m][num_dihedral[m]] = ids[restrain][3];
    num_dihedral[m]++;
  }

  if ((m = atom->map(ids[restrain][2])) >= 0) {
    if (num_dihedral[m] == atom->dihedral_per_atom)
      error->one(FLERR,
                 "New dihedral exceeded dihedrals per atom in fix topo");
    dihedral_type[m][num_dihedral[m]] = type[restrain];
    dihedral_atom1[m][num_dihedral[m]] = ids[restrain][0];
    dihedral_atom2[m][num_dihedral[m]] = ids[restrain][1];
    dihedral_atom3[m][num_dihedral[m]] = ids[restrain][2];
    dihedral_atom4[m][num_dihedral[m]] = ids[restrain][3];
    num_dihedral[m]++;
  }

  if ((m = atom->map(ids[restrain][3])) >= 0) {
    if (num_dihedral[m] == atom->dihedral_per_atom)
      error->one(FLERR,
                 "New dihedral exceeded dihedrals per atom in fix topo");
    dihedral_type[m][num_dihedral[m]] = type[restrain];
    dihedral_atom1[m][num_dihedral[m]] = ids[restrain][0];
    dihedral_atom2[m][num_dihedral[m]] = ids[restrain][1];
    dihedral_atom3[m][num_dihedral[m]] = ids[restrain][2];
    dihedral_atom4[m][num_dihedral[m]] = ids[restrain][3];
    num_dihedral[m]++;
  }
}

/* ----------------------------------------------------------------------
   remove dihedral restraints
---------------------------------------------------------------------- */

void FixTopo::break_dihedral(int restrain)
{
  /* break a dihedral from the restrain. this is done by looking for an
  angle in which all three atoms of the restrain are in an arbitrary order
  defined and which has the desired restrain type.
  NOTE: multiple atoms arranged differently with the same dihedral type
  will be deleted at once. this is accounted for in the counts but not if
  atoms are ordered differently in another restrain dihedral. normally the
  code would write an error that the dihedral has not been previously found
  which is deactived for now b/c is causes trouble in the epoxy crosslinking */

  int j,m,n,found;

  // create angle once or 3x if newton_bond set

  int *num_dihedral = atom->num_dihedral;
  int **dihedral_type = atom->dihedral_type;
  tagint **dihedral_atom1 = atom->dihedral_atom1;
  tagint **dihedral_atom2 = atom->dihedral_atom2;
  tagint **dihedral_atom3 = atom->dihedral_atom3;
  tagint **dihedral_atom4 = atom->dihedral_atom4;

  int nbreak = 0;
  for (int k = 0; k < 4; k++){
    if ((m = atom->map(ids[restrain][k])) >= 0) {
      j = 0;
      while (j < num_dihedral[m]) {
        found = 0;
        for (int i = 0; i < 4; i++)
          if ((dihedral_atom1[m][j] == ids[restrain][i]) ||
              (dihedral_atom2[m][j] == ids[restrain][i]) ||
              (dihedral_atom3[m][j] == ids[restrain][i]) ||
              (dihedral_atom4[m][j] == ids[restrain][i])) found++;
        if ( (found == 4) && (dihedral_type[m][j] == abs(type[restrain])) ) {
          n = num_dihedral[m];
          dihedral_type[m][j] = dihedral_type[m][n-1];
          dihedral_atom1[m][j] = dihedral_atom1[m][n-1];
          dihedral_atom2[m][j] = dihedral_atom2[m][n-1];
          dihedral_atom3[m][j] = dihedral_atom3[m][n-1];
          dihedral_atom4[m][j] = dihedral_atom4[m][n-1];
          num_dihedral[m]--;
          nbreak++;
        } else j++;
      }
    }
  }

  int breakcount;
  MPI_Allreduce(&nbreak,&breakcount,1,MPI_INT,MPI_SUM,world);
  // epoxy ring makes problems here ...
  // if (force->newton_bond && (allcount < 1))
  //   error->one(FLERR,"Dihedral has not been defined previously in fix topo");
  // if (!force->newton_bond && (allcount < 4))
  //   error->one(FLERR,"Dihedral has not been defined previously in fix topo");

  if (force->newton_bond)
    atom->ndihedrals -= breakcount;
  else
    atom->ndihedrals -= breakcount/4;
}

/* ----------------------------------------------------------------------
   apply improper restraints
---------------------------------------------------------------------- */

void FixTopo::create_improper(int restrain)
{
  int m;

  // check that 4 atoms exist

  const int nlocal = atom->nlocal;
  const int idx1 = atom->map(ids[restrain][0]);
  const int idx2 = atom->map(ids[restrain][1]);
  const int idx3 = atom->map(ids[restrain][2]);
  const int idx4 = atom->map(ids[restrain][3]);

  int count = 0;
  if ((idx1 >= 0) && (idx1 < nlocal)) count++;
  if ((idx2 >= 0) && (idx2 < nlocal)) count++;
  if ((idx3 >= 0) && (idx3 < nlocal)) count++;
  if ((idx4 >= 0) && (idx4 < nlocal)) count++;

  int allcount;
  MPI_Allreduce(&count,&allcount,1,MPI_INT,MPI_SUM,world);
  if (allcount != 4)
    error->all(FLERR,"Dihedral atoms do not exist in fix topo");

  // create improper once or 4x if newton_bond set

  int *num_improper = atom->num_improper;
  int **improper_type = atom->improper_type;
  tagint **improper_atom1 = atom->improper_atom1;
  tagint **improper_atom2 = atom->improper_atom2;
  tagint **improper_atom3 = atom->improper_atom3;
  tagint **improper_atom4 = atom->improper_atom4;

  if ((m = idx2) >= 0) {
    if (num_improper[m] == atom->improper_per_atom)
      error->one(FLERR,
                 "New improper exceeded impropers per atom in fix topo");
    improper_type[m][num_improper[m]] = type[restrain];
    improper_atom1[m][num_improper[m]] = ids[restrain][0];
    improper_atom2[m][num_improper[m]] = ids[restrain][1];
    improper_atom3[m][num_improper[m]] = ids[restrain][2];
    improper_atom4[m][num_improper[m]] = ids[restrain][3];
    num_improper[m]++;
  }
  atom->nimpropers++;

  if (force->newton_bond) return;

  if ((m = idx1) >= 0) {
    if (num_improper[m] == atom->improper_per_atom)
      error->one(FLERR,
                 "New improper exceeded impropers per atom in fix topo");
    improper_type[m][num_improper[m]] = type[restrain];
    improper_atom1[m][num_improper[m]] = ids[restrain][0];
    improper_atom2[m][num_improper[m]] = ids[restrain][1];
    improper_atom3[m][num_improper[m]] = ids[restrain][2];
    improper_atom4[m][num_improper[m]] = ids[restrain][3];
    num_improper[m]++;
  }

  if ((m = idx3) >= 0) {
    if (num_improper[m] == atom->improper_per_atom)
      error->one(FLERR,
                 "New improper exceeded impropers per atom in fix topo");
    improper_type[m][num_improper[m]] = type[restrain];
    improper_atom1[m][num_improper[m]] = ids[restrain][0];
    improper_atom2[m][num_improper[m]] = ids[restrain][1];
    improper_atom3[m][num_improper[m]] = ids[restrain][2];
    improper_atom4[m][num_improper[m]] = ids[restrain][3];
    num_improper[m]++;
  }

  if ((m = idx4) >= 0) {
    if (num_improper[m] == atom->improper_per_atom)
      error->one(FLERR,
                 "New improper exceeded impropers per atom in fix topo");
    improper_type[m][num_improper[m]] = type[restrain];
    improper_atom1[m][num_improper[m]] = ids[restrain][0];
    improper_atom2[m][num_improper[m]] = ids[restrain][1];
    improper_atom3[m][num_improper[m]] = ids[restrain][2];
    improper_atom4[m][num_improper[m]] = ids[restrain][3];
    num_improper[m]++;
  }
}

/* ----------------------------------------------------------------------
   remove improper restraints
---------------------------------------------------------------------- */

void FixTopo::break_improper(int restrain)
{
  int j,m,n;

  // check that 3 atoms exist

  const int nlocal = atom->nlocal;
  const int idx1 = atom->map(ids[restrain][0]);
  const int idx2 = atom->map(ids[restrain][1]);
  const int idx3 = atom->map(ids[restrain][2]);
  const int idx4 = atom->map(ids[restrain][3]);

  int count = 0;
  if ((idx1 >= 0) && (idx1 < nlocal)) count++;
  if ((idx2 >= 0) && (idx2 < nlocal)) count++;
  if ((idx3 >= 0) && (idx3 < nlocal)) count++;
  if ((idx4 >= 0) && (idx4 < nlocal)) count++;

  int allcount;
  MPI_Allreduce(&count,&allcount,1,MPI_INT,MPI_SUM,world);
  if (allcount != 4)
    error->all(FLERR,"Dihedral atoms do not exist in fix topo");

  // create angle once or 3x if newton_bond set

  int *num_improper = atom->num_improper;
  int **improper_type = atom->improper_type;
  tagint **improper_atom1 = atom->improper_atom1;
  tagint **improper_atom2 = atom->improper_atom2;
  tagint **improper_atom3 = atom->improper_atom3;
  tagint **improper_atom4 = atom->improper_atom4;

  int nbreak = 0;
  if ((m = idx2) >= 0) {
    j = 0;
    while (j < atom->num_improper[m])
      if ((improper_atom1[m][j] == ids[restrain][0]) &&
          (improper_atom2[m][j] == ids[restrain][1]) &&
          (improper_atom3[m][j] == ids[restrain][2]) &&
          (improper_atom4[m][j] == ids[restrain][3]) &&
          (improper_type[m][j] == abs(type[restrain]))) {
        n = num_improper[m];
        improper_type[m][j] = improper_type[m][n-1];
        improper_atom1[m][j] = improper_atom1[m][n-1];
        improper_atom2[m][j] = improper_atom2[m][n-1];
        improper_atom3[m][j] = improper_atom3[m][n-1];
        improper_atom4[m][j] = improper_atom4[m][n-1];
        num_improper[m]--;
        nbreak++;
      } else j++;
  }

  if (nbreak < 1)
    error->one(FLERR,"Dihedral has not been previously defined in fix topo");

  atom->nimpropers--;

  if (force->newton_bond) return;

  if ((m = idx1) >= 0) {
    j = 0;
    while (j < atom->num_improper[m])
      if ((improper_atom1[m][j] == ids[restrain][0]) &&
          (improper_atom2[m][j] == ids[restrain][1]) &&
          (improper_atom3[m][j] == ids[restrain][2]) &&
          (improper_atom4[m][j] == ids[restrain][3]) &&
          (improper_type[m][j] == abs(type[restrain]))) {
        n = num_improper[m];
        improper_type[m][j] = improper_type[m][n-1];
        improper_atom1[m][j] = improper_atom1[m][n-1];
        improper_atom2[m][j] = improper_atom2[m][n-1];
        improper_atom3[m][j] = improper_atom3[m][n-1];
        improper_atom4[m][j] = improper_atom4[m][n-1];
        num_improper[m]--;
        nbreak++;
      } else j++;
  }

  if ((m = idx3) >= 0) {
    j = 0;
    while (j < atom->num_improper[m])
      if ((improper_atom1[m][j] == ids[restrain][0]) &&
          (improper_atom2[m][j] == ids[restrain][1]) &&
          (improper_atom3[m][j] == ids[restrain][2]) &&
          (improper_atom4[m][j] == ids[restrain][3]) &&
          (improper_type[m][j] == abs(type[restrain]))) {
        n = num_improper[m];
        improper_type[m][j] = improper_type[m][n-1];
        improper_atom1[m][j] = improper_atom1[m][n-1];
        improper_atom2[m][j] = improper_atom2[m][n-1];
        improper_atom3[m][j] = improper_atom3[m][n-1];
        improper_atom4[m][j] = improper_atom4[m][n-1];
        num_improper[m]--;
        nbreak++;
      } else j++;
  }


  if ((m = idx4) >= 0) {
    j = 0;
    while (j < atom->num_improper[m])
      if ((improper_atom1[m][j] == ids[restrain][0]) &&
          (improper_atom2[m][j] == ids[restrain][1]) &&
          (improper_atom3[m][j] == ids[restrain][2]) &&
          (improper_atom4[m][j] == ids[restrain][3]) &&
          (improper_type[m][j] == abs(type[restrain]))) {
        n = num_improper[m];
        improper_type[m][j] = improper_type[m][n-1];
        improper_atom1[m][j] = improper_atom1[m][n-1];
        improper_atom2[m][j] = improper_atom2[m][n-1];
        improper_atom3[m][j] = improper_atom3[m][n-1];
        improper_atom4[m][j] = improper_atom4[m][n-1];
        num_improper[m]--;
        nbreak++;
      } else j++;
  }
}

/* ----------------------------------------------------------------------
   calculate difference of forces and energies of both topologies
     forces are stored in global array f
     forces are temporary stored and resetted to not mess up additional
        forces from other fixes
---------------------------------------------------------------------- */

double FixTopo::topo_eval(int vflag)
{
  // no matter what, rebuild neighbor list
  if (domain->triclinic) domain->x2lamda(atom->nlocal);
  domain->pbc();
  comm->exchange();
  atom->nghost = 0;
  comm->borders();
  if (domain->triclinic) domain->lamda2x(atom->nlocal+atom->nghost);
  if (modify->n_pre_neighbor) modify->pre_neighbor();
  neighbor->build(1);
  int eflag = 1;

  // create array to store old forces and energies and to restore them
  double **f_tmp;
  memory->create(f_tmp,atom->natoms,3,"topo:f_tmp");
  // store forces on atoms
  for (int i = 0; i < atom->nlocal; i++) {
    f_tmp[i][0] = atom->f[i][0];
    f_tmp[i][1] = atom->f[i][1];
    f_tmp[i][2] = atom->f[i][2];
  }
  // clear forces so we have a fresh array to calculate the forces
  size_t nbytes = sizeof(double) * (atom->nlocal + atom->nghost);
  if (nbytes) memset(&atom->f[0][0],0,3*nbytes);

  // forces and energies of new topology
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

  // store forces on atoms from actual topology
  for (int i = 0; i < atom->nlocal; i++) {
    f[i][0] = atom->f[i][0];
    f[i][1] = atom->f[i][1];
    f[i][2] = atom->f[i][2];
  }

  // restore forces to previous forces
  for (int i = 0; i < atom->nlocal; i++) {
    atom->f[i][0] = f_tmp[i][0];
    atom->f[i][1] = f_tmp[i][1];
    atom->f[i][2] = f_tmp[i][2];
  }

  update->eflag_global = update->ntimestep;
  double total_energy = c_pe->compute_scalar();

  // cleanup
  memory->destroy(f_tmp);

  return total_energy;
}

/* ----------------------------------------------------------------------
   re-build special list of atom M
   does not affect 1-2 neighs (already include effects of new bond)
   affects 1-3 and 1-4 neighs due to other atom's augmented 1-2 neighs
------------------------------------------------------------------------- */

void FixTopo::rebuild_special_one(int m)
{
  int i,j,n,n1,cn1,cn2,cn3;
  tagint *slist;

  tagint *tag = atom->tag;
  int **nspecial = atom->nspecial;
  tagint **special = atom->special;

  // existing 1-2 neighs of atom M

  slist = special[m];
  n1 = nspecial[m][0];
  cn1 = 0;
  for (i = 0; i < n1; i++)
    copy[cn1++] = slist[i];

  // new 1-3 neighs of atom M, based on 1-2 neighs of 1-2 neighs
  // exclude self
  // remove duplicates after adding all possible 1-3 neighs

  cn2 = cn1;
  for (i = 0; i < cn1; i++) {
    n = atom->map(copy[i]);
    if (n < 0)
      error->one(FLERR,"Fix topo needs ghost atoms from further away");
    slist = special[n];
    n1 = nspecial[n][0];
    for (j = 0; j < n1; j++)
      if (slist[j] != tag[m]) copy[cn2++] = slist[j];
  }

  cn2 = dedup(cn1,cn2,copy);
  if (cn2 > atom->maxspecial)
    error->one(FLERR,"Special list size exceeded in fix topo");

  // new 1-4 neighs of atom M, based on 1-2 neighs of 1-3 neighs
  // exclude self
  // remove duplicates after adding all possible 1-4 neighs

  cn3 = cn2;
  for (i = cn1; i < cn2; i++) {
    n = atom->map(copy[i]);
    if (n < 0)
      error->one(FLERR,"Fix topo needs ghost atoms from further away");
    slist = special[n];
    n1 = nspecial[n][0];
    for (j = 0; j < n1; j++)
      if (slist[j] != tag[m]) copy[cn3++] = slist[j];
  }

  cn3 = dedup(cn2,cn3,copy);
  if (cn3 > atom->maxspecial)
    error->one(FLERR,"Special list size exceeded in fix topo");

  // store new special list with atom M

  nspecial[m][0] = cn1;
  nspecial[m][1] = cn2;
  nspecial[m][2] = cn3;
  memcpy(special[m],copy,cn3*sizeof(int));
}

/* ----------------------------------------------------------------------
   remove all ID duplicates in copy from Nstart:Nstop-1
   compare to all previous values in copy
   return N decremented by any discarded duplicates
------------------------------------------------------------------------- */

int FixTopo::dedup(int nstart, int nstop, tagint *copy)
{
  int i;

  int m = nstart;
  while (m < nstop) {
    for (i = 0; i < m; i++)
      if (copy[i] == copy[m]) {
        copy[m] = copy[nstop-1];
        nstop--;
        break;
      }
    if (i == m) m++;
  }

  return nstop;
}

/* ----------------------------------------------------------------------
   potential energy of added force
------------------------------------------------------------------------- */

double FixTopo::compute_scalar()
{
  return energy;
}

/* ----------------------------------------------------------------------
  return energies of old an new topologies
------------------------------------------------------------------------- */

double FixTopo::compute_vector(int n)
{
  if (n == 0) {
    return energy_old;
  } else if (n == 1) {
    return energy_new;
  } else {
    return 0.0;
  }
}

int FixTopo::pack_forward_comm(int n, int *list, double *buf,
                                     int pbc_flag, int *pbc)
{
  int i,j,k,m,ns;

  int **nspecial = atom->nspecial;
  tagint **special = atom->special;

  m = 0;
  for (i = 0; i < n; i++) {
    j = list[i];
    ns = nspecial[j][0];
    buf[m++] = ubuf(ns).d;
    for (k = 0; k < ns; k++)
      buf[m++] = ubuf(special[j][k]).d;
  }
  return m;
}
