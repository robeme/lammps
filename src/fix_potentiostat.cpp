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
#define _USE_MATH_DEFINES

#include "fix_potentiostat.h"

#include <cmath>
#include <cstring>
#include <iostream>
#include <vector>

#include "atom.h"
#include "comm.h"
#include "compute.h"
#include "domain.h"
#include "error.h"
#include "force.h"
#include "group.h"
#include "kspace.h"
#include "math_const.h"
#include "memory.h"
#include "modify.h"
#include "neigh_list.h"
#include "pair.h"
#include "random_mars.h"
#include "update.h"

#define EWALD_F 1.12837917
#define EWALD_P 0.3275911
#define A1 0.254829592
#define A2 -0.284496736
#define A3 1.421413741
#define A4 -1.453152027
#define A5 1.061405429

using namespace LAMMPS_NS;
using namespace FixConst;
using namespace MathConst;
using namespace std;

/* ---------------------------------------------------------------------- */

//     0  1          2            3          4   5     6     7   8
// fix id group-id-1 potentiostat group-id-2 eta u1[V] u2[V] tau tq
// general TODO:
// check not Respa integrator

FixElPotential::FixElPotential(LAMMPS *lmp, int narg, char **arg)
    : Fix(lmp, narg, arg) {
  tag2eleall = eleall2tag = curr_tag2eleall = NULL;
  dynamic_group_allow = 0;
  scalar_flag = 0;
  nevery = 1;
  global_freq = nevery;
  extscalar = 1;
  elenum = 0;
  bbb_all = NULL;
  randomf = new RanMars(lmp, 112358 + comm->me);

  if (narg != 9 && narg != 11)
    error->all(FLERR, "Illegal fix potentiostat command");
  igroup2 = group->find(arg[3]);
  eta = utils::numeric(FLERR, arg[4], false, lmp);
  utarget1 =
      utils::numeric(FLERR, arg[5], false, lmp) * force->qe2f;  // V -> kcal / (mol e)
  utarget2 =
      utils::numeric(FLERR, arg[6], false, lmp) * force->qe2f;  // V -> kcal / (mol e)
  tau = utils::numeric(FLERR, arg[7], false, lmp);
  tq = utils::numeric(FLERR, arg[8], false, lmp);
  if (narg == 11) {
    c0_set = true;
    c0 = utils::numeric(FLERR, arg[10], false, lmp);
  } else {
    c0_set = false;
  }

  // error checks
  if (igroup2 == -1)
    error->all(FLERR, "Fix potentiostat 2nd group ID does not exist");
  if (igroup2 == -1)
    error->all(FLERR, "Fix potentiostat 2nd group ID does not exist");
  if (igroup2 == igroup)
    error->all(FLERR, "Two groups cannot be the same in fix potentiostat");
  group2bit = group->bitmask[igroup2];
  int *mask = atom->mask;
  for (int i = 0; i < atom->nlocal; i++) {
    if ((mask[i] & group2bit) && (mask[i] & groupbit))
      error->all(FLERR,
                 "One atom cannot be in both groups in fix potentiostat");
  }
  if (group->count(igroup) == 0 || group->count(igroup2) == 0)
    error->all(FLERR,
               "count of atoms in group cannot be zero in fix potentiostat");
}

/* ---------------------------------------------------------------------- */

FixElPotential::~FixElPotential() {
  memory->destroy3d_offset(cs, -kmax_created);
  memory->destroy3d_offset(sn, -kmax_created);
  delete[] bbb_all;
  delete[] curr_tag2eleall;
  delete[] tag2eleall;
  delete[] eleall2tag;
  delete[] kxvecs;
  delete[] kyvecs;
  delete[] kzvecs;
  delete[] ug;
  delete[] sfacrl;
  delete[] sfacim;
  delete[] sfacrl_all;
  delete[] sfacim_all;
  delete[] eleids_all;
  delete[] displs;
  delete[] elenum_list;
  delete randomf;
}
/* ---------------------------------------------------------------------- */
int FixElPotential::setmask() {
  int mask = 0;
  mask |= PRE_FORCE;
  return mask;
}

/* ---------------------------------------------------------------------- */
void FixElPotential::setup(int) {
  g_ewald = force->kspace->g_ewald;
  slab_volfactor = force->kspace->slab_volfactor;
  double accuracy = force->kspace->accuracy;
  int const nlocal = atom->nlocal;
  int *tag = atom->tag;
  double qsqsum = 0.0;
  for (int i = 0; i < nlocal; i++) {
    qsqsum += atom->q[i] * atom->q[i];
  }
  double tmp, q2;
  MPI_Allreduce(&qsqsum, &tmp, 1, MPI_DOUBLE, MPI_SUM, world);
  qsqsum = tmp;
  q2 = qsqsum * force->qqr2e;

  // capacitance per atom
  if (!c0_set) c0 = capacitance_cal();
  if (comm->me == 0) cout << "CAPACITANCE: " << c0 << endl;
  int elenum1 = 0, elenum2 = 0;
  int elenum1_all = 0, elenum2_all = 0;
  for (int i = 0; i < nlocal; i++) {
    if (electrode_check(i) == 1)
      elenum1++;
    else if (electrode_check(i) == -1)
      elenum2++;
  }
  MPI_Allreduce(&elenum1, &elenum1_all, 1, MPI_INT, MPI_SUM, world);
  MPI_Allreduce(&elenum2, &elenum2_all, 1, MPI_INT, MPI_SUM, world);
  c0_atom1 = c0 / elenum1_all;
  c0_atom2 = c0 / elenum2_all;

  // Copied from ewald.cpp
  double xprd = domain->xprd;
  double yprd = domain->yprd;
  double zprd = domain->zprd;
  double zprd_slab = zprd * slab_volfactor;
  volume = xprd * yprd * zprd_slab;

  unitk[0] = 2.0 * MY_PI / xprd;
  unitk[1] = 2.0 * MY_PI / yprd;
  unitk[2] = 2.0 * MY_PI / zprd_slab;

  bigint natoms = atom->natoms;
  double err;
  kxmax = 1;
  kymax = 1;
  kzmax = 1;

  err = rms(kxmax, xprd, natoms, q2);
  while (err > accuracy) {
    kxmax++;
    err = rms(kxmax, xprd, natoms, q2);
  }

  err = rms(kymax, yprd, natoms, q2);
  while (err > accuracy) {
    kymax++;
    err = rms(kymax, yprd, natoms, q2);
  }

  err = rms(kzmax, zprd_slab, natoms, q2);
  while (err > accuracy) {
    kzmax++;
    err = rms(kzmax, zprd_slab, natoms, q2);
  }

  kmax = MAX(kxmax, kymax);
  kmax = MAX(kmax, kzmax);
  kmax3d = 4 * kmax * kmax * kmax + 6 * kmax * kmax + 3 * kmax;

  kxvecs = new int[kmax3d];
  kyvecs = new int[kmax3d];
  kzvecs = new int[kmax3d];
  ug = new double[kmax3d];

  double gsqxmx = unitk[0] * unitk[0] * kxmax * kxmax;
  double gsqymx = unitk[1] * unitk[1] * kymax * kymax;
  double gsqzmx = unitk[2] * unitk[2] * kzmax * kzmax;
  gsqmx = MAX(gsqxmx, gsqymx);
  gsqmx = MAX(gsqmx, gsqzmx);

  gsqmx *= 1.00001;

  coeffs();
  kmax_created = kmax;
  // copied from ewald.cpp end

  int nmax = atom->nmax;

  curr_tag2eleall = new int[natoms + 1];
  tag2eleall = new int[natoms + 1];
  int const nprocs = comm->nprocs;
  elenum = 0;
  for (int i = 0; i < nlocal; i++) {
    if (electrode_check(i)) elenum++;
  }
  MPI_Allreduce(&elenum, &elenum_all, 1, MPI_INT, MPI_SUM, world);
  eleall2tag = new int[elenum_all];
  eleids_all = new int[elenum_all];
  elenum_list = new int[nprocs];
  MPI_Allgather(&elenum, 1, MPI_INT, elenum_list, 1, MPI_INT, world);

  // from a_cal
  displs = new int[nprocs];
  displs[0] = 0;
  int displssum = 0;
  for (int i = 1; i < nprocs; i++) {
    displssum += elenum_list[i - 1];
    displs[i] = displssum;
  }
  int *ele2tag = new int[elenum];
  for (int i = 0, j = 0; i < nlocal; i++) {
    if (electrode_check(i)) {
      ele2tag[j] = tag[i];
      j++;
    }
  }

  // gather tag,x and q
  MPI_Allgatherv(ele2tag, elenum, MPI_INT, eleall2tag, elenum_list, displs,
                 MPI_INT, world);
  delete[] ele2tag;

  for (int i = 0; i < natoms + 1; i++) tag2eleall[i] = -1;
  for (int i = 0; i < elenum_all; i++) {
    tag2eleall[eleall2tag[i]] = i;
  }
  // end a_cal

  memory->create3d_offset(cs, -kmax, kmax, 3, nmax, "fixpotential:cs");
  memory->create3d_offset(sn, -kmax, kmax, 3, nmax, "fixpotential:sn");
  sfacrl = new double[kmax3d];
  sfacim = new double[kmax3d];
  sfacrl_all = new double[kmax3d];
  sfacim_all = new double[kmax3d];
  bbb_all = new double[elenum_all];

  for (int i = 0; i < natoms + 1; i++) curr_tag2eleall[i] = -1;
}
/* ---------------------------------------------------------------------- */
void FixElPotential::pre_force(int /*vflag*/) {
  double const dt = update->dt;
  int const nall = atom->nlocal + atom->nghost;
  int const nlocal = atom->nlocal;
  int const nprocs = comm->nprocs;
  // int const mpitag = 0;
  int *tag = atom->tag;
  // double fluct_atom;
  elenum = 0;
  for (int i = 0; i < nlocal; i++) {
    if (electrode_check(i)) elenum++;
  }
  MPI_Allgather(&elenum, 1, MPI_INT, elenum_list, 1, MPI_INT, world);
  displs[0] = 0;
  for (int i = 1, displssum = 0; i < nprocs; ++i) {
    displssum += elenum_list[i - 1];
    displs[i] = displssum;
  }
  auto eleid = [this, tag](int atomid) { return tag2eleall[tag[atomid]]; };

  b_cal();
  double const exp_dt = 1 - std::exp(-dt / tau);
  double const fluct =
      std::sqrt(force->boltz * tq * (1 - std::exp(-2 * dt / tau)));
  double qqrd2e = force->qqrd2e;
  for (int i = 0; i < nall; i++) {
    if (electrode_check(i)) {
      double phi = -bbb_all[eleid(i)] * qqrd2e;
      double phi0 = (atom->mask[i] & groupbit) ? utarget1 : utarget2;
      double c0_atom = (atom->mask[i] & groupbit) ? c0_atom1 : c0_atom2;
      atom->q[i] += -c0_atom * (phi - phi0) * exp_dt +
                    randomf->gaussian() * std::sqrt(c0_atom) * fluct;
    }
  }

  force_cal();
}

/* ---------------------------------------------------------------------- */
void FixElPotential::force_cal() {
  if (force->kspace->energy) {
    double eleqsqsum = 0.0;
    for (int i = 0; i < atom->nlocal; i++) {
      if (electrode_check(i)) {
        eleqsqsum += atom->q[i] * atom->q[i];
      }
    }
    double tmp;
    MPI_Allreduce(&eleqsqsum, &tmp, 1, MPI_DOUBLE, MPI_SUM, world);
    eleqsqsum = tmp;
    double scale = 1.0;
    double qscale = force->qqrd2e * scale;
    force->kspace->energy += qscale * eta * eleqsqsum / (sqrt(2) * MY_PIS);
  }
  coul_cal(0, NULL, NULL);
}
/* ---------------------------------------------------------------------- */
void FixElPotential::coul_cal(int coulcalflag, double *m, int *ele2tag) {
  // coulcalflag = 2: a_cal; 1: b_cal; 0: force_cal
  double qtmp, xtmp, ytmp, ztmp, delx, dely, delz;
  double r, r2inv, rsq, grij, etarij, expm2, t, erfc, dudq;
  double forcecoul, ecoul, prefactor, fpair;

  int inum = force->pair->list->inum;
  int nlocal = atom->nlocal;
  int newton_pair = force->newton_pair;
  int *atomtype = atom->type;
  int *tag = atom->tag;
  int *ilist = force->pair->list->ilist;
  int *jlist;
  int *numneigh = force->pair->list->numneigh;
  int **firstneigh = force->pair->list->firstneigh;

  double qqrd2e = force->qqrd2e;
  double **cutsq = force->pair->cutsq;
  int itmp;
  double *p_cut_coul = (double *)force->pair->extract("cut_coul", itmp);
  double cut_coulsq = (*p_cut_coul) * (*p_cut_coul);
  double **x = atom->x;
  double **f = atom->f;
  double *q = atom->q;

  for (int ii = 0; ii < inum; ii++) {
    int i = ilist[ii];
    qtmp = q[i];
    xtmp = x[i][0];
    ytmp = x[i][1];
    ztmp = x[i][2];
    int itype = atomtype[i];
    jlist = firstneigh[i];
    int jnum = numneigh[i];
    for (int jj = 0; jj < jnum; jj++) {
      int j = jlist[jj];
      j &= NEIGHMASK;
      int checksum = abs(electrode_check(i)) + abs(electrode_check(j));
      if (checksum == 1 || checksum == 2) {
        delx = xtmp - x[j][0];
        dely = ytmp - x[j][1];
        delz = ztmp - x[j][2];
        rsq = delx * delx + dely * dely + delz * delz;
        int jtype = atomtype[j];
        if (rsq < cutsq[itype][jtype]) {
          r2inv = 1.0 / rsq;
          if (rsq < cut_coulsq) {
            dudq = 0.0;
            r = sqrt(rsq);
            if (coulcalflag != 0) {
              grij = g_ewald * r;
              expm2 = exp(-grij * grij);
              t = 1.0 / (1.0 + EWALD_P * grij);
              erfc = t * (A1 + t * (A2 + t * (A3 + t * (A4 + t * A5)))) * expm2;
              dudq = erfc / r;
            }
            if (checksum == 1)
              etarij = eta * r;
            else if (checksum == 2)
              etarij = eta * r / sqrt(2);
            expm2 = exp(-etarij * etarij);
            t = 1.0 / (1.0 + EWALD_P * etarij);
            erfc = t * (A1 + t * (A2 + t * (A3 + t * (A4 + t * A5)))) * expm2;

            if (coulcalflag == 0) {
              prefactor = qqrd2e * qtmp * q[j] / r;
              forcecoul = -prefactor * (erfc + EWALD_F * etarij * expm2);
              fpair = forcecoul * r2inv;
              f[i][0] += delx * forcecoul;
              f[i][1] += dely * forcecoul;
              f[i][2] += delz * forcecoul;
              if (newton_pair || j < nlocal) {
                f[j][0] -= delx * forcecoul;
                f[j][1] -= dely * forcecoul;
                f[j][2] -= delz * forcecoul;
              }
              ecoul = -prefactor * erfc;
              force->pair->ev_tally(i, j, nlocal, newton_pair, 0, ecoul, fpair,
                                    delx, dely, delz);  // evdwl=0
            } else {
              dudq -= erfc / r;
              for (int k = 0; k < elenum; ++k) {
                if (i < nlocal) {
                  if (ele2tag[k] == tag[i]) {
                    m[k] -= q[j] * dudq;
                  }
                }
                if (j < nlocal) {
                  if (ele2tag[k] == tag[j]) {
                    m[k] -= q[i] * dudq;
                  }
                }
                if (i < nlocal && j < nlocal) {
                  if (ele2tag[k] == tag[i] && ele2tag[k] == tag[j]) {
                    break;
                  }
                }
              }
            }
          }
        }
      }
    }
  }
}
/* ---------------------------------------------------------------------- */
int FixElPotential::electrode_check(int atomid) {
  if (atom->mask[atomid] & groupbit)
    return 1;
  else if (atom->mask[atomid] & group2bit)
    return -1;
  else
    return 0;
}
/* ---------------------------------------------------------------------- */
double FixElPotential::capacitance_cal() {
  double z1max, z2max, z1min, z2min;
  z1max = z2max = domain->boxlo[2];
  z1min = z2min = domain->boxhi[2];
  int *mask = atom->mask;
  double **x = atom->x;
  for (int i = 0; i < atom->nlocal; i++) {
    double z = x[i][2];
    if (mask[i] & groupbit) {
      if (z < z1min) z1min = z;
      if (z > z1max) z1max = z;
    } else if (mask[i] & group2bit) {
      if (z < z2min) z2min = z;
      if (z > z2max) z2max = z;
    }
  }
  double z1max_all, z2max_all, z1min_all, z2min_all;
  MPI_Allreduce(&z1min, &z1min_all, 1, MPI_DOUBLE, MPI_MIN, world);
  MPI_Allreduce(&z1max, &z1max_all, 1, MPI_DOUBLE, MPI_MAX, world);
  MPI_Allreduce(&z2min, &z2min_all, 1, MPI_DOUBLE, MPI_MIN, world);
  MPI_Allreduce(&z2max, &z2max_all, 1, MPI_DOUBLE, MPI_MAX, world);
  if ((z2max_all > z1min_all) != (z2min_all > z1max_all)) {
    error->all(FLERR, "electrodes are not seperated along z-axis");
  }
  double distance =
      (z2min_all > z1max_all) ? z2min_all - z1max_all : z1min_all - z2max_all;
  double area = domain->xprd * domain->yprd;
  return 1. / (MY_4PI * force->qqr2e) * area / distance;  // e^2 / (kcal/mol)
}

/* ----------------------------------------------------------------------*/
void FixElPotential::b_cal() {
  double const CON_s2overPIS = sqrt(2.0) / MY_PIS;
  double const CON_2overPIS = 2.0 / MY_PIS;
  int nmax = atom->nmax;
  if (atom->nlocal > nmax) {
    memory->destroy3d_offset(cs, -kmax_created);
    memory->destroy3d_offset(sn, -kmax_created);
    nmax = atom->nmax;
    kmax_created = kmax;
  }
  sincos_b();
  MPI_Allreduce(sfacrl, sfacrl_all, kcount, MPI_DOUBLE, MPI_SUM, world);
  MPI_Allreduce(sfacim, sfacim_all, kcount, MPI_DOUBLE, MPI_SUM, world);
  double **x = atom->x;
  double *q = atom->q;
  int *tag = atom->tag;
  int nlocal = atom->nlocal;
  int kx, ky, kz;
  double cypz, sypz, exprl, expim;
  double bbb[elenum];
  for (int i = 0; i < elenum; i++) bbb[i] = 0;
  for (int k = 0; k < kcount; k++) {
    kx = kxvecs[k];
    ky = kyvecs[k];
    kz = kzvecs[k];
    for (int i = 0, j = 0; i < nlocal; i++) {
      if (electrode_check(i)) {
        cypz = cs[ky][1][i] * cs[kz][2][i] - sn[ky][1][i] * sn[kz][2][i];
        sypz = sn[ky][1][i] * cs[kz][2][i] + cs[ky][1][i] * sn[kz][2][i];
        exprl = cs[kx][0][i] * cypz - sn[kx][0][i] * sypz;
        expim = sn[kx][0][i] * cypz + cs[kx][0][i] * sypz;
        bbb[j] -= 2.0 * ug[k] * (exprl * sfacrl_all[k] + expim * sfacim_all[k]);
        j++;
      }
    }
  }

  // slabcorrection and create ele tag list in current timestep
  double slabcorrtmp = 0.0;
  double slabcorrtmp_all = 0.0;
  for (int i = 0; i < nlocal; i++) {
    slabcorrtmp += 4 * q[i] * MY_PI * x[i][2] / volume;
  }
  int *ele2tag = new int[elenum];
  MPI_Allreduce(&slabcorrtmp, &slabcorrtmp_all, 1, MPI_DOUBLE, MPI_SUM, world);
  for (int i = 0, j = 0; i < nlocal; i++) {
    if (electrode_check(i)) {
      bbb[j] -= x[i][2] * slabcorrtmp_all;

      bbb[j] -= q[i] * (CON_s2overPIS * eta - CON_2overPIS * g_ewald);

      ele2tag[j] = tag[i];
      j++;
    }
  }

  coul_cal(1, bbb, ele2tag);

  // gather ele tag list
  int ele_taglist_all[elenum_all];
  int tagi;
  MPI_Allgatherv(ele2tag, elenum, MPI_INT, &ele_taglist_all, elenum_list,
                 displs, MPI_INT, world);
  delete[] ele2tag;
  for (int i = 0; i < elenum_all; i++) {
    tagi = ele_taglist_all[i];
    curr_tag2eleall[tagi] = i;
  }

  // gather b to bbb_all and sort in the same order as aaa_all
  double bbb_buf[elenum_all];
  MPI_Allgatherv(&bbb, elenum, MPI_DOUBLE, &bbb_buf, elenum_list, displs,
                 MPI_DOUBLE, world);
  int elei;
  for (int i = 0; i < elenum_all; i++) {
    tagi = eleall2tag[i];
    elei = curr_tag2eleall[tagi];
    bbb_all[i] = bbb_buf[elei];
  }
}
/*--------------------------------------------------------------*/
void FixElPotential::sincos_b() {
  double cstr1, sstr1, cstr2, sstr2, cstr3, sstr3, cstr4, sstr4;
  double sqk, clpm, slpm;

  double **x = atom->x;
  double *q = atom->q;
  int nlocal = atom->nlocal;

  int n = 0;

  // (k,0,0), (0,l,0), (0,0,m)

  for (int ic = 0; ic < 3; ic++) {
    sqk = unitk[ic] * unitk[ic];
    if (sqk <= gsqmx) {
      cstr1 = 0.0;
      sstr1 = 0.0;
      for (int i = 0; i < nlocal; i++) {
        cs[0][ic][i] = 1.0;
        sn[0][ic][i] = 0.0;
        cs[1][ic][i] = cos(unitk[ic] * x[i][ic]);
        sn[1][ic][i] = sin(unitk[ic] * x[i][ic]);
        cs[-1][ic][i] = cs[1][ic][i];
        sn[-1][ic][i] = -sn[1][ic][i];
        cstr1 += q[i] * cs[1][ic][i];
        sstr1 += q[i] * sn[1][ic][i];
      }
      sfacrl[n] = cstr1;
      sfacim[n++] = sstr1;
    }
  }
  for (int m = 2; m <= kmax; m++) {
    for (int ic = 0; ic < 3; ic++) {
      sqk = m * unitk[ic] * m * unitk[ic];
      if (sqk <= gsqmx) {
        cstr1 = 0.0;
        sstr1 = 0.0;
        for (int i = 0; i < nlocal; i++) {
          cs[m][ic][i] =
              cs[m - 1][ic][i] * cs[1][ic][i] - sn[m - 1][ic][i] * sn[1][ic][i];
          sn[m][ic][i] =
              sn[m - 1][ic][i] * cs[1][ic][i] + cs[m - 1][ic][i] * sn[1][ic][i];
          cs[-m][ic][i] = cs[m][ic][i];
          sn[-m][ic][i] = -sn[m][ic][i];
          cstr1 += q[i] * cs[m][ic][i];
          sstr1 += q[i] * sn[m][ic][i];
        }
        sfacrl[n] = cstr1;
        sfacim[n++] = sstr1;
      }
    }
  }

  // 1 = (k,l,0), 2 = (k,-l,0)
  for (int k = 1; k <= kxmax; k++) {
    for (int l = 1; l <= kymax; l++) {
      sqk = (k * unitk[0] * k * unitk[0]) + (l * unitk[1] * l * unitk[1]);
      if (sqk <= gsqmx) {
        cstr1 = 0.0;
        sstr1 = 0.0;
        cstr2 = 0.0;
        sstr2 = 0.0;
        for (int i = 0; i < nlocal; i++) {
          cstr1 +=
              q[i] * (cs[k][0][i] * cs[l][1][i] - sn[k][0][i] * sn[l][1][i]);
          sstr1 +=
              q[i] * (sn[k][0][i] * cs[l][1][i] + cs[k][0][i] * sn[l][1][i]);
          cstr2 +=
              q[i] * (cs[k][0][i] * cs[l][1][i] + sn[k][0][i] * sn[l][1][i]);
          sstr2 +=
              q[i] * (sn[k][0][i] * cs[l][1][i] - cs[k][0][i] * sn[l][1][i]);
        }
        sfacrl[n] = cstr1;
        sfacim[n++] = sstr1;
        sfacrl[n] = cstr2;
        sfacim[n++] = sstr2;
      }
    }
  }

  // 1 = (0,l,m), 2 = (0,l,-m)

  for (int l = 1; l <= kymax; l++) {
    for (int m = 1; m <= kzmax; m++) {
      sqk = (l * unitk[1] * l * unitk[1]) + (m * unitk[2] * m * unitk[2]);
      if (sqk <= gsqmx) {
        cstr1 = 0.0;
        sstr1 = 0.0;
        cstr2 = 0.0;
        sstr2 = 0.0;
        for (int i = 0; i < nlocal; i++) {
          cstr1 +=
              q[i] * (cs[l][1][i] * cs[m][2][i] - sn[l][1][i] * sn[m][2][i]);
          sstr1 +=
              q[i] * (sn[l][1][i] * cs[m][2][i] + cs[l][1][i] * sn[m][2][i]);
          cstr2 +=
              q[i] * (cs[l][1][i] * cs[m][2][i] + sn[l][1][i] * sn[m][2][i]);
          sstr2 +=
              q[i] * (sn[l][1][i] * cs[m][2][i] - cs[l][1][i] * sn[m][2][i]);
        }
        sfacrl[n] = cstr1;
        sfacim[n++] = sstr1;
        sfacrl[n] = cstr2;
        sfacim[n++] = sstr2;
      }
    }
  }

  // 1 = (k,0,m), 2 = (k,0,-m)

  for (int k = 1; k <= kxmax; k++) {
    for (int m = 1; m <= kzmax; m++) {
      sqk = (k * unitk[0] * k * unitk[0]) + (m * unitk[2] * m * unitk[2]);
      if (sqk <= gsqmx) {
        cstr1 = 0.0;
        sstr1 = 0.0;
        cstr2 = 0.0;
        sstr2 = 0.0;
        for (int i = 0; i < nlocal; i++) {
          cstr1 +=
              q[i] * (cs[k][0][i] * cs[m][2][i] - sn[k][0][i] * sn[m][2][i]);
          sstr1 +=
              q[i] * (sn[k][0][i] * cs[m][2][i] + cs[k][0][i] * sn[m][2][i]);
          cstr2 +=
              q[i] * (cs[k][0][i] * cs[m][2][i] + sn[k][0][i] * sn[m][2][i]);
          sstr2 +=
              q[i] * (sn[k][0][i] * cs[m][2][i] - cs[k][0][i] * sn[m][2][i]);
        }
        sfacrl[n] = cstr1;
        sfacim[n++] = sstr1;
        sfacrl[n] = cstr2;
        sfacim[n++] = sstr2;
      }
    }
  }

  // 1 = (k,l,m), 2 = (k,-l,m), 3 = (k,l,-m), 4 = (k,-l,-m)

  for (int k = 1; k <= kxmax; k++) {
    for (int l = 1; l <= kymax; l++) {
      for (int m = 1; m <= kzmax; m++) {
        sqk = (k * unitk[0] * k * unitk[0]) + (l * unitk[1] * l * unitk[1]) +
              (m * unitk[2] * m * unitk[2]);
        if (sqk <= gsqmx) {
          cstr1 = 0.0;
          sstr1 = 0.0;
          cstr2 = 0.0;
          sstr2 = 0.0;
          cstr3 = 0.0;
          sstr3 = 0.0;
          cstr4 = 0.0;
          sstr4 = 0.0;
          for (int i = 0; i < nlocal; i++) {
            clpm = cs[l][1][i] * cs[m][2][i] - sn[l][1][i] * sn[m][2][i];
            slpm = sn[l][1][i] * cs[m][2][i] + cs[l][1][i] * sn[m][2][i];
            cstr1 += q[i] * (cs[k][0][i] * clpm - sn[k][0][i] * slpm);
            sstr1 += q[i] * (sn[k][0][i] * clpm + cs[k][0][i] * slpm);

            clpm = cs[l][1][i] * cs[m][2][i] + sn[l][1][i] * sn[m][2][i];
            slpm = -sn[l][1][i] * cs[m][2][i] + cs[l][1][i] * sn[m][2][i];
            cstr2 += q[i] * (cs[k][0][i] * clpm - sn[k][0][i] * slpm);
            sstr2 += q[i] * (sn[k][0][i] * clpm + cs[k][0][i] * slpm);

            clpm = cs[l][1][i] * cs[m][2][i] + sn[l][1][i] * sn[m][2][i];
            slpm = sn[l][1][i] * cs[m][2][i] - cs[l][1][i] * sn[m][2][i];
            cstr3 += q[i] * (cs[k][0][i] * clpm - sn[k][0][i] * slpm);
            sstr3 += q[i] * (sn[k][0][i] * clpm + cs[k][0][i] * slpm);

            clpm = cs[l][1][i] * cs[m][2][i] - sn[l][1][i] * sn[m][2][i];
            slpm = -sn[l][1][i] * cs[m][2][i] - cs[l][1][i] * sn[m][2][i];
            cstr4 += q[i] * (cs[k][0][i] * clpm - sn[k][0][i] * slpm);
            sstr4 += q[i] * (sn[k][0][i] * clpm + cs[k][0][i] * slpm);
          }
          sfacrl[n] = cstr1;
          sfacim[n++] = sstr1;
          sfacrl[n] = cstr2;
          sfacim[n++] = sstr2;
          sfacrl[n] = cstr3;
          sfacim[n++] = sstr3;
          sfacrl[n] = cstr4;
          sfacim[n++] = sstr4;
        }
      }
    }
  }
}
/* ---------------------------------------------------------------------- */
double FixElPotential::rms(int km, double prd, bigint natoms, double q2) {
  return 2.0 * q2 * g_ewald / prd * sqrt(1.0 / (MY_PI * km * natoms)) *
         exp(-MY_PI * MY_PI * km * km / (g_ewald * g_ewald * prd * prd));
}
/*--------------------------------------------------------------*/

void FixElPotential::coeffs() {
  double sqk;

  double const g_ewald_sq_inv = 1.0 / (g_ewald * g_ewald);
  double const preu = 4.0 * MY_PI / volume;

  kcount = 0;

  // (k,0,0), (0,l,0), (0,0,m)

  for (int m = 1; m <= kmax; m++) {
    sqk = (m * unitk[0]) * (m * unitk[0]);
    if (sqk <= gsqmx) {
      kxvecs[kcount] = m;
      kyvecs[kcount] = 0;
      kzvecs[kcount] = 0;
      ug[kcount] = preu * exp(-0.25 * sqk * g_ewald_sq_inv) / sqk;
      kcount++;
    }
    sqk = (m * unitk[1]) * (m * unitk[1]);
    if (sqk <= gsqmx) {
      kxvecs[kcount] = 0;
      kyvecs[kcount] = m;
      kzvecs[kcount] = 0;
      ug[kcount] = preu * exp(-0.25 * sqk * g_ewald_sq_inv) / sqk;
      kcount++;
    }
    sqk = (m * unitk[2]) * (m * unitk[2]);
    if (sqk <= gsqmx) {
      kxvecs[kcount] = 0;
      kyvecs[kcount] = 0;
      kzvecs[kcount] = m;
      ug[kcount] = preu * exp(-0.25 * sqk * g_ewald_sq_inv) / sqk;
      kcount++;
    }
  }

  // 1 = (k,l,0), 2 = (k,-l,0)

  for (int k = 1; k <= kxmax; k++) {
    for (int l = 1; l <= kymax; l++) {
      sqk = (unitk[0] * k) * (unitk[0] * k) + (unitk[1] * l) * (unitk[1] * l);
      if (sqk <= gsqmx) {
        kxvecs[kcount] = k;
        kyvecs[kcount] = l;
        kzvecs[kcount] = 0;
        ug[kcount] = preu * exp(-0.25 * sqk * g_ewald_sq_inv) / sqk;
        kcount++;

        kxvecs[kcount] = k;
        kyvecs[kcount] = -l;
        kzvecs[kcount] = 0;
        ug[kcount] = preu * exp(-0.25 * sqk * g_ewald_sq_inv) / sqk;
        kcount++;
        ;
      }
    }
  }

  // 1 = (0,l,m), 2 = (0,l,-m)

  for (int l = 1; l <= kymax; l++) {
    for (int m = 1; m <= kzmax; m++) {
      sqk = (unitk[1] * l) * (unitk[1] * l) + (unitk[2] * m) * (unitk[2] * m);
      if (sqk <= gsqmx) {
        kxvecs[kcount] = 0;
        kyvecs[kcount] = l;
        kzvecs[kcount] = m;
        ug[kcount] = preu * exp(-0.25 * sqk * g_ewald_sq_inv) / sqk;
        kcount++;

        kxvecs[kcount] = 0;
        kyvecs[kcount] = l;
        kzvecs[kcount] = -m;
        ug[kcount] = preu * exp(-0.25 * sqk * g_ewald_sq_inv) / sqk;
        kcount++;
      }
    }
  }

  // 1 = (k,0,m), 2 = (k,0,-m)

  for (int k = 1; k <= kxmax; k++) {
    for (int m = 1; m <= kzmax; m++) {
      sqk = (unitk[0] * k) * (unitk[0] * k) + (unitk[2] * m) * (unitk[2] * m);
      if (sqk <= gsqmx) {
        kxvecs[kcount] = k;
        kyvecs[kcount] = 0;
        kzvecs[kcount] = m;
        ug[kcount] = preu * exp(-0.25 * sqk * g_ewald_sq_inv) / sqk;
        kcount++;

        kxvecs[kcount] = k;
        kyvecs[kcount] = 0;
        kzvecs[kcount] = -m;
        ug[kcount] = preu * exp(-0.25 * sqk * g_ewald_sq_inv) / sqk;
        kcount++;
      }
    }
  }

  // 1 = (k,l,m), 2 = (k,-l,m), 3 = (k,l,-m), 4 = (k,-l,-m)

  for (int k = 1; k <= kxmax; k++) {
    for (int l = 1; l <= kymax; l++) {
      for (int m = 1; m <= kzmax; m++) {
        sqk = (unitk[0] * k) * (unitk[0] * k) +
              (unitk[1] * l) * (unitk[1] * l) + (unitk[2] * m) * (unitk[2] * m);
        if (sqk <= gsqmx) {
          kxvecs[kcount] = k;
          kyvecs[kcount] = l;
          kzvecs[kcount] = m;
          ug[kcount] = preu * exp(-0.25 * sqk * g_ewald_sq_inv) / sqk;
          kcount++;

          kxvecs[kcount] = k;
          kyvecs[kcount] = -l;
          kzvecs[kcount] = m;
          ug[kcount] = preu * exp(-0.25 * sqk * g_ewald_sq_inv) / sqk;
          kcount++;

          kxvecs[kcount] = k;
          kyvecs[kcount] = l;
          kzvecs[kcount] = -m;
          ug[kcount] = preu * exp(-0.25 * sqk * g_ewald_sq_inv) / sqk;
          kcount++;

          kxvecs[kcount] = k;
          kyvecs[kcount] = -l;
          kzvecs[kcount] = -m;
          ug[kcount] = preu * exp(-0.25 * sqk * g_ewald_sq_inv) / sqk;
          kcount++;
        }
      }
    }
  }
}
