/* ----------------------------------------------------------------------
   LAMMPS - Large-scale Atomic/Molecular Massively Parallel Simulator
   https://lammps.sandia.gov/, Sandia National Laboratories
   Steve Plimpton, sjplimp@sandia.gov

   Copyright (2003) Sandia Corporation.  Under the terms of Contract
   DE-AC04-94AL85000 with Sandia Corporation, the U.S. Government retains
   certain rights in this software.  This software is distributed under
   the GNU General Public License.

   See the README file in the top-level LAMMPS directory.
------------------------------------------------------------------------- */

/* ----------------------------------------------------------------------
   Contributing authors: Roy Pollock (LLNL), Paul Crozier (SNL)
     per-atom energy/virial added by German Samolyuk (ORNL), Stan Moore (BYU)
     group/group energy/force added by Stan Moore (BYU)
     triclinic added by Stan Moore (SNL)
------------------------------------------------------------------------- */

#include "ewald_conp.h"

#include "atom.h"
#include "comm.h"
#include "domain.h"
#include "error.h"
#include "force.h"
#include "math_const.h"
#include "memory.h"
#include "pair.h"

#include <cmath>

using namespace LAMMPS_NS;
using namespace MathConst;

#define SMALL 0.00001

/* ---------------------------------------------------------------------- */

EwaldConp::EwaldConp(LAMMPS *lmp) : KSpace(lmp),
  kxvecs(nullptr), kyvecs(nullptr), kzvecs(nullptr), ug(nullptr), eg(nullptr), vg(nullptr),
  ek(nullptr), sfacrl(nullptr), sfacim(nullptr), sfacrl_all(nullptr), sfacim_all(nullptr),
  cs(nullptr), sn(nullptr), sfacrl_A(nullptr), sfacim_A(nullptr), sfacrl_A_all(nullptr),
  sfacim_A_all(nullptr), sfacrl_B(nullptr), sfacim_B(nullptr), sfacrl_B_all(nullptr),
  sfacim_B_all(nullptr)
{
  group_allocate_flag = 0;
  kmax_created = 0;
  ewaldflag = 1;
  group_group_enable = 1;

  accuracy_relative = 0.0;

  kmax = 0;
  kxvecs = kyvecs = kzvecs = nullptr;
  
  nprd_dim = -1;
  
  ug = nullptr;
  eg = vg = nullptr;
  sfacrl = sfacim = sfacrl_all = sfacim_all = nullptr;

  nmax = 0;
  ek = nullptr;
  cs = sn = nullptr;

  kcount = 0;
}

void EwaldConp::settings(int narg, char **arg)
{
  if (narg != 1) error->all(FLERR,"Illegal kspace_style ewald command");

  accuracy_relative = fabs(utils::numeric(FLERR,arg[0],false,lmp));
}

/* ----------------------------------------------------------------------
   free all memory
------------------------------------------------------------------------- */

EwaldConp::~EwaldConp()
{
  deallocate();
  if (group_allocate_flag) deallocate_groups();
  memory->destroy(ek);
  memory->destroy3d_offset(cs,-kmax_created);
  memory->destroy3d_offset(sn,-kmax_created);
}

/* ---------------------------------------------------------------------- */

void EwaldConp::init()
{
  if (comm->me == 0) utils::logmesg(lmp,"Ewald initialization ...\n");

  // error check
  triclinic_check();
  if (domain->dimension == 2)
    error->all(FLERR,"Cannot use Ewald with 2d simulation");

  if (!atom->q_flag) error->all(FLERR,"Kspace style requires atom attribute q");

  if (slabflag == 0 && domain->nonperiodic > 0)
    error->all(FLERR,"Cannot use non-periodic boundaries with Ewald");
  if (slabflag) {
    if (slab_volfactor == 1.0) {
      int ndim = 0;
      if (domain->xperiodic != 1) { nprd_dim = 0; ndim++; }
      if (domain->yperiodic != 1) { nprd_dim = 1; ndim++; }
      if (domain->zperiodic != 1) { nprd_dim = 2; ndim++; }
      if (ndim>1) error->all(FLERR,"Cannot use < 2 periodic boundaries with EW2D");
    } else if (domain->xperiodic != 1 || domain->yperiodic != 1 ||
        domain->boundary[2][0] != 1 || domain->boundary[2][1] != 1)
      error->all(FLERR,"Incorrect boundaries with slab Ewald");
    if (domain->triclinic)
      error->all(FLERR,"Cannot (yet) use Ewald with triclinic box "
                 "and slab correction nor with EW2D");
  }

  // compute two charge force

  two_charge();

  // extract short-range Coulombic cutoff from pair style

  triclinic = domain->triclinic;
  pair_check();

  int itmp;
  double *p_cutoff = (double *) force->pair->extract("cut_coul",itmp);
  if (p_cutoff == nullptr)
    error->all(FLERR,"KSpace style is incompatible with Pair style");
  double cutoff = *p_cutoff;

  // compute qsum & qsqsum and warn if not charge-neutral

  scale = 1.0;
  qqrd2e = force->qqrd2e;
  qsum_qsq();
  natoms_original = atom->natoms;

  // set accuracy (force units) from accuracy_relative or accuracy_absolute

  if (accuracy_absolute >= 0.0) accuracy = accuracy_absolute;
  else accuracy = accuracy_relative * two_charge_force;

  // setup K-space resolution

  bigint natoms = atom->natoms;

  // use xprd,yprd,zprd even if triclinic so grid size is the same
  // adjust z dimension for 2d slab Ewald
  // 3d Ewald just uses zprd since slab_volfactor = 1.0

  double xprd = domain->xprd;
  double yprd = domain->yprd;
  double zprd = domain->zprd;
  double zprd_slab = zprd*slab_volfactor;

  // make initial g_ewald estimate
  // based on desired accuracy and real space cutoff
  // fluid-occupied volume used to estimate real-space error
  // zprd used rather than zprd_slab

  if (!gewaldflag) {
    if (accuracy <= 0.0)
      error->all(FLERR,"KSpace accuracy must be > 0");
    if (q2 == 0.0)
      error->all(FLERR,"Must use 'kspace_modify gewald' for uncharged system");
    g_ewald = accuracy*sqrt(natoms*cutoff*xprd*yprd*zprd) / (2.0*q2);
    if (g_ewald >= 1.0) g_ewald = (1.35 - 0.15*log(accuracy))/cutoff;
    else g_ewald = sqrt(-log(g_ewald)) / cutoff;
  }

  // setup Ewald coefficients so can print stats

  setup();

  // final RMS accuracy

  double lprx = rms(kxmax_orig,xprd,natoms,q2);
  double lpry = rms(kymax_orig,yprd,natoms,q2);
  double lprz = rms(kzmax_orig,zprd_slab,natoms,q2);
  double lpr = sqrt(lprx*lprx + lpry*lpry + lprz*lprz) / sqrt(3.0);
  double q2_over_sqrt = q2 / sqrt(natoms*cutoff*xprd*yprd*zprd_slab);
  double spr = 2.0 *q2_over_sqrt * exp(-g_ewald*g_ewald*cutoff*cutoff);
  double tpr = estimate_table_accuracy(q2_over_sqrt,spr);
  double estimated_accuracy = sqrt(lpr*lpr + spr*spr + tpr*tpr);

  // stats

  if (comm->me == 0) {
    std::string mesg = fmt::format("  G vector (1/distance) = {:.8g}\n",g_ewald);
    mesg += fmt::format("  estimated absolute RMS force accuracy = {:.8g}\n",
                       estimated_accuracy);
    mesg += fmt::format("  estimated relative force accuracy = {:.8g}\n",
                       estimated_accuracy/two_charge_force);
    mesg += fmt::format("  KSpace vectors: actual max1d max3d = {} {} {}\n",
                        kcount,kmax,kmax3d);
    mesg += fmt::format("                  kxmax kymax kzmax  = {} {} {}\n",
                        kxmax,kymax,kzmax);
    utils::logmesg(lmp,mesg);
  }
}

/* ----------------------------------------------------------------------
   adjust Ewald coeffs, called initially and whenever volume has changed
------------------------------------------------------------------------- */

void EwaldConp::setup()
{
  // volume-dependent factors

  double xprd = domain->xprd;
  double yprd = domain->yprd;
  double zprd = domain->zprd;

  // adjustment of z dimension for 2d slab Ewald
  // 3d Ewald just uses zprd since slab_volfactor = 1.0

  double zprd_slab = zprd*slab_volfactor;
  volume = xprd * yprd * zprd_slab;
  if (nprd_dim == 0) area = yprd * zprd;
  if (nprd_dim == 1) area = xprd * zprd;
  if (nprd_dim == 2) area = xprd * yprd;

  unitk[0] = 2.0*MY_PI/xprd;
  unitk[1] = 2.0*MY_PI/yprd;
  unitk[2] = 2.0*MY_PI/zprd_slab;

  int kmax_old = kmax;

  if (kewaldflag == 0) {

    // determine kmax
    // function of current box size, accuracy, G_ewald (short-range cutoff)

    bigint natoms = atom->natoms;
    double err;
    kxmax = 1;
    kymax = 1;
    kzmax = 1;

    err = rms(kxmax,xprd,natoms,q2);
    while (err > accuracy) {
      kxmax++;
      err = rms(kxmax,xprd,natoms,q2);
    }

    err = rms(kymax,yprd,natoms,q2);
    while (err > accuracy) {
      kymax++;
      err = rms(kymax,yprd,natoms,q2);
    }

    err = rms(kzmax,zprd_slab,natoms,q2);
    while (err > accuracy) {
      kzmax++;
      err = rms(kzmax,zprd_slab,natoms,q2);
    }

    kmax = MAX(kxmax,kymax);
    kmax = MAX(kmax,kzmax);
    kmax3d = 4*kmax*kmax*kmax + 6*kmax*kmax + 3*kmax;

    double gsqxmx = unitk[0]*unitk[0]*kxmax*kxmax;
    double gsqymx = unitk[1]*unitk[1]*kymax*kymax;
    double gsqzmx = unitk[2]*unitk[2]*kzmax*kzmax;
    gsqmx = MAX(gsqxmx,gsqymx);
    gsqmx = MAX(gsqmx,gsqzmx);

    kxmax_orig = kxmax;
    kymax_orig = kymax;
    kzmax_orig = kzmax;

    // scale lattice vectors for triclinic skew

    if (triclinic) {
      double tmp[3];
      tmp[0] = kxmax/xprd;
      tmp[1] = kymax/yprd;
      tmp[2] = kzmax/zprd;
      lamda2xT(&tmp[0],&tmp[0]);
      kxmax = MAX(1,static_cast<int>(tmp[0]));
      kymax = MAX(1,static_cast<int>(tmp[1]));
      kzmax = MAX(1,static_cast<int>(tmp[2]));

      kmax = MAX(kxmax,kymax);
      kmax = MAX(kmax,kzmax);
      kmax3d = 4*kmax*kmax*kmax + 6*kmax*kmax + 3*kmax;
    }

  } else {

    kxmax = kx_ewald;
    kymax = ky_ewald;
    kzmax = kz_ewald;

    kxmax_orig = kxmax;
    kymax_orig = kymax;
    kzmax_orig = kzmax;

    kmax = MAX(kxmax,kymax);
    kmax = MAX(kmax,kzmax);
    kmax3d = 4*kmax*kmax*kmax + 6*kmax*kmax + 3*kmax;

    double gsqxmx = unitk[0]*unitk[0]*kxmax*kxmax;
    double gsqymx = unitk[1]*unitk[1]*kymax*kymax;
    double gsqzmx = unitk[2]*unitk[2]*kzmax*kzmax;
    gsqmx = MAX(gsqxmx,gsqymx);
    gsqmx = MAX(gsqmx,gsqzmx);
  }

  gsqmx *= 1.00001;

  // if size has grown, reallocate k-dependent and nlocal-dependent arrays

  if (kmax > kmax_old) {
    deallocate();
    allocate();
    group_allocate_flag = 0;

    memory->destroy(ek);
    memory->destroy3d_offset(cs,-kmax_created);
    memory->destroy3d_offset(sn,-kmax_created);
    nmax = atom->nmax;
    memory->create(ek,nmax,3,"ewald/conp:ek");
    memory->create3d_offset(cs,-kmax,kmax,3,nmax,"ewald/conp:cs");
    memory->create3d_offset(sn,-kmax,kmax,3,nmax,"ewald/conp:sn");
    kmax_created = kmax;
  }

  // pre-compute Ewald coefficients

  if (triclinic == 0)
    coeffs();
  else
    coeffs_triclinic();
}

/* ----------------------------------------------------------------------
   compute RMS accuracy for a dimension
------------------------------------------------------------------------- */

double EwaldConp::rms(int km, double prd, bigint natoms, double q2)
{
  if (natoms == 0) natoms = 1;   // avoid division by zero
  double value = 2.0*q2*g_ewald/prd *
    sqrt(1.0/(MY_PI*km*natoms)) *
    exp(-MY_PI*MY_PI*km*km/(g_ewald*g_ewald*prd*prd));

  return value;
}

/* ----------------------------------------------------------------------
   compute the Ewald long-range force, energy, virial
------------------------------------------------------------------------- */

void EwaldConp::compute(int eflag, int vflag)
{
  int i,j,k;

  // set energy/virial flags

  ev_init(eflag,vflag);

  // if atom count has changed, update qsum and qsqsum 
  // and fetch new z positions if necessary

  if (atom->natoms != natoms_original) {
    qsum_qsq(); 
    natoms_original = atom->natoms;
  }

  // return if there are no charges

  if (qsqsum == 0.0) return;

  // extend size of per-atom arrays if necessary

  if (atom->nmax > nmax) {
    memory->destroy(ek);
    memory->destroy3d_offset(cs,-kmax_created);
    memory->destroy3d_offset(sn,-kmax_created);
    nmax = atom->nmax;
    memory->create(ek,nmax,3,"ewald/conp:ek");
    memory->create3d_offset(cs,-kmax,kmax,3,nmax,"ewald/conp:cs");
    memory->create3d_offset(sn,-kmax,kmax,3,nmax,"ewald/conp:sn");
    kmax_created = kmax;
  }

  // partial structure factors on each processor
  // total structure factor by summing over procs

  if (triclinic == 0)
    eik_dot_r();
  else
    eik_dot_r_triclinic();

  MPI_Allreduce(sfacrl,sfacrl_all,kcount,MPI_DOUBLE,MPI_SUM,world);
  MPI_Allreduce(sfacim,sfacim_all,kcount,MPI_DOUBLE,MPI_SUM,world);

  // K-space portion of electric field
  // double loop over K-vectors and local atoms
  // perform per-atom calculations if needed

  double **f = atom->f;
  double *q = atom->q;
  int nlocal = atom->nlocal;

  int kx,ky,kz;
  double cypz,sypz,exprl,expim,partial,partial_peratom;

  for (i = 0; i < nlocal; i++) {
    ek[i][0] = 0.0;
    ek[i][1] = 0.0;
    ek[i][2] = 0.0;
  }

  for (k = 0; k < kcount; k++) {
    kx = kxvecs[k];
    ky = kyvecs[k];
    kz = kzvecs[k];

    for (i = 0; i < nlocal; i++) {
      cypz = cs[ky][1][i]*cs[kz][2][i] - sn[ky][1][i]*sn[kz][2][i];
      sypz = sn[ky][1][i]*cs[kz][2][i] + cs[ky][1][i]*sn[kz][2][i];
      exprl = cs[kx][0][i]*cypz - sn[kx][0][i]*sypz;
      expim = sn[kx][0][i]*cypz + cs[kx][0][i]*sypz;
      partial = expim*sfacrl_all[k] - exprl*sfacim_all[k];
      ek[i][0] += partial*eg[k][0];
      ek[i][1] += partial*eg[k][1];
      ek[i][2] += partial*eg[k][2];

      if (evflag_atom) {
        partial_peratom = exprl*sfacrl_all[k] + expim*sfacim_all[k];
        if (eflag_atom) eatom[i] += q[i]*ug[k]*partial_peratom;
        if (vflag_atom)
          for (j = 0; j < 6; j++)
            vatom[i][j] += ug[k]*vg[k][j]*partial_peratom;
      }
    }
  }

  // convert E-field to force

  const double qscale = qqrd2e * scale;

  for (i = 0; i < nlocal; i++) {
    f[i][0] += qscale * q[i]*ek[i][0];
    f[i][1] += qscale * q[i]*ek[i][1];
    if (slabflag != 2) f[i][2] += qscale * q[i]*ek[i][2];
  }

  // sum global energy across Kspace vevs and add in volume-dependent term

  if (eflag_global) {
    for (k = 0; k < kcount; k++)
      energy += ug[k] * (sfacrl_all[k]*sfacrl_all[k] +
                         sfacim_all[k]*sfacim_all[k]);

    energy -= g_ewald*qsqsum/MY_PIS +
      MY_PI2*qsum*qsum / (g_ewald*g_ewald*volume);
    energy *= qscale;
  }

  // global virial

  if (vflag_global) {
    double uk;
    for (k = 0; k < kcount; k++) {
      uk = ug[k] * (sfacrl_all[k]*sfacrl_all[k] + sfacim_all[k]*sfacim_all[k]);
      for (j = 0; j < 6; j++) virial[j] += uk*vg[k][j];
    }
    for (j = 0; j < 6; j++) virial[j] *= qscale;
  }

  // per-atom energy/virial
  // energy includes self-energy correction

  if (evflag_atom) {
    if (eflag_atom) {
      for (i = 0; i < nlocal; i++) {
        eatom[i] -= g_ewald*q[i]*q[i]/MY_PIS + MY_PI2*q[i]*qsum /
          (g_ewald*g_ewald*volume);
        eatom[i] *= qscale;
      }
    }

    if (vflag_atom)
      for (i = 0; i < nlocal; i++)
        for (j = 0; j < 6; j++) vatom[i][j] *= q[i]*qscale;
  }

  // 2d slab correction

  if (slabflag == 1 && slab_volfactor > 1.0) slabcorr();
  else if (slabflag == 1 && slab_volfactor == 1.0) ew2dcorr();
}

/* ---------------------------------------------------------------------- */

void EwaldConp::eik_dot_r()
{
  int i,k,l,m,n,ic;
  double cstr1,sstr1,cstr2,sstr2,cstr3,sstr3,cstr4,sstr4;
  double sqk,clpm,slpm;

  double **x = atom->x;
  double *q = atom->q;
  int nlocal = atom->nlocal;

  n = 0;

  // (k,0,0), (0,l,0), (0,0,m)

  for (ic = 0; ic < 3; ic++) {
    sqk = unitk[ic]*unitk[ic];
    if (sqk <= gsqmx) {
      cstr1 = 0.0;
      sstr1 = 0.0;
      for (i = 0; i < nlocal; i++) {
        cs[0][ic][i] = 1.0;
        sn[0][ic][i] = 0.0;
        cs[1][ic][i] = cos(unitk[ic]*x[i][ic]);
        sn[1][ic][i] = sin(unitk[ic]*x[i][ic]);
        cs[-1][ic][i] = cs[1][ic][i];
        sn[-1][ic][i] = -sn[1][ic][i];
        cstr1 += q[i]*cs[1][ic][i];
        sstr1 += q[i]*sn[1][ic][i];
      }
      sfacrl[n] = cstr1;
      sfacim[n++] = sstr1;
    }
  }

  for (m = 2; m <= kmax; m++) {
    for (ic = 0; ic < 3; ic++) {
      sqk = m*unitk[ic] * m*unitk[ic];
      if (sqk <= gsqmx) {
        cstr1 = 0.0;
        sstr1 = 0.0;
        for (i = 0; i < nlocal; i++) {
          cs[m][ic][i] = cs[m-1][ic][i]*cs[1][ic][i] -
            sn[m-1][ic][i]*sn[1][ic][i];
          sn[m][ic][i] = sn[m-1][ic][i]*cs[1][ic][i] +
            cs[m-1][ic][i]*sn[1][ic][i];
          cs[-m][ic][i] = cs[m][ic][i];
          sn[-m][ic][i] = -sn[m][ic][i];
          cstr1 += q[i]*cs[m][ic][i];
          sstr1 += q[i]*sn[m][ic][i];
        }
        sfacrl[n] = cstr1;
        sfacim[n++] = sstr1;
      }
    }
  }

  // 1 = (k,l,0), 2 = (k,-l,0)

  for (k = 1; k <= kxmax; k++) {
    for (l = 1; l <= kymax; l++) {
      sqk = (k*unitk[0] * k*unitk[0]) + (l*unitk[1] * l*unitk[1]);
      if (sqk <= gsqmx) {
        cstr1 = 0.0;
        sstr1 = 0.0;
        cstr2 = 0.0;
        sstr2 = 0.0;
        for (i = 0; i < nlocal; i++) {
          cstr1 += q[i]*(cs[k][0][i]*cs[l][1][i] - sn[k][0][i]*sn[l][1][i]);
          sstr1 += q[i]*(sn[k][0][i]*cs[l][1][i] + cs[k][0][i]*sn[l][1][i]);
          cstr2 += q[i]*(cs[k][0][i]*cs[l][1][i] + sn[k][0][i]*sn[l][1][i]);
          sstr2 += q[i]*(sn[k][0][i]*cs[l][1][i] - cs[k][0][i]*sn[l][1][i]);
        }
        sfacrl[n] = cstr1;
        sfacim[n++] = sstr1;
        sfacrl[n] = cstr2;
        sfacim[n++] = sstr2;
      }
    }
  }

  // 1 = (0,l,m), 2 = (0,l,-m)

  for (l = 1; l <= kymax; l++) {
    for (m = 1; m <= kzmax; m++) {
      sqk = (l*unitk[1] * l*unitk[1]) + (m*unitk[2] * m*unitk[2]);
      if (sqk <= gsqmx) {
        cstr1 = 0.0;
        sstr1 = 0.0;
        cstr2 = 0.0;
        sstr2 = 0.0;
        for (i = 0; i < nlocal; i++) {
          cstr1 += q[i]*(cs[l][1][i]*cs[m][2][i] - sn[l][1][i]*sn[m][2][i]);
          sstr1 += q[i]*(sn[l][1][i]*cs[m][2][i] + cs[l][1][i]*sn[m][2][i]);
          cstr2 += q[i]*(cs[l][1][i]*cs[m][2][i] + sn[l][1][i]*sn[m][2][i]);
          sstr2 += q[i]*(sn[l][1][i]*cs[m][2][i] - cs[l][1][i]*sn[m][2][i]);
        }
        sfacrl[n] = cstr1;
        sfacim[n++] = sstr1;
        sfacrl[n] = cstr2;
        sfacim[n++] = sstr2;
      }
    }
  }

  // 1 = (k,0,m), 2 = (k,0,-m)

  for (k = 1; k <= kxmax; k++) {
    for (m = 1; m <= kzmax; m++) {
      sqk = (k*unitk[0] * k*unitk[0]) + (m*unitk[2] * m*unitk[2]);
      if (sqk <= gsqmx) {
        cstr1 = 0.0;
        sstr1 = 0.0;
        cstr2 = 0.0;
        sstr2 = 0.0;
        for (i = 0; i < nlocal; i++) {
          cstr1 += q[i]*(cs[k][0][i]*cs[m][2][i] - sn[k][0][i]*sn[m][2][i]);
          sstr1 += q[i]*(sn[k][0][i]*cs[m][2][i] + cs[k][0][i]*sn[m][2][i]);
          cstr2 += q[i]*(cs[k][0][i]*cs[m][2][i] + sn[k][0][i]*sn[m][2][i]);
          sstr2 += q[i]*(sn[k][0][i]*cs[m][2][i] - cs[k][0][i]*sn[m][2][i]);
        }
        sfacrl[n] = cstr1;
        sfacim[n++] = sstr1;
        sfacrl[n] = cstr2;
        sfacim[n++] = sstr2;
      }
    }
  }

  // 1 = (k,l,m), 2 = (k,-l,m), 3 = (k,l,-m), 4 = (k,-l,-m)

  for (k = 1; k <= kxmax; k++) {
    for (l = 1; l <= kymax; l++) {
      for (m = 1; m <= kzmax; m++) {
        sqk = (k*unitk[0] * k*unitk[0]) + (l*unitk[1] * l*unitk[1]) +
          (m*unitk[2] * m*unitk[2]);
        if (sqk <= gsqmx) {
          cstr1 = 0.0;
          sstr1 = 0.0;
          cstr2 = 0.0;
          sstr2 = 0.0;
          cstr3 = 0.0;
          sstr3 = 0.0;
          cstr4 = 0.0;
          sstr4 = 0.0;
          for (i = 0; i < nlocal; i++) {
            clpm = cs[l][1][i]*cs[m][2][i] - sn[l][1][i]*sn[m][2][i];
            slpm = sn[l][1][i]*cs[m][2][i] + cs[l][1][i]*sn[m][2][i];
            cstr1 += q[i]*(cs[k][0][i]*clpm - sn[k][0][i]*slpm);
            sstr1 += q[i]*(sn[k][0][i]*clpm + cs[k][0][i]*slpm);

            clpm = cs[l][1][i]*cs[m][2][i] + sn[l][1][i]*sn[m][2][i];
            slpm = -sn[l][1][i]*cs[m][2][i] + cs[l][1][i]*sn[m][2][i];
            cstr2 += q[i]*(cs[k][0][i]*clpm - sn[k][0][i]*slpm);
            sstr2 += q[i]*(sn[k][0][i]*clpm + cs[k][0][i]*slpm);

            clpm = cs[l][1][i]*cs[m][2][i] + sn[l][1][i]*sn[m][2][i];
            slpm = sn[l][1][i]*cs[m][2][i] - cs[l][1][i]*sn[m][2][i];
            cstr3 += q[i]*(cs[k][0][i]*clpm - sn[k][0][i]*slpm);
            sstr3 += q[i]*(sn[k][0][i]*clpm + cs[k][0][i]*slpm);

            clpm = cs[l][1][i]*cs[m][2][i] - sn[l][1][i]*sn[m][2][i];
            slpm = -sn[l][1][i]*cs[m][2][i] - cs[l][1][i]*sn[m][2][i];
            cstr4 += q[i]*(cs[k][0][i]*clpm - sn[k][0][i]*slpm);
            sstr4 += q[i]*(sn[k][0][i]*clpm + cs[k][0][i]*slpm);
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

void EwaldConp::eik_dot_r_triclinic()
{
  int i,k,l,m,n,ic;
  double cstr1,sstr1;
  double sqk,clpm,slpm;

  double **x = atom->x;
  double *q = atom->q;
  int nlocal = atom->nlocal;

  double unitk_lamda[3];

  double max_kvecs[3];
  max_kvecs[0] = kxmax;
  max_kvecs[1] = kymax;
  max_kvecs[2] = kzmax;

  // (k,0,0), (0,l,0), (0,0,m)

  for (ic = 0; ic < 3; ic++) {
    unitk_lamda[0] = 0.0;
    unitk_lamda[1] = 0.0;
    unitk_lamda[2] = 0.0;
    unitk_lamda[ic] = 2.0*MY_PI;
    x2lamdaT(&unitk_lamda[0],&unitk_lamda[0]);
    sqk = unitk_lamda[ic]*unitk_lamda[ic];
    if (sqk <= gsqmx) {
      for (i = 0; i < nlocal; i++) {
        cs[0][ic][i] = 1.0;
        sn[0][ic][i] = 0.0;
        cs[1][ic][i] = cos(unitk_lamda[0]*x[i][0] + unitk_lamda[1]*x[i][1] + unitk_lamda[2]*x[i][2]);
        sn[1][ic][i] = sin(unitk_lamda[0]*x[i][0] + unitk_lamda[1]*x[i][1] + unitk_lamda[2]*x[i][2]);
        cs[-1][ic][i] = cs[1][ic][i];
        sn[-1][ic][i] = -sn[1][ic][i];
      }
    }
  }

  for (ic = 0; ic < 3; ic++) {
    for (m = 2; m <= max_kvecs[ic]; m++) {
      unitk_lamda[0] = 0.0;
      unitk_lamda[1] = 0.0;
      unitk_lamda[2] = 0.0;
      unitk_lamda[ic] = 2.0*MY_PI*m;
      x2lamdaT(&unitk_lamda[0],&unitk_lamda[0]);
      sqk = unitk_lamda[ic]*unitk_lamda[ic];
      for (i = 0; i < nlocal; i++) {
        cs[m][ic][i] = cs[m-1][ic][i]*cs[1][ic][i] -
          sn[m-1][ic][i]*sn[1][ic][i];
        sn[m][ic][i] = sn[m-1][ic][i]*cs[1][ic][i] +
          cs[m-1][ic][i]*sn[1][ic][i];
        cs[-m][ic][i] = cs[m][ic][i];
        sn[-m][ic][i] = -sn[m][ic][i];
      }
    }
  }

  for (n = 0; n < kcount; n++) {
    k = kxvecs[n];
    l = kyvecs[n];
    m = kzvecs[n];
    cstr1 = 0.0;
    sstr1 = 0.0;
    for (i = 0; i < nlocal; i++) {
      clpm = cs[l][1][i]*cs[m][2][i] - sn[l][1][i]*sn[m][2][i];
      slpm = sn[l][1][i]*cs[m][2][i] + cs[l][1][i]*sn[m][2][i];
      cstr1 += q[i]*(cs[k][0][i]*clpm - sn[k][0][i]*slpm);
      sstr1 += q[i]*(sn[k][0][i]*clpm + cs[k][0][i]*slpm);
    }
    sfacrl[n] = cstr1;
    sfacim[n] = sstr1;
  }
}

/* ----------------------------------------------------------------------
   pre-compute coefficients for each Ewald K-vector
------------------------------------------------------------------------- */

void EwaldConp::coeffs()
{
  int k,l,m;
  double sqk,vterm;

  double g_ewald_sq_inv = 1.0 / (g_ewald*g_ewald);
  double preu = 4.0*MY_PI/volume;

  kcount = 0;

  // (k,0,0), (0,l,0), (0,0,m), however skip (0,0) in case os EW2D
  for (m = 1; m <= kmax; m++) {
    sqk = (m*unitk[0]) * (m*unitk[0]);
    if (sqk <= gsqmx && nprd_dim != 0) {
      kxvecs[kcount] = m;
      kyvecs[kcount] = 0;
      kzvecs[kcount] = 0;
      ug[kcount] = preu*exp(-0.25*sqk*g_ewald_sq_inv)/sqk;
      eg[kcount][0] = 2.0*unitk[0]*m*ug[kcount];
      eg[kcount][1] = 0.0;
      eg[kcount][2] = 0.0;
      vterm = -2.0*(1.0/sqk + 0.25*g_ewald_sq_inv);
      vg[kcount][0] = 1.0 + vterm*(unitk[0]*m)*(unitk[0]*m);
      vg[kcount][1] = 1.0;
      vg[kcount][2] = 1.0;
      vg[kcount][3] = 0.0;
      vg[kcount][4] = 0.0;
      vg[kcount][5] = 0.0;
      kcount++;
    }
    sqk = (m*unitk[1]) * (m*unitk[1]);
    if (sqk <= gsqmx && nprd_dim != 1) {
      kxvecs[kcount] = 0;
      kyvecs[kcount] = m;
      kzvecs[kcount] = 0;
      ug[kcount] = preu*exp(-0.25*sqk*g_ewald_sq_inv)/sqk;
      eg[kcount][0] = 0.0;
      eg[kcount][1] = 2.0*unitk[1]*m*ug[kcount];
      eg[kcount][2] = 0.0;
      vterm = -2.0*(1.0/sqk + 0.25*g_ewald_sq_inv);
      vg[kcount][0] = 1.0;
      vg[kcount][1] = 1.0 + vterm*(unitk[1]*m)*(unitk[1]*m);
      vg[kcount][2] = 1.0;
      vg[kcount][3] = 0.0;
      vg[kcount][4] = 0.0;
      vg[kcount][5] = 0.0;
      kcount++;
    }
    sqk = (m*unitk[2]) * (m*unitk[2]);
    if (sqk <= gsqmx && nprd_dim != 2) {
      kxvecs[kcount] = 0;
      kyvecs[kcount] = 0;
      kzvecs[kcount] = m;
      ug[kcount] = preu*exp(-0.25*sqk*g_ewald_sq_inv)/sqk;
      eg[kcount][0] = 0.0;
      eg[kcount][1] = 0.0;
      eg[kcount][2] = 2.0*unitk[2]*m*ug[kcount];
      vterm = -2.0*(1.0/sqk + 0.25*g_ewald_sq_inv);
      vg[kcount][0] = 1.0;
      vg[kcount][1] = 1.0;
      vg[kcount][2] = 1.0 + vterm*(unitk[2]*m)*(unitk[2]*m);
      vg[kcount][3] = 0.0;
      vg[kcount][4] = 0.0;
      vg[kcount][5] = 0.0;
      kcount++;
    }
  }

  // 1 = (k,l,0), 2 = (k,-l,0)

  for (k = 1; k <= kxmax; k++) {
    for (l = 1; l <= kymax; l++) {
      sqk = (unitk[0]*k) * (unitk[0]*k) + (unitk[1]*l) * (unitk[1]*l);
      if (sqk <= gsqmx) {
        kxvecs[kcount] = k;
        kyvecs[kcount] = l;
        kzvecs[kcount] = 0;
        ug[kcount] = preu*exp(-0.25*sqk*g_ewald_sq_inv)/sqk;
        eg[kcount][0] = 2.0*unitk[0]*k*ug[kcount];
        eg[kcount][1] = 2.0*unitk[1]*l*ug[kcount];
        eg[kcount][2] = 0.0;
        vterm = -2.0*(1.0/sqk + 0.25*g_ewald_sq_inv);
        vg[kcount][0] = 1.0 + vterm*(unitk[0]*k)*(unitk[0]*k);
        vg[kcount][1] = 1.0 + vterm*(unitk[1]*l)*(unitk[1]*l);
        vg[kcount][2] = 1.0;
        vg[kcount][3] = vterm*unitk[0]*k*unitk[1]*l;
        vg[kcount][4] = 0.0;
        vg[kcount][5] = 0.0;
        kcount++;

        kxvecs[kcount] = k;
        kyvecs[kcount] = -l;
        kzvecs[kcount] = 0;
        ug[kcount] = preu*exp(-0.25*sqk*g_ewald_sq_inv)/sqk;
        eg[kcount][0] = 2.0*unitk[0]*k*ug[kcount];
        eg[kcount][1] = -2.0*unitk[1]*l*ug[kcount];
        eg[kcount][2] = 0.0;
        vg[kcount][0] = 1.0 + vterm*(unitk[0]*k)*(unitk[0]*k);
        vg[kcount][1] = 1.0 + vterm*(unitk[1]*l)*(unitk[1]*l);
        vg[kcount][2] = 1.0;
        vg[kcount][3] = -vterm*unitk[0]*k*unitk[1]*l;
        vg[kcount][4] = 0.0;
        vg[kcount][5] = 0.0;
        kcount++;;
      }
    }
  }

  // 1 = (0,l,m), 2 = (0,l,-m)

  for (l = 1; l <= kymax; l++) {
    for (m = 1; m <= kzmax; m++) {
      sqk = (unitk[1]*l) * (unitk[1]*l) + (unitk[2]*m) * (unitk[2]*m);
      if (sqk <= gsqmx) {
        kxvecs[kcount] = 0;
        kyvecs[kcount] = l;
        kzvecs[kcount] = m;
        ug[kcount] = preu*exp(-0.25*sqk*g_ewald_sq_inv)/sqk;
        eg[kcount][0] =  0.0;
        eg[kcount][1] =  2.0*unitk[1]*l*ug[kcount];
        eg[kcount][2] =  2.0*unitk[2]*m*ug[kcount];
        vterm = -2.0*(1.0/sqk + 0.25*g_ewald_sq_inv);
        vg[kcount][0] = 1.0;
        vg[kcount][1] = 1.0 + vterm*(unitk[1]*l)*(unitk[1]*l);
        vg[kcount][2] = 1.0 + vterm*(unitk[2]*m)*(unitk[2]*m);
        vg[kcount][3] = 0.0;
        vg[kcount][4] = 0.0;
        vg[kcount][5] = vterm*unitk[1]*l*unitk[2]*m;
        kcount++;

        kxvecs[kcount] = 0;
        kyvecs[kcount] = l;
        kzvecs[kcount] = -m;
        ug[kcount] = preu*exp(-0.25*sqk*g_ewald_sq_inv)/sqk;
        eg[kcount][0] =  0.0;
        eg[kcount][1] =  2.0*unitk[1]*l*ug[kcount];
        eg[kcount][2] = -2.0*unitk[2]*m*ug[kcount];
        vg[kcount][0] = 1.0;
        vg[kcount][1] = 1.0 + vterm*(unitk[1]*l)*(unitk[1]*l);
        vg[kcount][2] = 1.0 + vterm*(unitk[2]*m)*(unitk[2]*m);
        vg[kcount][3] = 0.0;
        vg[kcount][4] = 0.0;
        vg[kcount][5] = -vterm*unitk[1]*l*unitk[2]*m;
        kcount++;
      }
    }
  }

  // 1 = (k,0,m), 2 = (k,0,-m)

  for (k = 1; k <= kxmax; k++) {
    for (m = 1; m <= kzmax; m++) {
      sqk = (unitk[0]*k) * (unitk[0]*k) + (unitk[2]*m) * (unitk[2]*m);
      if (sqk <= gsqmx) {
        kxvecs[kcount] = k;
        kyvecs[kcount] = 0;
        kzvecs[kcount] = m;
        ug[kcount] = preu*exp(-0.25*sqk*g_ewald_sq_inv)/sqk;
        eg[kcount][0] =  2.0*unitk[0]*k*ug[kcount];
        eg[kcount][1] =  0.0;
        eg[kcount][2] =  2.0*unitk[2]*m*ug[kcount];
        vterm = -2.0*(1.0/sqk + 0.25*g_ewald_sq_inv);
        vg[kcount][0] = 1.0 + vterm*(unitk[0]*k)*(unitk[0]*k);
        vg[kcount][1] = 1.0;
        vg[kcount][2] = 1.0 + vterm*(unitk[2]*m)*(unitk[2]*m);
        vg[kcount][3] = 0.0;
        vg[kcount][4] = vterm*unitk[0]*k*unitk[2]*m;
        vg[kcount][5] = 0.0;
        kcount++;

        kxvecs[kcount] = k;
        kyvecs[kcount] = 0;
        kzvecs[kcount] = -m;
        ug[kcount] = preu*exp(-0.25*sqk*g_ewald_sq_inv)/sqk;
        eg[kcount][0] =  2.0*unitk[0]*k*ug[kcount];
        eg[kcount][1] =  0.0;
        eg[kcount][2] = -2.0*unitk[2]*m*ug[kcount];
        vg[kcount][0] = 1.0 + vterm*(unitk[0]*k)*(unitk[0]*k);
        vg[kcount][1] = 1.0;
        vg[kcount][2] = 1.0 + vterm*(unitk[2]*m)*(unitk[2]*m);
        vg[kcount][3] = 0.0;
        vg[kcount][4] = -vterm*unitk[0]*k*unitk[2]*m;
        vg[kcount][5] = 0.0;
        kcount++;
      }
    }
  }

  // 1 = (k,l,m), 2 = (k,-l,m), 3 = (k,l,-m), 4 = (k,-l,-m)

  for (k = 1; k <= kxmax; k++) {
    for (l = 1; l <= kymax; l++) {
      for (m = 1; m <= kzmax; m++) {
        sqk = (unitk[0]*k) * (unitk[0]*k) + (unitk[1]*l) * (unitk[1]*l) +
          (unitk[2]*m) * (unitk[2]*m);
        if (sqk <= gsqmx) {
          kxvecs[kcount] = k;
          kyvecs[kcount] = l;
          kzvecs[kcount] = m;
          ug[kcount] = preu*exp(-0.25*sqk*g_ewald_sq_inv)/sqk;
          eg[kcount][0] = 2.0*unitk[0]*k*ug[kcount];
          eg[kcount][1] = 2.0*unitk[1]*l*ug[kcount];
          eg[kcount][2] = 2.0*unitk[2]*m*ug[kcount];
          vterm = -2.0*(1.0/sqk + 0.25*g_ewald_sq_inv);
          vg[kcount][0] = 1.0 + vterm*(unitk[0]*k)*(unitk[0]*k);
          vg[kcount][1] = 1.0 + vterm*(unitk[1]*l)*(unitk[1]*l);
          vg[kcount][2] = 1.0 + vterm*(unitk[2]*m)*(unitk[2]*m);
          vg[kcount][3] = vterm*unitk[0]*k*unitk[1]*l;
          vg[kcount][4] = vterm*unitk[0]*k*unitk[2]*m;
          vg[kcount][5] = vterm*unitk[1]*l*unitk[2]*m;
          kcount++;

          kxvecs[kcount] = k;
          kyvecs[kcount] = -l;
          kzvecs[kcount] = m;
          ug[kcount] = preu*exp(-0.25*sqk*g_ewald_sq_inv)/sqk;
          eg[kcount][0] = 2.0*unitk[0]*k*ug[kcount];
          eg[kcount][1] = -2.0*unitk[1]*l*ug[kcount];
          eg[kcount][2] = 2.0*unitk[2]*m*ug[kcount];
          vg[kcount][0] = 1.0 + vterm*(unitk[0]*k)*(unitk[0]*k);
          vg[kcount][1] = 1.0 + vterm*(unitk[1]*l)*(unitk[1]*l);
          vg[kcount][2] = 1.0 + vterm*(unitk[2]*m)*(unitk[2]*m);
          vg[kcount][3] = -vterm*unitk[0]*k*unitk[1]*l;
          vg[kcount][4] = vterm*unitk[0]*k*unitk[2]*m;
          vg[kcount][5] = -vterm*unitk[1]*l*unitk[2]*m;
          kcount++;

          kxvecs[kcount] = k;
          kyvecs[kcount] = l;
          kzvecs[kcount] = -m;
          ug[kcount] = preu*exp(-0.25*sqk*g_ewald_sq_inv)/sqk;
          eg[kcount][0] = 2.0*unitk[0]*k*ug[kcount];
          eg[kcount][1] = 2.0*unitk[1]*l*ug[kcount];
          eg[kcount][2] = -2.0*unitk[2]*m*ug[kcount];
          vg[kcount][0] = 1.0 + vterm*(unitk[0]*k)*(unitk[0]*k);
          vg[kcount][1] = 1.0 + vterm*(unitk[1]*l)*(unitk[1]*l);
          vg[kcount][2] = 1.0 + vterm*(unitk[2]*m)*(unitk[2]*m);
          vg[kcount][3] = vterm*unitk[0]*k*unitk[1]*l;
          vg[kcount][4] = -vterm*unitk[0]*k*unitk[2]*m;
          vg[kcount][5] = -vterm*unitk[1]*l*unitk[2]*m;
          kcount++;

          kxvecs[kcount] = k;
          kyvecs[kcount] = -l;
          kzvecs[kcount] = -m;
          ug[kcount] = preu*exp(-0.25*sqk*g_ewald_sq_inv)/sqk;
          eg[kcount][0] = 2.0*unitk[0]*k*ug[kcount];
          eg[kcount][1] = -2.0*unitk[1]*l*ug[kcount];
          eg[kcount][2] = -2.0*unitk[2]*m*ug[kcount];
          vg[kcount][0] = 1.0 + vterm*(unitk[0]*k)*(unitk[0]*k);
          vg[kcount][1] = 1.0 + vterm*(unitk[1]*l)*(unitk[1]*l);
          vg[kcount][2] = 1.0 + vterm*(unitk[2]*m)*(unitk[2]*m);
          vg[kcount][3] = -vterm*unitk[0]*k*unitk[1]*l;
          vg[kcount][4] = -vterm*unitk[0]*k*unitk[2]*m;
          vg[kcount][5] = vterm*unitk[1]*l*unitk[2]*m;
          kcount++;
        }
      }
    }
  }
}

/* ----------------------------------------------------------------------
   pre-compute coefficients for each Ewald K-vector for a triclinic
   system
------------------------------------------------------------------------- */

void EwaldConp::coeffs_triclinic()
{
  int k,l,m;
  double sqk,vterm;

  double g_ewald_sq_inv = 1.0 / (g_ewald*g_ewald);
  double preu = 4.0*MY_PI/volume;

  double unitk_lamda[3];

  kcount = 0;

  // 1 = (k,l,m), 2 = (k,-l,m), 3 = (k,l,-m), 4 = (k,-l,-m)

  for (k = 1; k <= kxmax; k++) {
    for (l = -kymax; l <= kymax; l++) {
      for (m = -kzmax; m <= kzmax; m++) {
        unitk_lamda[0] = 2.0*MY_PI*k;
        unitk_lamda[1] = 2.0*MY_PI*l;
        unitk_lamda[2] = 2.0*MY_PI*m;
        x2lamdaT(&unitk_lamda[0],&unitk_lamda[0]);
        sqk = unitk_lamda[0]*unitk_lamda[0] + unitk_lamda[1]*unitk_lamda[1] +
          unitk_lamda[2]*unitk_lamda[2];
        if (sqk <= gsqmx) {
          kxvecs[kcount] = k;
          kyvecs[kcount] = l;
          kzvecs[kcount] = m;
          ug[kcount] = preu*exp(-0.25*sqk*g_ewald_sq_inv)/sqk;
          eg[kcount][0] = 2.0*unitk_lamda[0]*ug[kcount];
          eg[kcount][1] = 2.0*unitk_lamda[1]*ug[kcount];
          eg[kcount][2] = 2.0*unitk_lamda[2]*ug[kcount];
          vterm = -2.0*(1.0/sqk + 0.25*g_ewald_sq_inv);
          vg[kcount][0] = 1.0 + vterm*unitk_lamda[0]*unitk_lamda[0];
          vg[kcount][1] = 1.0 + vterm*unitk_lamda[1]*unitk_lamda[1];
          vg[kcount][2] = 1.0 + vterm*unitk_lamda[2]*unitk_lamda[2];
          vg[kcount][3] = vterm*unitk_lamda[0]*unitk_lamda[1];
          vg[kcount][4] = vterm*unitk_lamda[0]*unitk_lamda[2];
          vg[kcount][5] = vterm*unitk_lamda[1]*unitk_lamda[2];
          kcount++;
        }
      }
    }
  }

  // 1 = (0,l,m), 2 = (0,l,-m)

  for (l = 1; l <= kymax; l++) {
    for (m = -kzmax; m <= kzmax; m++) {
      unitk_lamda[0] = 0.0;
      unitk_lamda[1] = 2.0*MY_PI*l;
      unitk_lamda[2] = 2.0*MY_PI*m;
      x2lamdaT(&unitk_lamda[0],&unitk_lamda[0]);
      sqk = unitk_lamda[1]*unitk_lamda[1] + unitk_lamda[2]*unitk_lamda[2];
      if (sqk <= gsqmx) {
        kxvecs[kcount] = 0;
        kyvecs[kcount] = l;
        kzvecs[kcount] = m;
        ug[kcount] = preu*exp(-0.25*sqk*g_ewald_sq_inv)/sqk;
        eg[kcount][0] =  0.0;
        eg[kcount][1] =  2.0*unitk_lamda[1]*ug[kcount];
        eg[kcount][2] =  2.0*unitk_lamda[2]*ug[kcount];
        vterm = -2.0*(1.0/sqk + 0.25*g_ewald_sq_inv);
        vg[kcount][0] = 1.0;
        vg[kcount][1] = 1.0 + vterm*unitk_lamda[1]*unitk_lamda[1];
        vg[kcount][2] = 1.0 + vterm*unitk_lamda[2]*unitk_lamda[2];
        vg[kcount][3] = 0.0;
        vg[kcount][4] = 0.0;
        vg[kcount][5] = vterm*unitk_lamda[1]*unitk_lamda[2];
        kcount++;
      }
    }
  }

  // (0,0,m)

  for (m = 1; m <= kmax; m++) {
    unitk_lamda[0] = 0.0;
    unitk_lamda[1] = 0.0;
    unitk_lamda[2] = 2.0*MY_PI*m;
    x2lamdaT(&unitk_lamda[0],&unitk_lamda[0]);
    sqk = unitk_lamda[2]*unitk_lamda[2];
    if (sqk <= gsqmx) {
      kxvecs[kcount] = 0;
      kyvecs[kcount] = 0;
      kzvecs[kcount] = m;
      ug[kcount] = preu*exp(-0.25*sqk*g_ewald_sq_inv)/sqk;
      eg[kcount][0] = 0.0;
      eg[kcount][1] = 0.0;
      eg[kcount][2] = 2.0*unitk_lamda[2]*ug[kcount];
      vterm = -2.0*(1.0/sqk + 0.25*g_ewald_sq_inv);
      vg[kcount][0] = 1.0;
      vg[kcount][1] = 1.0;
      vg[kcount][2] = 1.0 + vterm*unitk_lamda[2]*unitk_lamda[2];
      vg[kcount][3] = 0.0;
      vg[kcount][4] = 0.0;
      vg[kcount][5] = 0.0;
      kcount++;
    }
  }
}

/* ----------------------------------------------------------------------
   allocate memory that depends on # of K-vectors
------------------------------------------------------------------------- */

void EwaldConp::allocate()
{
  kxvecs = new int[kmax3d];
  kyvecs = new int[kmax3d];
  kzvecs = new int[kmax3d];

  ug = new double[kmax3d];
  memory->create(eg,kmax3d,3,"ewald/conp:eg");
  memory->create(vg,kmax3d,6,"ewald/conp:vg");

  sfacrl = new double[kmax3d];
  sfacim = new double[kmax3d];
  sfacrl_all = new double[kmax3d];
  sfacim_all = new double[kmax3d];
}

/* ----------------------------------------------------------------------
   deallocate memory that depends on # of K-vectors
------------------------------------------------------------------------- */

void EwaldConp::deallocate()
{
  delete [] kxvecs;
  delete [] kyvecs;
  delete [] kzvecs;

  delete [] ug;
  memory->destroy(eg);
  memory->destroy(vg);

  delete [] sfacrl;
  delete [] sfacim;
  delete [] sfacrl_all;
  delete [] sfacim_all;
}

/* ----------------------------------------------------------------------
   Slab-geometry correction term to dampen inter-slab interactions between
   periodically repeating slabs.  Yields good approximation to 2D Ewald if
   adequate empty space is left between repeating slabs (J. Chem. Phys.
   111, 3155).  Slabs defined here to be parallel to the xy plane. Also
   extended to non-neutral systems (J. Chem. Phys. 131, 094107).
------------------------------------------------------------------------- */

void EwaldConp::slabcorr()
{
  // compute local contribution to global dipole moment

  double *q = atom->q;
  double **x = atom->x;
  double zprd = domain->zprd;
  int nlocal = atom->nlocal;

  double dipole = 0.0;
  for (int i = 0; i < nlocal; i++) dipole += q[i]*x[i][2];

  // sum local contributions to get global dipole moment

  double dipole_all;
  MPI_Allreduce(&dipole,&dipole_all,1,MPI_DOUBLE,MPI_SUM,world);

  // need to make non-neutral systems and/or
  //  per-atom energy translationally invariant

  double dipole_r2 = 0.0;
  if (eflag_atom || fabs(qsum) > SMALL) {
    for (int i = 0; i < nlocal; i++)
      dipole_r2 += q[i]*x[i][2]*x[i][2];

    // sum local contributions

    double tmp;
    MPI_Allreduce(&dipole_r2,&tmp,1,MPI_DOUBLE,MPI_SUM,world);
    dipole_r2 = tmp;
  }

  // compute corrections

  const double e_slabcorr = MY_2PI*(dipole_all*dipole_all -
    qsum*dipole_r2 - qsum*qsum*zprd*zprd/12.0)/volume;
  const double qscale = qqrd2e * scale;

  if (eflag_global) energy += qscale * e_slabcorr;

  // per-atom energy

  if (eflag_atom) {
    double efact = qscale * MY_2PI/volume;
    for (int i = 0; i < nlocal; i++)
      eatom[i] += efact * q[i]*(x[i][2]*dipole_all - 0.5*(dipole_r2 +
        qsum*x[i][2]*x[i][2]) - qsum*zprd*zprd/12.0);
  }

  // add on force corrections

  double ffact = qscale * (-4.0*MY_PI/volume);
  double **f = atom->f;

  for (int i = 0; i < nlocal; i++) f[i][2] += ffact * q[i]*(dipole_all - qsum*x[i][2]);
}

/* ----------------------------------------------------------------------
   Slab-geometry correction term (k=0) of EW2D. See Hu, JCTC 10:12 (2014)
   pp. 5254-5264 or metalwalls ewald and parallelization documentation.
------------------------------------------------------------------------- */

void EwaldConp::ew2dcorr()
{
  double *q = atom->q;
  double **x = atom->x;
  double **f = atom->f;
  tagint *tag = atom->tag;
  int nlocal = atom->nlocal;  
  bigint natoms = atom->natoms;
  int nprocs = comm->nprocs;

  int *recvcounts,*displs;
  double *nprd, *nprd_all, *q_all;  
  
  // gather q and non-periodic positions on all procs
  
  memory->create(recvcounts,nprocs,"ewald/conp:recvcounts");
  memory->create(displs,nprocs,"ewald/conp:displs");
  
  MPI_Allgather(&nlocal,1,MPI_INT,recvcounts,1,MPI_INT,world);
 
  displs[0] = 0;
  for (int i = 1; i < nprocs; i++)
    displs[i] = displs[i-1] + recvcounts[i-1];
  
  memory->create(nprd,nlocal,"ewald/conp:nprd");
  
  for (int i = 0; i < nlocal; i++)
    nprd[i] = x[i][nprd_dim];
  
  memory->create(nprd_all,natoms,"ewald/conp:nprd_all");
  memory->create(q_all,natoms,"ewald/conp:q_all");

  MPI_Allgatherv(q,nlocal,MPI_DOUBLE,q_all,recvcounts,displs,MPI_DOUBLE,world);
  MPI_Allgatherv(nprd,nlocal,MPI_DOUBLE,nprd_all,recvcounts,displs,MPI_DOUBLE,world);
  
  memory->destroy(nprd);
  memory->destroy(recvcounts);
  memory->destroy(displs);
  
  const double g_ewald_inv = 1.0 / g_ewald;
  const double g_ewald_sq = g_ewald*g_ewald;
  const double qscale = qqrd2e * scale;
  const double efact = 2.0 * MY_PIS/area;
  const double ffact = qscale * MY_2PI/area;
   
  double pot_ij, aij, dij, e_keq0_all;
  
  // loop over ALL atom interactions
  
  double e_keq0 = 0.0;
  for (int i = 0; i < nlocal; i++) {
    pot_ij = 0.0;
    for (bigint j = 0; j < natoms; j++) {
      
      dij = nprd_all[j] - x[i][nprd_dim];
      
      // resembles (aij) matrix component in constant potential
      aij = efact * ( exp( -dij*dij * g_ewald_sq ) * g_ewald_inv + 
                         MY_PIS * dij * erf( dij * g_ewald ) );
      // coulomb potential; see eq. (4) in metalwalls parallelization doc
      pot_ij += q_all[j] * aij;
      
      // add on force corrections
      
      f[i][nprd_dim] -= ffact * q[i]*q_all[j] * erf( g_ewald*dij );
    }
    
    // per-atom energy; see eq. (20) in metalwalls ewald doc

    if (eflag_atom) eatom[i] -= qscale * q[i] * pot_ij * 0.5;
    
    e_keq0 += q[i] * pot_ij;
  }
  
  memory->destroy(nprd_all);
  memory->destroy(q_all);
  
  // sum local contributions; see eq. (20) in metalwalls ewald doc
  
  MPI_Allreduce(&e_keq0,&e_keq0_all,1,MPI_DOUBLE,MPI_SUM,world);

  if (eflag_global) energy -= qscale * e_keq0_all * 0.5;
}

/* ----------------------------------------------------------------------
   memory usage of local arrays
------------------------------------------------------------------------- */

double EwaldConp::memory_usage()
{
  double bytes = 3 * kmax3d * sizeof(int);
  bytes += (1 + 3 + 6) * kmax3d * sizeof(double);
  bytes += 4 * kmax3d * sizeof(double);
  bytes += nmax*3 * sizeof(double);
  bytes += 2 * (2*kmax+1)*3*nmax * sizeof(double);
  return bytes;
}

/* ----------------------------------------------------------------------
   group-group interactions
 ------------------------------------------------------------------------- */

/* ----------------------------------------------------------------------
   compute the Ewald total long-range force and energy for groups A and B
 ------------------------------------------------------------------------- */

void EwaldConp::compute_group_group(int groupbit_A, int groupbit_B, int AA_flag)
{ 
  if (slabflag && triclinic)
    error->all(FLERR,"Cannot (yet) use K-space slab "
               "correction with compute group/group for triclinic systems");

  int i,k;

  if (!group_allocate_flag) {
    allocate_groups();
    group_allocate_flag = 1;
  }

  e2group = 0.0; //energy
  f2group[0] = 0.0; //force in x-direction
  f2group[1] = 0.0; //force in y-direction
  f2group[2] = 0.0; //force in z-direction

  // partial and total structure factors for groups A and B

  for (k = 0; k < kcount; k++) {

    // group A

    sfacrl_A[k] = 0.0;
    sfacim_A[k] = 0.0;
    sfacrl_A_all[k] = 0.0;
    sfacim_A_all[k] = 0.0;

    // group B

    sfacrl_B[k] = 0.0;
    sfacim_B[k] = 0.0;
    sfacrl_B_all[k] = 0.0;
    sfacim_B_all[k] = 0.0;
  }

  double *q = atom->q;
  int nlocal = atom->nlocal;
  int *mask = atom->mask;

  int kx,ky,kz;
  double cypz,sypz,exprl,expim;

  // partial structure factors for groups A and B on each processor

  for (k = 0; k < kcount; k++) {
    kx = kxvecs[k];
    ky = kyvecs[k];
    kz = kzvecs[k];

    for (i = 0; i < nlocal; i++) {

      if (!((mask[i] & groupbit_A) && (mask[i] & groupbit_B)))
        if (AA_flag) continue;

      if ((mask[i] & groupbit_A) || (mask[i] & groupbit_B)) {

        cypz = cs[ky][1][i]*cs[kz][2][i] - sn[ky][1][i]*sn[kz][2][i];
        sypz = sn[ky][1][i]*cs[kz][2][i] + cs[ky][1][i]*sn[kz][2][i];
        exprl = cs[kx][0][i]*cypz - sn[kx][0][i]*sypz;
        expim = sn[kx][0][i]*cypz + cs[kx][0][i]*sypz;

        // group A

        if (mask[i] & groupbit_A) {
          sfacrl_A[k] += q[i]*exprl;
          sfacim_A[k] += q[i]*expim;
        }

        // group B

        if (mask[i] & groupbit_B) {
          sfacrl_B[k] += q[i]*exprl;
          sfacim_B[k] += q[i]*expim;
        }
      }
    }
  }

  // total structure factor by summing over procs

  MPI_Allreduce(sfacrl_A,sfacrl_A_all,kcount,MPI_DOUBLE,MPI_SUM,world);
  MPI_Allreduce(sfacim_A,sfacim_A_all,kcount,MPI_DOUBLE,MPI_SUM,world);

  MPI_Allreduce(sfacrl_B,sfacrl_B_all,kcount,MPI_DOUBLE,MPI_SUM,world);
  MPI_Allreduce(sfacim_B,sfacim_B_all,kcount,MPI_DOUBLE,MPI_SUM,world);

  const double qscale = qqrd2e * scale;
  double partial_group;

  // total group A <--> group B energy
  // self and boundary correction terms are in compute_group_group.cpp

  for (k = 0; k < kcount; k++) {
    partial_group = sfacrl_A_all[k]*sfacrl_B_all[k] +
      sfacim_A_all[k]*sfacim_B_all[k];
    e2group += ug[k]*partial_group;
  }

  e2group *= qscale;

  // total group A <--> group B force

  for (k = 0; k < kcount; k++) {
    partial_group = sfacim_A_all[k]*sfacrl_B_all[k] -
      sfacrl_A_all[k]*sfacim_B_all[k];
    f2group[0] += eg[k][0]*partial_group;
    f2group[1] += eg[k][1]*partial_group;
    if (slabflag != 2) f2group[2] += eg[k][2]*partial_group;
  }

  f2group[0] *= qscale;
  f2group[1] *= qscale;
  f2group[2] *= qscale;

  // 2d slab correction

  if (slabflag == 1 && slab_volfactor > 1.0) 
    slabcorr_groups(groupbit_A, groupbit_B, AA_flag);
  else if (slabflag == 1 && slab_volfactor == 1.0)
    ew2dcorr_groups(groupbit_A, groupbit_B, AA_flag);
}

/* ----------------------------------------------------------------------
   Slab-geometry correction term to dampen inter-slab interactions between
   periodically repeating slabs.  Yields good approximation to 2D Ewald if
   adequate empty space is left between repeating slabs (J. Chem. Phys.
   111, 3155).  Slabs defined here to be parallel to the xy plane. Also
   extended to non-neutral systems (J. Chem. Phys. 131, 094107).
------------------------------------------------------------------------- */

void EwaldConp::slabcorr_groups(int groupbit_A, int groupbit_B, int AA_flag)
{
  // compute local contribution to global dipole moment

  double *q = atom->q;
  double **x = atom->x;
  double zprd = domain->zprd;
  int *mask = atom->mask;
  int nlocal = atom->nlocal;

  double qsum_A = 0.0;
  double qsum_B = 0.0;
  double dipole_A = 0.0;
  double dipole_B = 0.0;
  double dipole_r2_A = 0.0;
  double dipole_r2_B = 0.0;

  for (int i = 0; i < nlocal; i++) {
    if (!((mask[i] & groupbit_A) && (mask[i] & groupbit_B)))
      if (AA_flag) continue;

    if (mask[i] & groupbit_A) {
      qsum_A += q[i];
      dipole_A += q[i]*x[i][2];
      dipole_r2_A += q[i]*x[i][2]*x[i][2];
    }

    if (mask[i] & groupbit_B) {
      qsum_B += q[i];
      dipole_B += q[i]*x[i][2];
      dipole_r2_B += q[i]*x[i][2]*x[i][2];
    }
  }

  // sum local contributions to get total charge and global dipole moment
  //  for each group

  double tmp;
  MPI_Allreduce(&qsum_A,&tmp,1,MPI_DOUBLE,MPI_SUM,world);
  qsum_A = tmp;

  MPI_Allreduce(&qsum_B,&tmp,1,MPI_DOUBLE,MPI_SUM,world);
  qsum_B = tmp;

  MPI_Allreduce(&dipole_A,&tmp,1,MPI_DOUBLE,MPI_SUM,world);
  dipole_A = tmp;

  MPI_Allreduce(&dipole_B,&tmp,1,MPI_DOUBLE,MPI_SUM,world);
  dipole_B = tmp;

  MPI_Allreduce(&dipole_r2_A,&tmp,1,MPI_DOUBLE,MPI_SUM,world);
  dipole_r2_A = tmp;

  MPI_Allreduce(&dipole_r2_B,&tmp,1,MPI_DOUBLE,MPI_SUM,world);
  dipole_r2_B = tmp;

  // compute corrections

  const double qscale = qqrd2e * scale;
  const double efact = qscale * MY_2PI/volume;

  e2group += efact * (dipole_A*dipole_B - 0.5*(qsum_A*dipole_r2_B +
    qsum_B*dipole_r2_A) - qsum_A*qsum_B*zprd*zprd/12.0);

  // add on force corrections

  const double ffact = qscale * (-4.0*MY_PI/volume);
  f2group[2] += ffact * (qsum_A*dipole_B - qsum_B*dipole_A);
}

/* ----------------------------------------------------------------------
   computes infinite boundary term with EW2D (i.e. k = 0).
------------------------------------------------------------------------- */

void EwaldConp::ew2dcorr_groups(int groupbit_A, int groupbit_B, int AA_flag)
{
  if (slabflag == 1 && slab_volfactor == 1.0) 
    error->all(FLERR,"Cannot (yet) use EW2D correction with compute group/group");
}

/* ----------------------------------------------------------------------
   compute individual interactions between all pairs of atoms in group A 
   and B. see lammps_gather_atoms_concat() on how all sn and cs have been 
   obtained.
 ------------------------------------------------------------------------- */

void EwaldConp::compute_matrix(int groupbit_A, int groupbit_B, bigint *imat, double **matrix)
{ 
  if (slabflag && triclinic)
    error->all(FLERR,"Cannot (yet) use K-space slab "
               "correction with compute group/group for triclinic systems");

  int nlocal = atom->nlocal;
  tagint *tag = atom->tag;
  int *mask = atom->mask;
  
  double **csx,**csy,**csz,**snx,**sny,**snz;
  double *csx_all,*csy_all,*csz_all;
  double *snx_all,*sny_all,*snz_all;

  int i,j,k;

  // how many local group atoms owns each proc?
  
  bigint ngrouplocal = 0;
  bigint ngroup;
  for (i = 0; i < nlocal; i++)
    if ((mask[i] & groupbit_A) || (mask[i] & groupbit_B))
      ngrouplocal++;
      
  // how many atoms do we have in total in groups    
      
  MPI_Allreduce(&ngrouplocal,&ngroup,1,MPI_LMP_BIGINT,MPI_SUM,world);

  // gather only subset of sn and cs on each proc
  
  memory->create(csx,kxmax+1,ngrouplocal,"ewald/conp:csx");
  memory->create(snx,kxmax+1,ngrouplocal,"ewald/conp:snx");
  memory->create(csy,kymax+1,ngrouplocal,"ewald/conp:csy");
  memory->create(sny,kymax+1,ngrouplocal,"ewald/conp:sny");
  memory->create(snz,kzmax+1,ngrouplocal,"ewald/conp:snz");
  memory->create(csz,kzmax+1,ngrouplocal,"ewald/conp:csz");
  
  memory->create(csx_all,(kxmax+1)*ngroup,"ewald/conp:csx_all");
  memory->create(snx_all,(kxmax+1)*ngroup,"ewald/conp:snx_all");
  memory->create(csy_all,(kymax+1)*ngroup,"ewald/conp:csy_all");
  memory->create(sny_all,(kymax+1)*ngroup,"ewald/conp:sny_all");
  memory->create(csz_all,(kzmax+1)*ngroup,"ewald/conp:csz_all");
  memory->create(snz_all,(kzmax+1)*ngroup,"ewald/conp:snz_all");
  
  // copy sn and cn to new local arrays 
  
  j = 0;
  for (i = 0; i < nlocal; i++) {
    if ((mask[i] & groupbit_A) || (mask[i] & groupbit_B)) {
      for (k = 0; k <= kxmax; k++) {  
        csx[k][j] = cs[k][0][i];
        snx[k][j] = sn[k][0][i];
      }
      for (k = 0; k <= kymax; k++) { 
        csy[k][j] = cs[k][1][i];
        sny[k][j] = sn[k][1][i];
      }
      for (k = 0; k <= kzmax; k++) { 
        csz[k][j] = cs[k][2][i];
        snz[k][j] = sn[k][2][i];
      }
      j++;
    }
  }

  int nprocs = comm->nprocs;
  int n[3];
  int *recvcounts,**displs;

  memory->create(recvcounts,nprocs,"ewald/conp:recvcounts");
  memory->create(displs,3,nprocs,"ewald/conp:displs");
  
  for (int idim = 0; idim < 3; idim++) {
    
    n[idim] = (kxmax+1)*ngrouplocal;
    
    MPI_Allgather(&n[idim],1,MPI_INT,recvcounts,1,MPI_INT,world);
    
    displs[idim][0] = 0;
    for (i = 1; i < nprocs; i++)
      displs[idim][i] = displs[idim][i-1] + recvcounts[i-1];
      
    MPI_Allgatherv(&csx[0][0],n[idim],MPI_DOUBLE,csx_all,recvcounts,displs[idim],MPI_DOUBLE,world);
    MPI_Allgatherv(&snx[0][0],n[idim],MPI_DOUBLE,snx_all,recvcounts,displs[idim],MPI_DOUBLE,world);
    MPI_Allgatherv(&csy[0][0],n[idim],MPI_DOUBLE,csy_all,recvcounts,displs[idim],MPI_DOUBLE,world);
    MPI_Allgatherv(&sny[0][0],n[idim],MPI_DOUBLE,sny_all,recvcounts,displs[idim],MPI_DOUBLE,world);
    MPI_Allgatherv(&csz[0][0],n[idim],MPI_DOUBLE,csz_all,recvcounts,displs[idim],MPI_DOUBLE,world);
    MPI_Allgatherv(&snz[0][0],n[idim],MPI_DOUBLE,snz_all,recvcounts,displs[idim],MPI_DOUBLE,world);
  }
  
  // consistency check between local global arrays

//  FILE *pFile;
//  char fn[12];
//  sprintf(fn,"k.%d",comm->me);
//  pFile = fopen(fn,"w");
//  for (int m = 0; m < nprocs; m++)
//    if (comm->me == m)
//      for (k = 0; k < kxmax+1; k++)
//        for (i = 0; i < ngrouplocal; i++) 
//          fprintf(pFile,"%f %f\n",csx[k][i],csx_all[i+k*ngrouplocal+displs[0][m]]);   
//  fclose(pFile);

  // create matrix indexing for j atoms; resulting list is sorted by rank of process
  // and according to csx_all, snx_all, etc.
  
  bigint *jmat, *jmat_local;
    
  int *ndispls;
  memory->create(ndispls,nprocs,"ewald/conp:ndispls");
  MPI_Allgather(&ngrouplocal,1,MPI_INT,recvcounts,1,MPI_INT,world);
  ndispls[0] = 0;
    for (i = 1; i < nprocs; i++)
      ndispls[i] = ndispls[i-1] + recvcounts[i-1];
  
  memory->create(jmat_local,ngrouplocal,"ewald/conp:jmat_local");    
  bigint count = 0;  
  for (i = 0; i < nlocal; i++)
    if (imat[i] != -1) jmat_local[count++] = imat[i];
  
  memory->create(jmat,ngroup,"ewald/conp:jmat");    
  MPI_Allgatherv(jmat_local,ngrouplocal,MPI_LMP_BIGINT,jmat,recvcounts,ndispls,MPI_LMP_BIGINT,world);
  memory->destroy(jmat_local);
  memory->destroy(ndispls);
  
  int kx,ky,kz,kxj,kyj,kzj,kyabs,kzabs,sign_ky,sign_kz;
  double aij,cos_kxky,sin_kxky,cos_kxkykz_i,sin_kxkykz_i,cos_kxkykz_j,sin_kxkykz_j;
  
  // aij for each atom pair in groups; first loop over i,j then over k to reduce memory access

  for (int i = 0; i < nlocal; i++) {

    if ((mask[i] & groupbit_A) || (mask[i] & groupbit_B)) {
         
      // matrix is symmetric, use imat to skip interactions
      
      for (int j = imat[i]; j < ngroup; j++) {
        
        aij = 0.0;
        
        for (int k = 0; k < kcount; k++) {
        
          kx = kxvecs[k];
          ky = kyvecs[k];
          kz = kzvecs[k];
          
          kyabs = abs(ky);
          sign_ky = (ky > 0) - (ky < 0);
          
          kzabs = abs(kz);
          sign_kz = (kz > 0) - (kz < 0);
          
          // TODO blocking approach from metalwalls? pre-compute and store local cos_kxkykz_i and sin_kxkykz_i?
          
          cos_kxky = cs[kx][0][i] * cs[kyabs][1][i] - sn[kx][0][i] * sn[kyabs][1][i] * sign_ky;
          sin_kxky = sn[kx][0][i] * cs[kyabs][1][i] + cs[kx][0][i] * sn[kyabs][1][i] * sign_ky;

          cos_kxkykz_i = cos_kxky * cs[kzabs][2][i] - sin_kxky * sn[kzabs][2][i] * sign_kz;
          sin_kxkykz_i = sin_kxky * cs[kzabs][2][i] + cos_kxky * sn[kzabs][2][i] * sign_kz;
          
          // global indexing  csx_all[kx][j]     <>        csx_all[kx+j*(kxmax+1)]
          // local  indexing  cs_idim[k_idim][i] <> csx_all[i+k*ngrouplocal+displs[idim][comm->me]]]
          
          kxj = kx+j*(kxmax+1);
          kyj = kyabs+j*(kymax+1);
          kzj = kzabs+j*(kzmax+1);
        
          cos_kxky = csx_all[kxj] * csy_all[kyj] - snx_all[kxj] * sny_all[kyj] * sign_ky;
          sin_kxky = snx_all[kxj] * csy_all[kyj] + csx_all[kxj] * sny_all[kyj] * sign_ky;

          cos_kxkykz_j = cos_kxky * csz_all[kzj] - sin_kxky * snz_all[kzj] * sign_kz;
          sin_kxkykz_j = sin_kxky * csz_all[kzj] + cos_kxky * snz_all[kzj] * sign_kz;
          
          aij += 2.0*ug[k] * (cos_kxkykz_i*cos_kxkykz_j + sin_kxkykz_i*sin_kxkykz_j);
        }
        
        matrix[imat[i]][jmat[j]] += aij;
        if (imat[i] != j) matrix[jmat[j]][imat[i]] += aij;
      }
    }
    if ((i+1) % 100 == 0) printf("%d (%d/%d)\n",comm->me,i+1,nlocal);
  }
  
  memory->destroy(jmat);
  memory->destroy(displs);
  memory->destroy(recvcounts);
  memory->destroy(csx_all);
  memory->destroy(snx_all);
  memory->destroy(csy_all);
  memory->destroy(sny_all);
  memory->destroy(csz_all);
  memory->destroy(snz_all);
  memory->destroy(csx);
  memory->destroy(snx);
  memory->destroy(csy);
  memory->destroy(sny);
  memory->destroy(csz);
  memory->destroy(snz);
}

///* ----------------------------------------------------------------------
//   compute the Ewald long-range matrix for given list of tags
// ------------------------------------------------------------------------- */

//void EwaldConp::compute_matrix(bigint nmat, tagint *mat2tag, double **matrix)
//{ 
//  if (slabflag && triclinic)
//    error->all(FLERR,"Cannot (yet) use K-space slab "
//               "correction with compute group/group for triclinic systems");
// 
//  int nlocal = atom->nlocal;
//  tagint *tag = atom->tag;
//  double **csx_local, **csy_local, **csz_local;
//  double **snx_local, **sny_local, **snz_local;
//  double **csx_glob, **csy_glob, **csz_glob;
//  double **snx_glob, **sny_glob, **snz_glob;
//  double **matrix_local;
//  int i,k,l,m,n,mabs,nabs,sign_m,sign_n;
//  bigint ii,j;
//  
//  // gather only cs and sn of atoms in taglist
//  
//  memory->create(csx_local,kxmax+1,nmat,"ewald/conp:csx_local");
//  memory->create(csy_local,kymax+1,nmat,"ewald/conp:csy_local");
//  memory->create(csz_local,kzmax+1,nmat,"ewald/conp:csz_local");
//  memory->create(snx_local,kxmax+1,nmat,"ewald/conp:snx_local");
//  memory->create(sny_local,kymax+1,nmat,"ewald/conp:sny_local");
//  memory->create(snz_local,kzmax+1,nmat,"ewald/conp:snz_local");
//  
//  // set all local arrays to zero
//  
//  size_t nbytes = sizeof(double) * nmat;
//  
//  if (nbytes) {
//    for (k = 0; k < kcount; k++) {         
//      
//      // only positive m,n are required (see metalwalls)
//          
//      l = kxvecs[k];
//      m = abs(kyvecs[k]);
//      n = abs(kzvecs[k]);
//    
//      memset(&csx_local[l][0],0,nbytes);
//      memset(&snx_local[l][0],0,nbytes);
//      memset(&csy_local[m][0],0,nbytes);
//      memset(&sny_local[m][0],0,nbytes);
//      memset(&csz_local[n][0],0,nbytes);
//      memset(&snz_local[n][0],0,nbytes);
//    }
//  }
//  
//  // copy local sn and cn to taglist atoms
//  
//  for (i = 0; i < nlocal; i++) {
//    for (j = 0; j < nmat; j++) {
//      if (tag[i] == mat2tag[j]) {
//        for (k = 0; k < kcount; k++) {  
//          l = kxvecs[k];
//          m = abs(kyvecs[k]);
//          n = abs(kzvecs[k]);
//          
//          csx_local[l][j] = cs[l][0][i];
//          snx_local[l][j] = sn[l][0][i];
//          csy_local[m][j] = cs[m][1][i];
//          sny_local[m][j] = sn[m][1][i];
//          csz_local[n][j] = cs[n][2][i];
//          snz_local[n][j] = sn[n][2][i];
//        }
//      }
//    }
//  }
//  
//  memory->create(csx_glob,kxmax+1,nmat,"ewald/conp:csx_glob");
//  memory->create(csy_glob,kymax+1,nmat,"ewald/conp:csy_glob");
//  memory->create(csz_glob,kzmax+1,nmat,"ewald/conp:csz_glob");
//  memory->create(snx_glob,kxmax+1,nmat,"ewald/conp:snx_glob");
//  memory->create(sny_glob,kymax+1,nmat,"ewald/conp:sny_glob");
//  memory->create(snz_glob,kzmax+1,nmat,"ewald/conp:snz_glob");
//  
//  MPI_Allreduce(&csx_local[0][0], &csx_glob[0][0], (kxmax+1)*nmat, MPI_DOUBLE, MPI_SUM, world);
//  MPI_Allreduce(&csy_local[0][0], &csy_glob[0][0], (kymax+1)*nmat, MPI_DOUBLE, MPI_SUM, world);
//  MPI_Allreduce(&csz_local[0][0], &csz_glob[0][0], (kzmax+1)*nmat, MPI_DOUBLE, MPI_SUM, world);
//  MPI_Allreduce(&snx_local[0][0], &snx_glob[0][0], (kxmax+1)*nmat, MPI_DOUBLE, MPI_SUM, world);
//  MPI_Allreduce(&sny_local[0][0], &sny_glob[0][0], (kymax+1)*nmat, MPI_DOUBLE, MPI_SUM, world);
//  MPI_Allreduce(&snz_local[0][0], &snz_glob[0][0], (kzmax+1)*nmat, MPI_DOUBLE, MPI_SUM, world);
//  
//  memory->destroy(csx_local);
//  memory->destroy(csy_local);
//  memory->destroy(csz_local);
//  memory->destroy(snx_local);
//  memory->destroy(sny_local);
//  memory->destroy(snz_local);
//  
//  memory->create(matrix_local,nmat,nmat,"ewald/conp:matrix_local");
//   
//  double cos_kxky,sin_kxky,aij;
//  double cos_kxkykz_i,sin_kxkykz_i;
//  double cos_kxkykz_j,sin_kxkykz_j;
//  for (k = 0; k < kcount; k++) {
//    for (i = 0; i < nlocal; i++) { 
//      for (ii = 0; ii < nmat; ii++) {
//        if (tag[i] == mat2tag[ii]) { 
//        
//          // take care of k-vector sign
//        
//          l = kxvecs[k];
//          m = kyvecs[k];
//          n = kzvecs[k];
//          
//          mabs = abs(m);
//          sign_m = (m > 0) - (m < 0);
//          
//          nabs = abs(n);
//          sign_n = (n > 0) - (n < 0);


//          cos_kxky = cs[l][0][i] * cs[mabs][1][i] - sn[l][0][i] * sn[mabs][1][i] * sign_m;
//          sin_kxky = sn[l][0][i] * cs[mabs][1][i] + cs[l][0][i] * sn[mabs][1][i] * sign_m;

//          cos_kxkykz_i = cos_kxky * cs[nabs][2][i] - sin_kxky * sn[nabs][2][i] * sign_n;
//          sin_kxkykz_i = sin_kxky * cs[nabs][2][i] + cos_kxky * sn[nabs][2][i] * sign_n;
//        
//          // matrix is symmetric, compute only lower triangular
//          
//          for (j = ii; j < nmat; j++) {
//          
//            cos_kxky = csx_glob[0][j] * csy_glob[0][j] - snx_glob[0][j] * sny_glob[0][j] * sign_m;
//            sin_kxky = snx_glob[0][j] * csy_glob[0][j] + csx_glob[0][j] * sny_glob[0][j] * sign_m;

//            cos_kxkykz_j = cos_kxky * csz_glob[0][j] - sin_kxky * snz_glob[0][j] * sign_n;
//            sin_kxkykz_j = sin_kxky * csz_glob[0][j] + cos_kxky * snz_glob[0][j] * sign_n;
//          
//            printf("%d on %d: %f %f %f %f\n", tag[i],comm->me,
//                                            cos_kxkykz_i,
//                                            sin_kxkykz_i,
//                                            cos_kxkykz_j,
//                                            sin_kxkykz_j,);
//          
//            aij = 2.0*ug[k] * ( cos_kxkykz_i*cos_kxkykz_j + sin_kxkykz_i*sin_kxkykz_j );
//            
//            matrix[ii][j] += aij;
//            if (ii != j) 
//              matrix[j][ii] += aij;
//          }
//        }
//      }
//    }
//    if ((k+1) % 1000 == 0) printf("%d (%d/%d)\n",comm->me,k+1,kcount);
//  }  
//  
//  MPI_Allreduce(&matrix_local[0][0], &matrix[0][0], nmat*nmat, MPI_DOUBLE, MPI_SUM, world);
//  
//  memory->destroy(matrix_local);
//  memory->destroy(csx_glob);
//  memory->destroy(csy_glob);
//  memory->destroy(csz_glob);
//  memory->destroy(snx_glob);
//  memory->destroy(sny_glob);
//  memory->destroy(snz_glob);
//}

/* ----------------------------------------------------------------------
   allocate group-group memory that depends on # of K-vectors
------------------------------------------------------------------------- */

void EwaldConp::allocate_groups()
{
  // group A

  sfacrl_A = new double[kmax3d];
  sfacim_A = new double[kmax3d];
  sfacrl_A_all = new double[kmax3d];
  sfacim_A_all = new double[kmax3d];

  // group B

  sfacrl_B = new double[kmax3d];
  sfacim_B = new double[kmax3d];
  sfacrl_B_all = new double[kmax3d];
  sfacim_B_all = new double[kmax3d];
}

/* ----------------------------------------------------------------------
   deallocate group-group memory that depends on # of K-vectors
------------------------------------------------------------------------- */

void EwaldConp::deallocate_groups()
{
  // group A

  delete [] sfacrl_A;
  delete [] sfacim_A;
  delete [] sfacrl_A_all;
  delete [] sfacim_A_all;

  // group B

  delete [] sfacrl_B;
  delete [] sfacim_B;
  delete [] sfacrl_B_all;
  delete [] sfacim_B_all;
}
