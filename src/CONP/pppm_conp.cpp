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
     per-atom energy/virial & group/group energy/force added by Stan Moore (BYU)
     analytic diff (2 FFT) option added by Rolf Isele-Holder (Aachen University)
     triclinic added by Stan Moore (SNL)
------------------------------------------------------------------------- */

#include "pppm_conp.h"

#include <cmath>
#include <cstring>
#include <iostream>

#include "angle.h"
#include "assert.h"
#include "atom.h"
#include "bond.h"
#include "comm.h"
#include "domain.h"
#include "error.h"
#include "fft3d_wrap.h"
#include "force.h"
#include "gridcomm.h"
#include "math_const.h"
#include "math_special.h"
#include "memory.h"
#include "neighbor.h"
#include "pair.h"
#include "remap_wrap.h"
#include "update.h"

using namespace LAMMPS_NS;
using namespace MathConst;
using namespace MathSpecial;
using namespace std;

#define MAXORDER 7
#define OFFSET 16384
#define LARGE 10000.0
#define SMALL 0.00001
#define EPS_HOC 1.0e-7

enum { REVERSE_RHO };
enum { FORWARD_IK, FORWARD_AD, FORWARD_IK_PERATOM, FORWARD_AD_PERATOM };

#ifdef FFT_SINGLE
#define ZEROF 0.0f
#define ONEF 1.0f
#else
#define ZEROF 0.0
#define ONEF 1.0
#endif

/* ---------------------------------------------------------------------- */

PPPMConp::PPPMConp(LAMMPS *lmp)
    : PPPM(lmp),
      electrolyte_density_brick(nullptr),
      electrolyte_density_fft(nullptr) {
  electrolyte_density_brick = nullptr;
  electrolyte_density_fft = nullptr;
}
/* ----------------------------------------------------------------------
   free all memory
------------------------------------------------------------------------- */

PPPMConp::~PPPMConp() {
  if (copymode) return;

  deallocate();
  if (peratom_allocate_flag) deallocate_peratom();
  if (group_allocate_flag) deallocate_groups();
  memory->destroy(part2grid);
  memory->destroy(acons);
}

/* ----------------------------------------------------------------------
   called once before run
------------------------------------------------------------------------- */

void PPPMConp::init() {
  if (me == 0) utils::logmesg(lmp, "PPPM/conp initialization ...\n");

  // error check

  triclinic_check();

  if (triclinic != domain->triclinic)
    error->all(FLERR,
               "Must redefine kspace_style after changing to triclinic box");

  if (domain->triclinic && differentiation_flag == 1)
    error->all(FLERR,
               "Cannot (yet) use PPPM/conp with triclinic box "
               "and kspace_modify diff ad");
  if (domain->triclinic && slabflag)
    error->all(FLERR,
               "Cannot (yet) use PPPM/conp with triclinic box and "
               "slab correction");
  if (domain->triclinic && wireflag)
    error->all(FLERR,
               "Cannot (yet) use PPPM/conp with triclinic box and "
               "wire correction");
  if (domain->dimension == 2)
    error->all(FLERR, "Cannot use PPPM/conp with 2d simulation");

  if (!atom->q_flag)
    error->all(FLERR, "Kspace style requires atom attribute q");

  if (slabflag == 0 && wireflag == 0 && domain->nonperiodic > 0)
    error->all(FLERR, "Cannot use non-periodic boundaries with PPPM/conp");
  if (slabflag) {
    if (domain->xperiodic != 1 || domain->yperiodic != 1 ||
        domain->boundary[2][0] != 1 || domain->boundary[2][1] != 1)
      error->all(FLERR, "Incorrect boundaries with slab PPPM/conp");
  } else if (wireflag) {
    if (domain->zperiodic != 1 || domain->boundary[0][0] != 1 ||
        domain->boundary[0][1] != 1 || domain->boundary[1][0] != 1 ||
        domain->boundary[1][1] != 1)
      error->all(FLERR, "Incorrect boundaries with wire PPPM/conp");
  }

  if (order < 2 || order > MAXORDER)
    error->all(FLERR,
               fmt::format("PPPM/conp order cannot be < 2 or > {}", MAXORDER));

  // compute two charge force

  two_charge();

  // extract short-range Coulombic cutoff from pair style

  triclinic = domain->triclinic;
  pair_check();

  int itmp = 0;
  double *p_cutoff = (double *)force->pair->extract("cut_coul", itmp);
  if (p_cutoff == nullptr)
    error->all(FLERR, "KSpace style is incompatible with Pair style");
  cutoff = *p_cutoff;

  // if kspace is TIP4P, extract TIP4P params from pair style
  // bond/angle are not yet init(), so insure equilibrium request is valid

  qdist = 0.0;

  if (tip4pflag) {
    if (me == 0)
      utils::logmesg(lmp, "  extracting TIP4P info from pair style\n");

    double *p_qdist = (double *)force->pair->extract("qdist", itmp);
    int *p_typeO = (int *)force->pair->extract("typeO", itmp);
    int *p_typeH = (int *)force->pair->extract("typeH", itmp);
    int *p_typeA = (int *)force->pair->extract("typeA", itmp);
    int *p_typeB = (int *)force->pair->extract("typeB", itmp);
    if (!p_qdist || !p_typeO || !p_typeH || !p_typeA || !p_typeB)
      error->all(FLERR, "Pair style is incompatible with TIP4P KSpace style");
    qdist = *p_qdist;
    typeO = *p_typeO;
    typeH = *p_typeH;
    int typeA = *p_typeA;
    int typeB = *p_typeB;

    if (force->angle == nullptr || force->bond == nullptr ||
        force->angle->setflag == nullptr || force->bond->setflag == nullptr)
      error->all(FLERR, "Bond and angle potentials must be defined for TIP4P");
    if (typeA < 1 || typeA > atom->nangletypes ||
        force->angle->setflag[typeA] == 0)
      error->all(FLERR, "Bad TIP4P angle type for PPPM/TIP4P");
    if (typeB < 1 || typeB > atom->nbondtypes ||
        force->bond->setflag[typeB] == 0)
      error->all(FLERR, "Bad TIP4P bond type for PPPM/TIP4P");
    double theta = force->angle->equilibrium_angle(typeA);
    double blen = force->bond->equilibrium_distance(typeB);
    alpha = qdist / (cos(0.5 * theta) * blen);
  }

  // compute qsum & qsqsum and warn if not charge-neutral

  scale = 1.0;
  qqrd2e = force->qqrd2e;
  qsum_qsq();
  natoms_original = atom->natoms;

  // set accuracy (force units) from accuracy_relative or accuracy_absolute

  if (accuracy_absolute >= 0.0)
    accuracy = accuracy_absolute;
  else
    accuracy = accuracy_relative * two_charge_force;

  // free all arrays previously allocated

  deallocate();
  delete fft1;
  delete fft2;
  delete remap;
  delete gc;
  if (peratom_allocate_flag) deallocate_peratom();
  if (group_allocate_flag) deallocate_groups();

  // setup FFT grid resolution and g_ewald
  // normally one iteration thru while loop is all that is required
  // if grid stencil does not extend beyond neighbor proc
  //   or overlap is allowed, then done
  // else reduce order and try again

  GridComm *gctmp = nullptr;
  int iteration = 0;

  while (order >= minorder) {
    if (iteration && me == 0)
      error->warning(FLERR,
                     "Reducing PPPM/conp order b/c stencil extends "
                     "beyond nearest neighbor processor");

    if (stagger_flag && !differentiation_flag) compute_gf_denom();
    set_grid_global();
    set_grid_local();
    if (overlap_allowed) break;

    gctmp = new GridComm(lmp, world, nx_pppm, ny_pppm, nz_pppm, nxlo_in,
                         nxhi_in, nylo_in, nyhi_in, nzlo_in, nzhi_in, nxlo_out,
                         nxhi_out, nylo_out, nyhi_out, nzlo_out, nzhi_out);

    int tmp1, tmp2;
    gctmp->setup(tmp1, tmp2);
    if (gctmp->ghost_adjacent()) break;
    delete gctmp;

    order--;
    iteration++;
  }

  if (order < minorder)
    error->all(FLERR, "PPPM/conp order < minimum allowed order");
  if (!overlap_allowed && !gctmp->ghost_adjacent())
    error->all(FLERR,
               "PPPM/conp grid stencil extends "
               "beyond nearest neighbor processor");
  if (gctmp) delete gctmp;

  // adjust g_ewald

  if (!gewaldflag) adjust_gewald();

  // calculate the final accuracy

  double estimated_accuracy = final_accuracy();

  // print stats

  int ngrid_max, nfft_both_max;
  MPI_Allreduce(&ngrid, &ngrid_max, 1, MPI_INT, MPI_MAX, world);
  MPI_Allreduce(&nfft_both, &nfft_both_max, 1, MPI_INT, MPI_MAX, world);

  if (me == 0) {
    std::string mesg =
        fmt::format("  G vector (1/distance) = {:.8g}\n", g_ewald);
    mesg += fmt::format("  grid = {} {} {}\n", nx_pppm, ny_pppm, nz_pppm);
    mesg += fmt::format("  stencil order = {}\n", order);
    mesg += fmt::format("  estimated absolute RMS force accuracy = {:.8g}\n",
                        estimated_accuracy);
    mesg += fmt::format("  estimated relative force accuracy = {:.8g}\n",
                        estimated_accuracy / two_charge_force);
    mesg += "  using " LMP_FFT_PREC " precision " LMP_FFT_LIB "\n";
    mesg += fmt::format("  3d grid and FFT values/proc = {} {}\n", ngrid_max,
                        nfft_both_max);
    utils::logmesg(lmp, mesg);
  }

  // allocate K-space dependent memory
  // don't invoke allocate peratom() or group(), will be allocated when needed

  allocate();

  // pre-compute Green's function denomiator expansion
  // pre-compute 1d charge distribution coefficients

  compute_gf_denom();
  if (differentiation_flag == 1) compute_sf_precoeff();
  compute_rho_coeff();
  compute_step = -1;
}

/* ----------------------------------------------------------------------
   adjust PPPM coeffs, called initially and whenever volume has changed
------------------------------------------------------------------------- */

void PPPMConp::setup() {
  if (triclinic) {
    setup_triclinic();
    return;
  }

  // perform some checks to avoid illegal boundaries with read_data

  if (slabflag == 0 && wireflag == 0 && domain->nonperiodic > 0)
    error->all(FLERR, "Cannot use non-periodic boundaries with PPPM/conp");
  if (slabflag) {
    if (domain->xperiodic != 1 || domain->yperiodic != 1 ||
        domain->boundary[2][0] != 1 || domain->boundary[2][1] != 1)
      error->all(FLERR, "Incorrect boundaries with slab PPPM/conp");
  } else if (wireflag) {
    if (domain->zperiodic != 1 || domain->boundary[0][0] != 1 ||
        domain->boundary[0][1] != 1 || domain->boundary[1][0] != 1 ||
        domain->boundary[1][1] != 1)
      error->all(FLERR, "Incorrect boundaries with wire PPPM/conp");
  }

  int i, j, k, n;
  double *prd;

  // volume-dependent factors
  // adjust z dimension for 2d slab PPPM
  // z dimension for 3d PPPM is zprd since slab_volfactor = 1.0

  if (triclinic == 0)
    prd = domain->prd;
  else
    prd = domain->prd_lamda;

  double xprd = prd[0];
  double yprd = prd[1];
  double zprd = prd[2];
  double xprd_wire = xprd * wire_volfactor;
  double yprd_wire = yprd * wire_volfactor;
  double zprd_slab = zprd * slab_volfactor;
  volume = xprd_wire * yprd_wire * zprd_slab;

  delxinv = nx_pppm / xprd_wire;
  delyinv = ny_pppm / yprd_wire;
  delzinv = nz_pppm / zprd_slab;

  delvolinv = delxinv * delyinv * delzinv;

  double unitkx = (MY_2PI / xprd_wire);
  double unitky = (MY_2PI / yprd_wire);
  double unitkz = (MY_2PI / zprd_slab);

  // fkx,fky,fkz for my FFT grid pts

  for (i = nxlo_fft; i <= nxhi_fft; i++) {
    int per =
        i - nx_pppm * (2 * i / nx_pppm);  // TODO int division intentional?
    fkx[i] = unitkx * per;
  }

  for (i = nylo_fft; i <= nyhi_fft; i++) {
    int per = i - ny_pppm * (2 * i / ny_pppm);
    fky[i] = unitky * per;
  }

  for (i = nzlo_fft; i <= nzhi_fft; i++) {
    int per = i - nz_pppm * (2 * i / nz_pppm);
    fkz[i] = unitkz * per;
  }

  // virial coefficients

  double sqk, vterm;

  n = 0;
  for (k = nzlo_fft; k <= nzhi_fft; k++) {
    for (j = nylo_fft; j <= nyhi_fft; j++) {
      for (i = nxlo_fft; i <= nxhi_fft; i++) {
        sqk = fkx[i] * fkx[i] + fky[j] * fky[j] + fkz[k] * fkz[k];
        if (sqk == 0.0) {
          vg[n][0] = 0.0;
          vg[n][1] = 0.0;
          vg[n][2] = 0.0;
          vg[n][3] = 0.0;
          vg[n][4] = 0.0;
          vg[n][5] = 0.0;
        } else {
          vterm = -2.0 * (1.0 / sqk + 0.25 / (g_ewald * g_ewald));
          vg[n][0] = 1.0 + vterm * fkx[i] * fkx[i];
          vg[n][1] = 1.0 + vterm * fky[j] * fky[j];
          vg[n][2] = 1.0 + vterm * fkz[k] * fkz[k];
          vg[n][3] = vterm * fkx[i] * fky[j];
          vg[n][4] = vterm * fkx[i] * fkz[k];
          vg[n][5] = vterm * fky[j] * fkz[k];
        }
        n++;
      }
    }
  }

  if (differentiation_flag == 1)
    compute_gf_ad();
  else
    compute_gf_ik();
}

/* ----------------------------------------------------------------------
   adjust PPPM coeffs, called initially and whenever volume has changed
   for a triclinic system
------------------------------------------------------------------------- */

void PPPMConp::setup_triclinic() {
  int i, j, k, n;
  double *prd;

  // volume-dependent factors
  // adjust z dimension for 2d slab PPPM
  // z dimension for 3d PPPM is zprd since slab_volfactor = 1.0

  prd = domain->prd;

  double xprd = prd[0];
  double yprd = prd[1];
  double zprd = prd[2];
  double xprd_wire = xprd * wire_volfactor;
  double yprd_wire = yprd * wire_volfactor;
  double zprd_slab = zprd * slab_volfactor;
  volume = xprd_wire * yprd_wire * zprd_slab;

  // use lamda (0-1) coordinates

  delxinv = nx_pppm;
  delyinv = ny_pppm;
  delzinv = nz_pppm;
  delvolinv = delxinv * delyinv * delzinv / volume;

  // fkx,fky,fkz for my FFT grid pts

  n = 0;
  for (k = nzlo_fft; k <= nzhi_fft; k++) {
    // TODO int division intentional?
    int per_k = k - nz_pppm * (2 * k / nz_pppm);
    for (j = nylo_fft; j <= nyhi_fft; j++) {
      int per_j = j - ny_pppm * (2 * j / ny_pppm);
      for (i = nxlo_fft; i <= nxhi_fft; i++) {
        int per_i = i - nx_pppm * (2 * i / nx_pppm);

        double unitk_lamda[3];
        unitk_lamda[0] = 2.0 * MY_PI * per_i;
        unitk_lamda[1] = 2.0 * MY_PI * per_j;
        unitk_lamda[2] = 2.0 * MY_PI * per_k;
        x2lamdaT(&unitk_lamda[0], &unitk_lamda[0]);
        fkx[n] = unitk_lamda[0];
        fky[n] = unitk_lamda[1];
        fkz[n] = unitk_lamda[2];
        n++;
      }
    }
  }

  // virial coefficients

  double sqk, vterm;

  for (n = 0; n < nfft; n++) {
    sqk = fkx[n] * fkx[n] + fky[n] * fky[n] + fkz[n] * fkz[n];
    if (sqk == 0.0) {
      vg[n][0] = 0.0;
      vg[n][1] = 0.0;
      vg[n][2] = 0.0;
      vg[n][3] = 0.0;
      vg[n][4] = 0.0;
      vg[n][5] = 0.0;
    } else {
      vterm = -2.0 * (1.0 / sqk + 0.25 / (g_ewald * g_ewald));
      vg[n][0] = 1.0 + vterm * fkx[n] * fkx[n];
      vg[n][1] = 1.0 + vterm * fky[n] * fky[n];
      vg[n][2] = 1.0 + vterm * fkz[n] * fkz[n];
      vg[n][3] = vterm * fkx[n] * fky[n];
      vg[n][4] = vterm * fkx[n] * fkz[n];
      vg[n][5] = vterm * fky[n] * fkz[n];
    }
  }

  compute_gf_ik_triclinic();
}

/* ----------------------------------------------------------------------
   reset local grid arrays and communication stencils
   called by fix balance b/c it changed sizes of processor sub-domains
------------------------------------------------------------------------- */

void PPPMConp::setup_grid() {
  // free all arrays previously allocated

  deallocate();
  if (peratom_allocate_flag) deallocate_peratom();
  if (group_allocate_flag) deallocate_groups();

  // reset portion of global grid that each proc owns

  set_grid_local();

  // reallocate K-space dependent memory
  // check if grid communication is now overlapping if not allowed
  // don't invoke allocate peratom() or group(), will be allocated when needed

  allocate();

  if (!overlap_allowed && !gc->ghost_adjacent())
    error->all(FLERR,
               "PPPM/conp grid stencil extends "
               "beyond nearest neighbor processor");

  // pre-compute Green's function denomiator expansion
  // pre-compute 1d charge distribution coefficients

  compute_gf_denom();
  if (differentiation_flag == 1) compute_sf_precoeff();
  compute_rho_coeff();

  // pre-compute volume-dependent coeffs for portion of grid I now own

  setup();
}

/* ----------------------------------------------------------------------
   compute the PPPM long-range force, energy, virial
------------------------------------------------------------------------- */

void PPPMConp::compute(int eflag, int vflag) {
  int i, j;

  // set energy/virial flags
  // invoke allocate_peratom() if needed for first time

  ev_init(eflag, vflag);

  if (evflag_atom && !peratom_allocate_flag) allocate_peratom();

  // if atom count has changed, update qsum and qsqsum

  qsum_qsq();
  natoms_original = atom->natoms;

  // return if there are no charges

  // if (qsqsum == 0.0) return; TODO move back in

  start_compute();
  make_rho();

  // all procs communicate density values from their ghost cells
  //   to fully sum contribution in their 3d bricks
  // remap from 3d decomposition to FFT decomposition

  gc->reverse_comm_kspace(this, 1, sizeof(FFT_SCALAR), REVERSE_RHO, gc_buf1,
                          gc_buf2, MPI_FFT_SCALAR);
  brick2fft();

  // compute potential gradient on my FFT grid and
  //   portion of e_long on this proc's FFT grid
  // return gradients (electric fields) in 3d brick decomposition
  // also performs per-atom calculations via poisson_peratom()

  poisson();
  // cout << "POISSON ENERGY: " << energy << ", " << energy * 0.5 * volume << ",
  // "
  //<< energy * 0.5 * volume * qqrd2e * scale << endl;

  // all procs communicate E-field values
  // to fill ghost cells surrounding their 3d bricks

  if (differentiation_flag == 1)
    gc->forward_comm_kspace(this, 1, sizeof(FFT_SCALAR), FORWARD_AD, gc_buf1,
                            gc_buf2, MPI_FFT_SCALAR);
  else
    gc->forward_comm_kspace(this, 3, sizeof(FFT_SCALAR), FORWARD_IK, gc_buf1,
                            gc_buf2, MPI_FFT_SCALAR);

  // extra per-atom energy/virial communication

  if (evflag_atom) {
    if (differentiation_flag == 1 && vflag_atom)
      gc->forward_comm_kspace(this, 6, sizeof(FFT_SCALAR), FORWARD_AD_PERATOM,
                              gc_buf1, gc_buf2, MPI_FFT_SCALAR);
    else if (differentiation_flag == 0)
      gc->forward_comm_kspace(this, 7, sizeof(FFT_SCALAR), FORWARD_IK_PERATOM,
                              gc_buf1, gc_buf2, MPI_FFT_SCALAR);
  }

  // calculate the force on my particles

  fieldforce();

  // extra per-atom energy/virial communication

  if (evflag_atom) fieldforce_peratom();

  // sum global energy across procs and add in volume-dependent term

  const double qscale = qqrd2e * scale;

  if (eflag_global) {
    double energy_all;
    MPI_Allreduce(&energy, &energy_all, 1, MPI_DOUBLE, MPI_SUM, world);
    energy = energy_all;

    energy *= 0.5 * volume;
    energy -= g_ewald * qsqsum / MY_PIS +
              MY_PI2 * qsum * qsum / (g_ewald * g_ewald * volume);
    energy *= qscale;
  }

  // sum global virial across procs

  if (vflag_global) {
    double virial_all[6];
    MPI_Allreduce(virial, virial_all, 6, MPI_DOUBLE, MPI_SUM, world);
    for (i = 0; i < 6; i++) virial[i] = 0.5 * qscale * volume * virial_all[i];
  }

  // per-atom energy/virial
  // energy includes self-energy correction
  // ntotal accounts for TIP4P tallying eatom/vatom for ghost atoms

  if (evflag_atom) {
    double *q = atom->q;
    int nlocal = atom->nlocal;
    int ntotal = nlocal;
    if (tip4pflag) ntotal += atom->nghost;

    if (eflag_atom) {
      for (i = 0; i < nlocal; i++) {
        eatom[i] *= 0.5;
        eatom[i] -= g_ewald * q[i] * q[i] / MY_PIS +
                    MY_PI2 * q[i] * qsum / (g_ewald * g_ewald * volume);
        eatom[i] *= qscale;
      }
      for (i = nlocal; i < ntotal; i++) eatom[i] *= 0.5 * qscale;
    }

    if (vflag_atom) {
      for (i = 0; i < ntotal; i++)
        for (j = 0; j < 6; j++) vatom[i][j] *= 0.5 * qscale;
    }
  }

  // 2d slab correction

  if (slabflag == 1) slabcorr();

  // 1d wire correction

  if (wireflag == 1) wirecorr();

  // convert atoms back from lamda to box coords

  if (triclinic) domain->lamda2x(atom->nlocal);
}

/* ----------------------------------------------------------------------
------------------------------------------------------------------------- */
void PPPMConp::start_compute() {
  if (compute_step < update->ntimestep) {
    if (compute_step == -1) setup();
    // convert atoms from box to lamda coords
    if (triclinic == 0)
      boxlo = domain->boxlo;
    else {
      boxlo = domain->boxlo_lamda;
      domain->x2lamda(atom->nlocal);
    }
    // extend size of per-atom arrays if necessary
    if (atom->nmax > nmax) {
      memory->destroy(part2grid);
      nmax = atom->nmax;
      memory->create(part2grid, nmax, 3, "pppm:part2grid");
    }
    // find grid points for all my particles
    // map my particle charge onto my local 3d density grid
    particle_map();
    compute_step = update->ntimestep;
  }
}

/* ----------------------------------------------------------------------
------------------------------------------------------------------------- */
void PPPMConp::compute_vector(bigint *imat, double *vec) {
  start_compute();
  double const scaleinv = 1.0 / (nx_pppm * ny_pppm * nz_pppm);

  // temporarily store and switch pointers so we can use brick2fft() for
  // electrolyte density (without writing an additional function)
  FFT_SCALAR ***density_brick_real = density_brick;
  FFT_SCALAR *density_fft_real = density_fft;
  make_electrolyte_rho(imat);
  density_brick = electrolyte_density_brick;
  density_fft = electrolyte_density_fft;
  gc->reverse_comm_kspace(this, 1, sizeof(FFT_SCALAR), REVERSE_RHO, gc_buf1,
                          gc_buf2, MPI_FFT_SCALAR);
  brick2fft();
  // switch back pointers
  density_brick = density_brick_real;
  density_fft = density_fft_real;

  // transform electrolyte charge density (r -> k) (complex conjugate)
  for (int i = 0, n = 0; i < nfft; i++) {
    work1[n++] = electrolyte_density_fft[i];
    work1[n++] = ZEROF;
  }
  fft1->compute(work1, work1, -1);

  // k->r FFT of Green's * electrolyte density = brick_psi
  for (int i = 0, n = 0; i < nfft; i++) {
    work2[n] = work1[n] * greensfn[i];
    n++;
    work2[n] = work1[n] * greensfn[i];
    n++;
  }
  fft2->compute(work2, work2, 1);
  vector<double> brick_psi(nz_pppm * ny_pppm * nx_pppm, 0.);
  for (int k = nzlo_in, n = 0; k <= nzhi_in; k++)
    for (int j = nylo_in; j <= nyhi_in; j++)
      for (int i = nxlo_in; i <= nxhi_in; i++) {
        brick_psi[ny_pppm * nx_pppm * k + nx_pppm * j + i] = work2[n];
        n += 2;
      }
  MPI_Allreduce(MPI_IN_PLACE, &brick_psi.front(), nz_pppm * ny_pppm * nx_pppm,
                MPI_DOUBLE, MPI_SUM, world);

  // project brick_psi with weight matrix
  double **x = atom->x;
  for (int i = 0; i < atom->nlocal; i++) {
    int ipos = imat[i];
    if (ipos < 0) continue;
    double v = 0.;
    // (nx,ny,nz) = global coords of grid pt to "lower left" of charge
    // (dx,dy,dz) = distance to "lower left" grid pt
    // (mx,my,mz) = global coords of moving stencil pt
    int nix = part2grid[i][0];
    int niy = part2grid[i][1];
    int niz = part2grid[i][2];
    FFT_SCALAR dix = nix + shiftone - (x[i][0] - boxlo[0]) * delxinv;
    FFT_SCALAR diy = niy + shiftone - (x[i][1] - boxlo[1]) * delyinv;
    FFT_SCALAR diz = niz + shiftone - (x[i][2] - boxlo[2]) * delzinv;
    compute_rho1d(dix, diy, diz);
    for (int ni = nlower; ni <= nupper; ni++) {
      double iz0 = rho1d[2][ni];
      int miz = ni + niz;
      // int debug_miz = miz;
      while (miz < 0) miz += nz_pppm;
      miz = miz % nz_pppm;
      for (int mi = nlower; mi <= nupper; mi++) {
        double iy0 = iz0 * rho1d[1][mi];
        int miy = mi + niy;
        // int debug_miy = miy;
        while (miy < 0) miy += ny_pppm;
        miy = miy % ny_pppm;
        for (int li = nlower; li <= nupper; li++) {
          int mix = li + nix;
          // int debug_mix = mix;
          while (mix < 0) mix += nx_pppm;
          mix = mix % nx_pppm;
          double ix0 = iy0 * rho1d[0][li];
          v += ix0 * brick_psi[ny_pppm * nx_pppm * miz + nx_pppm * miy + mix];
        }
      }
    }
    vec[ipos] += v * scaleinv;
  }
}
/* ----------------------------------------------------------------------
-------------------------------------------------------------------------
*/

void PPPMConp::compute_matrix(bigint *imat, double **matrix) {
  if (comm->me == 0) cout << "MATRIX calculation" << endl;
  // TODO verify energies from real and k space are the same
  // TODO replace compute with required setup
  compute(1, 0);

  // fft green's funciton k -> r
  MPI_Barrier(world);
  double fft_time = MPI_Wtime();
  vector<double> greens_real(nz_pppm * ny_pppm * nx_pppm, 0.);
  for (int i = 0, n = 0; i < nfft; i++) {
    work2[n++] = greensfn[i];
    work2[n++] = ZEROF;  // *= greensfn[i];
  }
  fft2->compute(work2, work2, -1);
  for (int k = nzlo_in, n = 0; k <= nzhi_in; k++)
    for (int j = nylo_in; j <= nyhi_in; j++)
      for (int i = nxlo_in; i <= nxhi_in; i++) {
        greens_real[ny_pppm * nx_pppm * k + nx_pppm * j + i] = work2[n];
        n += 2;
      }
  MPI_Allreduce(MPI_IN_PLACE, &greens_real.front(), nz_pppm * ny_pppm * nx_pppm,
                MPI_DOUBLE, MPI_SUM, world);
  MPI_Barrier(world);
  if (comm->me == 0)
    utils::logmesg(lmp, fmt::format("FFT time: {}\n", MPI_Wtime() - fft_time));

  // (nx,ny,nz) = global coords of grid pt to "lower left" of charge
  // (dx,dy,dz) = distance to "lower left" grid pt
  // (mx,my,mz) = global coords of moving stencil pt
  int const nlocal = atom->nlocal;
  int nmat =
      std::count_if(&imat[0], &imat[nlocal], [](int x) { return x >= 0; });
  MPI_Allreduce(MPI_IN_PLACE, &nmat, 1, MPI_INT, MPI_SUM, world);
  double **x = atom->x;

  // map green's function in real space from mesh to particle positions
  // with matrix multiplication 'W^T G W' in two steps. gw is result of
  // first multiplication.
  MPI_Barrier(world);
  double step1_time = MPI_Wtime();
  int nx_conp = nxhi_out - nxlo_out + 1;  // nx_pppm + order + 1;
  int ny_conp = nyhi_out - nylo_out + 1;  // ny_pppm + order + 1;
  int nz_conp = nzhi_out - nzlo_out + 1;  // nz_pppm + order + 1;
  int nxyz = nx_conp * ny_conp * nz_conp;
  // int off = nlower - 1;
  vector<vector<double>> gw(nmat, vector<double>(nxyz, 0.));
  vector<vector<double>> x_ele(nmat, {0, 0, 0});
  for (int i = 0; i < nlocal; i++) {
    int ipos = imat[i];
    if (ipos < 0) continue;
    for (int dim = 0; dim < 3; dim++) x_ele[ipos][dim] = x[i][dim];
  }
  for (int i = 0; i < nmat; i++)
    MPI_Allreduce(MPI_IN_PLACE, &x_ele[i].front(), 3, MPI_DOUBLE, MPI_SUM,
                  world);

  for (int ipos = 0; ipos < nmat; ipos++) {
    vector<double> xi_ele = x_ele[ipos];
    // new calculation for nx, ny, nz because part2grid available for
    // nlocal, only
    int nix =
        static_cast<int>((xi_ele[0] - boxlo[0]) * delxinv + shift) - OFFSET;
    int niy =
        static_cast<int>((xi_ele[1] - boxlo[1]) * delyinv + shift) - OFFSET;
    int niz =
        static_cast<int>((xi_ele[2] - boxlo[2]) * delzinv + shift) - OFFSET;
    FFT_SCALAR dx = nix + shiftone - (xi_ele[0] - boxlo[0]) * delxinv;
    FFT_SCALAR dy = niy + shiftone - (xi_ele[1] - boxlo[1]) * delyinv;
    FFT_SCALAR dz = niz + shiftone - (xi_ele[2] - boxlo[2]) * delzinv;
    compute_rho1d(dx, dy, dz);
    for (int ni = nlower; ni <= nupper; ni++) {
      double iz0 = rho1d[2][ni];
      int miz = ni + niz;
      for (int mi = nlower; mi <= nupper; mi++) {
        double iy0 = iz0 * rho1d[1][mi];
        int miy = mi + niy;
        for (int li = nlower; li <= nupper; li++) {
          int mix = li + nix;
          double ix0 = iy0 * rho1d[0][li];
          for (int mjz = nzlo_out; mjz <= nzhi_out; mjz++) {
            int mz = abs(mjz - miz) % nz_pppm;
            for (int mjy = nylo_out; mjy <= nyhi_out; mjy++) {
              int my = abs(mjy - miy) % ny_pppm;
              for (int mjx = nxlo_out; mjx <= nxhi_out; mjx++) {
                int mx = abs(mjx - mix) % nx_pppm;
                gw[ipos][nx_conp * ny_conp * (mjz - nzlo_out) +
                         nx_conp * (mjy - nylo_out) + (mjx - nxlo_out)] +=
                    ix0 *
                    greens_real[mz * nx_pppm * ny_pppm + my * nx_pppm + mx];
              }
            }
          }
        }
      }
    }
  }
  MPI_Barrier(world);
  if (comm->me == 0)
    utils::logmesg(lmp,
                   fmt::format("step 1 time: {}\n", MPI_Wtime() - step1_time));

  double step2_time = MPI_Wtime();
  int min_miz = nzhi_out;
  int max_miz = nzlo_out;
  for (int i = 0; i < nlocal; i++) {
    int ipos = imat[i];
    if (ipos < 0) continue;
    int nix = part2grid[i][0];
    int niy = part2grid[i][1];
    int niz = part2grid[i][2];
    FFT_SCALAR dix = nix + shiftone - (x[i][0] - boxlo[0]) * delxinv;
    FFT_SCALAR diy = niy + shiftone - (x[i][1] - boxlo[1]) * delyinv;
    FFT_SCALAR diz = niz + shiftone - (x[i][2] - boxlo[2]) * delzinv;
    compute_rho1d(dix, diy, diz);
    for (int jpos = 0; jpos < nmat; jpos++) {
      double aij = 0.;
      for (int ni = nlower; ni <= nupper; ni++) {
        double iz0 = rho1d[2][ni];
        int miz = ni + niz;
        min_miz = MIN(min_miz, miz);
        max_miz = MAX(max_miz, miz);
        for (int mi = nlower; mi <= nupper; mi++) {
          double iy0 = iz0 * rho1d[1][mi];
          int miy = mi + niy;
          for (int li = nlower; li <= nupper; li++) {
            int mix = li + nix;
            double ix0 = iy0 * rho1d[0][li];
            int miz0 = miz - nzlo_out;
            int miy0 = miy - nylo_out;
            int mix0 = mix - nxlo_out;
            assert(miz0 >= 0);
            assert(miy0 >= 0);
            assert(mix0 >= 0);
            assert(miz0 < nz_conp);
            assert(miy0 < ny_conp);
            assert(mix0 < nx_conp);
            aij += ix0 *
                   gw[jpos][nx_conp * ny_conp * miz0 + nx_conp * miy0 + mix0];
          }
        }
      }
      matrix[ipos][jpos] += aij / volume;
    }
  }
  MPI_Barrier(world);
  if (comm->me == 0)
    utils::logmesg(lmp,
                   fmt::format("step 2 time: {}\n", MPI_Wtime() - step2_time));
}

/* ----------------------------------------------------------------------
   allocate memory that depends on # of K-vectors and order
-------------------------------------------------------------------------
*/

void PPPMConp::allocate() {
  memory->create3d_offset(electrolyte_density_brick, nzlo_out, nzhi_out,
                          nylo_out, nyhi_out, nxlo_out, nxhi_out,
                          "pppm/conp:electrolyte_density_brick");
  memory->create(electrolyte_density_fft, nfft_both,
                 "pppm/conp:electrolyte_density_fft");
  PPPM::allocate();
}

/* ----------------------------------------------------------------------
   deallocate memory that depends on # of K-vectors and order
-------------------------------------------------------------------------
*/

void PPPMConp::deallocate() {
  memory->destroy3d_offset(electrolyte_density_brick, nzlo_out, nylo_out,
                           nxlo_out);
  memory->destroy(electrolyte_density_fft);

  memory->destroy3d_offset(density_brick, nzlo_out, nylo_out, nxlo_out);
  if (differentiation_flag == 1) {
    memory->destroy3d_offset(u_brick, nzlo_out, nylo_out, nxlo_out);
    memory->destroy(sf_precoeff1);
    memory->destroy(sf_precoeff2);
    memory->destroy(sf_precoeff3);
    memory->destroy(sf_precoeff4);
    memory->destroy(sf_precoeff5);
    memory->destroy(sf_precoeff6);
  } else {
    memory->destroy3d_offset(vdx_brick, nzlo_out, nylo_out, nxlo_out);
    memory->destroy3d_offset(vdy_brick, nzlo_out, nylo_out, nxlo_out);
    memory->destroy3d_offset(vdz_brick, nzlo_out, nylo_out, nxlo_out);
  }

  memory->destroy(density_fft);
  memory->destroy(greensfn);
  memory->destroy(work1);
  memory->destroy(work2);
  memory->destroy(vg);

  if (triclinic == 0) {
    memory->destroy1d_offset(fkx, nxlo_fft);
    memory->destroy1d_offset(fky, nylo_fft);
    memory->destroy1d_offset(fkz, nzlo_fft);
  } else {
    memory->destroy(fkx);
    memory->destroy(fky);
    memory->destroy(fkz);
  }

  memory->destroy(gf_b);
  if (stagger_flag) gf_b = nullptr;
  memory->destroy2d_offset(rho1d, -order_allocated / 2);
  memory->destroy2d_offset(drho1d, -order_allocated / 2);
  memory->destroy2d_offset(rho_coeff, (1 - order_allocated) / 2);
  memory->destroy2d_offset(drho_coeff, (1 - order_allocated) / 2);

  memory->destroy(gc_buf1);
  memory->destroy(gc_buf2);
}

/* ----------------------------------------------------------------------
   set global size of PPPM grid = nx,ny,nz_pppm
   used for charge accumulation, FFTs, and electric field interpolation
-------------------------------------------------------------------------
*/

void PPPMConp::set_grid_global() {
  // use xprd,yprd,zprd (even if triclinic, and then scale later)
  // adjust z dimension for 2d slab PPPM
  // 3d PPPM just uses zprd since slab_volfactor = 1.0

  double xprd = domain->xprd;
  double yprd = domain->yprd;
  double zprd = domain->zprd;
  double xprd_wire = xprd * wire_volfactor;
  double yprd_wire = yprd * wire_volfactor;
  double zprd_slab = zprd * slab_volfactor;

  // make initial g_ewald estimate
  // based on desired accuracy and real space cutoff
  // fluid-occupied volume used to estimate real-space error
  // zprd used rather than zprd_slab

  double h;
  bigint natoms = atom->natoms;

  if (!gewaldflag) {
    if (accuracy <= 0.0) error->all(FLERR, "KSpace accuracy must be > 0");
    if (q2 == 0.0)
      error->all(FLERR, "Must use kspace_modify gewald for uncharged system");
    g_ewald =
        accuracy * sqrt(natoms * cutoff * xprd * yprd * zprd) / (2.0 * q2);
    if (g_ewald >= 1.0)
      g_ewald = (1.35 - 0.15 * log(accuracy)) / cutoff;
    else
      g_ewald = sqrt(-log(g_ewald)) / cutoff;
  }

  // set optimal nx_pppm,ny_pppm,nz_pppm based on order and accuracy
  // nz_pppm uses extended zprd_slab instead of zprd
  // reduce it until accuracy target is met

  if (!gridflag) {
    if (differentiation_flag == 1 || stagger_flag) {
      h = h_x = h_y = h_z = 4.0 / g_ewald;
      int count = 0;
      while (1) {
        // set grid dimensions

        nx_pppm = static_cast<int>(xprd_wire / h_x);
        ny_pppm = static_cast<int>(yprd_wire / h_y);
        nz_pppm = static_cast<int>(zprd_slab / h_z);

        if (nx_pppm <= 1) nx_pppm = 2;
        if (ny_pppm <= 1) ny_pppm = 2;
        if (nz_pppm <= 1) nz_pppm = 2;

        // estimate Kspace force error

        double df_kspace = compute_df_kspace();

        // break loop if the accuracy has been reached or
        // too many loops have been performed

        count++;
        if (df_kspace <= accuracy) break;

        if (count > 500) error->all(FLERR, "Could not compute grid size");
        h *= 0.95;
        h_x = h_y = h_z = h;
      }

    } else {
      double err;
      h_x = h_y = h_z = 1.0 / g_ewald;

      nx_pppm = static_cast<int>(xprd_wire / h_x) + 1;
      ny_pppm = static_cast<int>(yprd_wire / h_y) + 1;
      nz_pppm = static_cast<int>(zprd_slab / h_z) + 1;

      err = estimate_ik_error(h_x, xprd_wire, natoms);
      while (err > accuracy) {
        err = estimate_ik_error(h_x, xprd_wire, natoms);
        nx_pppm++;
        h_x = xprd_wire / nx_pppm;
      }

      err = estimate_ik_error(h_y, yprd_wire, natoms);
      while (err > accuracy) {
        err = estimate_ik_error(h_y, yprd_wire, natoms);
        ny_pppm++;
        h_y = yprd_wire / ny_pppm;
      }

      err = estimate_ik_error(h_z, zprd_slab, natoms);
      while (err > accuracy) {
        err = estimate_ik_error(h_z, zprd_slab, natoms);
        nz_pppm++;
        h_z = zprd_slab / nz_pppm;
      }
    }

    // scale grid for triclinic skew

    if (triclinic) {
      double tmp[3];
      tmp[0] = nx_pppm / xprd;
      tmp[1] = ny_pppm / yprd;
      tmp[2] = nz_pppm / zprd;
      lamda2xT(&tmp[0], &tmp[0]);
      nx_pppm = static_cast<int>(tmp[0]) + 1;
      ny_pppm = static_cast<int>(tmp[1]) + 1;
      nz_pppm = static_cast<int>(tmp[2]) + 1;
    }
  }

  // boost grid size until it is factorable

  while (!factorable(nx_pppm)) nx_pppm++;
  while (!factorable(ny_pppm)) ny_pppm++;
  while (!factorable(nz_pppm)) nz_pppm++;

  if (triclinic == 0) {
    h_x = xprd_wire / nx_pppm;
    h_y = yprd_wire / ny_pppm;
    h_z = zprd_slab / nz_pppm;
  } else {
    double tmp[3];
    tmp[0] = nx_pppm;
    tmp[1] = ny_pppm;
    tmp[2] = nz_pppm;
    x2lamdaT(&tmp[0], &tmp[0]);
    h_x = 1.0 / tmp[0];
    h_y = 1.0 / tmp[1];
    h_z = 1.0 / tmp[2];
  }

  if (nx_pppm >= OFFSET || ny_pppm >= OFFSET || nz_pppm >= OFFSET)
    error->all(FLERR, "PPPM/conp grid is too large");
}

/* ----------------------------------------------------------------------
   compute estimated kspace force error
-------------------------------------------------------------------------
*/

double PPPMConp::compute_df_kspace() {
  double xprd = domain->xprd;
  double yprd = domain->yprd;
  double zprd = domain->zprd;
  double xprd_wire = xprd * wire_volfactor;
  double yprd_wire = yprd * wire_volfactor;
  double zprd_slab = zprd * slab_volfactor;
  bigint natoms = atom->natoms;
  double df_kspace = 0.0;
  if (differentiation_flag == 1 || stagger_flag) {
    double qopt = compute_qopt();
    df_kspace = sqrt(qopt / natoms) * q2 / (xprd_wire * yprd_wire * zprd_slab);
  } else {
    double lprx = estimate_ik_error(h_x, xprd_wire, natoms);
    double lpry = estimate_ik_error(h_y, yprd_wire, natoms);
    double lprz = estimate_ik_error(h_z, zprd_slab, natoms);
    df_kspace = sqrt(lprx * lprx + lpry * lpry + lprz * lprz) / sqrt(3.0);
  }
  return df_kspace;
}

/* ----------------------------------------------------------------------
   compute qopt
-------------------------------------------------------------------------
*/

double PPPMConp::compute_qopt() {
  int k, l, m, nx, ny, nz;
  double argx, argy, argz, wx, wy, wz, sx, sy, sz, qx, qy, qz;
  double u1, u2, sqk;
  double sum1, sum2, sum3, sum4, dot2;

  double *prd = domain->prd;

  const double xprd = prd[0];
  const double yprd = prd[1];
  const double zprd = prd[2];
  const double xprd_wire = xprd * wire_volfactor;
  const double yprd_wire = yprd * wire_volfactor;
  const double zprd_slab = zprd * slab_volfactor;
  volume = xprd_wire * yprd_wire * zprd_slab;

  const double unitkx = (MY_2PI / xprd_wire);
  const double unitky = (MY_2PI / yprd_wire);
  const double unitkz = (MY_2PI / zprd_slab);

  const int twoorder = 2 * order;

  // loop over entire FFT grid
  // each proc calculates contributions from every Pth grid point

  bigint ngridtotal = (bigint)nx_pppm * ny_pppm * nz_pppm;
  int nxy_pppm = nx_pppm * ny_pppm;

  double qopt = 0.0;

  for (bigint i = me; i < ngridtotal; i += nprocs) {
    k = i % nx_pppm;
    l = (i / nx_pppm) % ny_pppm;
    m = i / nxy_pppm;

    const int kper = k - nx_pppm * (2 * k / nx_pppm);
    const int lper = l - ny_pppm * (2 * l / ny_pppm);
    const int mper = m - nz_pppm * (2 * m / nz_pppm);

    sqk = square(unitkx * kper) + square(unitky * lper) + square(unitkz * mper);
    if (sqk == 0.0) continue;

    sum1 = sum2 = sum3 = sum4 = 0.0;

    for (nx = -2; nx <= 2; nx++) {
      qx = unitkx * (kper + nx_pppm * nx);
      sx = exp(-0.25 * square(qx / g_ewald));
      argx = 0.5 * qx * xprd_wire / nx_pppm;
      wx = powsinxx(argx, twoorder);
      qx *= qx;

      for (ny = -2; ny <= 2; ny++) {
        qy = unitky * (lper + ny_pppm * ny);
        sy = exp(-0.25 * square(qy / g_ewald));
        argy = 0.5 * qy * yprd_wire / ny_pppm;
        wy = powsinxx(argy, twoorder);
        qy *= qy;

        for (nz = -2; nz <= 2; nz++) {
          qz = unitkz * (mper + nz_pppm * nz);
          sz = exp(-0.25 * square(qz / g_ewald));
          argz = 0.5 * qz * zprd_slab / nz_pppm;
          wz = powsinxx(argz, twoorder);
          qz *= qz;

          dot2 = qx + qy + qz;
          u1 = sx * sy * sz;
          u2 = wx * wy * wz;

          sum1 += u1 * u1 / dot2 * MY_4PI * MY_4PI;
          sum2 += u1 * u2 * MY_4PI;
          sum3 += u2;
          sum4 += dot2 * u2;
        }
      }
    }

    sum2 *= sum2;
    qopt += sum1 - sum2 / (sum3 * sum4);
  }

  // sum qopt over all procs

  double qopt_all;
  MPI_Allreduce(&qopt, &qopt_all, 1, MPI_DOUBLE, MPI_SUM, world);
  return qopt_all;
}

/* ----------------------------------------------------------------------
   set local subset of PPPM/FFT grid that I own
   n xyz lo/hi in = 3d brick that I own (inclusive)
   n xyz lo/hi out = 3d brick + ghost cells in 6 directions (inclusive)
   n xyz lo/hi fft = FFT columns that I own (all of x dim, 2d decomp in
yz)
-------------------------------------------------------------------------
*/

void PPPMConp::set_grid_local() {
  // global indices of PPPM grid range from 0 to N-1
  // nlo_in,nhi_in = lower/upper limits of the 3d sub-brick of
  //   global PPPM grid that I own without ghost cells
  // for slab PPPM, assign z grid as if it were not extended
  // both non-tiled and tiled proc layouts use 0-1 fractional sumdomain
  // info

  if (comm->layout != Comm::LAYOUT_TILED) {
    nxlo_in = static_cast<int>(comm->xsplit[comm->myloc[0]] * nx_pppm /
                               wire_volfactor);
    nxhi_in = static_cast<int>(comm->xsplit[comm->myloc[0] + 1] * nx_pppm /
                               wire_volfactor) -
              1;

    nylo_in = static_cast<int>(comm->ysplit[comm->myloc[1]] * ny_pppm /
                               wire_volfactor);
    nyhi_in = static_cast<int>(comm->ysplit[comm->myloc[1] + 1] * ny_pppm /
                               wire_volfactor) -
              1;

    nzlo_in = static_cast<int>(comm->zsplit[comm->myloc[2]] * nz_pppm /
                               slab_volfactor);
    nzhi_in = static_cast<int>(comm->zsplit[comm->myloc[2] + 1] * nz_pppm /
                               slab_volfactor) -
              1;

  } else {
    nxlo_in = static_cast<int>(comm->mysplit[0][0] * nx_pppm / wire_volfactor);
    nxhi_in =
        static_cast<int>(comm->mysplit[0][1] * nx_pppm / wire_volfactor) - 1;

    nylo_in = static_cast<int>(comm->mysplit[1][0] * ny_pppm / wire_volfactor);
    nyhi_in =
        static_cast<int>(comm->mysplit[1][1] * ny_pppm / wire_volfactor) - 1;

    nzlo_in = static_cast<int>(comm->mysplit[2][0] * nz_pppm / slab_volfactor);
    nzhi_in =
        static_cast<int>(comm->mysplit[2][1] * nz_pppm / slab_volfactor) - 1;
  }

  // nlower,nupper = stencil size for mapping particles to PPPM grid

  nlower = -(order - 1) / 2;
  nupper = order / 2;

  // shift values for particle <-> grid mapping
  // add/subtract OFFSET to avoid int(-0.75) = 0 when want it to be -1

  if (order % 2)
    shift = OFFSET + 0.5;
  else
    shift = OFFSET;
  if (order % 2)
    shiftone = 0.0;
  else
    shiftone = 0.5;

  // nlo_out,nhi_out = lower/upper limits of the 3d sub-brick of
  //   global PPPM grid that my particles can contribute charge to
  // effectively nlo_in,nhi_in + ghost cells
  // nlo,nhi = global coords of grid pt to "lower left" of
  // smallest/largest
  //           position a particle in my box can be at
  // dist[3] = particle position bound = subbox + skin/2.0 + qdist
  //   qdist = offset due to TIP4P fictitious charge
  //   convert to triclinic if necessary
  // nlo_out,nhi_out = nlo,nhi + stencil size for particle mapping
  // for slab PPPM, assign z grid as if it were not extended

  double *prd, *sublo, *subhi;

  if (triclinic == 0) {
    prd = domain->prd;
    boxlo = domain->boxlo;
    sublo = domain->sublo;
    subhi = domain->subhi;
  } else {
    prd = domain->prd_lamda;
    boxlo = domain->boxlo_lamda;
    sublo = domain->sublo_lamda;
    subhi = domain->subhi_lamda;
  }

  double xprd = prd[0];
  double yprd = prd[1];
  double zprd = prd[2];
  double xprd_wire = xprd * wire_volfactor;
  double yprd_wire = yprd * wire_volfactor;
  double zprd_slab = zprd * slab_volfactor;

  double dist[3] = {0.0, 0.0, 0.0};
  double cuthalf = 0.5 * neighbor->skin + qdist;
  if (triclinic == 0)
    dist[0] = dist[1] = dist[2] = cuthalf;
  else
    kspacebbox(cuthalf, &dist[0]);

  int nlo, nhi;
  nlo = nhi = 0;

  nlo = static_cast<int>((sublo[0] - dist[0] - boxlo[0]) * nx_pppm / xprd_wire +
                         shift) -
        OFFSET;
  nhi = static_cast<int>((subhi[0] + dist[0] - boxlo[0]) * nx_pppm / xprd_wire +
                         shift) -
        OFFSET;
  nxlo_out = nlo + nlower;
  nxhi_out = nhi + nupper;

  nlo = static_cast<int>((sublo[1] - dist[1] - boxlo[1]) * ny_pppm / yprd_wire +
                         shift) -
        OFFSET;
  nhi = static_cast<int>((subhi[1] + dist[1] - boxlo[1]) * ny_pppm / yprd_wire +
                         shift) -
        OFFSET;
  nylo_out = nlo + nlower;
  nyhi_out = nhi + nupper;

  nlo = static_cast<int>((sublo[2] - dist[2] - boxlo[2]) * nz_pppm / zprd_slab +
                         shift) -
        OFFSET;
  nhi = static_cast<int>((subhi[2] + dist[2] - boxlo[2]) * nz_pppm / zprd_slab +
                         shift) -
        OFFSET;
  nzlo_out = nlo + nlower;
  nzhi_out = nhi + nupper;

  if (stagger_flag) {
    nxhi_out++;
    nyhi_out++;
    nzhi_out++;
  }

  // for slab PPPM, change the grid boundary for processors at +z end
  //   to include the empty volume between periodically repeating slabs
  // for slab PPPM, want charge data communicated from -z proc to +z proc,
  //   but not vice versa, also want field data communicated from +z proc
  //   to -z proc, but not vice versa
  // this is accomplished by nzhi_in = nzhi_out on +z end (no ghost cells)
  // also insure no other procs use ghost cells beyond +z limit
  // differnet logic for non-tiled vs tiled decomposition

  if (slabflag == 1) {
    if (comm->layout != Comm::LAYOUT_TILED) {
      if (comm->myloc[2] == comm->procgrid[2] - 1)
        nzhi_in = nzhi_out = nz_pppm - 1;
    } else {
      if (comm->mysplit[2][1] == 1.0) nzhi_in = nzhi_out = nz_pppm - 1;
    }
    nzhi_out = MIN(nzhi_out, nz_pppm - 1);
  }

  if (wireflag == 1) {
    if (comm->layout != Comm::LAYOUT_TILED) {
      if (comm->myloc[0] == comm->procgrid[0] - 1)
        nxhi_in = nxhi_out = nx_pppm - 1;
      if (comm->myloc[1] == comm->procgrid[1] - 1)
        nyhi_in = nyhi_out = ny_pppm - 1;
    } else {
      if (comm->mysplit[0][1] == 1.0) nxhi_in = nxhi_out = nx_pppm - 1;
      if (comm->mysplit[1][1] == 1.0) nyhi_in = nyhi_out = ny_pppm - 1;
    }
    nxhi_out = MIN(nxhi_out, nx_pppm - 1);
    nyhi_out = MIN(nyhi_out, ny_pppm - 1);
  }

  // x-pencil decomposition of FFT mesh
  // global indices range from 0 to N-1
  // each proc owns entire x-dimension, clumps of columns in y,z
  // dimensions npey_fft,npez_fft = # of procs in y,z dims if nprocs is
  // small enough, proc can own 1 or more entire xy planes,
  //   else proc owns 2d sub-blocks of yz plane
  // me_y,me_z = which proc (0-npe_fft-1) I am in y,z dimensions
  // nlo_fft,nhi_fft = lower/upper limit of the section
  //   of the global FFT mesh that I own in x-pencil decomposition

  int npey_fft, npez_fft;
  if (nz_pppm >= nprocs) {
    npey_fft = 1;
    npez_fft = nprocs;
  } else
    procs2grid2d(nprocs, ny_pppm, nz_pppm, &npey_fft, &npez_fft);

  int me_y = me % npey_fft;
  int me_z = me / npey_fft;

  nxlo_fft = 0;
  nxhi_fft = nx_pppm - 1;
  nylo_fft = me_y * ny_pppm / npey_fft;
  nyhi_fft = (me_y + 1) * ny_pppm / npey_fft - 1;
  nzlo_fft = me_z * nz_pppm / npez_fft;
  nzhi_fft = (me_z + 1) * nz_pppm / npez_fft - 1;

  // ngrid = count of PPPM grid pts owned by this proc, including ghosts

  ngrid = (nxhi_out - nxlo_out + 1) * (nyhi_out - nylo_out + 1) *
          (nzhi_out - nzlo_out + 1);

  // count of FFT grids pts owned by this proc, without ghosts
  // nfft = FFT points in x-pencil FFT decomposition on this proc
  // nfft_brick = FFT points in 3d brick-decomposition on this proc
  // nfft_both = greater of 2 values

  nfft = (nxhi_fft - nxlo_fft + 1) * (nyhi_fft - nylo_fft + 1) *
         (nzhi_fft - nzlo_fft + 1);
  int nfft_brick = (nxhi_in - nxlo_in + 1) * (nyhi_in - nylo_in + 1) *
                   (nzhi_in - nzlo_in + 1);
  nfft_both = MAX(nfft, nfft_brick);
}

/* ----------------------------------------------------------------------
   pre-compute modified (Hockney-Eastwood) Coulomb Green's function
-------------------------------------------------------------------------
*/

void PPPMConp::compute_gf_ik() {
  const double *const prd = domain->prd;

  const double xprd = prd[0];
  const double yprd = prd[1];
  const double zprd = prd[2];
  const double xprd_wire = xprd * wire_volfactor;
  const double yprd_wire = yprd * wire_volfactor;
  const double zprd_slab = zprd * slab_volfactor;
  const double unitkx = (MY_2PI / xprd_wire);
  const double unitky = (MY_2PI / yprd_wire);
  const double unitkz = (MY_2PI / zprd_slab);

  double snx, sny, snz;
  double argx, argy, argz, wx, wy, wz, sx, sy, sz, qx, qy, qz;
  double sum1, dot1, dot2;
  double numerator, denominator;
  double sqk;

  int k, l, m, n, nx, ny, nz, kper, lper, mper;

  const int nbx = static_cast<int>((g_ewald * xprd_wire / (MY_PI * nx_pppm)) *
                                   pow(-log(EPS_HOC), 0.25));
  const int nby = static_cast<int>((g_ewald * yprd_wire / (MY_PI * ny_pppm)) *
                                   pow(-log(EPS_HOC), 0.25));
  const int nbz = static_cast<int>((g_ewald * zprd_slab / (MY_PI * nz_pppm)) *
                                   pow(-log(EPS_HOC), 0.25));
  const int twoorder = 2 * order;

  n = 0;
  for (m = nzlo_fft; m <= nzhi_fft; m++) {
    mper = m - nz_pppm * (2 * m / nz_pppm);
    snz = square(sin(0.5 * unitkz * mper * zprd_slab / nz_pppm));

    for (l = nylo_fft; l <= nyhi_fft; l++) {
      lper = l - ny_pppm * (2 * l / ny_pppm);
      sny = square(sin(0.5 * unitky * lper * yprd_wire / ny_pppm));

      for (k = nxlo_fft; k <= nxhi_fft; k++) {
        kper = k - nx_pppm * (2 * k / nx_pppm);
        snx = square(sin(0.5 * unitkx * kper * xprd_wire / nx_pppm));

        sqk = square(unitkx * kper) + square(unitky * lper) +
              square(unitkz * mper);

        if (sqk != 0.0) {
          numerator = 12.5663706 / sqk;  // 4 pi / k^2
          denominator = gf_denom(snx, sny, snz);
          sum1 = 0.0;

          for (nx = -nbx; nx <= nbx; nx++) {
            qx = unitkx * (kper + nx_pppm * nx);
            sx = exp(-0.25 * square(qx / g_ewald));
            argx = 0.5 * qx * xprd_wire / nx_pppm;
            wx = powsinxx(argx, twoorder);

            for (ny = -nby; ny <= nby; ny++) {
              qy = unitky * (lper + ny_pppm * ny);
              sy = exp(-0.25 * square(qy / g_ewald));
              argy = 0.5 * qy * yprd_wire / ny_pppm;
              wy = powsinxx(argy, twoorder);

              for (nz = -nbz; nz <= nbz; nz++) {
                qz = unitkz * (mper + nz_pppm * nz);
                sz = exp(-0.25 * square(qz / g_ewald));
                argz = 0.5 * qz * zprd_slab / nz_pppm;
                wz = powsinxx(argz, twoorder);

                dot1 = unitkx * kper * qx + unitky * lper * qy +
                       unitkz * mper * qz;
                dot2 = qx * qx + qy * qy + qz * qz;
                sum1 += (dot1 / dot2) * sx * sy * sz * wx * wy * wz;
              }
            }
          }
          greensfn[n++] = numerator * sum1 / denominator;
        } else
          greensfn[n++] = 0.0;
      }
    }
  }
}

/* ----------------------------------------------------------------------
   compute optimized Green's function for energy calculation
-------------------------------------------------------------------------
*/

void PPPMConp::compute_gf_ad() {
  const double *const prd = domain->prd;

  const double xprd = prd[0];
  const double yprd = prd[1];
  const double zprd = prd[2];
  const double xprd_wire = xprd * wire_volfactor;
  const double yprd_wire = yprd * wire_volfactor;
  const double zprd_slab = zprd * slab_volfactor;
  const double unitkx = (MY_2PI / xprd_wire);
  const double unitky = (MY_2PI / yprd_wire);
  const double unitkz = (MY_2PI / zprd_slab);

  double snx, sny, snz, sqk;
  double argx, argy, argz, wx, wy, wz, sx, sy, sz, qx, qy, qz;
  double numerator, denominator;
  int k, l, m, n, kper, lper, mper;

  const int twoorder = 2 * order;

  for (int i = 0; i < 6; i++) sf_coeff[i] = 0.0;

  n = 0;
  for (m = nzlo_fft; m <= nzhi_fft; m++) {
    mper = m - nz_pppm * (2 * m / nz_pppm);
    qz = unitkz * mper;
    snz = square(sin(0.5 * qz * zprd_slab / nz_pppm));
    sz = exp(-0.25 * square(qz / g_ewald));
    argz = 0.5 * qz * zprd_slab / nz_pppm;
    wz = powsinxx(argz, twoorder);

    for (l = nylo_fft; l <= nyhi_fft; l++) {
      lper = l - ny_pppm * (2 * l / ny_pppm);
      qy = unitky * lper;
      sny = square(sin(0.5 * qy * yprd_wire / ny_pppm));
      sy = exp(-0.25 * square(qy / g_ewald));
      argy = 0.5 * qy * yprd_wire / ny_pppm;
      wy = powsinxx(argy, twoorder);

      for (k = nxlo_fft; k <= nxhi_fft; k++) {
        kper = k - nx_pppm * (2 * k / nx_pppm);
        qx = unitkx * kper;
        snx = square(sin(0.5 * qx * xprd_wire / nx_pppm));
        sx = exp(-0.25 * square(qx / g_ewald));
        argx = 0.5 * qx * xprd_wire / nx_pppm;
        wx = powsinxx(argx, twoorder);

        sqk = qx * qx + qy * qy + qz * qz;

        if (sqk != 0.0) {
          numerator = MY_4PI / sqk;
          denominator = gf_denom(snx, sny, snz);
          greensfn[n] = numerator * sx * sy * sz * wx * wy * wz / denominator;
          sf_coeff[0] += sf_precoeff1[n] * greensfn[n];
          sf_coeff[1] += sf_precoeff2[n] * greensfn[n];
          sf_coeff[2] += sf_precoeff3[n] * greensfn[n];
          sf_coeff[3] += sf_precoeff4[n] * greensfn[n];
          sf_coeff[4] += sf_precoeff5[n] * greensfn[n];
          sf_coeff[5] += sf_precoeff6[n] * greensfn[n];
          n++;
        } else {
          greensfn[n] = 0.0;
          sf_coeff[0] += sf_precoeff1[n] * greensfn[n];
          sf_coeff[1] += sf_precoeff2[n] * greensfn[n];
          sf_coeff[2] += sf_precoeff3[n] * greensfn[n];
          sf_coeff[3] += sf_precoeff4[n] * greensfn[n];
          sf_coeff[4] += sf_precoeff5[n] * greensfn[n];
          sf_coeff[5] += sf_precoeff6[n] * greensfn[n];
          n++;
        }
      }
    }
  }

  // compute the coefficients for the self-force correction

  double prex, prey, prez;
  prex = prey = prez = MY_PI / volume;
  prex *= nx_pppm / xprd_wire;
  prey *= ny_pppm / yprd_wire;
  prez *= nz_pppm / zprd_slab;
  sf_coeff[0] *= prex;
  sf_coeff[1] *= prex * 2;
  sf_coeff[2] *= prey;
  sf_coeff[3] *= prey * 2;
  sf_coeff[4] *= prez;
  sf_coeff[5] *= prez * 2;

  // communicate values with other procs

  double tmp[6];
  MPI_Allreduce(sf_coeff, tmp, 6, MPI_DOUBLE, MPI_SUM, world);
  for (n = 0; n < 6; n++) sf_coeff[n] = tmp[n];
}

/* ----------------------------------------------------------------------
   create discretized "density" of electrolyte particles (c.f. make_rho())
   density(x,y,z) = charge "density" at grid points of my 3d brick
   (nxlo:nxhi,nylo:nyhi,nzlo:nzhi) is extent of my brick (including
ghosts) in global grid
-------------------------------------------------------------------------
*/

void PPPMConp::make_electrolyte_rho(bigint *imat) {
  int l, m, n, nx, ny, nz, mx, my, mz;
  FFT_SCALAR dx, dy, dz, x0, y0, z0;

  // clear 3d density array

  memset(&(electrolyte_density_brick[nzlo_out][nylo_out][nxlo_out]), 0,
         ngrid * sizeof(FFT_SCALAR));

  // loop over my charges, add their contribution to nearby grid points
  // (nx,ny,nz) = global coords of grid pt to "lower left" of charge
  // (dx,dy,dz) = distance to "lower left" grid pt
  // (mx,my,mz) = global coords of moving stencil pt

  double *q = atom->q;
  double **x = atom->x;
  int nlocal = atom->nlocal;

  for (int i = 0; i < nlocal; i++) {
    if (imat[i] >= 0) continue;
    nx = part2grid[i][0];
    ny = part2grid[i][1];
    nz = part2grid[i][2];
    dx = nx + shiftone - (x[i][0] - boxlo[0]) * delxinv;
    dy = ny + shiftone - (x[i][1] - boxlo[1]) * delyinv;
    dz = nz + shiftone - (x[i][2] - boxlo[2]) * delzinv;

    compute_rho1d(dx, dy, dz);

    z0 = delvolinv * q[i];
    for (n = nlower; n <= nupper; n++) {
      mz = n + nz;
      y0 = z0 * rho1d[2][n];
      for (m = nlower; m <= nupper; m++) {
        my = m + ny;
        x0 = y0 * rho1d[1][m];
        for (l = nlower; l <= nupper; l++) {
          mx = l + nx;
          electrolyte_density_brick[mz][my][mx] += x0 * rho1d[0][l];
        }
      }
    }
  }
}

/* ----------------------------------------------------------------------
   interpolate from grid to get electric field & force on my particles
-------------------------------------------------------------------------
*/

void PPPMConp::fieldforce() {
  if (differentiation_flag == 1)
    fieldforce_ad();
  else
    fieldforce_ik();
}

/* ----------------------------------------------------------------------
   interpolate from grid to get electric field & force on my particles for
ik
-------------------------------------------------------------------------
*/

void PPPMConp::fieldforce_ik() {
  int i, l, m, n, nx, ny, nz, mx, my, mz;
  FFT_SCALAR dx, dy, dz, x0, y0, z0;
  FFT_SCALAR ekx, eky, ekz;

  // loop over my charges, interpolate electric field from nearby grid
  // points (nx,ny,nz) = global coords of grid pt to "lower left" of
  // charge (dx,dy,dz) = distance to "lower left" grid pt (mx,my,mz) =
  // global coords of moving stencil pt ek = 3 components of E-field on
  // particle

  double *q = atom->q;
  double **x = atom->x;
  double **f = atom->f;

  int nlocal = atom->nlocal;

  for (i = 0; i < nlocal; i++) {
    nx = part2grid[i][0];
    ny = part2grid[i][1];
    nz = part2grid[i][2];
    dx = nx + shiftone - (x[i][0] - boxlo[0]) * delxinv;
    dy = ny + shiftone - (x[i][1] - boxlo[1]) * delyinv;
    dz = nz + shiftone - (x[i][2] - boxlo[2]) * delzinv;

    compute_rho1d(dx, dy, dz);

    ekx = eky = ekz = ZEROF;
    for (n = nlower; n <= nupper; n++) {
      mz = n + nz;
      z0 = rho1d[2][n];
      for (m = nlower; m <= nupper; m++) {
        my = m + ny;
        y0 = z0 * rho1d[1][m];
        for (l = nlower; l <= nupper; l++) {
          mx = l + nx;
          x0 = y0 * rho1d[0][l];
          ekx -= x0 * vdx_brick[mz][my][mx];
          eky -= x0 * vdy_brick[mz][my][mx];
          ekz -= x0 * vdz_brick[mz][my][mx];
        }
      }
    }

    // convert E-field to force

    const double qfactor = qqrd2e * scale * q[i];
    if (wireflag != 2) {
      f[i][0] += qfactor * ekx;
      f[i][1] += qfactor * eky;
    }
    if (slabflag != 2) f[i][2] += qfactor * ekz;
  }
}

/* ----------------------------------------------------------------------
   interpolate from grid to get electric field & force on my particles for
ad
-------------------------------------------------------------------------
*/

void PPPMConp::fieldforce_ad() {
  int i, l, m, n, nx, ny, nz, mx, my, mz;
  FFT_SCALAR dx, dy, dz;
  FFT_SCALAR ekx, eky, ekz;
  double s1, s2, s3;
  double sf = 0.0;
  double *prd;

  prd = domain->prd;
  double xprd = prd[0];
  double yprd = prd[1];
  double zprd = prd[2];

  double hx_inv = nx_pppm / xprd;
  double hy_inv = ny_pppm / yprd;
  double hz_inv = nz_pppm / zprd;

  // loop over my charges, interpolate electric field from nearby grid
  // points (nx,ny,nz) = global coords of grid pt to "lower left" of
  // charge (dx,dy,dz) = distance to "lower left" grid pt (mx,my,mz) =
  // global coords of moving stencil pt ek = 3 components of E-field on
  // particle

  double *q = atom->q;
  double **x = atom->x;
  double **f = atom->f;

  int nlocal = atom->nlocal;

  for (i = 0; i < nlocal; i++) {
    nx = part2grid[i][0];
    ny = part2grid[i][1];
    nz = part2grid[i][2];
    dx = nx + shiftone - (x[i][0] - boxlo[0]) * delxinv;
    dy = ny + shiftone - (x[i][1] - boxlo[1]) * delyinv;
    dz = nz + shiftone - (x[i][2] - boxlo[2]) * delzinv;

    compute_rho1d(dx, dy, dz);
    compute_drho1d(dx, dy, dz);

    ekx = eky = ekz = ZEROF;
    for (n = nlower; n <= nupper; n++) {
      mz = n + nz;
      for (m = nlower; m <= nupper; m++) {
        my = m + ny;
        for (l = nlower; l <= nupper; l++) {
          mx = l + nx;
          ekx += drho1d[0][l] * rho1d[1][m] * rho1d[2][n] * u_brick[mz][my][mx];
          eky += rho1d[0][l] * drho1d[1][m] * rho1d[2][n] * u_brick[mz][my][mx];
          ekz += rho1d[0][l] * rho1d[1][m] * drho1d[2][n] * u_brick[mz][my][mx];
        }
      }
    }
    ekx *= hx_inv;
    eky *= hy_inv;
    ekz *= hz_inv;

    // convert E-field to force and subtract self forces

    const double qfactor = qqrd2e * scale;

    s1 = x[i][0] * hx_inv;
    s2 = x[i][1] * hy_inv;
    s3 = x[i][2] * hz_inv;
    sf = sf_coeff[0] * sin(2 * MY_PI * s1);
    sf += sf_coeff[1] * sin(4 * MY_PI * s1);
    sf *= 2 * q[i] * q[i];
    if (wireflag != 2) f[i][0] += qfactor * (ekx * q[i] - sf);

    sf = sf_coeff[2] * sin(2 * MY_PI * s2);
    sf += sf_coeff[3] * sin(4 * MY_PI * s2);
    sf *= 2 * q[i] * q[i];
    if (wireflag != 2) f[i][1] += qfactor * (eky * q[i] - sf);

    sf = sf_coeff[4] * sin(2 * MY_PI * s3);
    sf += sf_coeff[5] * sin(4 * MY_PI * s3);
    sf *= 2 * q[i] * q[i];
    if (slabflag != 2) f[i][2] += qfactor * (ekz * q[i] - sf);
  }
}

/* ----------------------------------------------------------------------
   Slab-geometry correction term to dampen inter-slab interactions between
   periodically repeating slabs.  Yields good approximation to 2D Ewald if
   adequate empty space is left between repeating slabs (J. Chem. Phys.
   111, 3155).  Slabs defined here to be parallel to the xy plane. Also
   extended to non-neutral systems (J. Chem. Phys. 131, 094107).
-------------------------------------------------------------------------
*/

void PPPMConp::slabcorr() {
  // compute local contribution to global dipole moment

  double *q = atom->q;
  double **x = atom->x;
  double zprd = domain->zprd;
  int nlocal = atom->nlocal;

  double dipole = 0.0;
  for (int i = 0; i < nlocal; i++) dipole += q[i] * x[i][2];

  // sum local contributions to get global dipole moment

  double dipole_all;
  MPI_Allreduce(&dipole, &dipole_all, 1, MPI_DOUBLE, MPI_SUM, world);

  // need to make non-neutral systems and/or
  //  per-atom energy translationally invariant

  double dipole_r2 = 0.0;
  if (eflag_atom || fabs(qsum) > SMALL) {
    for (int i = 0; i < nlocal; i++) dipole_r2 += q[i] * x[i][2] * x[i][2];

    // sum local contributions

    double tmp;
    MPI_Allreduce(&dipole_r2, &tmp, 1, MPI_DOUBLE, MPI_SUM, world);
    dipole_r2 = tmp;
  }

  // compute corrections

  const double e_slabcorr = MY_2PI *
                            (dipole_all * dipole_all - qsum * dipole_r2 -
                             qsum * qsum * zprd * zprd / 12.0) /
                            volume;
  const double qscale = qqrd2e * scale;

  if (eflag_global) energy += qscale * e_slabcorr;

  // per-atom energy

  if (eflag_atom) {
    double efact = qscale * MY_2PI / volume;
    for (int i = 0; i < nlocal; i++)
      eatom[i] +=
          efact * q[i] *
          (x[i][2] * dipole_all - 0.5 * (dipole_r2 + qsum * x[i][2] * x[i][2]) -
           qsum * zprd * zprd / 12.0);
  }

  // add on force corrections

  double ffact = qscale * (-4.0 * MY_PI / volume);
  double **f = atom->f;

  for (int i = 0; i < nlocal; i++)
    f[i][2] += ffact * q[i] * (dipole_all - qsum * x[i][2]);
}

/* ----------------------------------------------------------------------
   Wire-geometry correction term to dampen inter-wire interactions between
   periodically repeating wires.  Yields good approximation to 1D Ewald if
   adequate empty space is left between repeating wires (J. Mol. Struct.
   704, 101). x and y are non-periodic.
-------------------------------------------------------------------------
*/

void PPPMConp::wirecorr() {
  // compute local contribution to global dipole moment

  double *q = atom->q;
  double **x = atom->x;
  int nlocal = atom->nlocal;

  double xdipole = 0.0;
  double ydipole = 0.0;
  for (int i = 0; i < nlocal; i++) {
    xdipole += q[i] * x[i][0];
    ydipole += q[i] * x[i][1];
  }

  // sum local contributions to get global dipole moment

  double xdipole_all, ydipole_all;
  MPI_Allreduce(&xdipole, &xdipole_all, 1, MPI_DOUBLE, MPI_SUM, world);
  MPI_Allreduce(&ydipole, &ydipole_all, 1, MPI_DOUBLE, MPI_SUM, world);

  // need to make per-atom energy translationally invariant

  double xdipole_r2 = 0.0;
  double ydipole_r2 = 0.0;
  if (eflag_atom) {
    for (int i = 0; i < nlocal; i++) {
      xdipole_r2 += q[i] * x[i][0] * x[i][0];
      ydipole_r2 += q[i] * x[i][1] * x[i][1];
    }

    // sum local contributions

    double tmp;
    MPI_Allreduce(&xdipole_r2, &tmp, 1, MPI_DOUBLE, MPI_SUM, world);
    xdipole_r2 = tmp;
    MPI_Allreduce(&ydipole_r2, &tmp, 1, MPI_DOUBLE, MPI_SUM, world);
    ydipole_r2 = tmp;
  }

  // compute corrections

  const double e_wirecorr =
      MY_PI * (xdipole_all * xdipole_all + ydipole_all * ydipole_all) / volume;
  const double qscale = qqrd2e * scale;

  if (eflag_global) energy += qscale * e_wirecorr;

  // per-atom energy

  if (eflag_atom) {
    double efact = qscale * MY_PI / volume;
    for (int i = 0; i < nlocal; i++)
      eatom[i] += efact * q[i] *
                  (x[i][0] * xdipole_all + x[i][1] * ydipole_all -
                   0.5 * (xdipole_r2 + ydipole_r2));
  }

  // add on force corrections

  double ffact = qscale * (-MY_2PI / volume);
  double **f = atom->f;

  for (int i = 0; i < nlocal; i++) {
    f[i][0] += ffact * q[i] * xdipole_all;
    f[i][1] += ffact * q[i] * ydipole_all;
  }
}

/* ----------------------------------------------------------------------
   group-group interactions
 -------------------------------------------------------------------------
*/

/* ----------------------------------------------------------------------
   compute the PPPM total long-range force and energy for groups A and B
 -------------------------------------------------------------------------
*/

void PPPMConp::compute_group_group(int groupbit_A, int groupbit_B,
                                   int AA_flag) {
  if ((wireflag || slabflag) && triclinic)
    error->all(FLERR,
               "Cannot (yet) use K-space slab or wire "
               "correction with compute group/group for triclinic systems");

  if (differentiation_flag)
    error->all(FLERR,
               "Cannot (yet) use kspace_modify "
               "diff ad with compute group/group");

  if (!group_allocate_flag) allocate_groups();

  // convert atoms from box to lamda coords

  if (triclinic == 0)
    boxlo = domain->boxlo;
  else {
    boxlo = domain->boxlo_lamda;
    domain->x2lamda(atom->nlocal);
  }

  e2group = 0.0;     // energy
  f2group[0] = 0.0;  // force in x-direction
  f2group[1] = 0.0;  // force in y-direction
  f2group[2] = 0.0;  // force in z-direction

  // map my particle charge onto my local 3d density grid

  make_rho_groups(groupbit_A, groupbit_B, AA_flag);

  // all procs communicate density values from their ghost cells
  //   to fully sum contribution in their 3d bricks
  // remap from 3d decomposition to FFT decomposition

  // temporarily store and switch pointers so we can
  //  use brick2fft() for groups A and B (without
  //  writing an additional function)

  FFT_SCALAR ***density_brick_real = density_brick;
  FFT_SCALAR *density_fft_real = density_fft;

  // group A

  density_brick = density_A_brick;
  density_fft = density_A_fft;

  gc->reverse_comm_kspace(this, 1, sizeof(FFT_SCALAR), REVERSE_RHO, gc_buf1,
                          gc_buf2, MPI_FFT_SCALAR);
  brick2fft();

  // group B

  density_brick = density_B_brick;
  density_fft = density_B_fft;

  gc->reverse_comm_kspace(this, 1, sizeof(FFT_SCALAR), REVERSE_RHO, gc_buf1,
                          gc_buf2, MPI_FFT_SCALAR);
  brick2fft();

  // switch back pointers

  density_brick = density_brick_real;
  density_fft = density_fft_real;

  // compute potential gradient on my FFT grid and
  //   portion of group-group energy/force on this proc's FFT grid

  poisson_groups(AA_flag);

  const double qscale = qqrd2e * scale;

  // total group A <--> group B energy
  // self and boundary correction terms are in compute_group_group.cpp

  double e2group_all;
  MPI_Allreduce(&e2group, &e2group_all, 1, MPI_DOUBLE, MPI_SUM, world);
  e2group = e2group_all;

  e2group *= qscale * 0.5 * volume;

  // total group A <--> group B force

  double f2group_all[3];
  MPI_Allreduce(f2group, f2group_all, 3, MPI_DOUBLE, MPI_SUM, world);

  f2group[0] = qscale * volume * f2group_all[0];
  if (wireflag != 2) {
    f2group[0] = qscale * volume * f2group_all[0];
    f2group[1] = qscale * volume * f2group_all[1];
  }
  if (slabflag != 2) f2group[2] = qscale * volume * f2group_all[2];

  // convert atoms back from lamda to box coords

  if (triclinic) domain->lamda2x(atom->nlocal);

  if (slabflag == 1) slabcorr_groups(groupbit_A, groupbit_B, AA_flag);

  if (wireflag == 1) wirecorr_groups(groupbit_A, groupbit_B, AA_flag);
}

/* ----------------------------------------------------------------------
   slab-geometry correction term to dampen inter-slab interactions between
   periodically repeating slabs.  Yields good approximation to 2D Ewald if
   adequate empty space is left between repeating slabs (J. Chem. Phys.
   111, 3155).  Slabs defined here to be parallel to the xy plane. Also
   extended to non-neutral systems (J. Chem. Phys. 131, 094107).
-------------------------------------------------------------------------
*/

void PPPMConp::slabcorr_groups(int groupbit_A, int groupbit_B, int AA_flag) {
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
      dipole_A += q[i] * x[i][2];
      dipole_r2_A += q[i] * x[i][2] * x[i][2];
    }

    if (mask[i] & groupbit_B) {
      qsum_B += q[i];
      dipole_B += q[i] * x[i][2];
      dipole_r2_B += q[i] * x[i][2] * x[i][2];
    }
  }

  // sum local contributions to get total charge and global dipole moment
  //  for each group

  double tmp;
  MPI_Allreduce(&qsum_A, &tmp, 1, MPI_DOUBLE, MPI_SUM, world);
  qsum_A = tmp;

  MPI_Allreduce(&qsum_B, &tmp, 1, MPI_DOUBLE, MPI_SUM, world);
  qsum_B = tmp;

  MPI_Allreduce(&dipole_A, &tmp, 1, MPI_DOUBLE, MPI_SUM, world);
  dipole_A = tmp;

  MPI_Allreduce(&dipole_B, &tmp, 1, MPI_DOUBLE, MPI_SUM, world);
  dipole_B = tmp;

  MPI_Allreduce(&dipole_r2_A, &tmp, 1, MPI_DOUBLE, MPI_SUM, world);
  dipole_r2_A = tmp;

  MPI_Allreduce(&dipole_r2_B, &tmp, 1, MPI_DOUBLE, MPI_SUM, world);
  dipole_r2_B = tmp;

  // compute corrections

  const double qscale = qqrd2e * scale;
  const double efact = qscale * MY_2PI / volume;

  e2group += efact * (dipole_A * dipole_B -
                      0.5 * (qsum_A * dipole_r2_B + qsum_B * dipole_r2_A) -
                      qsum_A * qsum_B * zprd * zprd / 12.0);

  // add on force corrections

  const double ffact = qscale * (-4.0 * MY_PI / volume);
  f2group[2] += ffact * (qsum_A * dipole_B - qsum_B * dipole_A);
}

/* ----------------------------------------------------------------------
   TODO wire-geometry correction term to dampen inter-wire interactions
between periodically repeating wires.  Yields good approximation to 2D
Ewald if adequate empty space is left between repeating wires (J. Chem.
Phys. 111, 3155).  Wires defined here to be parallel to the x axis. Also
extended to non-neutral systems (J. Chem. Phys. 131, 094107).
-------------------------------------------------------------------------
*/

void PPPMConp::wirecorr_groups(int groupbit_A, int groupbit_B, int AA_flag) {
  // compute local contribution to global dipole moment

  double *q = atom->q;
  double **x = atom->x;
  double yprd = domain->yprd;
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
      dipole_A += q[i] * x[i][1];
      dipole_r2_A += q[i] * x[i][1] * x[i][1];
    }

    if (mask[i] & groupbit_B) {
      qsum_B += q[i];
      dipole_B += q[i] * x[i][1];
      dipole_r2_B += q[i] * x[i][1] * x[i][1];
    }
  }

  // sum local contributions to get total charge and global dipole moment
  //  for each group

  double tmp;
  MPI_Allreduce(&qsum_A, &tmp, 1, MPI_DOUBLE, MPI_SUM, world);
  qsum_A = tmp;

  MPI_Allreduce(&qsum_B, &tmp, 1, MPI_DOUBLE, MPI_SUM, world);
  qsum_B = tmp;

  MPI_Allreduce(&dipole_A, &tmp, 1, MPI_DOUBLE, MPI_SUM, world);
  dipole_A = tmp;

  MPI_Allreduce(&dipole_B, &tmp, 1, MPI_DOUBLE, MPI_SUM, world);
  dipole_B = tmp;

  MPI_Allreduce(&dipole_r2_A, &tmp, 1, MPI_DOUBLE, MPI_SUM, world);
  dipole_r2_A = tmp;

  MPI_Allreduce(&dipole_r2_B, &tmp, 1, MPI_DOUBLE, MPI_SUM, world);
  dipole_r2_B = tmp;

  // compute corrections

  const double qscale = qqrd2e * scale;
  const double efact = qscale * MY_PI / volume;

  e2group += efact * (dipole_A * dipole_B -
                      0.5 * (qsum_A * dipole_r2_B + qsum_B * dipole_r2_A) -
                      qsum_A * qsum_B * yprd * yprd / 12.0);

  // add on force corrections

  const double ffact = qscale * (-2.0 * MY_PI / volume);
  f2group[1] += ffact * (qsum_A * dipole_B - qsum_B * dipole_A);
}

void PPPMConp::compute_matrix_corr(bigint *imat, double **matrix) {
  // copied from ewald_conp
  if (slabflag && triclinic)
    error->all(FLERR,
               "Cannot (yet) use K-space slab "
               "correction with compute coul/matrix for triclinic systems");

  int nprocs = comm->nprocs;
  int nlocal = atom->nlocal;
  double xprd = domain->xprd;
  double yprd = domain->yprd;
  double zprd = domain->zprd;
  double xprd_wire = xprd * wire_volfactor;
  double yprd_wire = yprd * wire_volfactor;
  double zprd_slab = zprd * slab_volfactor;
  double area = xprd_wire * yprd_wire;
  double vol = xprd_wire * yprd_wire * zprd_slab;

  bigint *jmat, *jmat_local;

  double **x = atom->x;

  // how many local and total group atoms?

  bigint ngroup = 0;

  int ngrouplocal = 0;
  for (int i = 0; i < nlocal; i++)
    if (imat[i] > -1) ngrouplocal++;
  MPI_Allreduce(&ngrouplocal, &ngroup, 1, MPI_INT, MPI_SUM, world);

  memory->create(jmat, ngroup, "pppm/conp:jmat");
  memory->create(jmat_local, ngrouplocal, "pppm/conp:jmat_local");

  for (int i = 0, n = 0; i < nlocal; i++) {
    if (imat[i] < 0) continue;

    // ... keep track of matrix index

    jmat_local[n] = imat[i];

    n++;
  }

  int *recvcounts, *displs;

  memory->create(recvcounts, nprocs, "pppm/conp:recvcounts");
  memory->create(displs, nprocs, "pppm/conp:displs");

  MPI_Allgather(&ngrouplocal, 1, MPI_INT, recvcounts, 1, MPI_INT, world);
  displs[0] = 0;
  for (int i = 1; i < nprocs; i++)
    displs[i] = displs[i - 1] + recvcounts[i - 1];

  // gather global matrix indexing

  MPI_Allgatherv(jmat_local, ngrouplocal, MPI_LMP_BIGINT, jmat, recvcounts,
                 displs, MPI_LMP_BIGINT, world);

  if (slabflag) {
    double *nprd_local, *nprd_all;

    memory->create(nprd_all, ngroup, "pppm/conp:nprd_all");
    memory->create(nprd_local, ngrouplocal, "pppm/conp:nprd_local");

    for (int i = 0, n = 0; i < nlocal; i++) {
      if (imat[i] < 0) continue;

      // gather non-periodic positions of groups

      nprd_local[n] = x[i][2];

      n++;
    }

    // gather subsets nprd positions

    MPI_Allgatherv(nprd_local, ngrouplocal, MPI_DOUBLE, nprd_all, recvcounts,
                   displs, MPI_DOUBLE, world);
    memory->destroy(nprd_local);

    double aij;

    if (slabflag == 1) {
      // use EW3DC slab correction on subset

      const double prefac = MY_4PI / vol;
      for (int i = 0; i < nlocal; i++) {
        if (imat[i] < 0) continue;
        for (bigint j = 0; j < ngroup; j++) {
          // matrix is symmetric
          if (jmat[j] > imat[i]) continue;
          aij = prefac * x[i][2] * nprd_all[j];

          // TODO add ELC corrections, needs sum over all kpoints but not
          // (0,0)

          matrix[imat[i]][jmat[j]] += aij;
          if (imat[i] != jmat[j]) matrix[jmat[j]][imat[i]] += aij;
        }
      }

    } else if (slabflag == 3) {
      // use EW2D infinite boundary correction

      const double g_ewald_inv = 1.0 / g_ewald;
      const double g_ewald_sq = g_ewald * g_ewald;
      const double prefac = 2.0 * MY_PIS / area;

      double dij;
      for (int i = 0; i < nlocal; i++) {
        if (imat[i] < 0) continue;
        for (bigint j = 0; j < ngroup; j++) {
          // matrix is symmetric
          if (jmat[j] > imat[i]) continue;
          dij = nprd_all[j] - x[i][2];
          // resembles (aij) matrix component in constant potential
          aij = prefac * (exp(-dij * dij * g_ewald_sq) * g_ewald_inv +
                          MY_PIS * dij * erf(dij * g_ewald));
          matrix[imat[i]][jmat[j]] -= aij;
          if (imat[i] != jmat[j]) matrix[jmat[j]][imat[i]] -= aij;
        }
      }
    }

    memory->destroy(nprd_all);
  } else if (wireflag) {
    // use EW3DC wire correction on subset

    double *xprd_local, *xprd_all;
    double *yprd_local, *yprd_all;

    memory->create(xprd_all, ngroup, "pppm/conp:xprd_all");
    memory->create(yprd_all, ngroup, "pppm/conp:yprd_all");
    memory->create(xprd_local, ngrouplocal, "pppm/conp:xprd_local");
    memory->create(yprd_local, ngrouplocal, "pppm/conp:yprd_local");

    for (int i = 0, n = 0; i < nlocal; i++) {
      if (imat[i] < 0) continue;

      // gather non-periodic positions of groups

      xprd_local[n] = x[i][0];
      yprd_local[n] = x[i][1];

      n++;
    }

    // gather subsets nprd positions

    MPI_Allgatherv(xprd_local, ngrouplocal, MPI_DOUBLE, xprd_all, recvcounts,
                   displs, MPI_DOUBLE, world);
    MPI_Allgatherv(yprd_local, ngrouplocal, MPI_DOUBLE, yprd_all, recvcounts,
                   displs, MPI_DOUBLE, world);
    memory->destroy(xprd_local);
    memory->destroy(yprd_local);

    double aij;

    const double prefac = MY_2PI / volume;
    for (int i = 0; i < nlocal; i++) {
      if (imat[i] < 0) continue;
      for (bigint j = 0; j < ngroup; j++) {
        // matrix is symmetric
        if (jmat[j] > imat[i]) continue;
        aij = prefac * (x[i][0] * xprd_all[j] + x[i][1] * yprd_all[j]);

        matrix[imat[i]][jmat[j]] += aij;
        if (imat[i] != jmat[j]) matrix[jmat[j]][imat[i]] += aij;
      }
    }

    memory->destroy(xprd_all);
    memory->destroy(yprd_all);
  }

  memory->destroy(recvcounts);
  memory->destroy(displs);
  memory->destroy(jmat);
  memory->destroy(jmat_local);
}

/* ----------------------------------------------------------------------
   compute b-vector EW3DC correction of constant potential approach
 ------------------------------------------------------------------------- */

void PPPMConp::compute_vector_corr(bigint *imat, double *vec) {
  int const nlocal = atom->nlocal;
  double **x = atom->x;
  double *q = atom->q;
  if (slabflag == 1) {
    // use EW3DC slab correction
    double dipole = 0.;
    for (int i = 0; i < nlocal; i++) {
      if (imat[i] < 0) dipole += q[i] * x[i][2];
    }
    MPI_Allreduce(MPI_IN_PLACE, &dipole, 1, MPI_DOUBLE, MPI_SUM, world);
    dipole *= 4.0 * MY_PI / volume;
    for (int i = 0; i < nlocal; i++) {
      int const pos = imat[i];
      if (pos >= 0) vec[pos] += x[i][2] * dipole;
    }
  } else if (slabflag == 3) {
    error->all(FLERR, "Cannot (yet) use PPPM CONP with EW2D ");
    // use EW2D infinite boundary correction
  } else if (wireflag) {
    error->all(FLERR, "Cannot (yet) use PPPM CONP with wire ");
  }
}
