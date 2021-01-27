
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

#include "compute_conp_vector.h"

#include "atom.h"
#include "comm.h"
#include "error.h"
#include "force.h"
#include "group.h"
#include "kspace.h"
#include "memory.h"
#include "neigh_list.h"
#include "neigh_request.h"
#include "neighbor.h"
#include "pair.h"

using namespace LAMMPS_NS;

#define EWALD_P 0.3275911
#define A1 0.254829592
#define A2 -0.284496736
#define A3 1.421413741
#define A4 -1.453152027
#define A5 1.061405429

ComputeConpVector::ComputeConpVector(LAMMPS *lmp, int narg, char **arg)
    : Compute(lmp, narg, arg), group2(nullptr), mpos(nullptr), fp(nullptr) {
  if (narg < 4) error->all(FLERR, "Illegal compute coul/vector command");

  vector_flag = 1;
  size_array_cols = 0;
  size_array_rows = 0;
  size_array_rows_variable = 0;
  extarray = 0;

  fp = nullptr;
  vector = nullptr;
  mpos = nullptr;

  pairflag = 1;
  kspaceflag = 1;
  boundaryflag = 1;  // include infite boundary correction term
  gaussians = 1;
  recalc_every = 0;
  overwrite = 1;
  assigned = false;

  g_ewald = 0.0;

  // get jgroup; igroup defined in parent class

  int n = strlen(arg[3]) + 1;
  group2 = new char[n];
  strcpy(group2, arg[3]);

  jgroup = group->find(group2);
  if (jgroup == -1)
    error->all(FLERR, "Compute coul/vector group ID does not exist");
  jgroupbit = group->bitmask[jgroup];

  // TODO recalculate coulomb vector every recalc_every

  recalc_every = utils::inumeric(FLERR, arg[4], false, lmp);
  eta =
      utils::numeric(FLERR, arg[5], false, lmp);  // TODO infer from pair_style!

  int iarg = 6;
  while (iarg < narg) {
    if (strcmp(arg[iarg], "pair") == 0) {
      if (iarg + 2 > narg)
        error->all(FLERR, "Illegal compute coul/vector command");
      if (strcmp(arg[iarg + 1], "yes") == 0)
        pairflag = 1;
      else if (strcmp(arg[iarg + 1], "no") == 0)
        pairflag = 0;
      else
        error->all(FLERR, "Illegal compute coul/vector command");
      iarg += 2;
    } else if (strcmp(arg[iarg], "kspace") == 0) {
      if (iarg + 2 > narg)
        error->all(FLERR, "Illegal compute coul/vector command");
      if (strcmp(arg[iarg + 1], "yes") == 0)
        kspaceflag = 1;
      else if (strcmp(arg[iarg + 1], "no") == 0)
        kspaceflag = 0;
      else
        error->all(FLERR, "Illegal compute coul/vector command");
      iarg += 2;
    } else if (strcmp(arg[iarg], "boundary") == 0) {
      if (iarg + 2 > narg)
        error->all(FLERR, "Illegal compute coul/vector command");
      if (strcmp(arg[iarg + 1], "yes") == 0)
        boundaryflag = 1;
      else if (strcmp(arg[iarg + 1], "no") == 0)
        boundaryflag = 0;
      else
        error->all(FLERR, "Illegal compute coul/vector command");
      iarg += 2;
    } else if (strcmp(arg[iarg], "overwrite") ==
               0) {  // TODO  if vector is recalculated overwrite or append
                     // output
      if (iarg + 2 > narg)
        error->all(FLERR, "Illegal compute coul/vector command");
      if (strcmp(arg[iarg + 1], "yes") == 0)
        overwrite = 1;
      else if (strcmp(arg[iarg + 1], "no") == 0)
        overwrite = 0;
      else
        error->all(FLERR, "Illegal compute coul/vector command");
      iarg += 2;
    } else if (strcmp(arg[iarg], "file") == 0) {
      if (iarg + 2 > narg)
        error->all(FLERR, "Illegal compute coul/vector command");
      if (comm->me == 0) {
        fp = fopen(arg[iarg + 1], "w");
        if (fp == nullptr)
          error->one(FLERR,
                     fmt::format("Cannot open compute coul/vector file {}: {}",
                                 arg[iarg + 1], utils::getsyserror()));
      }
      iarg += 2;
    } else
      error->all(FLERR, "Illegal compute coul/vector command");
  }

  // print file comment lines

  if (fp && comm->me == 0) {
    clearerr(fp);
    fprintf(fp, "# Constant potential coulomb vector\n");
    if (ferror(fp)) error->one(FLERR, "Error writing file header");
    filepos = ftell(fp);
  }
}

/* ---------------------------------------------------------------------- */

ComputeConpVector::~ComputeConpVector() {
  delete[] group2;

  deallocate();

  if (assigned) {
    memory->destroy(mpos);
  }

  if (fp && comm->me == 0) fclose(fp);
}

/* ---------------------------------------------------------------------- */

void ComputeConpVector::init_list(int /*id*/, NeighList *ptr) { list = ptr; }

/* ---------------------------------------------------------------------- */

void ComputeConpVector::init() {
  // if non-hybrid, then error if single_enable = 0
  // if hybrid, let hybrid determine if sub-style sets single_enable = 0

  // error if Kspace style does not compute coul/vectr interactions

  if ((boundaryflag || kspaceflag) && force->kspace == nullptr)
    error->all(FLERR, "No Kspace style defined for compute coul/vectr");

  // TODO need another flag since we don't use compute_group_group()
  if (kspaceflag && force->kspace->group_group_enable == 0)
    error->all(FLERR, "Kspace style does not support compute coul/vector");

  // check if coul pair style is active, no need for single() since done
  // explicitly

  if (pairflag) {
    int itmp;
    double *p_cutoff = (double *)force->pair->extract("cut_coul", itmp);
    if (p_cutoff == nullptr)
      error->all(FLERR, "compute coul/vector is incompatible with Pair style");
    pair = force->pair;
    cutsq = force->pair->cutsq;
  } else
    pair = nullptr;

  if (boundaryflag || kspaceflag) {
    kspace = force->kspace;
    g_ewald = force->kspace->g_ewald;
  } else
    kspace = nullptr;

  // recheck that group 2 has not been deleted

  jgroup = group->find(group2);
  if (jgroup == -1)
    error->all(FLERR, "Compute coul/vector group ID does not exist");
  jgroupbit = group->bitmask[jgroup];

  // need an occasional half neighbor list

  if (pairflag) {
    int irequest = neighbor->request(this, instance_me);
    neighbor->requests[irequest]->pair = 0;
    neighbor->requests[irequest]->compute = 1;
    neighbor->requests[irequest]->occasional = 1;
  }
}

/* ---------------------------------------------------------------------- */

void ComputeConpVector::setup() {
  igroupnum = group->count(igroup);
  jgroupnum = group->count(jgroup);
  ngroup = igroupnum + jgroupnum;

  // TODO could be useful to assign homogenously all atoms in both groups to
  // all procs for calculating vector to distribute evenly the workload

  // TODO would be nicer to have a local vector and gather it later
  // to reduce a little bit the memory consumption ... However, for
  // this wo work I think the atom ids must be from 1,...,N and consecutive

  allocate();

  // assign atom tags to vector locations and vice versa

  matrix_assignment();

  // initial calculation of coulomb matrix at setup of simulation

  compute_vector();
  MPI_Allreduce(MPI_IN_PLACE, vector, ngroup, MPI_DOUBLE, MPI_SUM, world);

  if (fp && comm->me == 0) write_vector(fp, vector);
}
/* ---------------------------------------------------------------------- */

void ComputeConpVector::compute_vector() {
  for (int i = 0; i < ngroup; i++) vector[i] = 0.;
  if (pairflag) pair_contribution();
  if (kspaceflag) kspace->compute_vector(mpos, vector);
  if (boundaryflag) kspace->compute_vector_corr(mpos, vector);
}

/* ---------------------------------------------------------------------- */

void ComputeConpVector::pair_contribution() {
  double **x = atom->x;
  double *q = atom->q;
  int *type = atom->type;
  int *mask = atom->mask;
  neighbor->build_one(list);
  int const nlocal = atom->nlocal;
  int const nghost = atom->nghost;
  int const inum = list->inum;
  int *ilist = list->ilist;
  int *numneigh = list->numneigh;
  int **firstneigh = list->firstneigh;

  for (int ii = 0; ii < inum; ii++) {
    int const i = ilist[ii];
    bool const i_in_electrode = (mask[i] & groupbit || mask[i] & jgroupbit);
    double const xtmp = x[i][0];
    double const ytmp = x[i][1];
    double const ztmp = x[i][2];
    int itype = type[i];
    int *jlist = firstneigh[i];
    int jnum = numneigh[i];
    for (int jj = 0; jj < jnum; jj++) {
      int const j = jlist[jj] & NEIGHMASK;
      bool const j_in_electrode = (mask[j] & groupbit || mask[j] & jgroupbit);
      if (i_in_electrode == j_in_electrode) continue;

      double const delx = xtmp - x[j][0];  // neighlists take care of pbc
      double const dely = ytmp - x[j][1];
      double const delz = ztmp - x[j][2];
      double const rsq = delx * delx + dely * dely + delz * delz;
      int jtype = type[j];
      if (rsq >= cutsq[itype][jtype]) continue;
      double const r = sqrt(rsq);
      double const rinv = 1.0 / r;
      double aij = rinv;
      if (kspaceflag || boundaryflag) {
        aij *= calc_erfc(g_ewald * r);
        // TODO real-space gaussians?
        if (gaussians) {
          // TODO infer eta from coeffs of pair coul/long/gauss
          aij -= calc_erfc(eta * r) * rinv;
        }
      }
      if (i_in_electrode && (i < nlocal)) {
        vector[mpos[i]] += aij * q[j];
      } else if (j_in_electrode && (j < nghost)) {
        vector[mpos[j]] += aij * q[i];
      }
    }
  }
}
/* ---------------------------------------------------------------------- */
void ComputeConpVector::matrix_assignment() {
  // assign local matrix indices to local atoms on each proc

  int *mask = atom->mask;
  int nlocal = atom->nlocal;
  int nghost = atom->nghost;
  int nprocs = comm->nprocs;
  tagint *tag = atom->tag;
  int igroupnum_local = 0, jgroupnum_local = 0;

  for (int i = 0; i < nlocal; i++) {
    if (mask[i] & groupbit && mask[i] & jgroupbit)
      error->all(FLERR, "Same atom in both groups in compute coul/matrix");
    else if (mask[i] & groupbit)
      igroupnum_local++;
    else if (mask[i] & jgroupbit)
      jgroupnum_local++;
  }

  std::vector<int> idispls(nprocs);
  std::vector<int> jdispls(nprocs);
  std::vector<int> igroupnum_list(nprocs);
  std::vector<int> jgroupnum_list(nprocs);

  MPI_Allgather(&igroupnum_local, 1, MPI_INT, &igroupnum_list.front(), 1,
                MPI_INT, world);
  MPI_Allgather(&jgroupnum_local, 1, MPI_INT, &jgroupnum_list.front(), 1,
                MPI_INT, world);

  idispls[0] = jdispls[0] = 0;
  for (int i = 1; i < nprocs; i++) {
    idispls[i] = idispls[i - 1] + igroupnum_list[i - 1];
    jdispls[i] = jdispls[i - 1] + jgroupnum_list[i - 1];
  }

  std::vector<int> itaglist_local(igroupnum_local);
  std::vector<int> jtaglist_local(jgroupnum_local);

  igroupnum_local = jgroupnum_local = 0;
  for (int i = 0; i < nlocal; i++) {
    if (mask[i] & groupbit) {
      itaglist_local[igroupnum_local] = tag[i];
      igroupnum_local++;
    } else if (mask[i] & jgroupbit) {
      jtaglist_local[jgroupnum_local] = tag[i];
      jgroupnum_local++;
    }
  }

  std::vector<int> itaglist(igroupnum);
  std::vector<int> jtaglist(jgroupnum);

  MPI_Allgatherv(&itaglist_local.front(), igroupnum_local, MPI_LMP_TAGINT,
                 &itaglist.front(), &igroupnum_list.front(), &idispls.front(),
                 MPI_LMP_TAGINT, world);
  MPI_Allgatherv(&jtaglist_local.front(), jgroupnum_local, MPI_LMP_TAGINT,
                 &jtaglist.front(), &jgroupnum_list.front(), &jdispls.front(),
                 MPI_LMP_TAGINT, world);

  // sort individual group taglists, first igroup than jgroup

  std::sort(itaglist.begin(), itaglist.end());
  std::sort(jtaglist.begin(), jtaglist.end());

  // if local+ghost matrix assignment already created, recreate

  if (assigned) {
    memory->destroy(mpos);
  }
  memory->create(mpos, nghost, "coul/vector:mpos");

  assigned = true;

  // local+ghost non-matrix atoms are -1 in mpos

  for (int i = 0; i < nghost; i++) {
    mpos[i] = -1;
  }

  // store which tag represents value in matrix

  for (bigint i = 0; i < igroupnum; i++) mat2tag[i] = itaglist[i];
  for (bigint j = 0; j < jgroupnum; j++) mat2tag[igroupnum + j] = jtaglist[j];

  // create global matrix indices for local+ghost atoms
  for (bigint ii = 0; ii < ngroup; ii++) {
    for (int i = 0; i < nghost; i++) {
      if (mat2tag[ii] == tag[i]) {
        mpos[i] = ii;
      }
    }
  }
}

/* ---------------------------------------------------------------------- */

void ComputeConpVector::allocate() {
  memory->create(mat2tag, ngroup, "coul/vector:mat2tag");
  vector = new double[ngroup]();  // init to zero
}

/* ---------------------------------------------------------------------- */

void ComputeConpVector::deallocate() {
  memory->destroy(mat2tag);
  delete[] vector;
}

/* ---------------------------------------------------------------------- */
double ComputeConpVector::calc_erfc(double x) {
  double expm2 = exp(-x * x);
  double t = 1.0 / (1.0 + EWALD_P * x);
  return t * (A1 + t * (A2 + t * (A3 + t * (A4 + t * A5)))) * expm2;
}
/* ---------------------------------------------------------------------- */

void ComputeConpVector::write_vector(FILE *file, double *v) {
  for (bigint i = 0; i < ngroup; i++) {
    fprintf(file, "%d, %E\n", mat2tag[i], v[i]);
  }
  fprintf(file, "\n");
}
