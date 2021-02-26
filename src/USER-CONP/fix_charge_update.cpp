#include "fix_charge_update.h"

#include <assert.h>

#include <fstream>
#include <iostream>
#include <numeric>
#include <sstream>

#include "atom.h"
#include "comm.h"
#include "compute.h"
#include "compute_conp_matrix.h"
#include "compute_conp_vector.h"
#include "error.h"
#include "force.h"
#include "group.h"
#include "memory.h"
#include "modify.h"

using namespace LAMMPS_NS;
using namespace FixConst;
using namespace std;

extern "C" {
void dgetrf_(const int *M, const int *N, double *A, const int *lda, int *ipiv,
             int *info);
void dgetri_(const int *N, double *A, const int *lda, const int *ipiv,
             double *work, const int *lwork, int *info);
}

//     0        1      2             3    4
// fix fxupdate group1 charge_update pot1 eta couple group2 pot2
FixChargeUpdate::FixChargeUpdate(LAMMPS *lmp, int narg, char **arg)
    : Fix(lmp, narg, arg) {
  // array_compute = vector_compute = nullptr;
  f_inv = f_mat = f_vec = nullptr;
  read_inv = read_mat = false;
  symm = false;

  // read fix command
  groups = std::vector<int>(1, igroup);
  group_bits = std::vector<int>(1, groupbit);
  group_psi = std::vector<double>(1, utils::numeric(FLERR, arg[3], false, lmp));
  char *eta = arg[4];  // TODO set via pair
  int iarg = 5;
  while (iarg < narg) {
    if ((strcmp(arg[iarg], "couple") == 0)) {
      if (iarg + 3 > narg)
        error->all(FLERR, "Need two arguments after couple keyword");
      int id = group->find(arg[++iarg]);
      if (id < 0) error->all(FLERR, "Group does not exist");
      groups.push_back(id);
      group_bits.push_back(group->bitmask[id]);
      group_psi.push_back(utils::numeric(FLERR, arg[++iarg], false, lmp));
    } else if ((strncmp(arg[iarg], "symm", 4) == 0)) {
      if (iarg + 2 > narg)
        error->all(FLERR, "Need yes/no command after symm keyword");
      char *symm_arg = arg[++iarg];
      if ((strcmp(symm_arg, "yes") == 0) || (strcmp(symm_arg, "on") == 0)) {
        symm = true;
      } else if ((strcmp(symm_arg, "no") == 0) ||
                 (strcmp(symm_arg, "off") == 0)) {
        symm = false;
      } else {
        error->all(FLERR, "Invalid argument after symm keyword");
      }

    } else if ((strncmp(arg[iarg], "write", 4) == 0)) {
      if (iarg + 2 > narg)
        error->all(FLERR, "Need one argument after write command");
      if (comm->me == 0) {
        if ((strcmp(arg[iarg], "write_inv") == 0)) {  // elastance matrix
          f_inv = fopen(arg[++iarg], "w");
          if (f_inv == nullptr)
            error->one(FLERR,
                       fmt::format("Cannot open elastance matrix file {}: {}",
                                   arg[iarg], utils::getsyserror()));
        } else if ((strcmp(arg[iarg], "write_mat") == 0)) {  // b vector
          f_mat = fopen(arg[++iarg], "w");
          if (f_mat == nullptr)
            error->one(FLERR,
                       fmt::format("Cannot open capacitance matrix file {}: {}",
                                   arg[iarg], utils::getsyserror()));
        } else if ((strcmp(arg[iarg], "write_vec") == 0)) {  // b vector
          f_vec = fopen(arg[++iarg], "w");
          if (f_vec == nullptr)
            error->one(FLERR, fmt::format("Cannot open vector file {}: {}",
                                          arg[iarg], utils::getsyserror()));
        } else {
          error->all(FLERR, "Illegal fix update_charge command with write");
        }
      } else {
        iarg++;
      }
    } else if ((strncmp(arg[iarg], "read", 4) == 0)) {
      if (iarg + 2 > narg)
        error->all(FLERR, "Need one argument after read command");
      if ((strcmp(arg[iarg], "read_inv") == 0)) {
        read_inv = true;
        input_file_inv = arg[++iarg];
      } else if ((strcmp(arg[iarg], "read_mat") == 0)) {
        read_mat = true;
        input_file_mat = arg[++iarg];
      } else {
        error->all(FLERR, "Illegal fix update_charge command");
      }
    } else {
      error->all(FLERR, "Illegal fix update_charge command");
    }
    iarg++;
  }

  // union of all coupled groups
  std::string union_group = "conp_group";
  std::string group_cmd = union_group + " union";
  for (int g : groups) {
    group_cmd += " ";
    group_cmd += group->names[g];
  }
  group->assign(group_cmd);
  igroup = group->find(union_group);
  if (igroup < 0) error->all(FLERR, "Failed to create union of groups");
  // construct computes
  int const narg_compute = 4;
  int iarg_compute = 0;
  char **vector_arg = new char *[narg_compute];
  vector_arg[iarg_compute++] = (char *)"conp_vec";
  vector_arg[iarg_compute++] = (char *)group->names[igroup];
  vector_arg[iarg_compute++] = (char *)"conp/vector";
  vector_arg[iarg_compute++] = eta;
  assert(iarg_compute == narg_compute);
  vector_compute =
      new LAMMPS_NS::ComputeConpVector(lmp, narg_compute, vector_arg);
  delete[] vector_arg;
  if (!(read_inv || read_mat)) {
    iarg_compute = 0;
    char **matrix_arg = new char *[narg_compute];
    matrix_arg[iarg_compute++] = (char *)"conp_mat";
    matrix_arg[iarg_compute++] = (char *)group->names[igroup];
    matrix_arg[iarg_compute++] = (char *)"conp/matrix";
    matrix_arg[iarg_compute++] = eta;
    assert(iarg_compute == narg_compute);
    array_compute =
        new LAMMPS_NS::ComputeConpMatrix(lmp, narg_compute, matrix_arg);
    delete[] matrix_arg;
  }

  // error checks
  assert(groups.size() == group_bits.size());
  assert(groups.size() == group_psi.size());
  assert(igroup == vector_compute->igroup);
  if (!(read_mat || read_inv)) assert(igroup == array_compute->igroup);
  if (read_inv && read_mat)
    error->all(FLERR, "Cannot read matrix from two files");
  if (f_mat && read_inv)
    error->all(FLERR,
               "Cannot write coulomb matrix if reading elastance matrix "
               "from file");

  // check groups are consistent
  int *mask = atom->mask;
  for (int i = 0; i < atom->nlocal; i++) {
    int m = mask[i];
    int matches = 0;
    for (int bit : group_bits)
      if (m & bit) matches++;
    if (matches > 1) {
      error->all(FLERR, "Groups may not overlap");
    } else {
      assert((matches == 0) == (m & group->bitmask[igroup]) == 0);
    }
  }
  groupbit = group->bitmask[igroup];
  ngroup = group->count(igroup);
}

/* ---------------------------------------------------------------------- */

void FixChargeUpdate::init() {
  vector_compute->init();
  if (!(read_mat || read_inv)) array_compute->init();
}

/* ---------------------------------------------------------------------- */

void FixChargeUpdate::setup(int) {
  vector_compute->setup();
  if (!(read_mat || read_inv)) array_compute->setup();
  int const nlocal = atom->nlocal;
  int *mask = atom->mask;

  create_taglist();

  // setup psi with target potentials
  std::vector<int> mpos = local_to_matrix();
  psi = std::vector<double>(ngroup);
  double const evscale = 0.069447;  // TODO do units properly
  for (int i = 0; i < nlocal; i++) {
    for (int g = 0; g < number_groups; g++) {
      if (mask[i] & group_bits[g]) psi[mpos[i]] = group_psi[g] * evscale;
    }
  }
  MPI_Allreduce(MPI_IN_PLACE, &psi.front(), ngroup, MPI_DOUBLE, MPI_SUM, world);

  // initial elastance matrix and b vector
  std::vector<std::vector<double>> capacitance;
  if (read_inv) {
    elastance = read_from_file(input_file_inv);
  } else if (read_mat) {
    capacitance = read_from_file(input_file_mat);
    invert(capacitance);
  } else {
    array_compute->compute_array();
    double **a = array_compute->array;
    for (bigint i = 0; i < ngroup; i++) {
      std::vector<double> vec;
      for (bigint j = 0; j < ngroup; j++) {
        vec.push_back(a[i][j]);
      }
      capacitance.push_back(vec);
    }
    invert(capacitance);
  }
  if (symm) symmetrize();

  // write to files, ordered by group
  auto const order_matrix = [](std::vector<tagint> order,
                               std::vector<std::vector<double>> mat) {
    size_t n = order.size();
    std::vector<std::vector<double>> ordered_mat(n, std::vector<double>(n));
    for (size_t i = 0; i < n; i++) {
      bigint const gi = order[i];
      for (size_t j = 0; j < n; j++) {
        ordered_mat[gi][order[j]] = mat[i][j];
      }
    }
    return ordered_mat;
  };
  vector_compute->compute_vector();  // TODO crash with "if (f_vec)"
  if (comm->me == 0) {
    if (f_vec) {
      double *b = vector_compute->vector;
      std::vector<std::vector<double>> vec(ngroup, std::vector<double>(1));
      for (int i = 0; i < ngroup; i++) {
        vec[group_idx[i]][0] = b[i];
      }
      write_to_file(f_vec, taglist_bygroup, vec);
    }
    if (f_inv) {
      write_to_file(f_inv, taglist_bygroup, order_matrix(group_idx, elastance));
    }
    if (f_mat && !(read_inv)) {
      write_to_file(f_mat, taglist_bygroup,
                    order_matrix(group_idx, capacitance));
    }
  }
}

/* ---------------------------------------------------------------------- */

void FixChargeUpdate::invert(std::vector<std::vector<double>> capacitance) {
  if (comm->me == 0) utils::logmesg(lmp, "CONP inverting matrix\n");
  int m = ngroup, n = ngroup, lda = ngroup;
  std::vector<int> ipiv(ngroup);
  int const lwork = ngroup * ngroup;
  std::vector<double> work(lwork);
  std::vector<double> tmp(lwork);

  for (int i = 0; i < ngroup; i++) {
    for (int j = 0; j < ngroup; j++) {
      int idx = i * ngroup + j;
      tmp[idx] = capacitance[i][j];
    }
  }

  int info_rf, info_ri;
  dgetrf_(&m, &n, &tmp.front(), &lda, &ipiv.front(), &info_rf);
  dgetri_(&n, &tmp.front(), &lda, &ipiv.front(), &work.front(), &lwork,
          &info_ri);
  if (info_rf != 0 || info_ri != 0)
    error->all(FLERR, "CONP matrix inversion failed!");
  elastance =
      std::vector<std::vector<double>>(ngroup, std::vector<double>(ngroup));
  for (int i = 0; i < ngroup; i++) {
    for (int j = 0; j < ngroup; j++) {
      int idx = i * ngroup + j;
      elastance[i][j] = tmp[idx];
    }
  }
}

/* ---------------------------------------------------------------------- */

void FixChargeUpdate::symmetrize() {
  // S matrix to enforce charge neutrality constraint
  std::vector<double> AinvE(ngroup, 0.);
  double EAinvE = 0.0;
  for (int i = 0; i < ngroup; i++) {
    for (double e : elastance[i]) {
      AinvE[i] += e;
    }
    EAinvE += AinvE[i];
  }
  for (int i = 0; i < ngroup; i++) {
    double iAinvE = AinvE[i];
    for (int j = 0; j < ngroup; j++) {
      elastance[i][j] -= AinvE[j] * iAinvE / EAinvE;
    }
  }
}

/* ---------------------------------------------------------------------- */

void FixChargeUpdate::create_taglist() {
  int *mask = atom->mask;
  int const nlocal = atom->nlocal;
  int const nprocs = comm->nprocs;
  tagint *tag = atom->tag;

  // assign a tag to each matrix index sorted by group and by tag
  taglist_bygroup = std::vector<tagint>();
  for (int gbit : group_bits) {
    std::vector<tagint> taglist_local;
    for (int i = 0; i < nlocal; i++) {
      if (mask[i] & gbit) taglist_local.push_back(tag[i]);
    }
    // gather from all cpus for this group
    int gnum_local = taglist_local.size();
    std::vector<int> idispls(nprocs);
    std::vector<int> gnum_list(nprocs);
    MPI_Allgather(&gnum_local, 1, MPI_INT, &gnum_list.front(), 1, MPI_INT,
                  world);
    idispls[0] = 0;
    for (int i = 1; i < nprocs; i++) {
      idispls[i] = idispls[i - 1] + gnum_list[i - 1];
    }
    int const gnum = accumulate(gnum_list.begin(), gnum_list.end(), 0);
    std::vector<tagint> taglist_all(gnum);
    MPI_Allgatherv(&taglist_local.front(), gnum_local, MPI_LMP_TAGINT,
                   &taglist_all.front(), &gnum_list.front(), &idispls.front(),
                   MPI_LMP_TAGINT, world);
    std::sort(taglist_all.begin(), taglist_all.end());
    for (tagint t : taglist_all) taglist_bygroup.push_back(t);
  }

  // taglist only sorted by tag not group, same order as in computes
  taglist = taglist_bygroup;
  std::sort(taglist.begin(), taglist.end());

  // group_idx allows mapping a vector that is sorted by taglist to being
  // ordered by taglist_bygroup
  group_idx = std::vector<tagint>();
  for (tagint t : taglist) {
    for (size_t i = 0; i < taglist_bygroup.size(); i++) {
      if (t == taglist_bygroup[i]) {
        group_idx.push_back(i);
        break;
      }
    }
  }
}

/* ---------------------------------------------------------------------- */

std::vector<int> FixChargeUpdate::local_to_matrix() {
  int const nall = atom->nlocal + atom->nghost;
  int *tag = atom->tag;
  std::vector<int> mpos(nall, -1);
  for (int i = 0; i < nall; i++) {
    for (bigint ii = 0; ii < ngroup; ii++) {
      if (taglist[ii] == tag[i]) {
        mpos[i] = ii;
      }
    }
  }
  return mpos;
}

/* ---------------------------------------------------------------------- */

void FixChargeUpdate::pre_force(int) {
  int *mask = atom->mask;
  std::vector<int> mpos = local_to_matrix();
  int const nall = atom->nlocal + atom->nghost;
  vector_compute->compute_vector();
  double *b = vector_compute->vector;
  for (int i = 0; i < nall; i++) {
    if (!(groupbit & mask[i])) continue;
    double q_tmp = 0;
    for (int j = 0; j < ngroup; j++) {
      q_tmp += elastance[mpos[i]][j] * (psi[j] - b[j]);
    }
    atom->q[i] = q_tmp;
  }
  forces_and_energies();
}

/* ---------------------------------------------------------------------- */

void FixChargeUpdate::forces_and_energies() {}

/* ---------------------------------------------------------------------- */

FixChargeUpdate::~FixChargeUpdate() {
  if (!(read_mat || read_inv)) delete array_compute;
  delete vector_compute;
}

/* ---------------------------------------------------------------------- */

int FixChargeUpdate::setmask() {
  int mask = 0;
  mask |= PRE_FORCE;
  return mask;
}

/* ---------------------------------------------------------------------- */

void FixChargeUpdate::write_to_file(FILE *file, std::vector<tagint> tags,
                                    std::vector<std::vector<double>> mat) {
  for (tagint t : tags) fprintf(file, "%20d", t);
  fprintf(file, "\n");
  for (std::vector<double> vec : mat) {
    for (double x : vec) fprintf(file, "%20.11e", x);
    fprintf(file, "\n");
  }
}

/*----------------------------------------------------------------------- */

std::vector<std::vector<double>> FixChargeUpdate::read_from_file(
    std::string input_file) {
  std::vector<double> matrix_1d(ngroup * ngroup);
  if (comm->me == 0) {
    bool got_tags = false;
    std::vector<std::vector<double>> matrix;
    std::vector<tagint> tags;
    std::ifstream input(input_file);
    if (!input.is_open())
      error->all(FLERR, fmt::format("Cannot open {} for reading", input_file));
    for (std::string line; getline(input, line);) {
      if (line.compare(0, 1, "#") == 0) continue;
      std::istringstream stream(line);
      std::string word;
      if (!got_tags) {
        while (std::getline(stream, word, ' ')) {
          if (word == "") continue;
          tags.push_back(stoi(word));
        }
        got_tags = true;
        if ((bigint)tags.size() != ngroup)
          error->all(FLERR,
                     fmt::format(
                         "Number of read tags {} not equal to group members {}",
                         tags.size(), ngroup));
      } else {
        std::vector<double> a_line;
        while (std::getline(stream, word, ' ')) {
          if (word == "") continue;
          a_line.push_back(stof(word));
        }
        if ((bigint)a_line.size() != ngroup)
          error->all(
              FLERR,
              fmt::format(
                  "Number of read entries {} not equal to group members {}",
                  a_line.size(), ngroup));
        matrix.push_back(a_line);
      }
    }
    if ((bigint)matrix.size() != ngroup)
      error->all(
          FLERR,
          fmt::format("Number of lines {} read not equal to group members {}",
                      matrix.size(), ngroup));

    std::vector<tagint> idx;
    for (tagint t : taglist) {
      for (size_t i = 0; i < tags.size(); i++) {
        if (t == tags[i]) {
          idx.push_back(i);
          break;
        }
      }
    }
    if ((bigint)idx.size() != ngroup)
      error->all(FLERR, fmt::format(
                            "Read tags do not match taglist of update_charge"));
    for (bigint i = 0; i < ngroup; i++) {
      bigint const ii = idx[i];
      for (bigint j = 0; j < ngroup; j++)
        matrix_1d[i * ngroup + j] = matrix[ii][idx[j]];
    }
  }
  MPI_Bcast(&matrix_1d.front(), ngroup * ngroup, MPI_DOUBLE, 0, world);
  std::vector<std::vector<double>> matrix_out(ngroup,
                                              std::vector<double>(ngroup));
  for (bigint i = 0; i < ngroup; i++) {
    for (bigint j = 0; j < ngroup; j++)
      matrix_out[i][j] = matrix_1d[i * ngroup + j];
  }
  return matrix_out;
}
