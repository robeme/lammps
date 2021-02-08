#include "fix_charge_update.h"

#include <assert.h>

#include <fstream>
#include <iostream>
#include <numeric>
#include <sstream>

#include "atom.h"
#include "comm.h"
#include "compute.h"
#include "error.h"
#include "force.h"
#include "group.h"
#include "memory.h"
#include "modify.h"

using namespace LAMMPS_NS;
using namespace FixConst;

extern "C" {
void dgetrf_(const int *M, const int *N, double *A, const int *lda, int *ipiv,
             int *info);
void dgetri_(const int *N, double *A, const int *lda, const int *ipiv,
             double *work, const int *lwork, int *info);
}

//     0        1   2             3
// fix fxupdate all charge_update group1 pot1 group2 pot2 c_matrix c_vector
FixChargeUpdate::FixChargeUpdate(LAMMPS *lmp, int narg, char **arg)
    : Fix(lmp, narg, arg) {
  array_compute = vector_compute = nullptr;
  f_inv = f_mat = f_vec = nullptr;
  read_inv = read_mat = false;

  groups = std::vector<int>();
  group_bits = std::vector<int>();
  group_psi = std::vector<double>();
  int iarg = 3;
  for (int i = 0; i < number_groups; i++) {
    int id = group->find(arg[iarg++]);
    double pot = utils::numeric(FLERR, arg[iarg++], false,
                                lmp);  //* force->qe2f; // V -> kcal / (mol e)
    if (id < 0) error->all(FLERR, "Group does not exist");
    groups.push_back(id);
    group_bits.push_back(group->bitmask[id]);
    group_psi.push_back(pot);
  }
  assert(groups.size() == group_bits.size());
  assert(groups.size() == group_psi.size());

  // read fix command
  std::vector<std::string> compute_ids;
  while (iarg < narg) {
    if ((strncmp(arg[iarg], "c_", 2) == 0)) {
      compute_ids.push_back(&arg[iarg][2]);
    } else if ((strncmp(arg[iarg], "write", 4) == 0)) {
      if (iarg + 2 > narg)
        error->all(FLERR, "Illegal compute coul/matrix command");
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
        error->all(FLERR, "Illegal compute coul/matrix command");
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

  // assign computes
  for (std::string id : compute_ids) {
    int i = modify->find_compute(id);
    if (i < 0)
      error->all(FLERR, "Compute ID for fix charge_update does not exist");
    Compute *c = modify->compute[i];
    if (c->array_flag) {
      if (read_inv || read_mat)
        error->all(FLERR,
                   "Fix charge_update is set to read array but a compute was "
                   "found, too");
      if (array_compute == nullptr) {
        array_compute = c;
      } else {
        error->all(FLERR, "Fix charge_update can only use one array compute");
      }
    }
    if (c->vector_flag) {
      if (vector_compute == nullptr) {
        vector_compute = c;
      } else {
        error->all(FLERR, "Fix charge_update can only use one vector compute");
      }
    }
  }
  // error checks
  if (array_compute == nullptr && !(read_inv || read_mat)) {
    error->all(FLERR, "Fix charge_update needs one matrix compute");
  } else {
  }
  if (vector_compute == nullptr)
    error->all(FLERR, "Fix charge_update needs one vector compute");
  for (Compute *c : {array_compute, vector_compute}) {
    if (c != nullptr && igroup != c->igroup) {
      error->all(
          FLERR,
          "Group of fix charge update does not match group of its compute");
    }
  }
  if (read_inv && read_mat)
    error->all(FLERR, "Cannot read matrix from two files");
  if (f_mat && read_inv)
    error->all(FLERR,
               "Cannot write coulomb matrix if reading elastance matrix "
               "from file");

  // init class arrays
  ngroup = group->count(igroup);

  // TODO more checks for computes
}

/* ---------------------------------------------------------------------- */

void FixChargeUpdate::init() {
  int const nlocal = atom->nlocal;
  int *mask = atom->mask;

  // check groups are consistent
  int sum = 0;
  for (int i : groups) {
    sum += group->count(i);
  }
  if (sum != ngroup) {
    error->all(FLERR,
               "Count of igroup not equal to sum of counts of other groups");
  }
  for (int i = 0; i < nlocal; i++) {
    int m = mask[i];
    if (m & groupbit) {
      int matches = 0;
      for (int bit : group_bits)
        if (m & bit) matches++;
      if (matches != 1) {
        error->all(
            FLERR,
            "All atoms in igroup must occur exactly once in other groups");
      }
    }
  }
}

/* ---------------------------------------------------------------------- */

void FixChargeUpdate::setup(int) {
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
  vector_compute->compute_vector();
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
  std::vector<int> ipiv(ngroup + 1);
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
  int const nmax = atom->nmax;
  int *tag = atom->tag;
  std::vector<int> mpos(nmax, -1);
  for (bigint ii = 0; ii < ngroup; ii++) {
    for (int i = 0; i < nmax; i++) {
      if (taglist[ii] == tag[i]) {
        mpos[i] = ii;
      }
    }
  }
  return mpos;
}

/* ---------------------------------------------------------------------- */

void FixChargeUpdate::pre_force(int) {
  std::vector<int> mpos = local_to_matrix();
  int const nmax = atom->nmax;
  vector_compute->compute_vector();
  double *b = vector_compute->vector;
  for (int i = 0; i < nmax; i++) {
    int const pos = mpos[i];
    if (pos < 0) continue;
    double q_tmp = 0;
    for (int j = 0; j < ngroup; j++) {
      q_tmp += elastance[pos][j] * (psi[j] - b[j]);
    }
    atom->q[i] = q_tmp;
  }
}

/* ---------------------------------------------------------------------- */

FixChargeUpdate::~FixChargeUpdate() {}

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
      std::cout << line << std::endl;
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
