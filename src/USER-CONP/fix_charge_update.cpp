#include "fix_charge_update.h"

#include <assert.h>

#include <numeric>

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

//     0        1   2             3
// fix fxupdate all charge_update group1 pot1 group2 pot2 c_matrix c_vector
FixChargeUpdate::FixChargeUpdate(LAMMPS *lmp, int narg, char **arg)
    : Fix(lmp, narg, arg) {
  array_compute = vector_compute = nullptr;
  f_ela = f_vec = nullptr;
  matrix_from_file = nullptr;
  read_matrix = false;

  int iarg = 3;
  for (int i = 0; i < number_groups; i++) {
    int id = group->find(arg[iarg++]);
    double pot = utils::numeric(FLERR, arg[iarg++], false,
                                lmp);  //* force->qe2f; // V -> kcal / (mol e)
    if (id < 0) error->all(FLERR, "Group does not exist");
    groups.push_back(id);
    group_bits.push_back(group->bitmask[id]);
    group_pots.push_back(pot);
  }
  assert(groups.size == group_bits.size);
  assert(groups.size == group_pots.size);

  // read fix command
  std::vector<std::string> compute_ids;
  while (iarg < narg) {
    if ((strncmp(arg[iarg], "c_", 2) == 0)) {
      compute_ids.push_back(&arg[iarg][2]);
    } else if ((strncmp(arg[iarg], "file", 4) == 0)) {
      if (iarg + 2 > narg)
        error->all(FLERR, "Illegal compute coul/matrix command");
      if (comm->me == 0) {
        if ((strcmp(arg[iarg], "file_elas") == 0)) {  // elastance matrix
          f_ela = fopen(arg[++iarg], "w");
          if (f_ela == nullptr)
            error->one(FLERR,
                       fmt::format("Cannot open elastance matrix file {}: {}",
                                   arg[iarg], utils::getsyserror()));
        } else if ((strcmp(arg[iarg], "file_vec") == 0)) {  // b vector
          f_vec = fopen(arg[++iarg], "w");
          if (f_vec == nullptr)
            error->one(FLERR, fmt::format("Cannot open vector file {}: {}",
                                          arg[iarg], utils::getsyserror()));
        } else {
          error->all(FLERR, "Illegal fix update_charge command");
        }
      } else {
        iarg++;
      }
    } else if ((strcmp(arg[iarg], "read_elas") == 0)) {
      if (iarg + 2 > narg)
        error->all(FLERR, "Illegal compute coul/matrix command");
      iarg++;
      read_matrix = true;
      if (comm->me == 0) {
        f_in = fopen(arg[iarg], "r");
        if (f_in == nullptr)
          error->one(FLERR,
                     fmt::format("Cannot open eleastance matrix file {}: {}",
                                 arg[iarg], utils::getsyserror()));
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
      if (read_matrix)
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
  if (array_compute == nullptr && !read_matrix) {
    error->all(FLERR, "Fix charge_update needs one array compute");
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

  // init class arrays
  ngroup = group->count(igroup);
  pots = new double[ngroup]();
  if (read_matrix) {
    matrix_from_file = new double *[ngroup];
    for (bigint i = 0; i < ngroup; i++)
      matrix_from_file[i] = new double[ngroup]();
  }
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

  // setup pots with target potentials
  std::vector<int> mpos = local_to_matrix();
  double const evscale = 0.069447;  // TODO do units properly
  for (int i = 0; i < nlocal; i++) {
    for (int g = 0; g < number_groups; g++) {
      if (mask[i] & group_bits[g]) pots[mpos[i]] = group_pots[g] * evscale;
    }
  }
  MPI_Allreduce(MPI_IN_PLACE, pots, ngroup, MPI_DOUBLE, MPI_SUM, world);

  // initial elastance matrix and b vector
  vector_compute->compute_vector();
  if (read_matrix) {
    read_from_file();
  } else {
    array_compute->compute_array();  // TODO turn off automatic calc in compute
  }

  // write to files, ordered by group
  if (comm->me == 0) {
    if (f_vec) {
      double *b = vector_compute->vector;
      std::vector<std::vector<double>> vec(ngroup, std::vector<double>(1));
      for (int i = 0; i < ngroup; i++) {
        vec[group_idx[i]][0] = b[i];
      }
      write_to_file(f_vec, taglist_bygroup, vec);
    }
    if (f_ela) {
      double **a = read_matrix ? matrix_from_file : array_compute->array;
      std::vector<std::vector<double>> mat(ngroup, std::vector<double>(ngroup));
      for (bigint i = 0; i < ngroup; i++) {
        bigint const gi = group_idx[i];
        for (bigint j = 0; j < ngroup; j++) {
          mat[gi][group_idx[j]] = a[i][j];
        }
      }
      write_to_file(f_ela, taglist_bygroup, mat);
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
  for (bigint ii = 0; ii < ngroup; ii++) {
    for (int i = 0; i < nall; i++) {
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
  int const nall = atom->nlocal + atom->nghost;
  double **a = read_matrix ? matrix_from_file : array_compute->array;
  vector_compute->compute_vector();
  double *b = vector_compute->vector;
  for (int i = 0; i < nall; i++) {
    int const pos = mpos[i];
    if (pos < 0) continue;
    double q_tmp = 0;
    for (int j = 0; j < ngroup; j++) {
      q_tmp += a[pos][j] * (pots[j] - b[j]);
    }
    atom->q[i] = q_tmp;
  }
}

/* ---------------------------------------------------------------------- */

FixChargeUpdate::~FixChargeUpdate() {
  delete[] pots;
  if (read_matrix) {
    for (bigint i = 0; i < ngroup; i++) delete[] matrix_from_file[i];
    delete[] matrix_from_file;
  }
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
  fprintf(file, "# atoms\n");
  for (tagint t : tags) fprintf(file, "%6d", t);
  fprintf(file, "\n");
  fprintf(file, "# matrix\n");
  for (std::vector<double> vec : mat) {
    for (double x : vec) fprintf(file, "%20.10f", x);
    fprintf(file, "\n");
  }
}

/*----------------------------------------------------------------------- */

void FixChargeUpdate::read_from_file() {
  double *matrix_1d = new double[ngroup * ngroup];
  if (comm->me == 0) {
    int maxchar = 21 * ngroup + 1;
    char line[maxchar];
    char *word;
    bool got_tags = false;
    std::vector<std::vector<double>> a;
    std::vector<tagint> tags;
    while (fgets(line, maxchar, f_in) != NULL) {
      word = strtok(line, " \t");
      if (strncmp(word, "#", 1) == 0) continue;
      if (!got_tags) {
        while (word != NULL) {
          tags.push_back(atoi(word));
          word = strtok(NULL, " \t");
        }
        got_tags = true;
        if ((bigint)tags.size() != ngroup)
          error->all(FLERR,
                     fmt::format(
                         "Number of read tags {} not equal to group members {}",
                         tags.size(), ngroup));
      } else {
        std::vector<double> a_line;
        while (word != NULL) {
          a_line.push_back(atof(word));
          word = strtok(NULL, " \t");
        }
        if ((bigint)a_line.size() != ngroup)
          error->all(
              FLERR,
              fmt::format(
                  "Number of read entries {} not equal to group members {}",
                  a_line.size(), ngroup));
        a.push_back(a_line);
      }
    }
    fclose(f_in);
    if ((bigint)a.size() != ngroup)
      error->all(
          FLERR,
          fmt::format("Number of lines {} read not equal to group members {}",
                      a.size(), ngroup));

    std::vector<tagint> idx;
    for (tagint t : taglist) {
      for (size_t i = 0; i < tags.size(); i++) {
        if (t == tags[i]) {
          idx.push_back(i);
          break;
        }
      }
    }
    for (bigint i = 0; i < ngroup; i++) {
      bigint const ii = idx[i];
      for (bigint j = 0; j < ngroup; j++)
        matrix_1d[i * ngroup + j] = a[ii][idx[j]];
    }
  }
  MPI_Bcast(matrix_1d, ngroup * ngroup, MPI_DOUBLE, 0, world);
  for (bigint i = 0; i < ngroup; i++) {
    for (bigint j = 0; j < ngroup; j++)
      matrix_from_file[i][j] = matrix_1d[i * ngroup + j];
  }
  delete[] matrix_1d;
}
