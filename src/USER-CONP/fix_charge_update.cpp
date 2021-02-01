#include "fix_charge_update.h"

#include <assert.h>


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
  assigned = false;

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

  std::vector<std::string> compute_ids;
  while (iarg < narg) {
    if ((strncmp(arg[iarg], "c_", 2) == 0)) {
      compute_ids.push_back(&arg[iarg][2]);
    }
    iarg++;
  }

  for (std::string id : compute_ids) {
    int i = modify->find_compute(id);
    if (i < 0)
      error->all(FLERR, "Compute ID for fix charge_update does not exist");
    Compute *c = modify->compute[i];
    if (c->array_flag) {
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
  if (array_compute == nullptr)
    error->all(FLERR, "Fix charge_update needs one array compute");
  if (vector_compute == nullptr)
    error->all(FLERR, "Fix charge_update needs one vector compute");
  for (Compute *c : {array_compute, vector_compute}) {
    if (igroup != c->igroup) {
      error->all(
          FLERR,
          "Group of fix charge update does not match group of its compute");
    }
  }
  // TODO more checks for computes
}

/* ---------------------------------------------------------------------- */

void FixChargeUpdate::init() {
  int const nlocal = atom->nlocal;
  int *mask = atom->mask;
  ngroup = group->count(igroup);
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

  create_taglist();
  std::vector<int> mpos = local_to_matrix();

  double const evscale = 0.069447; // TODO do units properly
  pots = new double[ngroup]();
  for (int i = 0; i < nlocal; i++) {
    for (int g = 0; g < number_groups; g++) {
      if (mask[i] & group_bits[g]) pots[mpos[i]] = group_pots[g] * evscale;
    }
  }
  MPI_Allreduce(MPI_IN_PLACE, pots, ngroup, MPI_DOUBLE, MPI_SUM, world);
}

/* ---------------------------------------------------------------------- */

void FixChargeUpdate::create_taglist() {
  // assign a tag to each matrix index

  int *mask = atom->mask;
  int const nlocal = atom->nlocal;
  int const nprocs = comm->nprocs;
  tagint *tag = atom->tag;

  std::vector<int> taglist_local;
  for (int i = 0; i < nlocal; i++) {
    if (mask[i] & groupbit) taglist_local.push_back(tag[i]);
  }
  int igroupnum_local = taglist_local.size();

  std::vector<int> idispls(nprocs);
  std::vector<int> igroupnum_list(nprocs);
  MPI_Allgather(&igroupnum_local, 1, MPI_INT, &igroupnum_list.front(), 1,
                MPI_INT, world);
  idispls[0] = 0;
  for (int i = 1; i < nprocs; i++) {
    idispls[i] = idispls[i - 1] + igroupnum_list[i - 1];
  }

  taglist = std::vector<int>(ngroup);
  MPI_Allgatherv(&taglist_local.front(), igroupnum_local, MPI_LMP_TAGINT,
                 &taglist.front(), &igroupnum_list.front(), &idispls.front(),
                 MPI_LMP_TAGINT, world);
  std::sort(taglist.begin(), taglist.end());
}

/* ---------------------------------------------------------------------- */

std::vector<int> FixChargeUpdate::local_to_matrix() {
  int const nlocal = atom->nlocal;
  int *tag = atom->tag;
  std::vector<int> mpos(nlocal, -1);
  for (bigint ii = 0; ii < ngroup; ii++) {
    for (int i = 0; i < nlocal; i++) {
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
  int const nlocal = atom->nlocal;
  double **a = array_compute->array;
  vector_compute->compute_vector();
  double *b = vector_compute->vector;
  for (int i = 0; i < nlocal; i++) {
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

FixChargeUpdate::~FixChargeUpdate() { delete[] pots; }

/* ---------------------------------------------------------------------- */

int FixChargeUpdate::setmask() {
  int mask = 0;
  mask |= PRE_FORCE;
  return mask;
}
