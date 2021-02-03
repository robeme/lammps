
#ifdef FIX_CLASS

FixStyle(charge_update, FixChargeUpdate)

#else

#ifndef LMP_FIX_CHARGE_UPDATE_H
#define LMP_FIX_CHARGE_UPDATE_H

#include "fix.h"

namespace LAMMPS_NS {

class FixChargeUpdate : public Fix {
 public:
  FixChargeUpdate(class LAMMPS *, int, char **);
  ~FixChargeUpdate();
  int setmask();
  // void post_constructor();
  void init();
  void setup(int);
  // void setup_pre_force(int);
  void pre_force(int);
  // void post_run();
  // void setup_pre_force_respa(int,int);
  // void pre_force_respa(int,int,int);
  // void set_arrays(int);
  // void write_restart(FILE *);
  // void restart(char *);

 private:
  FILE *f_ela, *f_vec;  // files for capacitance, eleastance and vector
  class Compute *array_compute, *vector_compute;
  static int const number_groups = 2;
  std::vector<int> groups, group_bits;
  std::vector<double> group_pots;
  double *pots;
  int ngroup;
  std::vector<tagint> taglist, taglist_bygroup, group_idx;
  bool assigned;
  void create_taglist();
  std::vector<int> local_to_matrix();
  void write_2d_vector(FILE *, std::vector<tagint>,
                       std::vector<std::vector<double> >);
};

}  // namespace LAMMPS_NS

#endif
#endif
