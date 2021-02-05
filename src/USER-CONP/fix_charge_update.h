
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
  FILE *f_ela, *f_vec, *f_cap;  // files for capacitance, eleastance and vector
  std::string input_file_elas, input_file_cap;
  class Compute *array_compute, *vector_compute;
  static int const number_groups = 2;
  std::vector<int> groups, group_bits;
  std::vector<double> group_pots;
  std::vector<double> pots;
  std::vector<std::vector<double> > elastance;
  bigint ngroup;
  std::vector<tagint> taglist, taglist_bygroup, group_idx;
  bool read_elas, read_cap;
  void create_taglist();
  void invert(std::vector<std::vector<double> >);
  std::vector<int> local_to_matrix();
  void write_to_file(FILE *, std::vector<tagint>,
                     std::vector<std::vector<double> >);
  std::vector<std::vector<double> > read_from_file(std::string input_file);
};

}  // namespace LAMMPS_NS

#endif
#endif
