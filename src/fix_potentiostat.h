/* -*- c++ -*- ----------------------------------------------------------
   LAMMPS - Large-scale Atomic/Molecular Massively Parallel Simulator
   http://lammps.sandia.gov, Sandia National Laboratories
   Steve Plimpton, sjplimp@sandia.gov

   Copyright (2003) Sandia Corporation.  Under the terms of Contract
   DE-AC04-94AL85000 with Sandia Corporation, the U.S. Government retains
   certain rights in this software.  This software is distributed under
   the GNU General Public License.

   See the README file in the top-level LAMMPS directory.
------------------------------------------------------------------------- */

#ifdef FIX_CLASS

FixStyle(potentiostat, FixElPotential)

#else

#include <vector>

#include "fix.h"
#define nchain 4

namespace LAMMPS_NS {

class FixElPotential : public Fix {
 public:
  FixElPotential(class LAMMPS *, int, char **);
  ~FixElPotential();
  int setmask();
  void pre_force(int);
  void setup(int);

 private:
  bool c0_set;
  double c0, c0_atom1, c0_atom2;
  double tau;
  double utarget1, utarget2;
  int igroup2, group2bit;
  double tq;        // potentiostat temperature;
  int *eleids_all;
  class RanMars *randomf;
  // from conp
  double *ug;
  double unitk[3];
  int *kxvecs, *kyvecs, *kzvecs;
  double ***cs, ***sn;
  int kxmax, kymax, kzmax;
  int kmax, kmax3d, kmax_created, kcount;
  double g_ewald, eta, gsqmx, volume, slab_volfactor;
  double *sfacrl, *sfacrl_all, *sfacim, *sfacim_all;
  double *bbb_all;
  int elenum, elenum_all;
  int *elenum_list, *displs;
  int *tag2eleall, *eleall2tag, *curr_tag2eleall;
  //
  std::vector<double> allgatherv_ordered(std::vector<double>);
  void allgatherv_ordered(std::vector<double>, double *);
  void force_cal();
  void coul_cal(int, double *, int *);
  double rms(int, double, bigint, double);
  void coeffs();
  void b_cal();
  void sincos_b();
  int electrode_check(int);
  double capacitance_cal();
};

}  // namespace LAMMPS_NS

#endif
    //#endif

    /* ERROR/WARNING messages:

    E: Illegal ... command

    Self-explanatory.  Check the input script syntax and compare to the
    documentation for the command.  You can use -echo screen as a
    command-line option when running LAMMPS to see the offending line.

    */
