/* -*- c++ -*- ----------------------------------------------------------
   LAMMPS - Large-scale Atomic/Molecular Massively Parallel Simulator
   https://www.lammps.org/, Sandia National Laboratories
   LAMMPS development team: developers@lammps.org

   Copyright (2003) Sandia Corporation.  Under the terms of Contract
   DE-AC04-94AL85000 with Sandia Corporation, the U.S. Government retains
   certain rights in this software.  This software is distributed under
   the GNU General Public License.

   See the README file in the top-level LAMMPS directory.
------------------------------------------------------------------------- */

#ifdef FIX_CLASS
// clang-format off
FixStyle(charge,FixCharge);
// clang-format on
#else

#ifndef LMP_FIX_CHARGE_H
#define LMP_FIX_CHARGE_H

#include "fix.h"

namespace LAMMPS_NS {

class FixCharge : public Fix {
 public:
  FixCharge(class LAMMPS *, int, char **);
  
  int setmask() override;
  void init() override;
  void init_list(int, class NeighList *) override;
  void setup_pre_force(int) override;
  void pre_force(int) override;
  
 protected:
  int ntype;
  double q0;
  double del, twodel, delpi;
  double cut;
  double cuthi, cutlo, cuthisq, cutlosq;
  
  bigint ngroup;
  
 private:
  class NeighList *list;
  void update_charges();
  double fc(double);
};

}    // namespace LAMMPS_NS

#endif
#endif
