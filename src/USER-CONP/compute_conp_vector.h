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

#ifdef COMPUTE_CLASS

ComputeStyle(conp/vector, ComputeConpVector)

#else

#ifndef LMP_COMPUTE_COUL_VECTOR_H
#define LMP_COMPUTE_COUL_VECTOR_H

#include "compute.h"

namespace LAMMPS_NS {

class ComputeConpVector : public Compute {
 public:
  ComputeConpVector(class LAMMPS *, int, char **);
  ~ComputeConpVector();
  void init();
  void setup();
  void init_list(int, class NeighList *);
  void compute_vector();

 private:
  bigint ngroup;
  int recalc_every;
  double **cutsq;
  double g_ewald, eta;
  int pairflag, kspaceflag, boundaryflag;
  int overwrite, gaussians;
  std::vector<int> taglist;
  bigint *mpos;
  bool assigned;
  class Pair *pair;
  class NeighList *list;
  class KSpace *kspace;
  class Ewald *ewald;
  FILE *fp;

  long filepos;

  void create_taglist();
  void update_mpos();

  void pair_contribution();
  double calc_erfc(double);
};

}  // namespace LAMMPS_NS

#endif
#endif