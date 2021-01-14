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

/* ----------------------------------------------------------------------
   Contributing author: Naveen Michaud-Agrawal (Johns Hopkins U)
     K-space terms added by Stan Moore (BYU)
------------------------------------------------------------------------- */

#include "compute_coul_matrix.h"

#include <cstring>
#include <cmath>
#include "atom.h"
#include "update.h"
#include "force.h"
#include "pair.h"
#include "neighbor.h"
#include "neigh_request.h"
#include "neigh_list.h"
#include "group.h"
#include "kspace.h"
#include "error.h"
#include "comm.h"
#include "memory.h"
#include "domain.h"
#include "math_const.h"

using namespace LAMMPS_NS;
using namespace MathConst;

#define EWALD_F   1.12837917
#define EWALD_P   0.3275911
#define A1        0.254829592
#define A2       -0.284496736
#define A3        1.421413741
#define A4       -1.453152027
#define A5        1.061405429

enum{OFF,INTER,INTRA};

/* ---------------------------------------------------------------------- */

ComputeCoulMatrix::ComputeCoulMatrix(LAMMPS *lmp, int narg, char **arg) :
  Compute(lmp, narg, arg),
  fp(nullptr), group2(nullptr), gradQ_V(nullptr), mpos(nullptr)
{
  if (narg < 4) error->all(FLERR,"Illegal compute coul/matrix command");
  
  array_flag = 1;
  size_array_cols = 0;
  size_array_rows = 0;
  size_array_rows_variable = 0;
  extarray = 0;
  
  fp = nullptr;
  gradQ_V = nullptr;
  mpos = nullptr;

  pairflag = 1;
  kspaceflag = 1; 
  boundaryflag = 1; // include infite boundary correction term
  selfflag = 1;
  gaussians = 1;
  recalc_every = 0; 
  overwrite = 1;
  assigned = 0;
  
  g_ewald = 0.0;
  
  // get jgroup; igroup defined in parent class
  
  int n = strlen(arg[3]) + 1;
  group2 = new char[n];
  strcpy(group2,arg[3]);

  jgroup = group->find(group2);
  if (jgroup == -1)
    error->all(FLERR,"Compute coul/matrix group ID does not exist"); 
  jgroupbit = group->bitmask[jgroup];
  
  // TODO recalculate coulomb matrix every recalc_every
  
  recalc_every = utils::inumeric(FLERR,arg[4],false,lmp);
  eta = utils::numeric(FLERR,arg[5],false,lmp); // TODO infer from pair_style!

  int iarg = 6;
  while (iarg < narg) {
    if (strcmp(arg[iarg],"pair") == 0) {
      if (iarg+2 > narg)
        error->all(FLERR,"Illegal compute coul/matrix command");
      if (strcmp(arg[iarg+1],"yes") == 0) pairflag = 1;
      else if (strcmp(arg[iarg+1],"no") == 0) pairflag = 0;
      else error->all(FLERR,"Illegal compute coul/matrix command");
      iarg += 2;
    } else if (strcmp(arg[iarg],"kspace") == 0) {
      if (iarg+2 > narg)
        error->all(FLERR,"Illegal compute coul/matrix command");
      if (strcmp(arg[iarg+1],"yes") == 0) kspaceflag = 1;
      else if (strcmp(arg[iarg+1],"no") == 0) kspaceflag = 0;
      else error->all(FLERR,"Illegal compute coul/matrix command");
      iarg += 2;
    } else if (strcmp(arg[iarg],"self") == 0) {
      if (iarg+2 > narg)
        error->all(FLERR,"Illegal compute coul/matrix command");
      if (strcmp(arg[iarg+1],"yes") == 0) selfflag = 1;
      else if (strcmp(arg[iarg+1],"no") == 0) selfflag  = 0;
      else error->all(FLERR,"Illegal compute coul/matrix command");
      iarg += 2;
    } else if (strcmp(arg[iarg],"boundary") == 0) {
      if (iarg+2 > narg)
        error->all(FLERR,"Illegal compute coul/matrix command");
      if (strcmp(arg[iarg+1],"yes") == 0) boundaryflag = 1;
      else if (strcmp(arg[iarg+1],"no") == 0) boundaryflag  = 0;
      else error->all(FLERR,"Illegal compute coul/matrix command");
      iarg += 2;
    } else if (strcmp(arg[iarg],"overwrite") == 0) { // TODO  if matrix is recalculated overwrite or append output
      if (iarg+2 > narg)
        error->all(FLERR,"Illegal compute coul/matrix command");
      if (strcmp(arg[iarg+1],"yes") == 0) overwrite = 1;
      else if (strcmp(arg[iarg+1],"no") == 0) overwrite  = 0;
      else error->all(FLERR,"Illegal compute coul/matrix command");                    
      iarg += 2;
    } else if (strcmp(arg[iarg],"file") == 0) {
      if (iarg+2 > narg) error->all(FLERR,"Illegal compute coul/matrix command");
      if (comm->me == 0) {
        fp = fopen(arg[iarg+1],"w");
        if (fp == nullptr)
          error->one(FLERR,fmt::format("Cannot open compute coul/matrix file {}: {}",
                                       arg[iarg+1], utils::getsyserror()));
      }
      iarg += 2;
    } else error->all(FLERR,"Illegal compute coul/matrix command");
  }
  
  // print file comment lines

  if (fp && comm->me == 0) {
    clearerr(fp);
    fprintf(fp,"# coulomb matrix for constant potential\n");
    if (ferror(fp))
      error->one(FLERR,"Error writing file header");
    filepos = ftell(fp);
  }
}

/* ---------------------------------------------------------------------- */

ComputeCoulMatrix::~ComputeCoulMatrix()
{
  delete [] group2;
  
  memory->destroy(mat2tag);
  memory->destroy(gradQ_V);
  
  if (fp && comm->me == 0) fclose(fp);
}

/* ---------------------------------------------------------------------- */

void ComputeCoulMatrix::init()
{
  // if non-hybrid, then error if single_enable = 0
  // if hybrid, let hybrid determine if sub-style sets single_enable = 0
  
  // TODO just check if a coul pair_style is chosen, we need just the cutsq
  // from pair_style and not pair->single as we do it here explicitly 
  
  if (pairflag && force->pair == nullptr) 
    error->all(FLERR,"No pair style defined for compute coul/matrix");
  if (force->pair_match("^hybrid",0) == nullptr
      && force->pair->single_enable == 0)
    error->all(FLERR,"Pair style does not support compute coul/matrix");

  // error if Kspace style does not compute coul/matrix interactions

  if (kspaceflag && force->kspace == nullptr)
    error->all(FLERR,"No Kspace style defined for compute coul/matrix");
  if (kspaceflag && force->kspace->group_group_enable == 0)
    error->all(FLERR,"Kspace style does not support compute coul/matrix");

  if (pairflag) {
    pair = force->pair;
    cutsq = force->pair->cutsq;
  } else pair = nullptr;

  if (kspaceflag) {
    kspace = force->kspace;
    g_ewald = force->kspace->g_ewald;
  } else kspace = nullptr;

  // recheck that group 2 has not been deleted

  jgroup = group->find(group2);
  if (jgroup == -1)
    error->all(FLERR,"Compute coul/matrix group ID does not exist");
  jgroupbit = group->bitmask[jgroup];

  // need an occasional half neighbor list

  if (pairflag) {
    int irequest = neighbor->request(this,instance_me);
    neighbor->requests[irequest]->pair = 0;
    neighbor->requests[irequest]->compute = 1;
    neighbor->requests[irequest]->occasional = 1;
  }  
}

/* ---------------------------------------------------------------------- */

void ComputeCoulMatrix::setup()
{
  double **matrix;
  
  igroupnum = group->count(igroup);
  jgroupnum = group->count(jgroup);
  natoms = igroupnum+jgroupnum;
  
  memory->create(mat2tag,natoms,"coul/matrix:mat2tag");
  
  // assign atom tags to matrix locations and vice versa
  
  matrix_assignment(); 
  
  // TODO would be nicer to have a local matrix and gather it later
  // to reduce a little bit the memory consumption ... However, for 
  // this wo work I think the atom ids must be from 1,...,N and consecutive
  
  memory->create(gradQ_V,natoms,natoms,"coul/matrix:gradQ_V");
  
  // setting all entries of coulomb matrix to zero
  
  size_t nbytes = sizeof(double) * natoms;

  if (nbytes)
    for (int i = 0; i < natoms; i++)
      memset(&gradQ_V[i][0],0,nbytes);
  
  // initial calculation of coulomb matrix at setup of simulation

  compute_array();
  
  // reduce coulomb matrix with contributions from all procs
  // all procs need to know full matrix for matrix inversion
  
  memory->create(matrix,natoms,natoms,"coul/matrix:matrix");
  
  MPI_Allreduce(&gradQ_V[0][0], &matrix[0][0], natoms*natoms, MPI_DOUBLE, MPI_SUM, world);
    
  if (fp && comm->me == 0) write_matrix(matrix);
  
  memory->destroy(matrix);
}

/* ---------------------------------------------------------------------- */

void ComputeCoulMatrix::init_list(int /*id*/, NeighList *ptr)
{
  list = ptr;
}

/* ---------------------------------------------------------------------- */

void ComputeCoulMatrix::compute_array()
{
  if (pairflag) pair_contribution();
  if (selfflag) self_contribution();
  if (kspaceflag) kspace_contribution(); 
  if (boundaryflag) kspace_correction();
}

/* ---------------------------------------------------------------------- */

void ComputeCoulMatrix::pair_contribution()
{ 
  int i,j,ii,jj,inum,jnum,itype,jtype;
  double xtmp,ytmp,ztmp,delx,dely,delz;
  double r,rinv,rsq,grij,etarij,expm2,t,erfc,aij;
  int *ilist,*jlist,*numneigh,**firstneigh;
  bigint ipos,jpos;

  double **x = atom->x;
  tagint *molecule = atom->molecule;
  int *type = atom->type;
  int *mask = atom->mask;
  tagint *tag = atom->tag;
  int nlocal = atom->nlocal;
  
  double etaij = 0.0;
  
  // invoke half neighbor list (will copy or build if necessary)
  
  neighbor->build_one(list);

  inum = list->inum;
  ilist = list->ilist;
  numneigh = list->numneigh;
  firstneigh = list->firstneigh;

  // loop over neighbors of my atoms
  // skip if I,J are not in 2 groups

  for (ii = 0; ii < inum; ii++) {
    i = ilist[ii];
    // skip if atom I is not in either group
    if (!(mask[i] & groupbit || mask[i] & jgroupbit)) continue;

    xtmp = x[i][0];
    ytmp = x[i][1];
    ztmp = x[i][2];
    itype = type[i];
    jlist = firstneigh[i];
    jnum = numneigh[i];
    
    // locate matrix position of first atom   
    for (ipos = 0; ipos < natoms; ipos++)
      if (mat2tag[ipos] == tag[i]) break;

    // real-space part of matrix is symmetric, start from jj == ii

    for (jj = ii; jj < jnum; jj++) {
      j = jlist[jj];
      j &= NEIGHMASK;
      
      // skip if atom J is not in either group

      if (!(mask[j] & groupbit || mask[j] & jgroupbit)) continue;

      delx = xtmp - x[j][0];  // neighlists take care of pbc
      dely = ytmp - x[j][1];
      delz = ztmp - x[j][2];
      rsq = delx*delx + dely*dely + delz*delz;
      jtype = type[j];

      if (rsq < cutsq[itype][jtype]) {
        r = sqrt(rsq);
        rinv = 1.0 / r;
        aij = rinv;
        if (kspaceflag) {
          grij = g_ewald * r;
          expm2 = exp(-grij*grij);
          t = 1.0 / (1.0 + EWALD_P*grij);
          erfc = t * (A1+t*(A2+t*(A3+t*(A4+t*A5)))) * expm2;
          
          aij *= erfc;
          
          // check if we have realspace gaussians
        
          if (gaussians) {
            // TODO infer eta from coeffs of pair coul/long/gauss
            
            etarij = eta * r;
            expm2 = exp(-etarij*etarij);
            t = 1.0 / (1.0 + EWALD_P*etarij);
            erfc = t * (A1+t*(A2+t*(A3+t*(A4+t*A5)))) * expm2;
            
            aij -= erfc*rinv;
          }
        }    
        
        // locate matrix position of secont atom
        for (jpos = 0; jpos < natoms; jpos++)
          if (mat2tag[jpos] == tag[j]) break;

        gradQ_V[ipos][jpos] += aij;
        if (tag[i] != tag[j]) gradQ_V[jpos][ipos] += aij;
      }
    }
  }
}

/* ---------------------------------------------------------------------- */

void ComputeCoulMatrix::self_contribution()
{ 
  const double selfint = -2.0/MY_PIS*g_ewald;
  const double preta = MY_SQRT2/MY_PIS;
  for (bigint i = 0; i < natoms; i++)
    // TODO infer eta from pair_coeffs
    gradQ_V[i][i] += selfint + preta*eta;
}

/* ---------------------------------------------------------------------- */

void ComputeCoulMatrix::kspace_contribution()
{ 
  kspace->compute_matrix(groupbit, jgroupbit, mpos, gradQ_V);
}

/* ---------------------------------------------------------------------- */

void ComputeCoulMatrix::kspace_correction()
{

}

/* ---------------------------------------------------------------------- */

void ComputeCoulMatrix::write_matrix(double **matrix)
{ 
  fprintf(fp,"# matrix -> atom id\n");
  for (bigint i = 0; i < natoms; i++)
    fprintf(fp,"%d ", mat2tag[i]);
  fprintf(fp,"\n");  

  fprintf(fp,"# matrix\n");
  for (bigint i = 0; i < natoms; i++) {
    for (bigint j = 0; j < natoms; j++) {
      fprintf(fp, "%.3f ", matrix[i][j]);
    }
    fprintf(fp,"\n");
  }
}

/* ---------------------------------------------------------------------- */

void ComputeCoulMatrix::reallocate()
{ 
  bigint igroupnum, jgroupnum, natoms;
  
  igroupnum = group->count(igroup);
  jgroupnum = group->count(jgroup);
  natoms = igroupnum+jgroupnum;
  
  memory->destroy(mat2tag);
  memory->destroy(gradQ_V);
  memory->create(gradQ_V,natoms,natoms,"coul/matrix:gradQ_V");
  memory->create(mat2tag,natoms,"coul/matrix:mat2tag");
  
  // since atom number has changed, reassign atom tags to matrix locations
  
  matrix_assignment(); 
}

/* ---------------------------------------------------------------------- 
   looks up to which proc each atom in each group belongs and creates a
   local array which locates the position of each local atom in the global 
   matrix. entries are sorted: first A then B. need to be so complex here
   b/c atom tags might not be be consecutive or sorted in any way. 
------------------------------------------------------------------------- */  

void ComputeCoulMatrix::matrix_assignment()
{
  // assign local matrix indices to local atoms on each proc
  
  int *mask = atom->mask;
  int nlocal = atom->nlocal;
  int nprocs = comm->nprocs;  
  tagint *tag = atom->tag;
  
  tagint *itaglist, *itaglist_local;
  tagint *jtaglist, *jtaglist_local;
  int igroupnum_local, jgroupnum_local;
  int *igroupnum_list, *jgroupnum_list;
  int *idispls, *jdispls;
  
  igroupnum_local = jgroupnum_local = 0;
  for (int i = 0; i < nlocal; i++) {
    if (mask[i] & groupbit && mask[i] & jgroupbit)
      error->all(FLERR,"Same atom in both groups in compute coul/matrix");
    else if (mask[i] & groupbit) 
      igroupnum_local++;
    else if (mask[i] & jgroupbit) 
      jgroupnum_local++;
  }
  
  memory->create(idispls,nprocs,"coul/matrix:idispls");
  memory->create(jdispls,nprocs,"coul/matrix:jdispls");
  memory->create(igroupnum_list,nprocs,"coul/matrix:ilist");
  memory->create(jgroupnum_list,nprocs,"coul/matrix:jlist");
    
  MPI_Allgather(&igroupnum_local,1,MPI_INT,igroupnum_list,1,MPI_INT,world);
  MPI_Allgather(&jgroupnum_local,1,MPI_INT,jgroupnum_list,1,MPI_INT,world);

  idispls[0] = jdispls[0] = 0;
  for (int i = 1; i < nprocs; i++) {
    idispls[i] = idispls[i-1] + igroupnum_list[i-1];
    jdispls[i] = jdispls[i-1] + jgroupnum_list[i-1];
  }
  
  memory->create(itaglist_local,igroupnum_local,"coul/matrix:itaglist_local");
  memory->create(jtaglist_local,jgroupnum_local,"coul/matrix:jtaglist_local");
  
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
  
  memory->create(itaglist,igroupnum,"coul/matrix:itaglist");
  memory->create(jtaglist,jgroupnum,"coul/matrix:jtaglist");
  
  MPI_Allgatherv(itaglist_local,igroupnum_local,MPI_LMP_TAGINT,
                 itaglist,igroupnum_list,idispls,MPI_LMP_TAGINT,world);
  MPI_Allgatherv(jtaglist_local,jgroupnum_local,MPI_LMP_TAGINT,
                 jtaglist,jgroupnum_list,jdispls,MPI_LMP_TAGINT,world);
  
  // if local matrix assignment already created, recreate
  
  if (assigned) {
    memory->destroy(mpos);
    memory->create(mpos,nlocal,"coul/matrix:mpos");
  } else memory->create(mpos,nlocal,"coul/matrix:mpos");
  
  assigned = 1;

  // set all local non-matrix values to -1
  
  size_t nbytes = sizeof(bigint) * nlocal;
  if (nbytes)
    memset(mpos,-1,nbytes);
  
  // sort itaglist and jtaglist
  
  std::sort(itaglist, itaglist + igroupnum);
  std::sort(jtaglist, jtaglist + igroupnum);
  
  memory->destroy(igroupnum_list);
  memory->destroy(jgroupnum_list);
  memory->destroy(idispls);
  memory->destroy(jdispls);
  memory->destroy(itaglist_local);
  memory->destroy(jtaglist_local);
  
  // store to which tag the value in the matrix belongs
  
  for (bigint i = 0; i < igroupnum; i++)
    mat2tag[i] = itaglist[i];
  for (bigint i = 0; i < jgroupnum; i++)
    mat2tag[igroupnum+i] = jtaglist[i];
  
  // create local array indexing global matrix
    
  int num = 0;
  for (int i = 0; i < igroupnum+jgroupnum; i++)
    for (int ii = 0; ii < nlocal; ii++)
      if (mat2tag[i] == tag[ii])
        mpos[ii] = num++;
  
  memory->destroy(itaglist);
  memory->destroy(jtaglist);
}
