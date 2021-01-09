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

#define SMALL 0.00001

enum{OFF,INTER,INTRA};

/* ---------------------------------------------------------------------- */

ComputeCoulMatrix::ComputeCoulMatrix(LAMMPS *lmp, int narg, char **arg) :
  Compute(lmp, narg, arg),
  fp(nullptr), group2(nullptr), gradQ_V(nullptr), mat2tag(nullptr), id2mat(nullptr)
{
  if (narg < 4) error->all(FLERR,"Illegal compute coul/matrix command");
  
  array_flag = 1;
  size_array_cols = 0;
  size_array_rows = 0;
  size_array_rows_variable = 0;
  extarray = 0;
  
  fp = nullptr;
  
  int n = strlen(arg[3]) + 1;
  group2 = new char[n];
  strcpy(group2,arg[3]);

  jgroup = group->find(group2);
  if (jgroup == -1)
    error->all(FLERR,"Compute coul/matrix group ID does not exist");
  jgroupbit = group->bitmask[jgroup];
  
  // igroup defined in parent class

  pairflag = 1;
  kspaceflag = 0;
  boundaryflag = 1;
  molflag = OFF;
  matrixflag = 0;
  overwrite = 0;
  
  igroupnum = jgroupnum = natoms_original = natoms = 0;

  int iarg = 4;
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
    } else if (strcmp(arg[iarg],"matrix") == 0) {
      if (iarg+2 > narg)
        error->all(FLERR,"Illegal compute coul/matrix command");
      if (strcmp(arg[iarg+1],"yes") == 0) matrixflag = 1;
      else if (strcmp(arg[iarg+1],"no") == 0) matrixflag = 0;
      else error->all(FLERR,"Illegal compute coul/matrix command");
      iarg += 2;
    } else if (strcmp(arg[iarg],"boundary") == 0) {
      if (iarg+2 > narg)
        error->all(FLERR,"Illegal compute coul/matrix command");
      if (strcmp(arg[iarg+1],"yes") == 0) boundaryflag = 1;
      else if (strcmp(arg[iarg+1],"no") == 0) boundaryflag  = 0;
      else error->all(FLERR,"Illegal compute coul/matrix command");
      iarg += 2;
    } else if (strcmp(arg[iarg],"molecule") == 0) {
      if (iarg+2 > narg)
        error->all(FLERR,"Illegal compute coul/matrix command");
      if (strcmp(arg[iarg+1],"off") == 0) molflag = OFF;
      else if (strcmp(arg[iarg+1],"inter") == 0) molflag = INTER;
      else if (strcmp(arg[iarg+1],"intra") == 0) molflag  = INTRA;
      else error->all(FLERR,"Illegal compute coul/matrix command");
      if (molflag != OFF && atom->molecule_flag == 0)
        error->all(FLERR,"Compute coul/matrix molecule requires molecule IDs");
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
    } else if (strcmp(arg[iarg],"overwrite") == 0) { // TODO overwrites output if used every n steps
      overwrite = 1;                                 // e.g. by end_of_step() etc.
      iarg += 1;
    } else error->all(FLERR,"Illegal compute coul/matrix command");
  }

  gradQ_V = nullptr;
  
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
  memory->destroy(id2mat);
  
  for (int i = 0; i < natoms; i++)
    delete [] gradQ_V[i];
  delete [] gradQ_V;
  
  if (fp && comm->me == 0) fclose(fp);
}

/* ---------------------------------------------------------------------- */

void ComputeCoulMatrix::init()
{
  // if non-hybrid, then error if single_enable = 0
  // if hybrid, let hybrid determine if sub-style sets single_enable = 0

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

  if (kspaceflag) kspace = force->kspace;
  else kspace = nullptr;

  // compute Kspace correction terms

  if (kspaceflag) {
    kspace_correction();
    if ((fabs(e_correction) > SMALL) && (comm->me == 0))
      error->warning(FLERR,"Both groups in compute coul/matrix have a net charge; "
                     "the Kspace boundary correction to energy will be non-zero");
  }

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
  
  // get number of atoms in each group
  
  igroupnum = group->count(igroup);
  jgroupnum = group->count(jgroup);
  natoms = igroupnum+jgroupnum;
  
  // TODO use method in library to create BIG array
  gradQ_V = new double*[natoms];
  for (int i = 0; i < natoms; i++)
  gradQ_V[i] = new double[natoms];
  
  memory->create(mat2tag,natoms,"coul/matrix:mat2tag");
  memory->create(id2mat,natoms,"coul/matrix:id2mat");
  
  // assign matrix indices to global tags
  
  assignment();
  
  natoms_original = natoms;
}

/* ---------------------------------------------------------------------- */

void ComputeCoulMatrix::setup()
{  
  
  // one-time calculation of coulomb matrix at simulation setup

  //compute_array();
  
  // store matrix in file
  
  if (fp && comm->me == 0) {
    fprintf(screen,"Writing out coulomb matrix\n");

    for (int i = 0; i < natoms; i++)
      fprintf(fp,"%d ", mat2tag[i]);
    fprintf(fp,"\n");  

    for (int i = 0; i < natoms; i++) {
      for (int j = 0; j < natoms; j++) {
        fprintf(fp, "%.3f ", gradQ_V[i][j]);
      }
      fprintf(fp,"\n");
    }
  } 
}

/* ---------------------------------------------------------------------- */

void ComputeCoulMatrix::init_list(int /*id*/, NeighList *ptr)
{
  list = ptr;
}

/* ---------------------------------------------------------------------- */

void ComputeCoulMatrix::compute_array()
{
  if (natoms != natoms_original) reallocate();
  
  // initialize gradQ_V to zero  
    
  size_t nbytes = sizeof(double) * natoms;
  if (nbytes)
    for (int i = 0; i < natoms; i++)
      memset(&gradQ_V[i][0],0,nbytes);

  if (pairflag) pair_contribution();
  if (kspaceflag) kspace_contribution(); 
}

/* ---------------------------------------------------------------------- */

void ComputeCoulMatrix::reallocate()
{
  igroupnum = group->count(igroup);
  jgroupnum = group->count(jgroup);
  natoms = igroupnum+jgroupnum;
  
  // grow coulomb matrix
  
  // TODO use memory->destroy()
  for (int i = 0; i < natoms; i++)
    delete [] gradQ_V[i];
  delete [] gradQ_V;
  
  // TODO use memory->create()
  gradQ_V = new double*[natoms];
  for (int i = 0; i < natoms; i++)
    gradQ_V[i] = new double[natoms];

  natoms_original = natoms;
  
  // reassignment matrix tags
  
  memory->destroy(mat2tag);
  memory->create(mat2tag,natoms,"coul/matrix:mat2tag");
  assignment();
}

/* ---------------------------------------------------------------------- */

void ComputeCoulMatrix::assignment()
{
  // assign matrix indices to global tags and local matrix position
  
  int *mask = atom->mask;
  int nlocal = atom->nlocal;
  int nprocs = comm->nprocs;  
  tagint *tag = atom->tag;
  tagint *itaglist, *itaglist_local;
  tagint *jtaglist, *jtaglist_local;
  
  int igroupnum_local, jgroupnum_local;
  
  int *igroupnum_list, *jgroupnum_list, *idispls, *jdispls;
  
  igroupnum_local = jgroupnum_local = 0;
  for (int i = 0; i < nlocal; i++) {
    if (mask[i] & groupbit && mask[i] & jgroupbit)
      error->all(FLERR,"Same atom in both groups in compute coul/matrix");
    else if (mask[i] & groupbit) 
      igroupnum_local++;
    else if (mask[i] & jgroupbit) 
      jgroupnum_local++;
  }
  
  memory->create(igroupnum_list,nprocs,"coul/matrix:ilist");
  memory->create(jgroupnum_list,nprocs,"coul/matrix:jlist");
  memory->create(idispls,nprocs,"coul/matrix:idispls");
  memory->create(jdispls,nprocs,"coul/matrix:jdispls");
  memory->create(itaglist,igroupnum,"coul/matrix:itaglist");
  memory->create(jtaglist,jgroupnum,"coul/matrix:jtaglist");
    
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
  
  MPI_Allgatherv(itaglist_local,igroupnum_local,MPI_LMP_TAGINT,
                 itaglist,igroupnum_list,idispls,MPI_LMP_TAGINT,world);
  MPI_Allgatherv(jtaglist_local,jgroupnum_local,MPI_LMP_TAGINT,
                 jtaglist,jgroupnum_list,jdispls,MPI_LMP_TAGINT,world);
  
  for (int i = 0; i < igroupnum; i++)
    mat2tag[i] = itaglist[i];
  for (int i = 0; i < jgroupnum; i++)
    mat2tag[igroupnum+i] = jtaglist[i];
  
  // TODO create also a id2mat for each procs based on mat2tag
  
  memory->destroy(igroupnum_list);
  memory->destroy(jgroupnum_list);
  memory->destroy(idispls);
  memory->destroy(jdispls);
  memory->destroy(itaglist);
  memory->destroy(jtaglist);
  memory->destroy(itaglist_local);
  memory->destroy(jtaglist_local);
}

/* ---------------------------------------------------------------------- */

void ComputeCoulMatrix::pair_contribution()
{
  int i,j,ii,jj,inum,jnum,itype,jtype;
  double xtmp,ytmp,ztmp,delx,dely,delz;
  double rsq,eng,fpair,factor_coul;
  int *ilist,*jlist,*numneigh,**firstneigh;

  double **x = atom->x;
  double *q = atom->q;
  tagint *molecule = atom->molecule;
  int *type = atom->type;
  int *mask = atom->mask;
  tagint *tag = atom->tag;
  int nlocal = atom->nlocal;
  double *special_coul = force->special_coul;

  double **gradQ_V_local = new double*[natoms];
    for (int i = 0; i < natoms; i++)
        gradQ_V_local[i] = new double[natoms];
      
  // initialize gradQ_V_local to zero      
        
  size_t nbytes = sizeof(double) * natoms;
  if (nbytes)
    for (int i = 0; i < natoms; i++)
      memset(&gradQ_V_local[i][0],0,nbytes);
  
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

    for (jj = 0; jj < jnum; jj++) {
      j = jlist[jj];
      factor_coul = special_coul[sbmask(j)];
      j &= NEIGHMASK;

      // skip if atom J is not in either group

      if (!(mask[j] & groupbit || mask[j] & jgroupbit)) continue;

      // skip if atoms I,J are only in the same group

      int ij_flag = 0;
      int ji_flag = 0;
      if (mask[i] & groupbit && mask[j] & jgroupbit) ij_flag = 1;
      if (mask[j] & groupbit && mask[i] & jgroupbit) ji_flag = 1;
      if (!ij_flag && !ji_flag) continue;

      // skip if molecule IDs of atoms I,J do not satisfy molflag setting

      if (molflag != OFF) {
        if (molflag == INTER) {
          if (molecule[i] == molecule[j]) continue;
        } else {
          if (molecule[i] != molecule[j]) continue;
        }
      }

      delx = xtmp - x[j][0];
      dely = ytmp - x[j][1];
      delz = ztmp - x[j][2];
      rsq = delx*delx + dely*dely + delz*delz;
      jtype = type[j];

      if (rsq < cutsq[itype][jtype]) {
         //gradQ_V_local[tag[i]-1][tag[j]-1] += pair->single(i,j,itype,jtype,rsq,factor_coul,0.0,fpair);
         //gradQ_V_local[tag[i]-1][tag[j]-1] *= 2.0/(atom->q[i]*atom->q[j]);
      }
    }
  }
  
  // need to gather data from all procs

  for (int i = 0; i < natoms; i++)
    MPI_Reduce(&gradQ_V_local[i],&gradQ_V[i],natoms,MPI_DOUBLE,MPI_SUM,0,world);
  
  // TODO do the BIG array stuff with memory create from library.cpp  
  for (int i = 0; i < natoms; i++)
    delete [] gradQ_V_local[i];
  delete [] gradQ_V_local;
}

/* ---------------------------------------------------------------------- */

void ComputeCoulMatrix::kspace_contribution()
{
  double *vector_kspace = force->kspace->f2group;

  force->kspace->compute_group_group(groupbit,jgroupbit,0);
  //scalar += 2.0*force->kspace->e2group;
  //vector[0] += vector_kspace[0];
  //vector[1] += vector_kspace[1];
  //vector[2] += vector_kspace[2];

  // subtract extra A <--> A Kspace interaction so energy matches
  //   real-space style of compute group-group
  // add extra Kspace term to energy

  force->kspace->compute_group_group(groupbit,jgroupbit,1);
  //scalar -= force->kspace->e2group;

  // self energy correction term

  //scalar -= e_self;

  // k=0 boundary correction term

  if (boundaryflag) {
    double xprd = domain->xprd;
    double yprd = domain->yprd;
    double zprd = domain->zprd;

    // adjustment of z dimension for 2d slab Ewald
    // 3d Ewald just uses zprd since slab_volfactor = 1.0

    double volume = xprd*yprd*zprd*force->kspace->slab_volfactor;
    //scalar -= e_correction/volume;
  }
}

/* ---------------------------------------------------------------------- */

void ComputeCoulMatrix::kspace_correction()
{

  // total charge of groups A & B, needed for correction term

  double qsqsum_group,qsum_A,qsum_B;
  qsqsum_group = qsum_A = qsum_B = 0.0;

  double *q = atom->q;
  int *mask = atom->mask;
  int groupbit_A = groupbit;
  int groupbit_B = jgroupbit;

  for (int i = 0; i < atom->nlocal; i++) {
    if ((mask[i] & groupbit_A) && (mask[i] & groupbit_B))
      qsqsum_group += q[i]*q[i];
    if (mask[i] & groupbit_A) qsum_A += q[i];
    if (mask[i] & groupbit_B) qsum_B += q[i];
  }

  double tmp;
  MPI_Allreduce(&qsqsum_group,&tmp,1,MPI_DOUBLE,MPI_SUM,world);
  qsqsum_group = tmp;

  MPI_Allreduce(&qsum_A,&tmp,1,MPI_DOUBLE,MPI_SUM,world);
  qsum_A = tmp;

  MPI_Allreduce(&qsum_B,&tmp,1,MPI_DOUBLE,MPI_SUM,world);
  qsum_B = tmp;

  double g_ewald = force->kspace->g_ewald;

  double scale = 1.0;
  const double qscale = force->qqrd2e * scale;

  // self-energy correction

  e_self = qscale * g_ewald*qsqsum_group/MY_PIS;
  e_correction = 2.0*qsum_A*qsum_B;

  // subtract extra AA terms

  qsum_A = qsum_B = 0.0;

  for (int i = 0; i < atom->nlocal; i++) {
    if (!((mask[i] & groupbit_A) && (mask[i] & groupbit_B)))
      continue;

    if (mask[i] & groupbit_A) qsum_A += q[i];
    if (mask[i] & groupbit_B) qsum_B += q[i];
  }

  MPI_Allreduce(&qsum_A,&tmp,1,MPI_DOUBLE,MPI_SUM,world);
  qsum_A = tmp;

  MPI_Allreduce(&qsum_B,&tmp,1,MPI_DOUBLE,MPI_SUM,world);
  qsum_B = tmp;

  // k=0 energy correction term (still need to divide by volume above)

  e_correction -= qsum_A*qsum_B;
  e_correction *= qscale * MY_PI2 / (g_ewald*g_ewald);
}
