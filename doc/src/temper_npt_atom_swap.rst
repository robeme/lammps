.. index:: temper/npt

temper/npt command
==================

Syntax
""""""

.. parsed-literal::

   temper/npt  N M temp fix-ID1 fix-ID2 seed1 seed2 pressure index

* N = total # of timesteps to run
* M = attempt a tempering swap every this many steps
* temp = initial temperature for this ensemble
* fix-ID1 = ID of the fix that will control temperature and pressure during the run
* fix-ID2 = ID of the fix that will control atom swapping during the run
* seed1 = random # seed used to decide on adjacent temperature to partner with
* seed2 = random # seed for Boltzmann factor in Metropolis swap
* pressure = setpoint pressure for the ensemble
* index = which temperature (0 to N-1) I am simulating (optional)

Examples
""""""""

.. code-block:: LAMMPS

   temper/npt/atom/swap 100000 100 $t nptfix swpfix 0 58728 1
   temper/npt/atom/swap 2500000 1000 300 nptfix swpfix 0 32285 $p
   temper/npt/atom/swap 5000000 2000 $t nptfix swpfix 0 12523 1 $w

Description
"""""""""""

Run a hybrid parallel tempering (PT) or replica exchange (RE) simulation using multiple
replicas (ensembles) of a system in the isothermal-isobaric (NPT)
ensemble.  The command temper/npt/atom/swap works like :doc:`temper <temper>`
 but requires running replicas in the NPT ensemble instead of the canonical
(NVT) ensemble and allows for pressure to be set in the ensembles. Additionally, Monte 
Carlo swaps of atoms of one given atom type with atoms of the other given atom types are
performed within each replica. These multiple ensembles can run in parallel at different 
temperatures or different pressures.  The acceptance criteria for temper/npt/atom/swap 
is specific to the NPT ensemble and can be found in references
:ref:`(Okabe) <Okabe2>` and :ref:`(Mori) <Mori2>`. 

Apart from the difference in acceptance criteria and the specification
of pressure, this command works much like the :doc:`temper <temper>`
command. See the documentation on :doc:`temper <temper>` for information
on how the parallel tempering is handled in general.

----------

Restrictions
""""""""""""

This command can only be used if LAMMPS was built with the USER-MISC
package.  See the :doc:`Build package <Build_package>` doc page for more
info.

This command should be used with a fix that maintains the
isothermal-isobaric (NPT) ensemble and a fix that performs atom swaps.

Related commands
""""""""""""""""

:doc:`temper <temper>`, :doc:`variable <variable>`, :doc:`fix_npt <fix_nh>` :doc:`fix_atom_swap <fix_atom_swap>`

**Default:** none

.. _Okabe2:

**(Okabe)** T. Okabe, M. Kawata, Y. Okamoto, M. Masuhiro, Chem. Phys. Lett., 335, 435-439 (2001).

.. _Mori2:

**(Mori)** Y. Mori, Y. Okamoto, J. Phys. Soc. Jpn., 7, 074003 (2010).
