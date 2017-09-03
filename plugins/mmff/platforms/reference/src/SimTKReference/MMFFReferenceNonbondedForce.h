
/* Portions copyright (c) 2006-2013 Stanford University and Simbios.
 * Contributors: Pande Group
 *
 * Permission is hereby granted, free of charge, to any person obtaining
 * a copy of this software and associated documentation files (the
 * "Software"), to deal in the Software without restriction, including
 * without limitation the rights to use, copy, modify, merge, publish,
 * distribute, sublicense, and/or sell copies of the Software, and to
 * permit persons to whom the Software is furnished to do so, subject
 * to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included
 * in all copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS
 * OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF
 * MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.
 * IN NO EVENT SHALL THE AUTHORS, CONTRIBUTORS OR COPYRIGHT HOLDERS BE
 * LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION
 * OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION
 * WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
 */

#ifndef __MMFFReferenceNonbondedForce_H__
#define __MMFFReferenceNonbondedForce_H__

#include "ReferencePairIxn.h"
#include "ReferenceNeighborList.h"

namespace OpenMM {

class MMFFReferenceNonbondedForce {

   public:

      // MMFF vdW constants
      
      static const double dhal;
      static const double ghal;

      /**---------------------------------------------------------------------------------------
      
         Constructor
      
         --------------------------------------------------------------------------------------- */

       MMFFReferenceNonbondedForce();

      /**---------------------------------------------------------------------------------------
      
         Destructor
      
         --------------------------------------------------------------------------------------- */

       ~MMFFReferenceNonbondedForce();

      /**---------------------------------------------------------------------------------------
      
         Set the force to use a cutoff.
      
         @param distance            the cutoff distance
         @param neighbors           the neighbor list to use
         @param solventDielectric   the dielectric constant of the bulk solvent
      
         --------------------------------------------------------------------------------------- */
      
      void setUseCutoff(double distance, const OpenMM::NeighborList& neighbors, double solventDielectric);

      /**---------------------------------------------------------------------------------------
      
         Set the force to use periodic boundary conditions.  This requires that a cutoff has
         already been set, and the smallest side of the periodic box is at least twice the cutoff
         distance.
      
         @param vectors    the vectors defining the periodic box
      
         --------------------------------------------------------------------------------------- */
      
      void setPeriodic(OpenMM::Vec3* vectors);
       
      /**---------------------------------------------------------------------------------------
      
         Set the force to use Ewald summation.
      
         @param alpha  the Ewald separation parameter
         @param kmaxx  the largest wave vector in the x direction
         @param kmaxy  the largest wave vector in the y direction
         @param kmaxz  the largest wave vector in the z direction
      
         --------------------------------------------------------------------------------------- */
      
      void setUseEwald(double alpha, int kmaxx, int kmaxy, int kmaxz);

     
      /**---------------------------------------------------------------------------------------

         Set the force to use Particle-Mesh Ewald (PME) summation.

         @param alpha    the Ewald separation parameter
         @param gridSize the dimensions of the mesh

         --------------------------------------------------------------------------------------- */
      
      void setUsePME(double alpha, int meshSize[3]);
      
      /**---------------------------------------------------------------------------------------
      
         Calculate vdW Coulomb pair ixn
      
         @param numberOfAtoms    number of atoms
         @param atomCoordinates  atom coordinates
         @param atomParameters   atom parameters (charges, sigma, ...)     atomParameters[atomIndex][parameterIndex]
         @param exclusions       atom exclusion indices
                                 exclusions[atomIndex] contains the list of exclusions for that atom
         @param fixedParameters  non atom parameters (not currently used)
         @param forces           force array (forces added)
         @param energyByAtom     atom energy
         @param totalEnergy      total energy
         @param includeDirect      true if direct space interactions should be included
         @param includeReciprocal  true if reciprocal space interactions should be included
      
         --------------------------------------------------------------------------------------- */
          
      void calculatePairIxn(int numberOfAtoms, std::vector<OpenMM::Vec3>& atomCoordinates,
                            double** atomParameters, std::vector<std::set<int> >& exclusions,
                            double* fixedParameters, std::vector<OpenMM::Vec3>& forces,
                            double* energyByAtom, double* totalEnergy, bool includeDirect, bool includeReciprocal) const;

private:
      bool cutoff;
      bool periodic;
      bool ewald;
      bool pme;
      const OpenMM::NeighborList* neighborList;
      OpenMM::Vec3 periodicBoxVectors[3];
      double cutoffDistance;
      double taperCutoff;
      double taperCoefficients[3];
      double krf, crf;
      double alphaEwald, alphaDispersionEwald;
      int numRx, numRy, numRz;
      int meshDim[3], dispersionMeshDim[3];

      // parameter indices

      static const int SigIndex;
      static const int GtaIndex;
      static const int adNIndex;
      static const int   QIndex;
            
      // taper coefficient indices

      static const int C3;
      static const int C4;
      static const int C5;

      /**---------------------------------------------------------------------------------------
    
         Set taper coefficients
    
         @param  distance     cutoff distance

       --------------------------------------------------------------------------------------- */
    
      void setTaperCoefficients(double distance);

      /**---------------------------------------------------------------------------------------
      
         Calculate Ewald ixn
      
         @param numberOfAtoms    number of atoms
         @param atomCoordinates  atom coordinates
         @param atomParameters   atom parameters (charges, sigma, ...)     atomParameters[atomIndex][parameterIndex]
         @param exclusions       atom exclusion indices
                                 exclusions[atomIndex] contains the list of exclusions for that atom
         @param fixedParameters  non atom parameters (not currently used)
         @param forces           force array (forces added)
         @param energyByAtom     atom energy
         @param totalEnergy      total energy
         @param includeDirect      true if direct space interactions should be included
         @param includeReciprocal  true if reciprocal space interactions should be included
            
         --------------------------------------------------------------------------------------- */
          
      void calculateEwaldIxn(int numberOfAtoms, std::vector<OpenMM::Vec3>& atomCoordinates,
                            double** atomParameters, std::vector<std::set<int> >& exclusions,
                            double* fixedParameters, std::vector<OpenMM::Vec3>& forces,
                            double* energyByAtom, double* totalEnergy, bool includeDirect, bool includeReciprocal) const;

      /**---------------------------------------------------------------------------------------
      
         Calculate vdW Coulomb pair ixn between two atoms
      
         @param atom1            the index of the first atom
         @param atom2            the index of the second atom
         @param atomCoordinates  atom coordinates
         @param atomParameters   atom parameters (charges, sigma, ...)     atomParameters[atomIndex][parameterIndex]
         @param forces           force array (forces added)
         @param energyByAtom     atom energy
         @param totalEnergy      total energy
            
         --------------------------------------------------------------------------------------- */
          
      void calculateOneIxn(int atom1, int atom2, std::vector<OpenMM::Vec3>& atomCoordinates,
                           double** atomParameters, std::vector<OpenMM::Vec3>& forces,
                           double* energyByAtom, double* totalEnergy) const;

};

} // namespace OpenMM

#endif // __MMFFReferenceNonbondedForce_H__
