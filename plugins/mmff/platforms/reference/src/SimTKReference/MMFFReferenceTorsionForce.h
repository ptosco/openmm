
/* Portions copyright (c) 2006-2016 Stanford University and Simbios.
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

#ifndef __MMFFReferenceTorsionForce_H__
#define __MMFFReferenceTorsionForce_H__

#include "openmm/Vec3.h"
#include <vector>

namespace OpenMM {

class MMFFReferenceTorsionForce {

public:
 
    /**---------------------------------------------------------------------------------------
       
       Constructor
       
       --------------------------------------------------------------------------------------- */
 
    MMFFReferenceTorsionForce() : usePeriodic(false) {};
 
    /**---------------------------------------------------------------------------------------
       
       Destructor
       
          --------------------------------------------------------------------------------------- */
 
    ~MMFFReferenceTorsionForce() {};
 
    /**---------------------------------------------------------------------------------------

       Set the force to use periodic boundary conditions.
      
       @param vectors    the vectors defining the periodic box
      
       --------------------------------------------------------------------------------------- */
      
    void setPeriodic(OpenMM::Vec3* vectors);

     /**---------------------------------------------------------------------------------------
     
        Calculate MMFF torsion ixns (force and energy)
     
        @param numTorsions             number of torsions
        @param posData                 particle positions
        @param particle1               particle 1 indices
        @param particle2               particle 2 indices
        @param particle3               particle 3 indices
        @param particle4               particle 4 indices
        @param torsionK1               dihedral torsion constant 1
        @param torsionK2               dihedral torsion constant 2
        @param torsionK3               dihedral torsion constant 3
        @param forces                  output force vector
     
        @return total energy

     
        --------------------------------------------------------------------------------------- */

    double calculateForceAndEnergy(int numTorsions, std::vector<OpenMM::Vec3>& posData,
                                   const std::vector<int>& particle1,
                                   const std::vector<int>&  particle2,
                                   const std::vector<int>&  particle3,
                                   const std::vector<int>&  particle4,
                                   const std::vector<double>& k1,
                                   const std::vector<double>& k2,
                                   const std::vector<double>& k3,
                                   std::vector<OpenMM::Vec3>& forceData) const;

private:

    bool usePeriodic;
    Vec3 boxVectors[3];

    /**---------------------------------------------------------------------------------------
    
       Calculate MMFF torsion ixn (force and energy)
    
       @param positionAtomA           Cartesian coordinates of atom A
       @param positionAtomB           Cartesian coordinates of atom B
       @param positionAtomC           Cartesian coordinates of atom C
       @param positionAtomD           Cartesian coordinates of atom D
       @param torsionK1               dihedral torsion constant 1
       @param torsionK2               dihedral torsion constant 2
       @param torsionK3               dihedral torsion constant 3
       @param forces                  force vector
    
       @return energy
    
       --------------------------------------------------------------------------------------- */
    
    double calculateTorsionIxn(const OpenMM::Vec3& positionAtomA, const OpenMM::Vec3& positionAtomB,
                               const OpenMM::Vec3& positionAtomC, const OpenMM::Vec3& positionAtomD,
                               double torsionK1, double torsionK2, double torsionK3,
                               OpenMM::Vec3* forces) const;
         
};

} // namespace OpenMM

#endif // _MMFFReferenceTorsionForce___
