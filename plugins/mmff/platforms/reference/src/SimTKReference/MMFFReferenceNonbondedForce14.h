
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

#ifndef __MMFFReferenceNonbondedForce14_H__
#define __MMFFReferenceNonbondedForce14_H__

#include "ReferenceBondIxn.h"
#include "openmm/internal/windowsExport.h"

namespace OpenMM {

class OPENMM_EXPORT MMFFReferenceNonbondedForce14 : public ReferenceBondIxn {

   public:

      /**---------------------------------------------------------------------------------------
      
         Constructor
      
         --------------------------------------------------------------------------------------- */

       MMFFReferenceNonbondedForce14();

      /**---------------------------------------------------------------------------------------
      
         Destructor
      
         --------------------------------------------------------------------------------------- */

       ~MMFFReferenceNonbondedForce14();

      /**---------------------------------------------------------------------------------------
      
         Calculate vdW 1-4 ixn
      
         @param atomIndices      atom indices of 2 interacting atoms
         @param atomCoordinates  atom coordinates
         @param parameters       four parameters:
                                        parameters[0]= sigma
                                        parameters[1]= G_t_alpha
                                        parameters[2]= alpha_d_N
                                        parameters[3]= charge
         @param forces           force array (forces added to current values)
         @param totalEnergy      if not null, the energy will be added to this
            
         --------------------------------------------------------------------------------------- */
      
      void calculateBondIxn(int* atomIndices, std::vector<OpenMM::Vec3>& atomCoordinates,
                            double* parameters, std::vector<OpenMM::Vec3>& forces,
                            double* totalEnergy, double* energyParamDerivs);

};

} // namespace OpenMM

#endif // __MMFFReferenceNonbondedForce14_H__
