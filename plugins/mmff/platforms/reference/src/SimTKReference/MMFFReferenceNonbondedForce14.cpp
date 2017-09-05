
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

#include <string.h>
#include <sstream>

#include "SimTKOpenMMUtilities.h"
#include "MMFFReferenceNonbondedForce.h"
#include "MMFFReferenceNonbondedForce14.h"
#include "ReferenceForce.h"

using std::vector;
using namespace OpenMM;

/**---------------------------------------------------------------------------------------

   MMFFReferenceNonbondedForce14 constructor

   --------------------------------------------------------------------------------------- */

MMFFReferenceNonbondedForce14::MMFFReferenceNonbondedForce14() {
}

/**---------------------------------------------------------------------------------------

   MMFFReferenceNonbondedForce14 destructor

   --------------------------------------------------------------------------------------- */

MMFFReferenceNonbondedForce14::~MMFFReferenceNonbondedForce14() {
}

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

void MMFFReferenceNonbondedForce14::calculateBondIxn(int* atomIndices, vector<Vec3>& atomCoordinates,
                                     double* parameters, vector<Vec3>& forces,
                                     double* totalEnergy, double* energyParamDerivs) {
    double deltaR[2][ReferenceForce::LastDeltaRIndex];

    // get deltaR, R2, and R between 2 atoms

    int atomAIndex = atomIndices[0];
    int atomBIndex = atomIndices[1];
    ReferenceForce::getDeltaR(atomCoordinates[atomBIndex], atomCoordinates[atomAIndex], deltaR[0]);  

    double inverseR     = 1.0/(deltaR[0][ReferenceForce::RIndex]);
    double r_ij         = deltaR[0][ReferenceForce::RIndex];
    double rBuf         = r_ij + MMFFReferenceNonbondedForce::eleBuf;
    double inverseRbuf  = 1.0/rBuf;
    double r_ij_2       = deltaR[0][ReferenceForce::R2Index];
    double sigma_7      = parameters[0]*parameters[0]*parameters[0];
           sigma_7      = sigma_7*sigma_7*parameters[0];

    double r_ij_6       = r_ij_2*r_ij_2*r_ij_2;
    double r_ij_7       = r_ij_6*r_ij;

    double rho          = r_ij_7 + MMFFReferenceNonbondedForce::ghal*sigma_7;

    double tau          = (MMFFReferenceNonbondedForce::dhal + 1.0)/(r_ij + MMFFReferenceNonbondedForce::dhal*parameters[0]);
    double tau_7        = tau*tau*tau;
           tau_7        = tau_7*tau_7*tau;

    double dtau         = tau/(MMFFReferenceNonbondedForce::dhal + 1.0);

    double ratio        = (sigma_7/rho);
    double gtau         = parameters[1]*tau_7*r_ij_6*(MMFFReferenceNonbondedForce::ghal+1.0)*ratio*ratio;

    double vdwEnergy    = parameters[1]*tau_7*sigma_7*((MMFFReferenceNonbondedForce::ghal+1.0)*sigma_7/rho - 2.0);
    double vdw_dEdR     = 7.0*(dtau*vdwEnergy + gtau);

    double coulEnergy   = ONE_4PI_EPS0*parameters[2]*inverseRbuf;
    double dEdR         = vdw_dEdR*inverseR + coulEnergy*inverseRbuf*inverseR;

   // accumulate forces

   for (int ii = 0; ii < 3; ii++) {
      double force        = dEdR*deltaR[0][ii];
      forces[atomAIndex][ii] += force;
      forces[atomBIndex][ii] -= force;
   }

   // accumulate energies

   if (totalEnergy != NULL)
       *totalEnergy += vdwEnergy + coulEnergy;
}
