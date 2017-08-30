
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

#include "ReferenceForce.h"
#include "ReferenceBondIxn.h"
#include "MMFFReferenceForce.h"
#include "MMFFReferenceTorsionForce.h"
#include "SimTKOpenMMRealType.h"
#include "SimTKOpenMMUtilities.h"
#include "openmm/MMFFConstants.h"

using std::vector;
using namespace OpenMM;

void MMFFReferenceTorsionForce::setPeriodic(OpenMM::Vec3* vectors) {
    usePeriodic = true;
    boxVectors[0] = vectors[0];
    boxVectors[1] = vectors[1];
    boxVectors[2] = vectors[2];
}

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

double MMFFReferenceTorsionForce::calculateTorsionIxn(const Vec3& positionAtomA, const Vec3& positionAtomB,
                                                      const Vec3& positionAtomC, const Vec3& positionAtomD,
                                                      double torsionK1, double torsionK2, double torsionK3,
                                                      Vec3* forces) const {
   double deltaR[3][ReferenceForce::LastDeltaRIndex];
   double crossProductMemory[6];

   // ---------------------------------------------------------------------------------------

   // get deltaR, R2, and R between 2 atoms

   if (usePeriodic) {
      ReferenceForce::getDeltaRPeriodic(positionAtomB, positionAtomA, boxVectors, deltaR[0]);
      ReferenceForce::getDeltaRPeriodic(positionAtomB, positionAtomC, boxVectors, deltaR[1]);
      ReferenceForce::getDeltaRPeriodic(positionAtomD, positionAtomC, boxVectors, deltaR[2]);
   }
   else {
      ReferenceForce::getDeltaR(positionAtomB, positionAtomA, deltaR[0]);
      ReferenceForce::getDeltaR(positionAtomB, positionAtomC, deltaR[1]);
      ReferenceForce::getDeltaR(positionAtomD, positionAtomC, deltaR[2]);
   }

   double cosPhi;
   double signOfAngle;
   int hasREntry = 1;

   std::vector<double> parameters(4);
   // Visual Studio complains if crossProduct declared as 'crossProduct[2][3]'

   double* crossProduct[2];
   crossProduct[0]           = crossProductMemory;
   crossProduct[1]           = crossProductMemory + 3;
   double dihedralAngle = ReferenceBondIxn::getDihedralAngleBetweenThreeVectors(deltaR[0], deltaR[1], deltaR[2],
                                                               crossProduct, &cosPhi, deltaR[0], 
                                                               &signOfAngle, hasREntry);

   // Gromacs: use polymer convention

   if (dihedralAngle < 0.0) {
      dihedralAngle += PI_M;
   } else {
      dihedralAngle -= PI_M;
   }
   cosPhi *= -1.0;

   // Ryckaert-Bellemans:

   // V = sum over i: { C_i*cos(psi)**i }, where psi = phi - PI, 
   //                                              C_i is ith RB coefficient
   parameters[0] = torsionK2 + 0.5*(torsionK1 + torsionK3);
   parameters[1] = 0.5*(3.0*torsionK3 - torsionK1);
   parameters[2] = -torsionK2;
   parameters[3] = -2.0*torsionK3;
   
   double dEdAngle       = 0.0;
   double energy         = parameters[0];
   double cosFactor      = 1.0;
   for (int ii = 1; ii < 4; ii++) {
      dEdAngle  -= ii*parameters[ii]*cosFactor;
      cosFactor *= cosPhi;
      energy    += cosFactor*parameters[ii];
   }

   dEdAngle *= SIN(dihedralAngle);

   double internalF[4][3];
   double forceFactors[4];
   double normCross1         = DOT3(crossProduct[0], crossProduct[0]);
   double normBC             = deltaR[1][ReferenceForce::RIndex];
          forceFactors[0]    = (-dEdAngle*normBC)/normCross1;

   double normCross2         = DOT3(crossProduct[1], crossProduct[1]);
          forceFactors[3]    = (dEdAngle*normBC)/normCross2;

          forceFactors[1]    = DOT3(deltaR[0], deltaR[1]);
          forceFactors[1]   /= deltaR[1][ReferenceForce::R2Index];

          forceFactors[2]    = DOT3(deltaR[2], deltaR[1]);
          forceFactors[2]   /= deltaR[1][ReferenceForce::R2Index];

   for (int ii = 0; ii < 3; ++ii) {

      internalF[0][ii]  = forceFactors[0]*crossProduct[0][ii];
      internalF[3][ii]  = forceFactors[3]*crossProduct[1][ii];

      double s          = forceFactors[1]*internalF[0][ii] - forceFactors[2]*internalF[3][ii]; 

      internalF[1][ii]  = internalF[0][ii] - s;
      internalF[2][ii]  = internalF[3][ii] + s;
   }

   // accumulate forces

   for (int jj = 0; jj < 4; jj++) {
      forces[jj][0] += internalF[jj][0];
      forces[jj][1] += internalF[jj][1];
      forces[jj][2] += internalF[jj][2];
   }

   return energy;
}

double MMFFReferenceTorsionForce::calculateForceAndEnergy(int numTorsions, std::vector<Vec3>& posData,
                                                          const std::vector<int>& particle1,
                                                          const std::vector<int>&  particle2,
                                                          const std::vector<int>&  particle3,
                                                          const std::vector<int>&  particle4,
                                                          const std::vector<double>& k1,
                                                          const std::vector<double>& k2,
                                                          const std::vector<double>& k3,
                                                          std::vector<Vec3>& forceData) const {
    double energy = 0.0;
    for (unsigned int ii = 0; ii < static_cast<unsigned int>(numTorsions); ii++) {
        int particle1Index = particle1[ii];
        int particle2Index = particle2[ii];
        int particle3Index = particle3[ii];
        int particle4Index = particle4[ii];
        double torsionK1 = k1[ii];
        double torsionK2 = k2[ii];
        double torsionK3 = k3[ii];
        Vec3 forces[4];
        energy += calculateTorsionIxn(posData[particle1Index], posData[particle2Index], posData[particle3Index],
                                      posData[particle4Index], torsionK1, torsionK2, torsionK3, forces);

        for (unsigned int jj = 0; jj < 3; jj++) {
            forceData[particle1Index][jj] += forces[0][jj];
            forceData[particle2Index][jj] += forces[1][jj];
            forceData[particle3Index][jj] += forces[2][jj];
            forceData[particle4Index][jj] += forces[3][jj];
        }
    }   
    return energy;
}
