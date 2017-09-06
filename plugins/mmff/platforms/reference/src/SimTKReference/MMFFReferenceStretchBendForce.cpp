
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

#include "MMFFReferenceForce.h"
#include "MMFFReferenceStretchBendForce.h"
#include "SimTKOpenMMRealType.h"
#include <vector>

using std::vector;
using namespace OpenMM;

void MMFFReferenceStretchBendForce::setPeriodic(OpenMM::Vec3* vectors) {
    usePeriodic = true;
    boxVectors[0] = vectors[0];
    boxVectors[1] = vectors[1];
    boxVectors[2] = vectors[2];
}

/**---------------------------------------------------------------------------------------

   Calculate MMFF stretch bend angle ixn (force and energy)

   @param positionAtomA           Cartesian coordinates of atom A
   @param positionAtomB           Cartesian coordinates of atom B
   @param positionAtomC           Cartesian coordinates of atom C
   @param lengthAB                ideal AB bondlength
   @param lengthCB                ideal CB bondlength
   @param idealAngle              ideal angle
   @param k1Parameter             k for distance A-B * angle A-B-C
   @param k2Parameter             k for distance B-C * angle A-B-C
   @param forces                  force vector

   @return energy

   --------------------------------------------------------------------------------------- */

double MMFFReferenceStretchBendForce::calculateStretchBendIxn(const Vec3& positionAtomA, const Vec3& positionAtomB,
                                                                const Vec3& positionAtomC,
                                                                double lengthAB,      double lengthCB,
                                                                double idealAngle,    double k1Parameter,
                                                                double k2Parameter, Vec3* forces) const {

   enum { A, B, C, LastAtomIndex };
   enum { AB, CB, CBxAB, ABxP, CBxP, LastDeltaIndex };

   // ---------------------------------------------------------------------------------------

   // get deltaR between various combinations of the 3 atoms
   // and various intermediate terms

    std::vector<double> deltaR[LastDeltaIndex];
    for (unsigned int ii = 0; ii < LastDeltaIndex; ii++) {
        deltaR[ii].resize(3);
    }
    if (usePeriodic) {
        MMFFReferenceForce::loadDeltaRPeriodic(positionAtomB, positionAtomA, deltaR[AB], boxVectors);
        MMFFReferenceForce::loadDeltaRPeriodic(positionAtomB, positionAtomC, deltaR[CB], boxVectors);
    }
    else {
        MMFFReferenceForce::loadDeltaR(positionAtomB, positionAtomA, deltaR[AB]);
        MMFFReferenceForce::loadDeltaR(positionAtomB, positionAtomC, deltaR[CB]);
    }
    double  rAB2 = MMFFReferenceForce::getNormSquared3(deltaR[AB]);
    double  rAB  = sqrt(rAB2);
    double  rCB2 = MMFFReferenceForce::getNormSquared3(deltaR[CB]);
    double  rCB  = sqrt(rCB2);

    MMFFReferenceForce::getCrossProduct(deltaR[CB], deltaR[AB], deltaR[CBxAB]);
    double  rP   = MMFFReferenceForce::getNorm3(deltaR[CBxAB]);
    if (rP <= 0.0) {
       return 0.0;
    }
    double dot    = MMFFReferenceForce::getDotProduct3(deltaR[CB], deltaR[AB]);
    double cosine = dot/(rAB*rCB);
 
    double angle;
    if (!(cosine < 1.0)) {
       angle = 0.0;
    } else if (!(cosine > -1.0)) {
       angle = PI_M;
    } else {
       angle = ACOS(cosine);
    }
 
    double termA = -1.0/(rAB2*rP);
    double termC =  1.0/(rCB2*rP);
 
    // P = CBxAB
 
    MMFFReferenceForce::getCrossProduct(deltaR[AB], deltaR[CBxAB], deltaR[ABxP]);
    MMFFReferenceForce::getCrossProduct(deltaR[CB], deltaR[CBxAB], deltaR[CBxP]);
    for (int ii = 0; ii < 3; ii++) {
       deltaR[ABxP][ii] *= termA;
       deltaR[CBxP][ii] *= termC;
    }
 
    double dr1   = rAB - lengthAB;
    double dr2   = rCB - lengthCB;
    double drkk  = dr1*k1Parameter + dr2*k2Parameter;
    termA            = 1.0/rAB;
    termC            = 1.0/rCB;
 
    // ---------------------------------------------------------------------------------------
 
    // forces
 
    // calculate forces for atoms a, b, c
    // the force for b is then -(a + c)
 
    std::vector<double> subForce[LastAtomIndex];
    for (int ii = 0; ii < LastAtomIndex; ii++) {
        subForce[ii].resize(3);
    }
    double dt = angle - idealAngle;
    for (int jj = 0; jj < 3; jj++) {
       subForce[A][jj] = k1Parameter*dt*termA*deltaR[AB][jj] + drkk*deltaR[ABxP][jj];
       subForce[C][jj] = k2Parameter*dt*termC*deltaR[CB][jj] + drkk*deltaR[CBxP][jj];
       subForce[B][jj] = -(subForce[A][jj] + subForce[C][jj]);
    }
 
    // add in forces
 
    for (int jj = 0; jj < LastAtomIndex; jj++) {
        forces[jj][0] = subForce[jj][0];
        forces[jj][1] = subForce[jj][1];
        forces[jj][2] = subForce[jj][2];
    }
 
    // ---------------------------------------------------------------------------------------
 
    return dt*drkk;
}

double MMFFReferenceStretchBendForce::calculateForceAndEnergy(int numStretchBends, vector<Vec3>& posData,
                                                                const std::vector<int>&  particle1,
                                                                const std::vector<int>&  particle2,
                                                                const std::vector<int>&  particle3,
                                                                const std::vector<double>& lengthABParameters,
                                                                const std::vector<double>& lengthCBParameters,
                                                                const std::vector<double>&  angle,
                                                                const std::vector<double>&  k1Quadratic,
                                                                const std::vector<double>&  k2Quadratic,
                                                                vector<Vec3>& forceData) const {
    double energy = 0.0; 
    for (unsigned int ii = 0; ii < static_cast<unsigned int>(numStretchBends); ii++) {
        int particle1Index  = particle1[ii];
        int particle2Index  = particle2[ii];
        int particle3Index  = particle3[ii];
        double abLength     = lengthABParameters[ii];
        double cbLength     = lengthCBParameters[ii];
        double idealAngle   = angle[ii];
        double angleK1      = k1Quadratic[ii];
        double angleK2      = k2Quadratic[ii];
        Vec3 forces[3];
        energy             += calculateStretchBendIxn(posData[particle1Index], posData[particle2Index], posData[particle3Index],
                                                      abLength, cbLength, idealAngle, angleK1, angleK2, forces);
        // accumulate forces
    
        for (int jj = 0; jj < 3; jj++) {
            forceData[particle1Index][jj] -= forces[0][jj];
            forceData[particle2Index][jj] -= forces[1][jj];
            forceData[particle3Index][jj] -= forces[2][jj];
        }

    }   
    return energy;
}

