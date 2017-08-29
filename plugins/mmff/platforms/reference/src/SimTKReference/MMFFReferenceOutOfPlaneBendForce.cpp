
/* Portions copyright (c) 2006 Stanford University and Simbios.
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
#include "MMFFReferenceOutOfPlaneBendForce.h"
#include "SimTKOpenMMRealType.h"
#include "openmm/MMFFConstants.h"

using std::vector;
using namespace OpenMM;

void MMFFReferenceOutOfPlaneBendForce::setPeriodic(OpenMM::Vec3* vectors) {
    usePeriodic = true;
    boxVectors[0] = vectors[0];
    boxVectors[1] = vectors[1];
    boxVectors[2] = vectors[2];
}

/**---------------------------------------------------------------------------------------

   Calculate MMFF Out-Of-Plane-Bend  ixn (force and energy)

   @param positionAtomA           Cartesian coordinates of atom A
   @param positionAtomB           Cartesian coordinates of atom B
   @param positionAtomC           Cartesian coordinates of atom C
   @param positionAtomD           Cartesian coordinates of atom D
   @param angleLength             angle
   @param angleK                  quadratic angle force
   @param forces                  force vector

   @return energy

   --------------------------------------------------------------------------------------- */

double MMFFReferenceOutOfPlaneBendForce::calculateOutOfPlaneBendIxn(const Vec3& positionAtomA, const Vec3& positionAtomB,
                                                                      const Vec3& positionAtomC, const Vec3& positionAtomD,
                                                                      double angleK, Vec3* forces) const {

    enum { A, B, C, D, LastAtomIndex };
    enum { AB, CB, DB, AD, CD, LastDeltaIndex };
 
    // get deltaR between various combinations of the 4 atoms
    // and various intermediate terms
 
    std::vector<double> deltaR[LastDeltaIndex];
    for (int ii = 0; ii < LastDeltaIndex; ii++) {
        deltaR[ii].resize(3);
    }
    if (usePeriodic) {
        MMFFReferenceForce::loadDeltaRPeriodic(positionAtomB, positionAtomA, deltaR[AB], boxVectors);
        MMFFReferenceForce::loadDeltaRPeriodic(positionAtomB, positionAtomC, deltaR[CB], boxVectors);
        MMFFReferenceForce::loadDeltaRPeriodic(positionAtomB, positionAtomD, deltaR[DB], boxVectors);
        MMFFReferenceForce::loadDeltaRPeriodic(positionAtomD, positionAtomA, deltaR[AD], boxVectors);
        MMFFReferenceForce::loadDeltaRPeriodic(positionAtomD, positionAtomC, deltaR[CD], boxVectors);
    }
    else {
        MMFFReferenceForce::loadDeltaR(positionAtomB, positionAtomA, deltaR[AB]);
        MMFFReferenceForce::loadDeltaR(positionAtomB, positionAtomC, deltaR[CB]);
        MMFFReferenceForce::loadDeltaR(positionAtomB, positionAtomD, deltaR[DB]);
        MMFFReferenceForce::loadDeltaR(positionAtomD, positionAtomA, deltaR[AD]);
        MMFFReferenceForce::loadDeltaR(positionAtomD, positionAtomC, deltaR[CD]);
    }

    double rDB2  = MMFFReferenceForce::getNormSquared3(deltaR[DB]);
    double rAD2  = MMFFReferenceForce::getNormSquared3(deltaR[AD]);
    double rCD2  = MMFFReferenceForce::getNormSquared3(deltaR[CD]);
 
    std::vector<double> tempVector(3);
    MMFFReferenceForce::getCrossProduct(deltaR[CB], deltaR[DB], tempVector);
    double eE = MMFFReferenceForce::getDotProduct3(deltaR[AB], tempVector);
    double dot = MMFFReferenceForce::getDotProduct3(deltaR[AD],  deltaR[CD]);
    double cc = rAD2*rCD2 - dot*dot;
 
    if (rDB2 <= 0.0 || cc == 0.0) {
       return 0.0;
    }
    double bkk2   = rDB2 - eE*eE/cc;
    double cosine = sqrt(bkk2/rDB2);
    double angle;
    if (!(cosine < 1.0)) {
       angle = 0.0;
    } else if (!(cosine > -1.0)) {
       angle = RADIAN*PI_M;
    } else {
       angle = RADIAN*ACOS(cosine);
    }
 
    // chain rule
 
    double dt    = angle;
    double dt2   = dt*dt;
 
    double dEdDt = MMFF_OOP_C1*angleK*dt;
    // calculate energy if 'energy' is set
    double energy = 0.5*dEdDt*dt;
    dEdDt *= RADIAN;
 
    double dEdCos  = dEdDt/sqrt(cc*bkk2);
    if (eE > 0.0) {
       dEdCos *= -1.0;
    }
 
    double term = eE/cc;
 
    std::vector<double> dccd[LastAtomIndex];
    std::vector<double> deed[LastAtomIndex];
    std::vector<double> subForce[LastAtomIndex];
    for (int ii = 0; ii < LastAtomIndex; ii++) {
        dccd[ii].resize(3);
        deed[ii].resize(3);
        subForce[ii].resize(3);
    }   
    for (int ii = 0; ii < 3; ii++) {
       dccd[A][ii] = (deltaR[AD][ii]*rCD2 - deltaR[CD][ii]*dot)*term;
       dccd[C][ii] = (deltaR[CD][ii]*rAD2 - deltaR[AD][ii]*dot)*term;
       dccd[D][ii] = -1.0*(dccd[A][ii] + dccd[C][ii]);
    }
 
    MMFFReferenceForce::getCrossProduct(deltaR[DB], deltaR[CB], deed[A]);
    MMFFReferenceForce::getCrossProduct(deltaR[AB], deltaR[DB], deed[C]);
    MMFFReferenceForce::getCrossProduct(deltaR[CB], deltaR[AB], deed[D]);
 
    term        = eE/rDB2;
    deed[D][0] += deltaR[DB][0]*term;
    deed[D][1] += deltaR[DB][1]*term;
    deed[D][2] += deltaR[DB][2]*term;
 
    // ---------------------------------------------------------------------------------------
 
    // forces
 
    // calculate forces for atoms a, c, d
    // the force for b is then -(a+ c + d)
 
 
    for (int jj = 0; jj < LastAtomIndex; jj++) {
 
       // A, C, D
 
       for (int ii = 0; ii < 3; ii++) {
          subForce[jj][ii] = dEdCos*(dccd[jj][ii] + deed[jj][ii]);
       }
 
       if (jj == 0)jj++; // skip B
 
       // now compute B
 
       if (jj == 3) {
          for (int ii = 0; ii < 3; ii++) {
             subForce[1][ii] = -1.0*(subForce[0][ii] + subForce[2][ii] + subForce[3][ii]);
          }
       }
    }
 
    // add in forces
 
    for (int jj = 0; jj < LastAtomIndex; jj++) {
       for (int ii = 0; ii < 3; ii++) {
          forces[jj][ii] = subForce[jj][ii];
       }
    }
 
    // ---------------------------------------------------------------------------------------
 
    return energy;
}

double MMFFReferenceOutOfPlaneBendForce::calculateForceAndEnergy(int numOutOfPlaneBends, vector<Vec3>& posData,
                                                                   const std::vector<int>&  particle1,
                                                                   const std::vector<int>&  particle2,
                                                                   const std::vector<int>&  particle3,
                                                                   const std::vector<int>&  particle4,
                                                                   const std::vector<double>&  kQuadratic,
                                                                   vector<Vec3>& forceData) const {
    double energy      = 0.0; 
    for (unsigned int ii = 0; ii < static_cast<unsigned int>(numOutOfPlaneBends); ii++) {
        int particle1Index      = particle1[ii];
        int particle2Index      = particle2[ii];
        int particle3Index      = particle3[ii];
        int particle4Index      = particle4[ii];
        double kAngle           = kQuadratic[ii];
        Vec3 forces[4];
        energy                 += calculateOutOfPlaneBendIxn(posData[particle1Index], posData[particle2Index], posData[particle3Index], posData[particle4Index],
                                                              kAngle, forces);
        for (int jj = 0; jj < 3; jj++) {
            forceData[particle1Index][jj] -= forces[0][jj];
            forceData[particle2Index][jj] -= forces[1][jj];
            forceData[particle3Index][jj] -= forces[2][jj];
            forceData[particle4Index][jj] -= forces[3][jj];
        }

    }   
    return energy;
}

