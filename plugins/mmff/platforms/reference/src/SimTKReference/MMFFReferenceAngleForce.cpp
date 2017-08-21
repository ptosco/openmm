
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
#include "MMFFReferenceAngleForce.h"
#include "SimTKOpenMMRealType.h"
#include "openmm/MMFFConstants.h"

using std::vector;
using namespace OpenMM;

void MMFFReferenceAngleForce::setPeriodic(OpenMM::Vec3* vectors) {
    usePeriodic = true;
    boxVectors[0] = vectors[0];
    boxVectors[1] = vectors[1];
    boxVectors[2] = vectors[2];
}

/**---------------------------------------------------------------------------------------

   Get dEdT and energy prefactor given cosine of angle :: the calculation for different
   the force types is identical 

   @param  cosine               cosine of angle
   @param  idealAngle           ideal angle
   @param  angleK               angle k (quadratic prefactor)
   @param  angleCubic           cubic prefactor
   @param  isLinear             true if the angle is linear
   @param  dEdR                 dEdR

   @return energy

   --------------------------------------------------------------------------------------- */

double MMFFReferenceAngleForce::getPrefactorsGivenAngleCosine(double cosine,
                                                                double idealAngle,     double angleK,
                                                                double angleCubic,     bool isLinear,
                                                                double* dEdR) const {

    double angle = -1.0;
    static const double MIN_SINE = 1.0e-8;
    double energy = MMFF_ANGLE_C1*angleK;
    if (!(cosine < 1.0)) {
        angle = 0.0;
        cosine = 1.0;
    } else if (!(cosine > -1.0)) {
        angle = RADIAN*PI_M;
        cosine = -1.0;
    }
    if (isLinear) {
        double sine = std::max(sqrt(1.0 - cosine*cosine), MIN_SINE);
        *dEdR                     = -energy*sine;
        energy                   *= 1.0 + cosine;
    } else {
        if (angle < 0.0)
            angle = RADIAN*ACOS(cosine);
        double deltaIdeal         = angle - idealAngle;
        double deltaIdeal2        = deltaIdeal*deltaIdeal;
        energy                   *= 0.5*deltaIdeal2 * (1.0 + MMFF_ANGLE_CUBIC_K*deltaIdeal);
        *dEdR                     = RADIAN*MMFF_ANGLE_C1*angleK*deltaIdeal*(1.0 + 1.5*MMFF_ANGLE_CUBIC_K*deltaIdeal);
    }

    return energy;

}

/**---------------------------------------------------------------------------------------

   Calculate MMFF angle ixn (force and energy)

   @param positionAtomA           Cartesian coordinates of atom A
   @param positionAtomB           Cartesian coordinates of atom B
   @param positionAtomC           Cartesian coordinates of atom C
   @param angleLength             angle
   @param angleK                  quadratic angle force
   @param angleCubic              cubic angle force parameter
   @param forces                  force vector

   @return energy

   --------------------------------------------------------------------------------------- */

double MMFFReferenceAngleForce::calculateAngleIxn(const Vec3& positionAtomA, const Vec3& positionAtomB,
                                                    const Vec3& positionAtomC,
                                                    double angle,          double angleK,
                                                    double angleCubic,     bool isLinear,
                                                    Vec3* forces) const {

    std::vector<double> deltaR[2];
    if (usePeriodic)
        MMFFReferenceForce::loadDeltaRPeriodic(positionAtomA, positionAtomB, deltaR[0], boxVectors);
    else
        MMFFReferenceForce::loadDeltaR(positionAtomA, positionAtomB, deltaR[0]);
    double rAB2      = MMFFReferenceForce::getNormSquared3(deltaR[0]);
    double rAB       = SQRT(rAB2);
 
    if (usePeriodic)
        MMFFReferenceForce::loadDeltaRPeriodic(positionAtomC, positionAtomB, deltaR[1], boxVectors);
    else
        MMFFReferenceForce::loadDeltaR(positionAtomC, positionAtomB, deltaR[1]);
    double rCB2      = MMFFReferenceForce::getNormSquared3(deltaR[1]);
    double rCB       = SQRT(rCB2);
 
    if (rAB <= 0.0 || rCB <= 0.0) {
       return 0.0;
    }
 
    std::vector<double> pVector(3);
    MMFFReferenceForce::getCrossProduct(deltaR[0], deltaR[1], pVector);
    double rp = MMFFReferenceForce::getNorm3(pVector);
    if (rp < 1.0e-06) {
       rp = 1.0e-06;
    }
    double dot    = MMFFReferenceForce::getDotProduct3(deltaR[0], deltaR[1]);
    double cosine = dot/(rAB*rCB);
 
    double dEdR;
    double energy = getPrefactorsGivenAngleCosine(cosine, angle, angleK, angleCubic, isLinear, &dEdR);
 
    double termA  =  dEdR/(rAB2*rp);
    double termC  = -dEdR/(rCB2*rp);
 
    std::vector<double> deltaCrossP[3];
    deltaCrossP[0].resize(3); 
    deltaCrossP[1].resize(3); 
    deltaCrossP[2].resize(3); 
    MMFFReferenceForce::getCrossProduct(deltaR[0], pVector, deltaCrossP[0]);
    MMFFReferenceForce::getCrossProduct(deltaR[1], pVector, deltaCrossP[2]);
    for (unsigned int ii = 0; ii < 3; ii++) {
       deltaCrossP[0][ii] *= termA;
       deltaCrossP[2][ii] *= termC;
       deltaCrossP[1][ii]  = -1.0f*(deltaCrossP[0][ii] + deltaCrossP[2][ii]);
    }
 
    // accumulate forces
 
    for (int jj = 0; jj < 3; jj++) {
        forces[jj][0] = deltaCrossP[jj][0];
        forces[jj][1] = deltaCrossP[jj][1];
        forces[jj][2] = deltaCrossP[jj][2];
    }
 
    return energy;
}

double MMFFReferenceAngleForce::calculateForceAndEnergy(int numAngles, vector<Vec3>& posData,
                                                          const std::vector<int>&  particle1,
                                                          const std::vector<int>&  particle2,
                                                          const std::vector<int>&  particle3,
                                                          const std::vector<double>&  angle,
                                                          const std::vector<double>&  kQuadratic,
                                                          const std::vector<bool>&  linear,
                                                          double angleCubic,
                                                          vector<Vec3>& forceData) const {
    double energy = 0.0; 
    for (unsigned int ii = 0; ii < static_cast<unsigned int>(numAngles); ii++) {
        int particle1Index = particle1[ii];
        int particle2Index = particle2[ii];
        int particle3Index = particle3[ii];
        double idealAngle = angle[ii];
        double angleK = kQuadratic[ii];
        bool isLinear = linear[ii];
        Vec3 forces[3];
        energy += calculateAngleIxn(posData[particle1Index], posData[particle2Index], posData[particle3Index],
                                    idealAngle, angleK, angleCubic, isLinear, forces);

        for (unsigned int jj = 0; jj < 3; jj++) {
            forceData[particle1Index][jj] += forces[0][jj];
            forceData[particle2Index][jj] += forces[1][jj];
            forceData[particle3Index][jj] += forces[2][jj];
        }
    }   
    return energy;
}
