/* -------------------------------------------------------------------------- *
 *                               OpenMMMMFF                                 *
 * -------------------------------------------------------------------------- *
 * This is part of the OpenMM molecular simulation toolkit originating from   *
 * Simbios, the NIH National Center for Physics-Based Simulation of           *
 * Biological Structures at Stanford, funded under the NIH Roadmap for        *
 * Medical Research, grant U54 GM072970. See https://simtk.org.               *
 *                                                                            *
 * Portions copyright (c) 2008-2016 Stanford University and the Authors.      *
 * Authors:                                                                   *
 * Contributors:                                                              *
 *                                                                            *
 * This program is free software: you can redistribute it and/or modify       *
 * it under the terms of the GNU Lesser General Public License as published   *
 * by the Free Software Foundation, either version 3 of the License, or       *
 * (at your option) any later version.                                        *
 *                                                                            *
 * This program is distributed in the hope that it will be useful,            *
 * but WITHOUT ANY WARRANTY; without even the implied warranty of             *
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the              *
 * GNU Lesser General Public License for more details.                        *
 *                                                                            *
 * You should have received a copy of the GNU Lesser General Public License   *
 * along with this program.  If not, see <http://www.gnu.org/licenses/>.      *
 * -------------------------------------------------------------------------- */

#include "MMFFReferenceKernels.h"
#include "MMFFReferenceBondForce.h"
#include "ReferenceBondForce.h"
#include "MMFFReferenceAngleForce.h"
#include "MMFFReferenceTorsionForce.h"
#include "MMFFReferenceStretchBendForce.h"
#include "MMFFReferenceOutOfPlaneBendForce.h"
#include "MMFFReferenceVdwForce.h"
#include "MMFFReferenceNonbondedForce.h"
#include "MMFFReferenceNonbondedForce14.h"
#include "ReferencePlatform.h"
#include "openmm/internal/ContextImpl.h"
#include "openmm/internal/MMFFVdwForceImpl.h"
#include "openmm/MMFFNonbondedForce.h"
#include "openmm/internal/MMFFNonbondedForceImpl.h"
#include "openmm/OpenMMException.h"

#include <cmath>
#ifdef _MSC_VER
#include <windows.h>
#endif

using namespace OpenMM;
using namespace std;

static int** allocateIntArray(int length, int width) {
    int** array = new int*[length];
    for (int i = 0; i < length; ++i)
        array[i] = new int[width];
    return array;
}

static double** allocateRealArray(int length, int width) {
    double** array = new double*[length];
    for (int i = 0; i < length; ++i)
        array[i] = new double[width];
    return array;
}

static void disposeIntArray(int** array, int size) {
    if (array) {
        for (int i = 0; i < size; ++i)
            delete[] array[i];
        delete[] array;
    }
}

static void disposeRealArray(double** array, int size) {
    if (array) {
        for (int i = 0; i < size; ++i)
            delete[] array[i];
        delete[] array;
    }
}

static vector<Vec3>& extractPositions(ContextImpl& context) {
    ReferencePlatform::PlatformData* data = reinterpret_cast<ReferencePlatform::PlatformData*>(context.getPlatformData());
    return *((vector<Vec3>*) data->positions);
}

static vector<Vec3>& extractVelocities(ContextImpl& context) {
    ReferencePlatform::PlatformData* data = reinterpret_cast<ReferencePlatform::PlatformData*>(context.getPlatformData());
    return *((vector<Vec3>*) data->velocities);
}

static vector<Vec3>& extractForces(ContextImpl& context) {
    ReferencePlatform::PlatformData* data = reinterpret_cast<ReferencePlatform::PlatformData*>(context.getPlatformData());
    return *((vector<Vec3>*) data->forces);
}

static Vec3& extractBoxSize(ContextImpl& context) {
    ReferencePlatform::PlatformData* data = reinterpret_cast<ReferencePlatform::PlatformData*>(context.getPlatformData());
    return *(Vec3*) data->periodicBoxSize;
}

static Vec3* extractBoxVectors(ContextImpl& context) {
    ReferencePlatform::PlatformData* data = reinterpret_cast<ReferencePlatform::PlatformData*>(context.getPlatformData());
    return (Vec3*) data->periodicBoxVectors;
}

// ***************************************************************************

ReferenceCalcMMFFBondForceKernel::ReferenceCalcMMFFBondForceKernel(std::string name, const Platform& platform, const System& system) : 
                CalcMMFFBondForceKernel(name, platform), system(system) {
}

ReferenceCalcMMFFBondForceKernel::~ReferenceCalcMMFFBondForceKernel() {
}

void ReferenceCalcMMFFBondForceKernel::initialize(const System& system, const MMFFBondForce& force) {

    numBonds = force.getNumBonds();
    for (int ii = 0; ii < numBonds; ii++) {

        int particle1Index, particle2Index;
        double lengthValue, kValue;
        force.getBondParameters(ii, particle1Index, particle2Index, lengthValue, kValue);

        particle1.push_back(particle1Index); 
        particle2.push_back(particle2Index); 
        length.push_back(static_cast<double>(lengthValue));
        kQuadratic.push_back(kValue);
    } 
    globalBondCubic   = force.getMMFFGlobalBondCubic();
    globalBondQuartic = force.getMMFFGlobalBondQuartic();
    usePeriodic = force.usesPeriodicBoundaryConditions();
}

double ReferenceCalcMMFFBondForceKernel::execute(ContextImpl& context, bool includeForces, bool includeEnergy) {
    vector<Vec3>& posData = extractPositions(context);
    vector<Vec3>& forceData = extractForces(context);
    MMFFReferenceBondForce mmffReferenceBondForce;
    if (usePeriodic)
        mmffReferenceBondForce.setPeriodic(extractBoxVectors(context));
    double energy = mmffReferenceBondForce.calculateForceAndEnergy(numBonds, posData, particle1, particle2, length, kQuadratic,
                                                                     globalBondCubic, globalBondQuartic,
                                                                     forceData);
    return static_cast<double>(energy);
}

void ReferenceCalcMMFFBondForceKernel::copyParametersToContext(ContextImpl& context, const MMFFBondForce& force) {
    if (numBonds != force.getNumBonds())
        throw OpenMMException("updateParametersInContext: The number of bonds has changed");

    // Record the values.

    for (int i = 0; i < numBonds; ++i) {
        int particle1Index, particle2Index;
        double lengthValue, kValue;
        force.getBondParameters(i, particle1Index, particle2Index, lengthValue, kValue);
        if (particle1Index != particle1[i] || particle2Index != particle2[i])
            throw OpenMMException("updateParametersInContext: The set of particles in a bond has changed");
        length[i] = lengthValue;
        kQuadratic[i] = kValue;
    }
}

// ***************************************************************************

ReferenceCalcMMFFAngleForceKernel::ReferenceCalcMMFFAngleForceKernel(std::string name, const Platform& platform, const System& system) :
            CalcMMFFAngleForceKernel(name, platform), system(system) {
}

ReferenceCalcMMFFAngleForceKernel::~ReferenceCalcMMFFAngleForceKernel() {
}

void ReferenceCalcMMFFAngleForceKernel::initialize(const System& system, const MMFFAngleForce& force) {

    numAngles = force.getNumAngles();

    for (int ii = 0; ii < numAngles; ii++) {
        int particle1Index, particle2Index, particle3Index;
        double angleValue, k;
        bool isLinear;
        force.getAngleParameters(ii, particle1Index, particle2Index, particle3Index, angleValue, k, isLinear);
        particle1.push_back(particle1Index); 
        particle2.push_back(particle2Index); 
        particle3.push_back(particle3Index); 
        angle.push_back(angleValue);
        kQuadratic.push_back(k);
        linear.push_back(isLinear);
    }
    globalAngleCubic    = force.getMMFFGlobalAngleCubic();
    usePeriodic = force.usesPeriodicBoundaryConditions();
}

double ReferenceCalcMMFFAngleForceKernel::execute(ContextImpl& context, bool includeForces, bool includeEnergy) {
    vector<Vec3>& posData = extractPositions(context);
    vector<Vec3>& forceData = extractForces(context);
    MMFFReferenceAngleForce mmffReferenceAngleForce;
    if (usePeriodic)
        mmffReferenceAngleForce.setPeriodic(extractBoxVectors(context));
    double energy = mmffReferenceAngleForce.calculateForceAndEnergy(numAngles, 
                                       posData, particle1, particle2, particle3, angle,
                                       kQuadratic, linear, globalAngleCubic, forceData);
    return static_cast<double>(energy);
}

void ReferenceCalcMMFFAngleForceKernel::copyParametersToContext(ContextImpl& context, const MMFFAngleForce& force) {
    if (numAngles != force.getNumAngles())
        throw OpenMMException("updateParametersInContext: The number of angles has changed");

    // Record the values.

    for (int i = 0; i < numAngles; ++i) {
        int particle1Index, particle2Index, particle3Index;
        double angleValue, k;
        bool isLinear;
        force.getAngleParameters(i, particle1Index, particle2Index, particle3Index, angleValue, k, isLinear);
        if (particle1Index != particle1[i] || particle2Index != particle2[i] || particle3Index != particle3[i])
            throw OpenMMException("updateParametersInContext: The set of particles in an angle has changed");
        angle[i] = angleValue;
        kQuadratic[i] = k;
        linear[i] = isLinear;
    }
}

ReferenceCalcMMFFTorsionForceKernel::ReferenceCalcMMFFTorsionForceKernel(std::string name, const Platform& platform, const System& system) :
         CalcMMFFTorsionForceKernel(name, platform), system(system) {
}

ReferenceCalcMMFFTorsionForceKernel::~ReferenceCalcMMFFTorsionForceKernel() {
}

void ReferenceCalcMMFFTorsionForceKernel::initialize(const System& system, const MMFFTorsionForce& force) {

    numTorsions                     = force.getNumTorsions();
    for (int ii = 0; ii < numTorsions; ii++) {

        int particle1Index, particle2Index, particle3Index, particle4Index;
        double k1TorsionParameter, k2TorsionParameter, k3TorsionParameter;
        force.getTorsionParameters(ii, particle1Index, particle2Index, particle3Index, particle4Index, k1TorsionParameter, k2TorsionParameter, k3TorsionParameter);
        particle1.push_back(particle1Index); 
        particle2.push_back(particle2Index); 
        particle3.push_back(particle3Index); 
        particle4.push_back(particle4Index); 
        k1Torsion.push_back(k1TorsionParameter);
        k2Torsion.push_back(k2TorsionParameter);
        k3Torsion.push_back(k3TorsionParameter);
    }
    usePeriodic = force.usesPeriodicBoundaryConditions();
}

double ReferenceCalcMMFFTorsionForceKernel::execute(ContextImpl& context, bool includeForces, bool includeEnergy) {
    vector<Vec3>& posData = extractPositions(context);
    vector<Vec3>& forceData = extractForces(context);
    MMFFReferenceTorsionForce mmffReferenceTorsionForce;
    if (usePeriodic)
        mmffReferenceTorsionForce.setPeriodic(extractBoxVectors(context));
    double energy = mmffReferenceTorsionForce.calculateForceAndEnergy(numTorsions, posData, particle1, particle2,
                                                                      particle3, particle4, 
                                                                      k1Torsion, k2Torsion, k3Torsion, forceData);
    return static_cast<double>(energy);
}

void ReferenceCalcMMFFTorsionForceKernel::copyParametersToContext(ContextImpl& context, const MMFFTorsionForce& force) {
    if (numTorsions != force.getNumTorsions())
        throw OpenMMException("updateParametersInContext: The number of torsions has changed");

    // Record the values.

    for (int i = 0; i < numTorsions; ++i) {
        int particle1Index, particle2Index, particle3Index, particle4Index;
        double k1TorsionParameter, k2TorsionParameter, k3TorsionParameter;
        force.getTorsionParameters(i, particle1Index, particle2Index, particle3Index, particle4Index, k1TorsionParameter, k2TorsionParameter, k3TorsionParameter);
        if (particle1Index != particle1[i] || particle2Index != particle2[i] || particle3Index != particle3[i] ||
            particle4Index != particle4[i])
            throw OpenMMException("updateParametersInContext: The set of particles in a torsion has changed");
        k1Torsion[i] = k1TorsionParameter;
        k2Torsion[i] = k2TorsionParameter;
        k3Torsion[i] = k3TorsionParameter;
    }
}

ReferenceCalcMMFFStretchBendForceKernel::ReferenceCalcMMFFStretchBendForceKernel(std::string name, const Platform& platform, const System& system) :
                   CalcMMFFStretchBendForceKernel(name, platform), system(system) {
}

ReferenceCalcMMFFStretchBendForceKernel::~ReferenceCalcMMFFStretchBendForceKernel() {
}

void ReferenceCalcMMFFStretchBendForceKernel::initialize(const System& system, const MMFFStretchBendForce& force) {

    numStretchBends = force.getNumStretchBends();
    for (int ii = 0; ii < numStretchBends; ii++) {
        int particle1Index, particle2Index, particle3Index;
        double lengthAB, lengthCB, angle, k1, k2;
        force.getStretchBendParameters(ii, particle1Index, particle2Index, particle3Index, lengthAB, lengthCB, angle, k1, k2);
        particle1.push_back(particle1Index); 
        particle2.push_back(particle2Index); 
        particle3.push_back(particle3Index); 
        lengthABParameters.push_back(lengthAB);
        lengthCBParameters.push_back(lengthCB);
        angleParameters.push_back(angle);
        k1Parameters.push_back(k1);
        k2Parameters.push_back(k2);
    }
    usePeriodic = force.usesPeriodicBoundaryConditions();
}

double ReferenceCalcMMFFStretchBendForceKernel::execute(ContextImpl& context, bool includeForces, bool includeEnergy) {
    vector<Vec3>& posData = extractPositions(context);
    vector<Vec3>& forceData = extractForces(context);
    MMFFReferenceStretchBendForce mmffReferenceStretchBendForce;
    if (usePeriodic)
        mmffReferenceStretchBendForce.setPeriodic(extractBoxVectors(context));
    double energy = mmffReferenceStretchBendForce.calculateForceAndEnergy(numStretchBends, posData, particle1, particle2, particle3,
                                                                                      lengthABParameters, lengthCBParameters, angleParameters, k1Parameters,
                                                                                      k2Parameters, forceData);
    return static_cast<double>(energy);
}

void ReferenceCalcMMFFStretchBendForceKernel::copyParametersToContext(ContextImpl& context, const MMFFStretchBendForce& force) {
    if (numStretchBends != force.getNumStretchBends())
        throw OpenMMException("updateParametersInContext: The number of stretch-bends has changed");

    // Record the values.

    for (int i = 0; i < numStretchBends; ++i) {
        int particle1Index, particle2Index, particle3Index;
        double lengthAB, lengthCB, angle, k1, k2;
        force.getStretchBendParameters(i, particle1Index, particle2Index, particle3Index, lengthAB, lengthCB, angle, k1, k2);
        if (particle1Index != particle1[i] || particle2Index != particle2[i] || particle3Index != particle3[i])
            throw OpenMMException("updateParametersInContext: The set of particles in a stretch-bend has changed");
        lengthABParameters[i] = lengthAB;
        lengthCBParameters[i] = lengthCB;
        angleParameters[i] = angle;
        k1Parameters[i] = k1;
        k2Parameters[i] = k2;
    }
}

ReferenceCalcMMFFOutOfPlaneBendForceKernel::ReferenceCalcMMFFOutOfPlaneBendForceKernel(std::string name, const Platform& platform, const System& system) :
          CalcMMFFOutOfPlaneBendForceKernel(name, platform), system(system) {
}

ReferenceCalcMMFFOutOfPlaneBendForceKernel::~ReferenceCalcMMFFOutOfPlaneBendForceKernel() {
}

void ReferenceCalcMMFFOutOfPlaneBendForceKernel::initialize(const System& system, const MMFFOutOfPlaneBendForce& force) {

    numOutOfPlaneBends = force.getNumOutOfPlaneBends();
    for (int ii = 0; ii < numOutOfPlaneBends; ii++) {

        int particle1Index, particle2Index, particle3Index, particle4Index;
        double k;

        force.getOutOfPlaneBendParameters(ii, particle1Index, particle2Index, particle3Index, particle4Index, k);
        particle1.push_back(particle1Index); 
        particle2.push_back(particle2Index); 
        particle3.push_back(particle3Index); 
        particle4.push_back(particle4Index); 
        kParameters.push_back(k);
    }
    usePeriodic = force.usesPeriodicBoundaryConditions();
}

double ReferenceCalcMMFFOutOfPlaneBendForceKernel::execute(ContextImpl& context, bool includeForces, bool includeEnergy) {
    vector<Vec3>& posData = extractPositions(context);
    vector<Vec3>& forceData = extractForces(context);
    MMFFReferenceOutOfPlaneBendForce mmffReferenceOutOfPlaneBendForce;
    if (usePeriodic)
        mmffReferenceOutOfPlaneBendForce.setPeriodic(extractBoxVectors(context));
    double energy = mmffReferenceOutOfPlaneBendForce.calculateForceAndEnergy(numOutOfPlaneBends, posData,
                                                                               particle1, particle2, particle3, particle4,
                                                                               kParameters, forceData); 
    return static_cast<double>(energy);
}

void ReferenceCalcMMFFOutOfPlaneBendForceKernel::copyParametersToContext(ContextImpl& context, const MMFFOutOfPlaneBendForce& force) {
    if (numOutOfPlaneBends != force.getNumOutOfPlaneBends())
        throw OpenMMException("updateParametersInContext: The number of out-of-plane bends has changed");

    // Record the values.

    for (int i = 0; i < numOutOfPlaneBends; ++i) {
        int particle1Index, particle2Index, particle3Index, particle4Index;
        double k;
        force.getOutOfPlaneBendParameters(i, particle1Index, particle2Index, particle3Index, particle4Index, k);
        if (particle1Index != particle1[i] || particle2Index != particle2[i] || particle3Index != particle3[i] || particle4Index != particle4[i])
            throw OpenMMException("updateParametersInContext: The set of particles in an out-of-plane bend has changed");
        kParameters[i] = k;
    }
}

/* -------------------------------------------------------------------------- *
 *                           MMFFVdw                                          *
 * -------------------------------------------------------------------------- */

ReferenceCalcMMFFVdwForceKernel::ReferenceCalcMMFFVdwForceKernel(std::string name, const Platform& platform, const System& system) :
       CalcMMFFVdwForceKernel(name, platform), system(system) {
    useCutoff = 0;
    usePBC = 0;
    cutoff = 1.0e+10;
    neighborList = NULL;
}

ReferenceCalcMMFFVdwForceKernel::~ReferenceCalcMMFFVdwForceKernel() {
    if (neighborList) {
        delete neighborList;
    } 
}

void ReferenceCalcMMFFVdwForceKernel::initialize(const System& system, const MMFFVdwForce& force) {

    // per-particle parameters

    numParticles = system.getNumParticles();

    allExclusions.resize(numParticles);
    sigmas.resize(numParticles);
    G_t_alphas.resize(numParticles);
    alpha_d_Ns.resize(numParticles);
    vdwDAs.resize(numParticles);

    for (int ii = 0; ii < numParticles; ii++) {

        int indexIV;
        double sigma, G_t_alpha, alpha_d_N;
        char vdwDA;
        std::vector<int> exclusions;

        force.getParticleParameters(ii, sigma, G_t_alpha, alpha_d_N, vdwDA);
        force.getParticleExclusions(ii, exclusions);
        for (unsigned int jj = 0; jj < exclusions.size(); jj++) {
           allExclusions[ii].insert(exclusions[jj]);
        }

        sigmas[ii]        = sigma;
        G_t_alphas[ii]    = G_t_alpha;
        alpha_d_Ns[ii]    = alpha_d_N;
        vdwDAs[ii]        = vdwDA;
    }   
    useCutoff              = (force.getNonbondedMethod() != MMFFVdwForce::NoCutoff);
    usePBC                 = (force.getNonbondedMethod() == MMFFVdwForce::CutoffPeriodic);
    cutoff                 = force.getCutoffDistance();
    neighborList           = useCutoff ? new NeighborList() : NULL;
    dispersionCoefficient  = force.getUseDispersionCorrection() ?  MMFFVdwForceImpl::calcDispersionCorrection(system, force) : 0.0;

}

double ReferenceCalcMMFFVdwForceKernel::execute(ContextImpl& context, bool includeForces, bool includeEnergy) {

    vector<Vec3>& posData   = extractPositions(context);
    vector<Vec3>& forceData = extractForces(context);
    MMFFReferenceVdwForce vdwForce;
    double energy;
    if (useCutoff) {
        vdwForce.setCutoff(cutoff);
        computeNeighborListVoxelHash(*neighborList, numParticles, posData, allExclusions, extractBoxVectors(context), usePBC, cutoff, 0.0);
        if (usePBC) {
            vdwForce.setNonbondedMethod(MMFFReferenceVdwForce::CutoffPeriodic);
            Vec3* boxVectors = extractBoxVectors(context);
            double minAllowedSize = 1.999999*cutoff;
            if (boxVectors[0][0] < minAllowedSize || boxVectors[1][1] < minAllowedSize || boxVectors[2][2] < minAllowedSize) {
                throw OpenMMException("The periodic box size has decreased to less than twice the cutoff.");
            }
            vdwForce.setPeriodicBox(boxVectors);
            energy  = vdwForce.calculateForceAndEnergy(numParticles, posData, sigmas, G_t_alphas, alpha_d_Ns, vdwDAs, *neighborList, forceData);
            energy += dispersionCoefficient/(boxVectors[0][0]*boxVectors[1][1]*boxVectors[2][2]);
        } else {
            vdwForce.setNonbondedMethod(MMFFReferenceVdwForce::CutoffNonPeriodic);
        }
    } else {
        vdwForce.setNonbondedMethod(MMFFReferenceVdwForce::NoCutoff);
        energy = vdwForce.calculateForceAndEnergy(numParticles, posData, sigmas, G_t_alphas, alpha_d_Ns, vdwDAs, allExclusions, forceData);
    }
    return static_cast<double>(energy);
}

void ReferenceCalcMMFFVdwForceKernel::copyParametersToContext(ContextImpl& context, const MMFFVdwForce& force) {
    if (numParticles != force.getNumParticles())
        throw OpenMMException("updateParametersInContext: The number of particles has changed");

    // Record the values.

    for (int i = 0; i < numParticles; ++i) {
        double sigma, G_t_alpha, alpha_d_N;
        char vdwDA;
        force.getParticleParameters(i, sigma, G_t_alpha, alpha_d_N, vdwDA);
        sigmas[i] = sigma;
        G_t_alphas[i] = G_t_alpha;
        alpha_d_Ns[i] = alpha_d_N;
        vdwDAs[i]= vdwDA;
    }
}

/* -------------------------------------------------------------------------- *
 *                           MMFFNonbonded                                    *
 * -------------------------------------------------------------------------- */

ReferenceCalcMMFFNonbondedForceKernel::ReferenceCalcMMFFNonbondedForceKernel(std::string name, const Platform& platform, const System& system) :
       CalcMMFFNonbondedForceKernel(name, platform), system(system) {
}

ReferenceCalcMMFFNonbondedForceKernel::~ReferenceCalcMMFFNonbondedForceKernel() {
    disposeRealArray(particleParamArray, numParticles);
    disposeIntArray(bonded14IndexArray, num14);
    disposeRealArray(bonded14ParamArray, num14);
    if (neighborList != NULL)
        delete neighborList;
}

void ReferenceCalcMMFFNonbondedForceKernel::initialize(const System& system, const MMFFNonbondedForce& force) {

    // Identify which exceptions are 1-4 interactions.

    numParticles = force.getNumParticles();
    exclusions.resize(numParticles);
    vector<int> nb14s;
    for (int i = 0; i < force.getNumExceptions(); i++) {
        int particle1, particle2;
        double chargeProd, sigma, epsilon;
        force.getExceptionParameters(i, particle1, particle2, chargeProd, sigma, epsilon);
        exclusions[particle1].insert(particle2);
        exclusions[particle2].insert(particle1);
        if (chargeProd != 0.0 || epsilon != 0.0)
            nb14s.push_back(i);
    }

    // Build the arrays.

    num14 = nb14s.size();
    bonded14IndexArray = allocateIntArray(num14, 2);
    bonded14ParamArray = allocateRealArray(num14, 3);
    particleParamArray = allocateRealArray(numParticles, 4);
    for (int i = 0; i < numParticles; ++i) {
        double charge, sigma, G_t_alpha, alpha_d_N;
        char vdwDA;
        force.getParticleParameters(i, charge, sigma, G_t_alpha, alpha_d_N, vdwDA);
        particleParamArray[i][0] = sigma;
        particleParamArray[i][1] = (vdwDA == 'D') ? -G_t_alpha : G_t_alpha;
        particleParamArray[i][2] = (vdwDA == 'A') ? -alpha_d_N : alpha_d_N;
        particleParamArray[i][3] = charge;
    }
    this->exclusions = exclusions;
    for (int i = 0; i < num14; ++i) {
        int particle1, particle2;
        double chargeProd, sigma, epsilon;
        force.getExceptionParameters(nb14s[i], particle1, particle2, chargeProd, sigma, epsilon);
        bonded14IndexArray[i][0] = particle1;
        bonded14IndexArray[i][1] = particle2;
        bonded14ParamArray[i][0] = sigma;
        bonded14ParamArray[i][1] = epsilon;
        bonded14ParamArray[i][2] = chargeProd;
    }
    nonbondedMethod = CalcMMFFNonbondedForceKernel::NonbondedMethod(force.getNonbondedMethod());
    nonbondedCutoff = force.getCutoffDistance();
    neighborList = (nonbondedMethod == NoCutoff) ? NULL : new NeighborList();
    if (nonbondedMethod == Ewald) {
        double alpha;
        MMFFNonbondedForceImpl::calcEwaldParameters(system, force, alpha, kmax[0], kmax[1], kmax[2]);
        ewaldAlpha = alpha;
    }
    else if (nonbondedMethod == PME) {
        double alpha;
        MMFFNonbondedForceImpl::calcPMEParameters(system, force, alpha, gridSize[0], gridSize[1], gridSize[2], false);
        ewaldAlpha = alpha;
    }
    rfDielectric = force.getReactionFieldDielectric();
    dispersionCoefficient = force.getUseDispersionCorrection() ? MMFFNonbondedForceImpl::calcDispersionCorrection(system, force) : 0.0;
}

double ReferenceCalcMMFFNonbondedForceKernel::execute(ContextImpl& context, bool includeForces, bool includeEnergy, bool includeDirect, bool includeReciprocal) {
    vector<Vec3>& posData = extractPositions(context);
    vector<Vec3>& forceData = extractForces(context);
    double energy = 0;
    MMFFReferenceNonbondedForce cvdw;
    bool periodic = (nonbondedMethod == CutoffPeriodic);
    bool ewald  = (nonbondedMethod == Ewald);
    bool pme  = (nonbondedMethod == PME);
    bool isPeriodicEwaldOrPme = periodic || ewald || pme;
    if (nonbondedMethod != NoCutoff) {
        computeNeighborListVoxelHash(*neighborList, numParticles, posData, exclusions, extractBoxVectors(context), isPeriodicEwaldOrPme, nonbondedCutoff, 0.0);
        cvdw.setUseCutoff(nonbondedCutoff, *neighborList, rfDielectric);
    }
    if (isPeriodicEwaldOrPme) {
        Vec3* boxVectors = extractBoxVectors(context);
        double minAllowedSize = 1.999999*nonbondedCutoff;
        if (boxVectors[0][0] < minAllowedSize || boxVectors[1][1] < minAllowedSize || boxVectors[2][2] < minAllowedSize)
            throw OpenMMException("The periodic box size has decreased to less than twice the nonbonded cutoff.");
        cvdw.setPeriodic(boxVectors);
    }
    if (ewald)
        cvdw.setUseEwald(ewaldAlpha, kmax[0], kmax[1], kmax[2]);
    if (pme)
        cvdw.setUsePME(ewaldAlpha, gridSize);
    cvdw.calculatePairIxn(numParticles, posData, particleParamArray, exclusions, 0, forceData, 0, includeEnergy ? &energy : NULL, includeDirect, includeReciprocal);
    if (includeDirect) {
        ReferenceBondForce refBondForce;
        MMFFReferenceNonbondedForce14 nonbonded14;
        refBondForce.calculateForce(num14, bonded14IndexArray, posData, bonded14ParamArray, forceData, includeEnergy ? &energy : NULL, nonbonded14);
        if (isPeriodicEwaldOrPme) {
            Vec3* boxVectors = extractBoxVectors(context);
            energy += dispersionCoefficient/(boxVectors[0][0]*boxVectors[1][1]*boxVectors[2][2]);
        }
    }
    return energy;
}

void ReferenceCalcMMFFNonbondedForceKernel::copyParametersToContext(ContextImpl& context, const MMFFNonbondedForce& force) {
    if (force.getNumParticles() != numParticles)
        throw OpenMMException("updateParametersInContext: The number of particles has changed");
    vector<int> nb14s;
    for (int i = 0; i < force.getNumExceptions(); i++) {
        int particle1, particle2;
        double chargeProd, sigma, epsilon;
        force.getExceptionParameters(i, particle1, particle2, chargeProd, sigma, epsilon);
        if (chargeProd != 0.0 || epsilon != 0.0)
            nb14s.push_back(i);
    }
    if (nb14s.size() != num14)
        throw OpenMMException("updateParametersInContext: The number of non-excluded exceptions has changed");

    // Record the values.

    for (int i = 0; i < numParticles; ++i) {
        double charge, sigma, G_t_alpha, alpha_d_N;
        char vdwDA;
        force.getParticleParameters(i, charge, sigma, G_t_alpha, alpha_d_N, vdwDA);
        particleParamArray[i][0] = sigma;
        particleParamArray[i][1] = (vdwDA == 'D') ? -G_t_alpha : G_t_alpha;
        particleParamArray[i][2] = (vdwDA == 'A') ? -alpha_d_N : alpha_d_N;
        particleParamArray[i][3] = charge;
    }
    for (int i = 0; i < num14; ++i) {
        int particle1, particle2;
        double charge, sigma, epsilon;
        force.getExceptionParameters(nb14s[i], particle1, particle2, charge, sigma, epsilon);
        bonded14IndexArray[i][0] = particle1;
        bonded14IndexArray[i][1] = particle2;
        bonded14ParamArray[i][0] = sigma;
        bonded14ParamArray[i][1] = epsilon;
        bonded14ParamArray[i][2] = charge;
    }
    
    // Recompute the coefficient for the dispersion correction.

    MMFFNonbondedForce::NonbondedMethod method = force.getNonbondedMethod();
    if (force.getUseDispersionCorrection() && (method == MMFFNonbondedForce::CutoffPeriodic || method == MMFFNonbondedForce::Ewald || method == MMFFNonbondedForce::PME))
        dispersionCoefficient = MMFFNonbondedForceImpl::calcDispersionCorrection(context.getSystem(), force);
}

void ReferenceCalcMMFFNonbondedForceKernel::getPMEParameters(double& alpha, int& nx, int& ny, int& nz) const {
    if (nonbondedMethod != PME)
        throw OpenMMException("getPMEParametersInContext: This Context is not using PME");
    alpha = ewaldAlpha;
    nx = gridSize[0];
    ny = gridSize[1];
    nz = gridSize[2];
}
