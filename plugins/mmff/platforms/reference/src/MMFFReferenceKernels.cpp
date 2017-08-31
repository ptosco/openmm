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
#include "MMFFReferenceAngleForce.h"
#include "MMFFReferenceTorsionForce.h"
#include "MMFFReferenceStretchBendForce.h"
#include "MMFFReferenceOutOfPlaneBendForce.h"
#include "MMFFReferenceVdwForce.h"
#include "MMFFReferenceWcaDispersionForce.h"
#include "MMFFReferenceGeneralizedKirkwoodForce.h"
#include "openmm/internal/MMFFWcaDispersionForceImpl.h"
#include "ReferencePlatform.h"
#include "openmm/internal/ContextImpl.h"
#include "openmm/MMFFMultipoleForce.h"
#include "openmm/internal/MMFFMultipoleForceImpl.h"
#include "openmm/internal/MMFFVdwForceImpl.h"
#include "openmm/internal/MMFFGeneralizedKirkwoodForceImpl.h"
#include "openmm/NonbondedForce.h"
#include "openmm/internal/NonbondedForceImpl.h"

#include <cmath>
#ifdef _MSC_VER
#include <windows.h>
#endif

using namespace OpenMM;
using namespace std;

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
 *                             MMFFMultipole                                *
 * -------------------------------------------------------------------------- */

ReferenceCalcMMFFMultipoleForceKernel::ReferenceCalcMMFFMultipoleForceKernel(std::string name, const Platform& platform, const System& system) : 
         CalcMMFFMultipoleForceKernel(name, platform), system(system), numMultipoles(0), mutualInducedMaxIterations(60), mutualInducedTargetEpsilon(1.0e-03),
                                                         usePme(false),alphaEwald(0.0), cutoffDistance(1.0) {  

}

ReferenceCalcMMFFMultipoleForceKernel::~ReferenceCalcMMFFMultipoleForceKernel() {
}

void ReferenceCalcMMFFMultipoleForceKernel::initialize(const System& system, const MMFFMultipoleForce& force) {

    numMultipoles   = force.getNumMultipoles();

    charges.resize(numMultipoles);
    dipoles.resize(3*numMultipoles);
    quadrupoles.resize(9*numMultipoles);
    tholes.resize(numMultipoles);
    dampingFactors.resize(numMultipoles);
    polarity.resize(numMultipoles);
    axisTypes.resize(numMultipoles);
    multipoleAtomZs.resize(numMultipoles);
    multipoleAtomXs.resize(numMultipoles);
    multipoleAtomYs.resize(numMultipoles);
    multipoleAtomCovalentInfo.resize(numMultipoles);

    int dipoleIndex      = 0;
    int quadrupoleIndex  = 0;
    double totalCharge   = 0.0;
    for (int ii = 0; ii < numMultipoles; ii++) {

        // multipoles

        int axisType, multipoleAtomZ, multipoleAtomX, multipoleAtomY;
        double charge, tholeD, dampingFactorD, polarityD;
        std::vector<double> dipolesD;
        std::vector<double> quadrupolesD;
        force.getMultipoleParameters(ii, charge, dipolesD, quadrupolesD, axisType, multipoleAtomZ, multipoleAtomX, multipoleAtomY,
                                     tholeD, dampingFactorD, polarityD);

        totalCharge                       += charge;
        axisTypes[ii]                      = axisType;
        multipoleAtomZs[ii]                = multipoleAtomZ;
        multipoleAtomXs[ii]                = multipoleAtomX;
        multipoleAtomYs[ii]                = multipoleAtomY;

        charges[ii]                        = charge;
        tholes[ii]                         = tholeD;
        dampingFactors[ii]                 = dampingFactorD;
        polarity[ii]                       = polarityD;

        dipoles[dipoleIndex++]             = dipolesD[0];
        dipoles[dipoleIndex++]             = dipolesD[1];
        dipoles[dipoleIndex++]             = dipolesD[2];
        
        quadrupoles[quadrupoleIndex++]     = quadrupolesD[0];
        quadrupoles[quadrupoleIndex++]     = quadrupolesD[1];
        quadrupoles[quadrupoleIndex++]     = quadrupolesD[2];
        quadrupoles[quadrupoleIndex++]     = quadrupolesD[3];
        quadrupoles[quadrupoleIndex++]     = quadrupolesD[4];
        quadrupoles[quadrupoleIndex++]     = quadrupolesD[5];
        quadrupoles[quadrupoleIndex++]     = quadrupolesD[6];
        quadrupoles[quadrupoleIndex++]     = quadrupolesD[7];
        quadrupoles[quadrupoleIndex++]     = quadrupolesD[8];

        // covalent info

        std::vector< std::vector<int> > covalentLists;
        force.getCovalentMaps(ii, covalentLists);
        multipoleAtomCovalentInfo[ii] = covalentLists;

    }

    polarizationType = force.getPolarizationType();
    if (polarizationType == MMFFMultipoleForce::Mutual) {
        mutualInducedMaxIterations = force.getMutualInducedMaxIterations();
        mutualInducedTargetEpsilon = force.getMutualInducedTargetEpsilon();
    } else if (polarizationType == MMFFMultipoleForce::Extrapolated) {
        extrapolationCoefficients = force.getExtrapolationCoefficients();
    }

    // PME

    nonbondedMethod  = force.getNonbondedMethod();
    if (nonbondedMethod == MMFFMultipoleForce::PME) {
        usePme     = true;
        pmeGridDimension.resize(3);
        force.getPMEParameters(alphaEwald, pmeGridDimension[0], pmeGridDimension[1], pmeGridDimension[2]);
        cutoffDistance = force.getCutoffDistance();
        if (pmeGridDimension[0] == 0 || alphaEwald == 0.0) {
            NonbondedForce nb;
            nb.setEwaldErrorTolerance(force.getEwaldErrorTolerance());
            nb.setCutoffDistance(force.getCutoffDistance());
            int gridSizeX, gridSizeY, gridSizeZ;
            NonbondedForceImpl::calcPMEParameters(system, nb, alphaEwald, gridSizeX, gridSizeY, gridSizeZ, false);
            pmeGridDimension[0] = gridSizeX;
            pmeGridDimension[1] = gridSizeY;
            pmeGridDimension[2] = gridSizeZ;
        }    
    } else {
        usePme = false;
    }
    return;
}

MMFFReferenceMultipoleForce* ReferenceCalcMMFFMultipoleForceKernel::setupMMFFReferenceMultipoleForce(ContextImpl& context)
{

    // mmffReferenceMultipoleForce is set to MMFFReferenceGeneralizedKirkwoodForce if MMFFGeneralizedKirkwoodForce is present
    // mmffReferenceMultipoleForce is set to MMFFReferencePmeMultipoleForce if 'usePme' is set
    // mmffReferenceMultipoleForce is set to MMFFReferenceMultipoleForce otherwise

    // check if MMFFGeneralizedKirkwoodForce is present 

    ReferenceCalcMMFFGeneralizedKirkwoodForceKernel* gkKernel = NULL;
    for (unsigned int ii = 0; ii < context.getForceImpls().size() && gkKernel == NULL; ii++) {
        MMFFGeneralizedKirkwoodForceImpl* gkImpl = dynamic_cast<MMFFGeneralizedKirkwoodForceImpl*>(context.getForceImpls()[ii]);
        if (gkImpl != NULL) {
            gkKernel = dynamic_cast<ReferenceCalcMMFFGeneralizedKirkwoodForceKernel*>(&gkImpl->getKernel().getImpl());
        }
    }    

    MMFFReferenceMultipoleForce* mmffReferenceMultipoleForce = NULL;
    if (gkKernel) {

        // mmffReferenceGeneralizedKirkwoodForce is deleted in MMFFReferenceGeneralizedKirkwoodMultipoleForce
        // destructor

        MMFFReferenceGeneralizedKirkwoodForce* mmffReferenceGeneralizedKirkwoodForce = new MMFFReferenceGeneralizedKirkwoodForce();
        mmffReferenceGeneralizedKirkwoodForce->setNumParticles(gkKernel->getNumParticles());
        mmffReferenceGeneralizedKirkwoodForce->setSoluteDielectric(gkKernel->getSoluteDielectric());
        mmffReferenceGeneralizedKirkwoodForce->setSolventDielectric(gkKernel->getSolventDielectric());
        mmffReferenceGeneralizedKirkwoodForce->setDielectricOffset(gkKernel->getDielectricOffset());
        mmffReferenceGeneralizedKirkwoodForce->setProbeRadius(gkKernel->getProbeRadius());
        mmffReferenceGeneralizedKirkwoodForce->setSurfaceAreaFactor(gkKernel->getSurfaceAreaFactor());
        mmffReferenceGeneralizedKirkwoodForce->setIncludeCavityTerm(gkKernel->getIncludeCavityTerm());
        mmffReferenceGeneralizedKirkwoodForce->setDirectPolarization(gkKernel->getDirectPolarization());

        vector<double> parameters; 
        gkKernel->getAtomicRadii(parameters);
        mmffReferenceGeneralizedKirkwoodForce->setAtomicRadii(parameters);

        gkKernel->getScaleFactors(parameters);
        mmffReferenceGeneralizedKirkwoodForce->setScaleFactors(parameters);

        gkKernel->getCharges(parameters);
        mmffReferenceGeneralizedKirkwoodForce->setCharges(parameters);

        // calculate Grycuk Born radii

        vector<Vec3>& posData   = extractPositions(context);
        mmffReferenceGeneralizedKirkwoodForce->calculateGrycukBornRadii(posData);

        mmffReferenceMultipoleForce = new MMFFReferenceGeneralizedKirkwoodMultipoleForce(mmffReferenceGeneralizedKirkwoodForce);

    } else if (usePme) {

        MMFFReferencePmeMultipoleForce* mmffReferencePmeMultipoleForce = new MMFFReferencePmeMultipoleForce();
        mmffReferencePmeMultipoleForce->setAlphaEwald(alphaEwald);
        mmffReferencePmeMultipoleForce->setCutoffDistance(cutoffDistance);
        mmffReferencePmeMultipoleForce->setPmeGridDimensions(pmeGridDimension);
        Vec3* boxVectors = extractBoxVectors(context);
        double minAllowedSize = 1.999999*cutoffDistance;
        if (boxVectors[0][0] < minAllowedSize || boxVectors[1][1] < minAllowedSize || boxVectors[2][2] < minAllowedSize) {
            throw OpenMMException("The periodic box size has decreased to less than twice the nonbonded cutoff.");
        }
        mmffReferencePmeMultipoleForce->setPeriodicBoxSize(boxVectors);
        mmffReferenceMultipoleForce = static_cast<MMFFReferenceMultipoleForce*>(mmffReferencePmeMultipoleForce);

    } else {
         mmffReferenceMultipoleForce = new MMFFReferenceMultipoleForce(MMFFReferenceMultipoleForce::NoCutoff);
    }

    // set polarization type

    if (polarizationType == MMFFMultipoleForce::Mutual) {
        mmffReferenceMultipoleForce->setPolarizationType(MMFFReferenceMultipoleForce::Mutual);
        mmffReferenceMultipoleForce->setMutualInducedDipoleTargetEpsilon(mutualInducedTargetEpsilon);
        mmffReferenceMultipoleForce->setMaximumMutualInducedDipoleIterations(mutualInducedMaxIterations);
    } else if (polarizationType == MMFFMultipoleForce::Direct) {
        mmffReferenceMultipoleForce->setPolarizationType(MMFFReferenceMultipoleForce::Direct);
    } else if (polarizationType == MMFFMultipoleForce::Extrapolated) {
        mmffReferenceMultipoleForce->setPolarizationType(MMFFReferenceMultipoleForce::Extrapolated);
        mmffReferenceMultipoleForce->setExtrapolationCoefficients(extrapolationCoefficients);
    } else {
        throw OpenMMException("Polarization type not recognzied.");
    }

    return mmffReferenceMultipoleForce;

}

double ReferenceCalcMMFFMultipoleForceKernel::execute(ContextImpl& context, bool includeForces, bool includeEnergy) {

    MMFFReferenceMultipoleForce* mmffReferenceMultipoleForce = setupMMFFReferenceMultipoleForce(context);

    vector<Vec3>& posData = extractPositions(context);
    vector<Vec3>& forceData = extractForces(context);
    double energy = mmffReferenceMultipoleForce->calculateForceAndEnergy(posData, charges, dipoles, quadrupoles, tholes,
                                                                           dampingFactors, polarity, axisTypes, 
                                                                           multipoleAtomZs, multipoleAtomXs, multipoleAtomYs,
                                                                           multipoleAtomCovalentInfo, forceData);

    delete mmffReferenceMultipoleForce;

    return static_cast<double>(energy);
}

void ReferenceCalcMMFFMultipoleForceKernel::getInducedDipoles(ContextImpl& context, vector<Vec3>& outputDipoles) {
    int numParticles = context.getSystem().getNumParticles();
    outputDipoles.resize(numParticles);

    // Create an MMFFReferenceMultipoleForce to do the calculation.
    
    MMFFReferenceMultipoleForce* mmffReferenceMultipoleForce = setupMMFFReferenceMultipoleForce(context);
    vector<Vec3>& posData = extractPositions(context);
    
    // Retrieve the induced dipoles.
    
    vector<Vec3> inducedDipoles;
    mmffReferenceMultipoleForce->calculateInducedDipoles(posData, charges, dipoles, quadrupoles, tholes,
            dampingFactors, polarity, axisTypes, multipoleAtomZs, multipoleAtomXs, multipoleAtomYs, multipoleAtomCovalentInfo, inducedDipoles);
    for (int i = 0; i < numParticles; i++)
        outputDipoles[i] = inducedDipoles[i];
    delete mmffReferenceMultipoleForce;
}

void ReferenceCalcMMFFMultipoleForceKernel::getLabFramePermanentDipoles(ContextImpl& context, vector<Vec3>& outputDipoles) {
    int numParticles = context.getSystem().getNumParticles();
    outputDipoles.resize(numParticles);

    // Create an MMFFReferenceMultipoleForce to do the calculation.
    
    MMFFReferenceMultipoleForce* mmffReferenceMultipoleForce = setupMMFFReferenceMultipoleForce(context);
    vector<Vec3>& posData = extractPositions(context);
    
    // Retrieve the permanent dipoles in the lab frame.
    
    vector<Vec3> labFramePermanentDipoles;
    mmffReferenceMultipoleForce->calculateLabFramePermanentDipoles(posData, charges, dipoles, quadrupoles, tholes, 
            dampingFactors, polarity, axisTypes, multipoleAtomZs, multipoleAtomXs, multipoleAtomYs, multipoleAtomCovalentInfo, labFramePermanentDipoles);
    for (int i = 0; i < numParticles; i++)
        outputDipoles[i] = labFramePermanentDipoles[i];
    delete mmffReferenceMultipoleForce;
}


void ReferenceCalcMMFFMultipoleForceKernel::getTotalDipoles(ContextImpl& context, vector<Vec3>& outputDipoles) {
    int numParticles = context.getSystem().getNumParticles();
    outputDipoles.resize(numParticles);

    // Create an MMFFReferenceMultipoleForce to do the calculation.
    
    MMFFReferenceMultipoleForce* mmffReferenceMultipoleForce = setupMMFFReferenceMultipoleForce(context);
    vector<Vec3>& posData = extractPositions(context);
    
    // Retrieve the permanent dipoles in the lab frame.
    
    vector<Vec3> totalDipoles;
    mmffReferenceMultipoleForce->calculateTotalDipoles(posData, charges, dipoles, quadrupoles, tholes,
            dampingFactors, polarity, axisTypes, multipoleAtomZs, multipoleAtomXs, multipoleAtomYs, multipoleAtomCovalentInfo, totalDipoles);

    for (int i = 0; i < numParticles; i++)
        outputDipoles[i] = totalDipoles[i];
    delete mmffReferenceMultipoleForce;
}



void ReferenceCalcMMFFMultipoleForceKernel::getElectrostaticPotential(ContextImpl& context, const std::vector< Vec3 >& inputGrid,
                                                                        std::vector< double >& outputElectrostaticPotential) {

    MMFFReferenceMultipoleForce* mmffReferenceMultipoleForce = setupMMFFReferenceMultipoleForce(context);
    vector<Vec3>& posData                                     = extractPositions(context);
    vector<Vec3> grid(inputGrid.size());
    vector<double> potential(inputGrid.size());
    for (unsigned int ii = 0; ii < inputGrid.size(); ii++) {
        grid[ii] = inputGrid[ii];
    }
    mmffReferenceMultipoleForce->calculateElectrostaticPotential(posData, charges, dipoles, quadrupoles, tholes,
                                                                   dampingFactors, polarity, axisTypes, 
                                                                   multipoleAtomZs, multipoleAtomXs, multipoleAtomYs,
                                                                   multipoleAtomCovalentInfo, grid, potential);

    outputElectrostaticPotential.resize(inputGrid.size());
    for (unsigned int ii = 0; ii < inputGrid.size(); ii++) {
        outputElectrostaticPotential[ii] = potential[ii];
    }

    delete mmffReferenceMultipoleForce;
}

void ReferenceCalcMMFFMultipoleForceKernel::getSystemMultipoleMoments(ContextImpl& context, std::vector< double >& outputMultipoleMoments) {

    // retrieve masses

    const System& system             = context.getSystem();
    vector<double> masses;
    for (int i = 0; i <  system.getNumParticles(); ++i) {
        masses.push_back(system.getParticleMass(i));
    }    

    MMFFReferenceMultipoleForce* mmffReferenceMultipoleForce = setupMMFFReferenceMultipoleForce(context);
    vector<Vec3>& posData                                     = extractPositions(context);
    mmffReferenceMultipoleForce->calculateMMFFSystemMultipoleMoments(masses, posData, charges, dipoles, quadrupoles, tholes,
                                                                         dampingFactors, polarity, axisTypes, 
                                                                         multipoleAtomZs, multipoleAtomXs, multipoleAtomYs,
                                                                         multipoleAtomCovalentInfo, outputMultipoleMoments);

    delete mmffReferenceMultipoleForce;
}

void ReferenceCalcMMFFMultipoleForceKernel::copyParametersToContext(ContextImpl& context, const MMFFMultipoleForce& force) {
    if (numMultipoles != force.getNumMultipoles())
        throw OpenMMException("updateParametersInContext: The number of multipoles has changed");

    // Record the values.

    int dipoleIndex = 0;
    int quadrupoleIndex = 0;
    for (int i = 0; i < numMultipoles; ++i) {
        int axisType, multipoleAtomZ, multipoleAtomX, multipoleAtomY;
        double charge, tholeD, dampingFactorD, polarityD;
        std::vector<double> dipolesD;
        std::vector<double> quadrupolesD;
        force.getMultipoleParameters(i, charge, dipolesD, quadrupolesD, axisType, multipoleAtomZ, multipoleAtomX, multipoleAtomY, tholeD, dampingFactorD, polarityD);
        axisTypes[i] = axisType;
        multipoleAtomZs[i] = multipoleAtomZ;
        multipoleAtomXs[i] = multipoleAtomX;
        multipoleAtomYs[i] = multipoleAtomY;
        charges[i] = charge;
        tholes[i] = tholeD;
        dampingFactors[i] = dampingFactorD;
        polarity[i] = polarityD;
        dipoles[dipoleIndex++] = dipolesD[0];
        dipoles[dipoleIndex++] = dipolesD[1];
        dipoles[dipoleIndex++] = dipolesD[2];
        quadrupoles[quadrupoleIndex++] = quadrupolesD[0];
        quadrupoles[quadrupoleIndex++] = quadrupolesD[1];
        quadrupoles[quadrupoleIndex++] = quadrupolesD[2];
        quadrupoles[quadrupoleIndex++] = quadrupolesD[3];
        quadrupoles[quadrupoleIndex++] = quadrupolesD[4];
        quadrupoles[quadrupoleIndex++] = quadrupolesD[5];
        quadrupoles[quadrupoleIndex++] = quadrupolesD[6];
        quadrupoles[quadrupoleIndex++] = quadrupolesD[7];
        quadrupoles[quadrupoleIndex++] = quadrupolesD[8];
    }
}

void ReferenceCalcMMFFMultipoleForceKernel::getPMEParameters(double& alpha, int& nx, int& ny, int& nz) const {
    if (!usePme)
        throw OpenMMException("getPMEParametersInContext: This Context is not using PME");
    alpha = alphaEwald;
    nx = pmeGridDimension[0];
    ny = pmeGridDimension[1];
    nz = pmeGridDimension[2];
}

/* -------------------------------------------------------------------------- *
 *                       MMFFGeneralizedKirkwood                            *
 * -------------------------------------------------------------------------- */

ReferenceCalcMMFFGeneralizedKirkwoodForceKernel::ReferenceCalcMMFFGeneralizedKirkwoodForceKernel(std::string name, const Platform& platform, const System& system) : 
           CalcMMFFGeneralizedKirkwoodForceKernel(name, platform), system(system) {
}

ReferenceCalcMMFFGeneralizedKirkwoodForceKernel::~ReferenceCalcMMFFGeneralizedKirkwoodForceKernel() {
}

int ReferenceCalcMMFFGeneralizedKirkwoodForceKernel::getNumParticles() const {
    return numParticles;
}

int ReferenceCalcMMFFGeneralizedKirkwoodForceKernel::getIncludeCavityTerm() const {
    return includeCavityTerm;
}

int ReferenceCalcMMFFGeneralizedKirkwoodForceKernel::getDirectPolarization() const {
    return directPolarization;
}

double ReferenceCalcMMFFGeneralizedKirkwoodForceKernel::getSoluteDielectric() const {
    return soluteDielectric;
}

double ReferenceCalcMMFFGeneralizedKirkwoodForceKernel::getSolventDielectric() const {
    return solventDielectric;
}

double ReferenceCalcMMFFGeneralizedKirkwoodForceKernel::getDielectricOffset() const {
    return dielectricOffset;
}

double ReferenceCalcMMFFGeneralizedKirkwoodForceKernel::getProbeRadius() const {
    return probeRadius;
}

double ReferenceCalcMMFFGeneralizedKirkwoodForceKernel::getSurfaceAreaFactor() const {
    return surfaceAreaFactor;
}

void ReferenceCalcMMFFGeneralizedKirkwoodForceKernel::getAtomicRadii(vector<double>& outputAtomicRadii) const {
    outputAtomicRadii.resize(atomicRadii.size());
    copy(atomicRadii.begin(), atomicRadii.end(), outputAtomicRadii.begin());
}

void ReferenceCalcMMFFGeneralizedKirkwoodForceKernel::getScaleFactors(vector<double>& outputScaleFactors) const {
    outputScaleFactors.resize(scaleFactors.size());
    copy(scaleFactors.begin(), scaleFactors.end(), outputScaleFactors.begin());
}

void ReferenceCalcMMFFGeneralizedKirkwoodForceKernel::getCharges(vector<double>& outputCharges) const {
    outputCharges.resize(charges.size());
    copy(charges.begin(), charges.end(), outputCharges.begin());
}

void ReferenceCalcMMFFGeneralizedKirkwoodForceKernel::initialize(const System& system, const MMFFGeneralizedKirkwoodForce& force) {

    // check that MMFFMultipoleForce is present

    const MMFFMultipoleForce* mmffMultipoleForce = NULL;
    for (int ii = 0; ii < system.getNumForces() && mmffMultipoleForce == NULL; ii++) {
        mmffMultipoleForce = dynamic_cast<const MMFFMultipoleForce*>(&system.getForce(ii));
    }

    if (mmffMultipoleForce == NULL) {
        throw OpenMMException("MMFFGeneralizedKirkwoodForce requires the System to also contain an MMFFMultipoleForce.");
    }

    if (mmffMultipoleForce->getNonbondedMethod() != MMFFMultipoleForce::NoCutoff) {
        throw OpenMMException("MMFFGeneralizedKirkwoodForce requires the MMFFMultipoleForce use the NoCutoff nonbonded method.");
    }

    numParticles = system.getNumParticles();

    for (int ii = 0; ii < numParticles; ii++) {

        double particleCharge, particleRadius, scalingFactor;
        force.getParticleParameters(ii, particleCharge, particleRadius, scalingFactor);
        atomicRadii.push_back(particleRadius);
        scaleFactors.push_back(scalingFactor);
        charges.push_back(particleCharge);

        // Make sure the charge matches the one specified by the MMFFMultipoleForce.

        double charge2, thole, damping, polarity;
        int axisType, atomX, atomY, atomZ;
        vector<double> dipole, quadrupole;
        mmffMultipoleForce->getMultipoleParameters(ii, charge2, dipole, quadrupole, axisType, atomZ, atomX, atomY, thole, damping, polarity);
        if (particleCharge != charge2) {
            throw OpenMMException("MMFFGeneralizedKirkwoodForce and MMFFMultipoleForce must specify the same charge for every atom.");
        }

    }   
    includeCavityTerm  = force.getIncludeCavityTerm();
    soluteDielectric   = force.getSoluteDielectric();
    solventDielectric  = force.getSolventDielectric();
    dielectricOffset   = 0.009;
    probeRadius        = force.getProbeRadius(), 
    surfaceAreaFactor  = force.getSurfaceAreaFactor(); 
    directPolarization = mmffMultipoleForce->getPolarizationType() == MMFFMultipoleForce::Direct ? 1 : 0;
}

double ReferenceCalcMMFFGeneralizedKirkwoodForceKernel::execute(ContextImpl& context, bool includeForces, bool includeEnergy) {
    // handled in MMFFReferenceGeneralizedKirkwoodMultipoleForce, a derived class of the class MMFFReferenceMultipoleForce
    return 0.0;
}

void ReferenceCalcMMFFGeneralizedKirkwoodForceKernel::copyParametersToContext(ContextImpl& context, const MMFFGeneralizedKirkwoodForce& force) {
    if (numParticles != force.getNumParticles())
        throw OpenMMException("updateParametersInContext: The number of particles has changed");

    // Record the values.

    for (int i = 0; i < numParticles; ++i) {
        double particleCharge, particleRadius, scalingFactor;
        force.getParticleParameters(i, particleCharge, particleRadius, scalingFactor);
        atomicRadii[i] = particleRadius;
        scaleFactors[i] = scalingFactor;
        charges[i] = particleCharge;
    }
}

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
 *                           MMFFWcaDispersion                              *
 * -------------------------------------------------------------------------- */

ReferenceCalcMMFFWcaDispersionForceKernel::ReferenceCalcMMFFWcaDispersionForceKernel(std::string name, const Platform& platform, const System& system) : 
           CalcMMFFWcaDispersionForceKernel(name, platform), system(system) {
}

ReferenceCalcMMFFWcaDispersionForceKernel::~ReferenceCalcMMFFWcaDispersionForceKernel() {
}

void ReferenceCalcMMFFWcaDispersionForceKernel::initialize(const System& system, const MMFFWcaDispersionForce& force) {

    // per-particle parameters

    numParticles = system.getNumParticles();
    radii.resize(numParticles);
    epsilons.resize(numParticles);
    for (int ii = 0; ii < numParticles; ii++) {

        double radius, epsilon;
        force.getParticleParameters(ii, radius, epsilon);

        radii[ii] = radius;
        epsilons[ii] = epsilon;
    }   

    totalMaximumDispersionEnergy = MMFFWcaDispersionForceImpl::getTotalMaximumDispersionEnergy(force);

    epso    = force.getEpso();
    epsh    = force.getEpsh();
    rmino   = force.getRmino();
    rminh   = force.getRminh();
    awater  = force.getAwater();
    shctd   = force.getShctd();
    dispoff = force.getDispoff();
    slevy   = force.getSlevy();
}

double ReferenceCalcMMFFWcaDispersionForceKernel::execute(ContextImpl& context, bool includeForces, bool includeEnergy) {
    vector<Vec3>& posData = extractPositions(context);
    vector<Vec3>& forceData = extractForces(context);
    MMFFReferenceWcaDispersionForce mmffReferenceWcaDispersionForce(epso, epsh, rmino, rminh, awater, shctd, dispoff, slevy);
    double energy = mmffReferenceWcaDispersionForce.calculateForceAndEnergy(numParticles, posData, radii, epsilons, totalMaximumDispersionEnergy, forceData);
    return static_cast<double>(energy);
}

void ReferenceCalcMMFFWcaDispersionForceKernel::copyParametersToContext(ContextImpl& context, const MMFFWcaDispersionForce& force) {
    if (numParticles != force.getNumParticles())
        throw OpenMMException("updateParametersInContext: The number of particles has changed");

    // Record the values.

    for (int i = 0; i < numParticles; ++i) {
        double radius, epsilon;
        force.getParticleParameters(i, radius, epsilon);
        radii[i] = radius;
        epsilons[i] = epsilon;
    }
    totalMaximumDispersionEnergy = MMFFWcaDispersionForceImpl::getTotalMaximumDispersionEnergy(force);
}
