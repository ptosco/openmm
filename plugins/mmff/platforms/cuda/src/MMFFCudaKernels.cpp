/* -------------------------------------------------------------------------- *
 *                               OpenMMMMFF                                 *
 * -------------------------------------------------------------------------- *
 * This is part of the OpenMM molecular simulation toolkit originating from   *
 * Simbios, the NIH National Center for Physics-Based Simulation of           *
 * Biological Structures at Stanford, funded under the NIH Roadmap for        *
 * Medical Research, grant U54 GM072970. See https://simtk.org.               *
 *                                                                            *
 * Portions copyright (c) 2008-2016 Stanford University and the Authors.      *
 * Authors: Peter Eastman, Mark Friedrichs                                    *
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

#ifdef WIN32
  #define _USE_MATH_DEFINES // Needed to get M_PI
#endif
#include "MMFFCudaKernels.h"
#include "CudaMMFFKernelSources.h"
#include "openmm/internal/ContextImpl.h"
#include "openmm/internal/MMFFGeneralizedKirkwoodForceImpl.h"
#include "openmm/internal/MMFFMultipoleForceImpl.h"
#include "openmm/internal/MMFFWcaDispersionForceImpl.h"
#include "openmm/internal/MMFFVdwForceImpl.h"
#include "openmm/internal/NonbondedForceImpl.h"
#include "openmm/MMFFConstants.h"
#include "CudaBondedUtilities.h"
#include "CudaFFT3D.h"
#include "CudaForceInfo.h"
#include "CudaKernelSources.h"
#include "CudaNonbondedUtilities.h"
#include "jama_lu.h"

#include <algorithm>
#include <cmath>
#ifdef _MSC_VER
#include <windows.h>
#endif

using namespace OpenMM;
using namespace std;

#define CHECK_RESULT(result, prefix) \
    if (result != CUDA_SUCCESS) { \
        std::stringstream m; \
        m<<prefix<<": "<<cu.getErrorString(result)<<" ("<<result<<")"<<" at "<<__FILE__<<":"<<__LINE__; \
        throw OpenMMException(m.str());\
    }

/* -------------------------------------------------------------------------- *
 *                            MMFFBondForce                                 *
 * -------------------------------------------------------------------------- */

class CudaCalcMMFFBondForceKernel::ForceInfo : public CudaForceInfo {
public:
    ForceInfo(const MMFFBondForce& force) : force(force) {
    }
    int getNumParticleGroups() {
        return force.getNumBonds();
    }
    void getParticlesInGroup(int index, std::vector<int>& particles) {
        int particle1, particle2;
        double length, k;
        force.getBondParameters(index, particle1, particle2, length, k);
        particles.resize(2);
        particles[0] = particle1;
        particles[1] = particle2;
    }
    bool areGroupsIdentical(int group1, int group2) {
        int particle1, particle2;
        double length1, length2, k1, k2;
        force.getBondParameters(group1, particle1, particle2, length1, k1);
        force.getBondParameters(group2, particle1, particle2, length2, k2);
        return (length1 == length2 && k1 == k2);
    }
private:
    const MMFFBondForce& force;
};

CudaCalcMMFFBondForceKernel::CudaCalcMMFFBondForceKernel(std::string name, const Platform& platform, CudaContext& cu, const System& system) : 
                CalcMMFFBondForceKernel(name, platform), cu(cu), system(system), params(NULL) {
}

CudaCalcMMFFBondForceKernel::~CudaCalcMMFFBondForceKernel() {
    cu.setAsCurrent();
    if (params != NULL)
        delete params;
}

void CudaCalcMMFFBondForceKernel::initialize(const System& system, const MMFFBondForce& force) {
    cu.setAsCurrent();
    int numContexts = cu.getPlatformData().contexts.size();
    int startIndex = cu.getContextIndex()*force.getNumBonds()/numContexts;
    int endIndex = (cu.getContextIndex()+1)*force.getNumBonds()/numContexts;
    numBonds = endIndex-startIndex;
    if (numBonds == 0)
        return;
    vector<vector<int> > atoms(numBonds, vector<int>(2));
    params = CudaArray::create<float2>(cu, numBonds, "bondParams");
    vector<float2> paramVector(numBonds);
    for (int i = 0; i < numBonds; i++) {
        double length, k;
        force.getBondParameters(startIndex+i, atoms[i][0], atoms[i][1], length, k);
        paramVector[i] = make_float2((float) length, (float) k);
    }
    params->upload(paramVector);
    map<string, string> replacements;
    replacements["APPLY_PERIODIC"] = (force.usesPeriodicBoundaryConditions() ? "1" : "0");
    replacements["COMPUTE_FORCE"] = CudaMMFFKernelSources::mmffBondForce;
    replacements["PARAMS"] = cu.getBondedUtilities().addArgument(params->getDevicePointer(), "float2");
    replacements["CUBIC_K"] = cu.doubleToString(force.getMMFFGlobalBondCubic());
    replacements["QUARTIC_K"] = cu.doubleToString(force.getMMFFGlobalBondQuartic());
    cu.getBondedUtilities().addInteraction(atoms, cu.replaceStrings(CudaKernelSources::bondForce, replacements), force.getForceGroup());
    cu.addForce(new ForceInfo(force));
}

double CudaCalcMMFFBondForceKernel::execute(ContextImpl& context, bool includeForces, bool includeEnergy) {
    return 0.0;
}

void CudaCalcMMFFBondForceKernel::copyParametersToContext(ContextImpl& context, const MMFFBondForce& force) {
    cu.setAsCurrent();
    int numContexts = cu.getPlatformData().contexts.size();
    int startIndex = cu.getContextIndex()*force.getNumBonds()/numContexts;
    int endIndex = (cu.getContextIndex()+1)*force.getNumBonds()/numContexts;
    if (numBonds != endIndex-startIndex)
        throw OpenMMException("updateParametersInContext: The number of bonds has changed");
    if (numBonds == 0)
        return;
    
    // Record the per-bond parameters.
    
    vector<float2> paramVector(numBonds);
    for (int i = 0; i < numBonds; i++) {
        int atom1, atom2;
        double length, k;
        force.getBondParameters(startIndex+i, atom1, atom2, length, k);
        paramVector[i] = make_float2((float) length, (float) k);
    }
    params->upload(paramVector);
    
    // Mark that the current reordering may be invalid.
    
    cu.invalidateMolecules();
}

/* -------------------------------------------------------------------------- *
 *                            MMFFAngleForce                                *
 * -------------------------------------------------------------------------- */

class CudaCalcMMFFAngleForceKernel::ForceInfo : public CudaForceInfo {
public:
    ForceInfo(const MMFFAngleForce& force) : force(force) {
    }
    int getNumParticleGroups() {
        return force.getNumAngles();
    }
    void getParticlesInGroup(int index, std::vector<int>& particles) {
        int particle1, particle2, particle3;
        double angle, k;
        bool isLinear;
        force.getAngleParameters(index, particle1, particle2, particle3, angle, k, isLinear);
        particles.resize(3);
        particles[0] = particle1;
        particles[1] = particle2;
        particles[2] = particle3;
    }
    bool areGroupsIdentical(int group1, int group2) {
        int particle1, particle2, particle3;
        double angle1, angle2, k1, k2;
        bool isLinear1, isLinear2;
        force.getAngleParameters(group1, particle1, particle2, particle3, angle1, k1, isLinear1);
        force.getAngleParameters(group2, particle1, particle2, particle3, angle2, k2, isLinear2);
        return (angle1 == angle2 && k1 == k2 && isLinear1 == isLinear2);
    }
private:
    const MMFFAngleForce& force;
};

CudaCalcMMFFAngleForceKernel::CudaCalcMMFFAngleForceKernel(std::string name, const Platform& platform, CudaContext& cu, const System& system) :
            CalcMMFFAngleForceKernel(name, platform), cu(cu), system(system), params(NULL) {
}

CudaCalcMMFFAngleForceKernel::~CudaCalcMMFFAngleForceKernel() {
    cu.setAsCurrent();
    if (params != NULL)
        delete params;
}

void CudaCalcMMFFAngleForceKernel::initialize(const System& system, const MMFFAngleForce& force) {
    cu.setAsCurrent();
    int numContexts = cu.getPlatformData().contexts.size();
    int startIndex = cu.getContextIndex()*force.getNumAngles()/numContexts;
    int endIndex = (cu.getContextIndex()+1)*force.getNumAngles()/numContexts;
    numAngles = endIndex-startIndex;
    if (numAngles == 0)
        return;
    vector<vector<int> > atoms(numAngles, vector<int>(3));
    params = CudaArray::create<float2>(cu, numAngles, "angleParams");
    vector<float2> paramVector(numAngles);
    for (int i = 0; i < numAngles; i++) {
        double angle, k;
        bool isLinear;
        force.getAngleParameters(startIndex+i, atoms[i][0], atoms[i][1], atoms[i][2], angle, k, isLinear);
        paramVector[i] = make_float2((float) angle, (float) (isLinear ? -k : k));
    }
    params->upload(paramVector);
    map<string, string> replacements;
    replacements["APPLY_PERIODIC"] = (force.usesPeriodicBoundaryConditions() ? "1" : "0");
    replacements["COMPUTE_FORCE"] = CudaMMFFKernelSources::mmffAngleForce;
    replacements["PARAMS"] = cu.getBondedUtilities().addArgument(params->getDevicePointer(), "float2");
    replacements["CUBIC_K"] = cu.doubleToString(force.getMMFFGlobalAngleCubic());
    replacements["RAD_TO_DEG"] = cu.doubleToString(180/M_PI);
    cu.getBondedUtilities().addInteraction(atoms, cu.replaceStrings(CudaKernelSources::angleForce, replacements), force.getForceGroup());
    cu.addForce(new ForceInfo(force));
}

double CudaCalcMMFFAngleForceKernel::execute(ContextImpl& context, bool includeForces, bool includeEnergy) {
    return 0.0;
}

void CudaCalcMMFFAngleForceKernel::copyParametersToContext(ContextImpl& context, const MMFFAngleForce& force) {
    cu.setAsCurrent();
    int numContexts = cu.getPlatformData().contexts.size();
    int startIndex = cu.getContextIndex()*force.getNumAngles()/numContexts;
    int endIndex = (cu.getContextIndex()+1)*force.getNumAngles()/numContexts;
    if (numAngles != endIndex-startIndex)
        throw OpenMMException("updateParametersInContext: The number of angles has changed");
    if (numAngles == 0)
        return;
    
    // Record the per-angle parameters.
    
    vector<float2> paramVector(numAngles);
    for (int i = 0; i < numAngles; i++) {
        int atom1, atom2, atom3;
        double angle, k;
        bool isLinear;
        force.getAngleParameters(startIndex+i, atom1, atom2, atom3, angle, k, isLinear);
        paramVector[i] = make_float2((float) angle, (float) (isLinear ? -k : k));
    }
    params->upload(paramVector);
    
    // Mark that the current reordering may be invalid.
    
    cu.invalidateMolecules();
}

/* -------------------------------------------------------------------------- *
  *                              MMFFTorsion                              *
 * -------------------------------------------------------------------------- */

class CudaCalcMMFFTorsionForceKernel::ForceInfo : public CudaForceInfo {
public:
    ForceInfo(const MMFFTorsionForce& force) : force(force) {
    }
    int getNumParticleGroups() {
        return force.getNumTorsions();
    }
    void getParticlesInGroup(int index, std::vector<int>& particles) {
        int particle1, particle2, particle3, particle4;
        double k1, k2 ,k3;
        force.getTorsionParameters(index, particle1, particle2, particle3, particle4, k1, k2, k3);
        particles.resize(4);
        particles[0] = particle1;
        particles[1] = particle2;
        particles[2] = particle3;
        particles[3] = particle4;
    }
    bool areGroupsIdentical(int group1, int group2) {
        int particle1, particle2, particle3, particle4;
        double k11, k21, k31;
        double k12, k22, k32;
        force.getTorsionParameters(group1, particle1, particle2, particle3, particle4, k11, k21, k31);
        force.getTorsionParameters(group2, particle1, particle2, particle3, particle4, k12, k22, k32);
        return (k11 == k12 && k21 == k22 && k31 == k32);
    }
private:
    const MMFFTorsionForce& force;
};

CudaCalcMMFFTorsionForceKernel::CudaCalcMMFFTorsionForceKernel(std::string name, const Platform& platform, CudaContext& cu, const System& system) :
         CalcMMFFTorsionForceKernel(name, platform), cu(cu), system(system), params(NULL) {
}

CudaCalcMMFFTorsionForceKernel::~CudaCalcMMFFTorsionForceKernel() {
    cu.setAsCurrent();
    if (params != NULL)
        delete params;
}

void CudaCalcMMFFTorsionForceKernel::initialize(const System& system, const MMFFTorsionForce& force) {
    cu.setAsCurrent();
    int numContexts = cu.getPlatformData().contexts.size();
    int startIndex = cu.getContextIndex()*force.getNumTorsions()/numContexts;
    int endIndex = (cu.getContextIndex()+1)*force.getNumTorsions()/numContexts;
    numTorsions = endIndex-startIndex;
    if (numTorsions == 0)
        return;
    vector<vector<int> > atoms(numTorsions, vector<int>(4));
    params = CudaArray::create<float3>(cu, numTorsions, "torsionParams");
    vector<float3> paramVector(numTorsions);
    for (int i = 0; i < numTorsions; i++) {
        double k1, k2, k3;
        force.getTorsionParameters(startIndex+i, atoms[i][0], atoms[i][1], atoms[i][2], atoms[i][3], k1, k2, k3);
        paramVector[i] = make_float3((float) k1, (float) k2, (float) k3);
    }
    params->upload(paramVector);
    map<string, string> replacements;
    replacements["APPLY_PERIODIC"] = (force.usesPeriodicBoundaryConditions() ? "1" : "0");
    replacements["COMPUTE_FORCE"] = CudaMMFFKernelSources::mmffTorsionForce;
    replacements["PARAMS"] = cu.getBondedUtilities().addArgument(params->getDevicePointer(), "float3");
    cu.getBondedUtilities().addInteraction(atoms, cu.replaceStrings(CudaKernelSources::torsionForce, replacements), force.getForceGroup());
    cu.addForce(new ForceInfo(force));
}

double CudaCalcMMFFTorsionForceKernel::execute(ContextImpl& context, bool includeForces, bool includeEnergy) {
    return 0.0;
}

void CudaCalcMMFFTorsionForceKernel::copyParametersToContext(ContextImpl& context, const MMFFTorsionForce& force) {
    cu.setAsCurrent();
    int numContexts = cu.getPlatformData().contexts.size();
    int startIndex = cu.getContextIndex()*force.getNumTorsions()/numContexts;
    int endIndex = (cu.getContextIndex()+1)*force.getNumTorsions()/numContexts;
    if (numTorsions != endIndex-startIndex)
        throw OpenMMException("updateParametersInContext: The number of torsions has changed");
    if (numTorsions == 0)
        return;
    
    // Record the per-torsion parameters.
    
    vector<float3> paramVector(numTorsions);
    for (int i = 0; i < numTorsions; i++) {
        int atom1, atom2, atom3, atom4;
        double k1, k2, k3;
        force.getTorsionParameters(startIndex+i, atom1, atom2, atom3, atom4, k1, k2, k3);
        paramVector[i] = make_float3((float) k1, (float) k2, (float) k3);
    }
    params->upload(paramVector);
    
    // Mark that the current reordering may be invalid.
    
    cu.invalidateMolecules();
}

/* -------------------------------------------------------------------------- *
 *                           MMFFStretchBend                                *
 * -------------------------------------------------------------------------- */

class CudaCalcMMFFStretchBendForceKernel::ForceInfo : public CudaForceInfo {
public:
    ForceInfo(const MMFFStretchBendForce& force) : force(force) {
    }
    int getNumParticleGroups() {
        return force.getNumStretchBends();
    }
    void getParticlesInGroup(int index, std::vector<int>& particles) {
        int particle1, particle2, particle3;
        double lengthAB, lengthCB, angle, k1, k2;
        force.getStretchBendParameters(index, particle1, particle2, particle3, lengthAB, lengthCB, angle, k1, k2);
        particles.resize(3);
        particles[0] = particle1;
        particles[1] = particle2;
        particles[2] = particle3;
    }
    bool areGroupsIdentical(int group1, int group2) {
        int particle1, particle2, particle3;
        double lengthAB1, lengthAB2, lengthCB1, lengthCB2, angle1, angle2, k11, k12, k21, k22;
        force.getStretchBendParameters(group1, particle1, particle2, particle3, lengthAB1, lengthCB1, angle1, k11, k12);
        force.getStretchBendParameters(group2, particle1, particle2, particle3, lengthAB2, lengthCB2, angle2, k21, k22);
        return (lengthAB1 == lengthAB2 && lengthCB1 == lengthCB2 && angle1 == angle2 && k11 == k21 && k12 == k22);
    }
private:
    const MMFFStretchBendForce& force;
};

CudaCalcMMFFStretchBendForceKernel::CudaCalcMMFFStretchBendForceKernel(std::string name, const Platform& platform, CudaContext& cu, const System& system) :
                   CalcMMFFStretchBendForceKernel(name, platform), cu(cu), system(system), params1(NULL), params2(NULL) {
}

CudaCalcMMFFStretchBendForceKernel::~CudaCalcMMFFStretchBendForceKernel() {
    cu.setAsCurrent();
    if (params1 != NULL)
        delete params1;
    if (params2 != NULL)
        delete params2;
}

void CudaCalcMMFFStretchBendForceKernel::initialize(const System& system, const MMFFStretchBendForce& force) {
    cu.setAsCurrent();
    int numContexts = cu.getPlatformData().contexts.size();
    int startIndex = cu.getContextIndex()*force.getNumStretchBends()/numContexts;
    int endIndex = (cu.getContextIndex()+1)*force.getNumStretchBends()/numContexts;
    numStretchBends = endIndex-startIndex;
    if (numStretchBends == 0)
        return;
    vector<vector<int> > atoms(numStretchBends, vector<int>(3));
    params1 = CudaArray::create<float3>(cu, numStretchBends, "stretchBendParams");
    params2 = CudaArray::create<float2>(cu, numStretchBends, "stretchBendForceConstants");
    vector<float3> paramVector(numStretchBends);
    vector<float2> paramVectorK(numStretchBends);
    for (int i = 0; i < numStretchBends; i++) {
        double lengthAB, lengthCB, angle, k1, k2;
        force.getStretchBendParameters(startIndex+i, atoms[i][0], atoms[i][1], atoms[i][2], lengthAB, lengthCB, angle, k1, k2);
        paramVector[i] = make_float3((float) lengthAB, (float) lengthCB, (float) angle);
        paramVectorK[i] = make_float2((float) k1, (float) k2);
    }
    params1->upload(paramVector);
    params2->upload(paramVectorK);
    map<string, string> replacements;
    replacements["APPLY_PERIODIC"] = (force.usesPeriodicBoundaryConditions() ? "1" : "0");
    replacements["PARAMS"] = cu.getBondedUtilities().addArgument(params1->getDevicePointer(), "float3");
    replacements["FORCE_CONSTANTS"] = cu.getBondedUtilities().addArgument(params2->getDevicePointer(), "float2");
    replacements["RAD_TO_DEG"] = cu.doubleToString(180/M_PI);
    cu.getBondedUtilities().addInteraction(atoms, cu.replaceStrings(CudaMMFFKernelSources::mmffStretchBendForce, replacements), force.getForceGroup());
    cu.addForce(new ForceInfo(force));
}

double CudaCalcMMFFStretchBendForceKernel::execute(ContextImpl& context, bool includeForces, bool includeEnergy) {
    return 0.0;
}

void CudaCalcMMFFStretchBendForceKernel::copyParametersToContext(ContextImpl& context, const MMFFStretchBendForce& force) {
    cu.setAsCurrent();
    int numContexts = cu.getPlatformData().contexts.size();
    int startIndex = cu.getContextIndex()*force.getNumStretchBends()/numContexts;
    int endIndex = (cu.getContextIndex()+1)*force.getNumStretchBends()/numContexts;
    if (numStretchBends != endIndex-startIndex)
        throw OpenMMException("updateParametersInContext: The number of bend-stretch terms has changed");
    if (numStretchBends == 0)
        return;
    
    // Record the per-stretch-bend parameters.
    
    vector<float3> paramVector(numStretchBends);
    vector<float2> paramVector1(numStretchBends);
    for (int i = 0; i < numStretchBends; i++) {
        int atom1, atom2, atom3;
        double lengthAB, lengthCB, angle, k1, k2;
        force.getStretchBendParameters(startIndex+i, atom1, atom2, atom3, lengthAB, lengthCB, angle, k1, k2);
        paramVector[i] = make_float3((float) lengthAB, (float) lengthCB, (float) angle);
        paramVector1[i] = make_float2((float) k1, (float) k2);
    }
    params1->upload(paramVector);
    params2->upload(paramVector1);
    
    // Mark that the current reordering may be invalid.
    
    cu.invalidateMolecules();
}

/* -------------------------------------------------------------------------- *
 *                           MMFFOutOfPlaneBend                             *
 * -------------------------------------------------------------------------- */

class CudaCalcMMFFOutOfPlaneBendForceKernel::ForceInfo : public CudaForceInfo {
public:
    ForceInfo(const MMFFOutOfPlaneBendForce& force) : force(force) {
    }
    int getNumParticleGroups() {
        return force.getNumOutOfPlaneBends();
    }
    void getParticlesInGroup(int index, std::vector<int>& particles) {
        int particle1, particle2, particle3, particle4;
        double k;
        force.getOutOfPlaneBendParameters(index, particle1, particle2, particle3, particle4, k);
        particles.resize(4);
        particles[0] = particle1;
        particles[1] = particle2;
        particles[2] = particle3;
        particles[3] = particle4;
    }
    bool areGroupsIdentical(int group1, int group2) {
        int particle1, particle2, particle3, particle4;
        double k1, k2;
        force.getOutOfPlaneBendParameters(group1, particle1, particle2, particle3, particle4, k1);
        force.getOutOfPlaneBendParameters(group2, particle1, particle2, particle3, particle4, k2);
        return (k1 == k2);
    }
private:
    const MMFFOutOfPlaneBendForce& force;
};

CudaCalcMMFFOutOfPlaneBendForceKernel::CudaCalcMMFFOutOfPlaneBendForceKernel(std::string name, const Platform& platform, CudaContext& cu, const System& system) :
          CalcMMFFOutOfPlaneBendForceKernel(name, platform), cu(cu), system(system), params(NULL) {
}

CudaCalcMMFFOutOfPlaneBendForceKernel::~CudaCalcMMFFOutOfPlaneBendForceKernel() {
    cu.setAsCurrent();
    if (params != NULL)
        delete params;
}

void CudaCalcMMFFOutOfPlaneBendForceKernel::initialize(const System& system, const MMFFOutOfPlaneBendForce& force) {
    cu.setAsCurrent();
    int numContexts = cu.getPlatformData().contexts.size();
    int startIndex = cu.getContextIndex()*force.getNumOutOfPlaneBends()/numContexts;
    int endIndex = (cu.getContextIndex()+1)*force.getNumOutOfPlaneBends()/numContexts;
    numOutOfPlaneBends = endIndex-startIndex;
    if (numOutOfPlaneBends == 0)
        return;
    vector<vector<int> > atoms(numOutOfPlaneBends, vector<int>(4));
    params = CudaArray::create<float>(cu, numOutOfPlaneBends, "outOfPlaneParams");
    vector<float> paramVector(numOutOfPlaneBends);
    for (int i = 0; i < numOutOfPlaneBends; i++) {
        double k;
        force.getOutOfPlaneBendParameters(startIndex+i, atoms[i][0], atoms[i][1], atoms[i][2], atoms[i][3], k);
        paramVector[i] = (float) k;
    }
    params->upload(paramVector);
    map<string, string> replacements;
    replacements["APPLY_PERIODIC"] = (force.usesPeriodicBoundaryConditions() ? "1" : "0");
    replacements["PARAMS"] = cu.getBondedUtilities().addArgument(params->getDevicePointer(), "float");
    replacements["RAD_TO_DEG"] = cu.doubleToString(180/M_PI);
    cu.getBondedUtilities().addInteraction(atoms, cu.replaceStrings(CudaMMFFKernelSources::mmffOutOfPlaneBendForce, replacements), force.getForceGroup());
    cu.addForce(new ForceInfo(force));
}

double CudaCalcMMFFOutOfPlaneBendForceKernel::execute(ContextImpl& context, bool includeForces, bool includeEnergy) {
    return 0.0;
}

void CudaCalcMMFFOutOfPlaneBendForceKernel::copyParametersToContext(ContextImpl& context, const MMFFOutOfPlaneBendForce& force) {
    cu.setAsCurrent();
    int numContexts = cu.getPlatformData().contexts.size();
    int startIndex = cu.getContextIndex()*force.getNumOutOfPlaneBends()/numContexts;
    int endIndex = (cu.getContextIndex()+1)*force.getNumOutOfPlaneBends()/numContexts;
    if (numOutOfPlaneBends != endIndex-startIndex)
        throw OpenMMException("updateParametersInContext: The number of out-of-plane bends has changed");
    if (numOutOfPlaneBends == 0)
        return;
    
    // Record the per-bend parameters.
    
    vector<float> paramVector(numOutOfPlaneBends);
    for (int i = 0; i < numOutOfPlaneBends; i++) {
        int atom1, atom2, atom3, atom4;
        double k;
        force.getOutOfPlaneBendParameters(startIndex+i, atom1, atom2, atom3, atom4, k);
        paramVector[i] = (float) k;
    }
    params->upload(paramVector);
    
    // Mark that the current reordering may be invalid.
    
    cu.invalidateMolecules();
}

/* -------------------------------------------------------------------------- *
 *                             MMFFMultipole                                *
 * -------------------------------------------------------------------------- */

class CudaCalcMMFFMultipoleForceKernel::ForceInfo : public CudaForceInfo {
public:
    ForceInfo(const MMFFMultipoleForce& force) : force(force) {
    }
    bool areParticlesIdentical(int particle1, int particle2) {
        double charge1, charge2, thole1, thole2, damping1, damping2, polarity1, polarity2;
        int axis1, axis2, multipole11, multipole12, multipole21, multipole22, multipole31, multipole32;
        vector<double> dipole1, dipole2, quadrupole1, quadrupole2;
        force.getMultipoleParameters(particle1, charge1, dipole1, quadrupole1, axis1, multipole11, multipole21, multipole31, thole1, damping1, polarity1);
        force.getMultipoleParameters(particle2, charge2, dipole2, quadrupole2, axis2, multipole12, multipole22, multipole32, thole2, damping2, polarity2);
        if (charge1 != charge2 || thole1 != thole2 || damping1 != damping2 || polarity1 != polarity2 || axis1 != axis2) {
            return false;
        }
        for (int i = 0; i < (int) dipole1.size(); ++i) {
            if (dipole1[i] != dipole2[i]) {
                return false;
            }
        }
        for (int i = 0; i < (int) quadrupole1.size(); ++i) {
            if (quadrupole1[i] != quadrupole2[i]) {
                return false;
            }
        }
        return true;
    }
    int getNumParticleGroups() {
        return 7*force.getNumMultipoles();
    }
    void getParticlesInGroup(int index, vector<int>& particles) {
        int particle = index/7;
        int type = index-7*particle;
        force.getCovalentMap(particle, MMFFMultipoleForce::CovalentType(type), particles);
    }
    bool areGroupsIdentical(int group1, int group2) {
        return ((group1%7) == (group2%7));
    }
private:
    const MMFFMultipoleForce& force;
};

CudaCalcMMFFMultipoleForceKernel::CudaCalcMMFFMultipoleForceKernel(std::string name, const Platform& platform, CudaContext& cu, const System& system) : 
        CalcMMFFMultipoleForceKernel(name, platform), cu(cu), system(system), hasInitializedScaleFactors(false), hasInitializedFFT(false), multipolesAreValid(false), hasCreatedEvent(false),
        multipoleParticles(NULL), molecularDipoles(NULL), molecularQuadrupoles(NULL), labFrameDipoles(NULL), labFrameQuadrupoles(NULL), sphericalDipoles(NULL), sphericalQuadrupoles(NULL),
        fracDipoles(NULL), fracQuadrupoles(NULL), field(NULL), fieldPolar(NULL), inducedField(NULL), inducedFieldPolar(NULL), torque(NULL), dampingAndThole(NULL), inducedDipole(NULL),
        diisCoefficients(NULL), inducedDipolePolar(NULL), inducedDipoleErrors(NULL), prevDipoles(NULL), prevDipolesPolar(NULL), prevDipolesGk(NULL),
        prevDipolesGkPolar(NULL), prevErrors(NULL), diisMatrix(NULL), polarizability(NULL), extrapolatedDipole(NULL), extrapolatedDipolePolar(NULL),
        extrapolatedDipoleGk(NULL), extrapolatedDipoleGkPolar(NULL), inducedDipoleFieldGradient(NULL), inducedDipoleFieldGradientPolar(NULL),
        inducedDipoleFieldGradientGk(NULL), inducedDipoleFieldGradientGkPolar(NULL), extrapolatedDipoleFieldGradient(NULL), extrapolatedDipoleFieldGradientPolar(NULL),
        extrapolatedDipoleFieldGradientGk(NULL), extrapolatedDipoleFieldGradientGkPolar(NULL), covalentFlags(NULL), polarizationGroupFlags(NULL),
        pmeGrid(NULL), pmeBsplineModuliX(NULL), pmeBsplineModuliY(NULL), pmeBsplineModuliZ(NULL), pmeIgrid(NULL), pmePhi(NULL),
        pmePhid(NULL), pmePhip(NULL), pmePhidp(NULL), pmeCphi(NULL), lastPositions(NULL), sort(NULL), gkKernel(NULL) {
}

CudaCalcMMFFMultipoleForceKernel::~CudaCalcMMFFMultipoleForceKernel() {
    cu.setAsCurrent();
    if (multipoleParticles != NULL)
        delete multipoleParticles;
    if (molecularDipoles != NULL)
        delete molecularDipoles;
    if (molecularQuadrupoles != NULL)
        delete molecularQuadrupoles;
    if (labFrameDipoles != NULL)
        delete labFrameDipoles;
    if (labFrameQuadrupoles != NULL)
        delete labFrameQuadrupoles;
    if (sphericalDipoles != NULL)
        delete sphericalDipoles;
    if (sphericalQuadrupoles != NULL)
        delete sphericalQuadrupoles;
    if (fracDipoles != NULL)
        delete fracDipoles;
    if (fracQuadrupoles != NULL)
        delete fracQuadrupoles;
    if (field != NULL)
        delete field;
    if (fieldPolar != NULL)
        delete fieldPolar;
    if (inducedField != NULL)
        delete inducedField;
    if (inducedFieldPolar != NULL)
        delete inducedFieldPolar;
    if (torque != NULL)
        delete torque;
    if (dampingAndThole != NULL)
        delete dampingAndThole;
    if (inducedDipole != NULL)
        delete inducedDipole;
    if (inducedDipolePolar != NULL)
        delete inducedDipolePolar;
    if (inducedDipoleErrors != NULL)
        delete inducedDipoleErrors;
    if (prevDipoles != NULL)
        delete prevDipoles;
    if (prevDipolesPolar != NULL)
        delete prevDipolesPolar;
    if (prevDipolesGk != NULL)
        delete prevDipolesGk;
    if (prevDipolesGkPolar != NULL)
        delete prevDipolesGkPolar;
    if (prevErrors != NULL)
        delete prevErrors;
    if (diisMatrix != NULL)
        delete diisMatrix;
    if (diisCoefficients != NULL)
        delete diisCoefficients;
    if (extrapolatedDipole != NULL)
        delete extrapolatedDipole;
    if (extrapolatedDipolePolar != NULL)
        delete extrapolatedDipolePolar;
    if (extrapolatedDipoleGk != NULL)
        delete extrapolatedDipoleGk;
    if (extrapolatedDipoleGkPolar != NULL)
        delete extrapolatedDipoleGkPolar;
    if (inducedDipoleFieldGradient != NULL)
        delete inducedDipoleFieldGradient;
    if (inducedDipoleFieldGradientPolar != NULL)
        delete inducedDipoleFieldGradientPolar;
    if (inducedDipoleFieldGradientGk != NULL)
        delete inducedDipoleFieldGradientGk;
    if (inducedDipoleFieldGradientGkPolar != NULL)
        delete inducedDipoleFieldGradientGkPolar;
    if (extrapolatedDipoleFieldGradient != NULL)
        delete extrapolatedDipoleFieldGradient;
    if (extrapolatedDipoleFieldGradientPolar != NULL)
        delete extrapolatedDipoleFieldGradientPolar;
    if (extrapolatedDipoleFieldGradientGk != NULL)
        delete extrapolatedDipoleFieldGradientGk;
    if (extrapolatedDipoleFieldGradientGkPolar != NULL)
        delete extrapolatedDipoleFieldGradientGkPolar;
    if (polarizability != NULL)
        delete polarizability;
    if (covalentFlags != NULL)
        delete covalentFlags;
    if (polarizationGroupFlags != NULL)
        delete polarizationGroupFlags;
    if (pmeGrid != NULL)
        delete pmeGrid;
    if (pmeBsplineModuliX != NULL)
        delete pmeBsplineModuliX;
    if (pmeBsplineModuliY != NULL)
        delete pmeBsplineModuliY;
    if (pmeBsplineModuliZ != NULL)
        delete pmeBsplineModuliZ;
    if (pmeIgrid != NULL)
        delete pmeIgrid;
    if (pmePhi != NULL)
        delete pmePhi;
    if (pmePhid != NULL)
        delete pmePhid;
    if (pmePhip != NULL)
        delete pmePhip;
    if (pmePhidp != NULL)
        delete pmePhidp;
    if (pmeCphi != NULL)
        delete pmeCphi;
    if (lastPositions != NULL)
        delete lastPositions;
    if (sort != NULL)
        delete sort;
    if (hasInitializedFFT)
        cufftDestroy(fft);
    if (hasCreatedEvent)
        cuEventDestroy(syncEvent);
}

void CudaCalcMMFFMultipoleForceKernel::initialize(const System& system, const MMFFMultipoleForce& force) {
    cu.setAsCurrent();

    // Initialize multipole parameters.

    numMultipoles = force.getNumMultipoles();
    CudaArray& posq = cu.getPosq();
    vector<double4> temp(posq.getSize());
    float4* posqf = (float4*) &temp[0];
    double4* posqd = (double4*) &temp[0];
    vector<float2> dampingAndTholeVec;
    vector<float> polarizabilityVec;
    vector<float> molecularDipolesVec;
    vector<float> molecularQuadrupolesVec;
    vector<int4> multipoleParticlesVec;
    for (int i = 0; i < numMultipoles; i++) {
        double charge, thole, damping, polarity;
        int axisType, atomX, atomY, atomZ;
        vector<double> dipole, quadrupole;
        force.getMultipoleParameters(i, charge, dipole, quadrupole, axisType, atomZ, atomX, atomY, thole, damping, polarity);
        if (cu.getUseDoublePrecision())
            posqd[i] = make_double4(0, 0, 0, charge);
        else
            posqf[i] = make_float4(0, 0, 0, (float) charge);
        dampingAndTholeVec.push_back(make_float2((float) damping, (float) thole));
        polarizabilityVec.push_back((float) polarity);
        multipoleParticlesVec.push_back(make_int4(atomX, atomY, atomZ, axisType));
        for (int j = 0; j < 3; j++)
            molecularDipolesVec.push_back((float) dipole[j]);
        molecularQuadrupolesVec.push_back((float) quadrupole[0]);
        molecularQuadrupolesVec.push_back((float) quadrupole[1]);
        molecularQuadrupolesVec.push_back((float) quadrupole[2]);
        molecularQuadrupolesVec.push_back((float) quadrupole[4]);
        molecularQuadrupolesVec.push_back((float) quadrupole[5]);
    }
    hasQuadrupoles = false;
    for (auto q : molecularQuadrupolesVec)
        if (q != 0.0)
            hasQuadrupoles = true;
    int paddedNumAtoms = cu.getPaddedNumAtoms();
    for (int i = numMultipoles; i < paddedNumAtoms; i++) {
        dampingAndTholeVec.push_back(make_float2(0, 0));
        polarizabilityVec.push_back(0);
        multipoleParticlesVec.push_back(make_int4(0, 0, 0, 0));
        for (int j = 0; j < 3; j++)
            molecularDipolesVec.push_back(0);
        for (int j = 0; j < 5; j++)
            molecularQuadrupolesVec.push_back(0);
    }
    dampingAndThole = CudaArray::create<float2>(cu, paddedNumAtoms, "dampingAndThole");
    polarizability = CudaArray::create<float>(cu, paddedNumAtoms, "polarizability");
    multipoleParticles = CudaArray::create<int4>(cu, paddedNumAtoms, "multipoleParticles");
    molecularDipoles = CudaArray::create<float>(cu, 3*paddedNumAtoms, "molecularDipoles");
    molecularQuadrupoles = CudaArray::create<float>(cu, 5*paddedNumAtoms, "molecularQuadrupoles");
    lastPositions = new CudaArray(cu, cu.getPosq().getSize(), cu.getPosq().getElementSize(), "lastPositions");
    dampingAndThole->upload(dampingAndTholeVec);
    polarizability->upload(polarizabilityVec);
    multipoleParticles->upload(multipoleParticlesVec);
    molecularDipoles->upload(molecularDipolesVec);
    molecularQuadrupoles->upload(molecularQuadrupolesVec);
    posq.upload(&temp[0]);
    
    // Create workspace arrays.
    
    polarizationType = force.getPolarizationType();
    int elementSize = (cu.getUseDoublePrecision() ? sizeof(double) : sizeof(float));
    labFrameDipoles = new CudaArray(cu, 3*paddedNumAtoms, elementSize, "labFrameDipoles");
    labFrameQuadrupoles = new CudaArray(cu, 5*paddedNumAtoms, elementSize, "labFrameQuadrupoles");
    sphericalDipoles = new CudaArray(cu, 3*paddedNumAtoms, elementSize, "sphericalDipoles");
    sphericalQuadrupoles = new CudaArray(cu, 5*paddedNumAtoms, elementSize, "sphericalQuadrupoles");
    fracDipoles = new CudaArray(cu, 3*paddedNumAtoms, elementSize, "fracDipoles");
    fracQuadrupoles = new CudaArray(cu, 6*paddedNumAtoms, elementSize, "fracQuadrupoles");
    field = new CudaArray(cu, 3*paddedNumAtoms, sizeof(long long), "field");
    fieldPolar = new CudaArray(cu, 3*paddedNumAtoms, sizeof(long long), "fieldPolar");
    torque = new CudaArray(cu, 3*paddedNumAtoms, sizeof(long long), "torque");
    inducedDipole = new CudaArray(cu, 3*paddedNumAtoms, elementSize, "inducedDipole");
    inducedDipolePolar = new CudaArray(cu, 3*paddedNumAtoms, elementSize, "inducedDipolePolar");
    if (polarizationType == MMFFMultipoleForce::Mutual) {
        inducedDipoleErrors = new CudaArray(cu, cu.getNumThreadBlocks(), sizeof(float2), "inducedDipoleErrors");
        prevDipoles = new CudaArray(cu, 3*numMultipoles*MaxPrevDIISDipoles, elementSize, "prevDipoles");
        prevDipolesPolar = new CudaArray(cu, 3*numMultipoles*MaxPrevDIISDipoles, elementSize, "prevDipolesPolar");
        prevErrors = new CudaArray(cu, 3*numMultipoles*MaxPrevDIISDipoles, elementSize, "prevErrors");
        diisMatrix = new CudaArray(cu, MaxPrevDIISDipoles*MaxPrevDIISDipoles, elementSize, "diisMatrix");
        diisCoefficients = new CudaArray(cu, MaxPrevDIISDipoles+1, sizeof(float), "diisMatrix");
        CHECK_RESULT(cuEventCreate(&syncEvent, CU_EVENT_DISABLE_TIMING), "Error creating event for MMFFMultipoleForce");
        hasCreatedEvent = true;
    }
    else if (polarizationType == MMFFMultipoleForce::Extrapolated) {
        int numOrders = force.getExtrapolationCoefficients().size();
        extrapolatedDipole = new CudaArray(cu, 3*numMultipoles*numOrders, elementSize, "extrapolatedDipole");
        extrapolatedDipolePolar = new CudaArray(cu, 3*numMultipoles*numOrders, elementSize, "extrapolatedDipolePolar");
        inducedDipoleFieldGradient = new CudaArray(cu, 6*paddedNumAtoms, sizeof(long long), "inducedDipoleFieldGradient");
        inducedDipoleFieldGradientPolar = new CudaArray(cu, 6*paddedNumAtoms, sizeof(long long), "inducedDipoleFieldGradientPolar");
        extrapolatedDipoleFieldGradient = new CudaArray(cu, 6*numMultipoles*(numOrders-1), elementSize, "extrapolatedDipoleFieldGradient");
        extrapolatedDipoleFieldGradientPolar = new CudaArray(cu, 6*numMultipoles*(numOrders-1), elementSize, "extrapolatedDipoleFieldGradientPolar");
    }
    cu.addAutoclearBuffer(*field);
    cu.addAutoclearBuffer(*fieldPolar);
    cu.addAutoclearBuffer(*torque);
    
    // Record which atoms should be flagged as exclusions based on covalent groups, and determine
    // the values for the covalent group flags.
    
    vector<vector<int> > exclusions(numMultipoles);
    for (int i = 0; i < numMultipoles; i++) {
        vector<int> atoms;
        set<int> allAtoms;
        allAtoms.insert(i);
        force.getCovalentMap(i, MMFFMultipoleForce::Covalent12, atoms);
        allAtoms.insert(atoms.begin(), atoms.end());
        force.getCovalentMap(i, MMFFMultipoleForce::Covalent13, atoms);
        allAtoms.insert(atoms.begin(), atoms.end());
        for (int atom : allAtoms)
            covalentFlagValues.push_back(make_int3(i, atom, 0));
        force.getCovalentMap(i, MMFFMultipoleForce::Covalent14, atoms);
        allAtoms.insert(atoms.begin(), atoms.end());
        for (int atom : atoms)
            covalentFlagValues.push_back(make_int3(i, atom, 1));
        force.getCovalentMap(i, MMFFMultipoleForce::Covalent15, atoms);
        for (int atom : atoms)
            covalentFlagValues.push_back(make_int3(i, atom, 2));
        allAtoms.insert(atoms.begin(), atoms.end());
        force.getCovalentMap(i, MMFFMultipoleForce::PolarizationCovalent11, atoms);
        allAtoms.insert(atoms.begin(), atoms.end());
        exclusions[i].insert(exclusions[i].end(), allAtoms.begin(), allAtoms.end());

        // Workaround for bug in TINKER: if an atom is listed in both the PolarizationCovalent11
        // and PolarizationCovalent12 maps, the latter takes precedence.

        vector<int> atoms12;
        force.getCovalentMap(i, MMFFMultipoleForce::PolarizationCovalent12, atoms12);
        for (int atom : atoms)
            if (find(atoms12.begin(), atoms12.end(), atom) == atoms12.end())
                polarizationFlagValues.push_back(make_int2(i, atom));
    }
    set<pair<int, int> > tilesWithExclusions;
    for (int atom1 = 0; atom1 < (int) exclusions.size(); ++atom1) {
        int x = atom1/CudaContext::TileSize;
        for (int atom2 : exclusions[atom1]) {
            int y = atom2/CudaContext::TileSize;
            tilesWithExclusions.insert(make_pair(max(x, y), min(x, y)));
        }
    }
    
    // Record other options.
    
    if (polarizationType == MMFFMultipoleForce::Mutual) {
        maxInducedIterations = force.getMutualInducedMaxIterations();
        inducedEpsilon = force.getMutualInducedTargetEpsilon();
    }
    else
        maxInducedIterations = 0;
    if (polarizationType != MMFFMultipoleForce::Direct) {
        inducedField = new CudaArray(cu, 3*paddedNumAtoms, sizeof(long long), "inducedField");
        inducedFieldPolar = new CudaArray(cu, 3*paddedNumAtoms, sizeof(long long), "inducedFieldPolar");
    }
    usePME = (force.getNonbondedMethod() == MMFFMultipoleForce::PME);
    
    // See whether there's an MMFFGeneralizedKirkwoodForce in the System.

    const MMFFGeneralizedKirkwoodForce* gk = NULL;
    for (int i = 0; i < system.getNumForces() && gk == NULL; i++)
        gk = dynamic_cast<const MMFFGeneralizedKirkwoodForce*>(&system.getForce(i));
    double innerDielectric = (gk == NULL ? 1.0 : gk->getSoluteDielectric());
    
    // Create the kernels.

    bool useShuffle = (cu.getComputeCapability() >= 3.0 && !cu.getUseDoublePrecision());
    double fixedThreadMemory = 19*elementSize+2*sizeof(float)+3*sizeof(int)/(double) cu.TileSize;
    double inducedThreadMemory = 15*elementSize+2*sizeof(float);
    if (polarizationType == MMFFMultipoleForce::Extrapolated)
        inducedThreadMemory += 12*elementSize;
    double electrostaticsThreadMemory = 0;
    if (!useShuffle)
        fixedThreadMemory += 3*elementSize;
    map<string, string> defines;
    defines["NUM_ATOMS"] = cu.intToString(numMultipoles);
    defines["PADDED_NUM_ATOMS"] = cu.intToString(cu.getPaddedNumAtoms());
    defines["NUM_BLOCKS"] = cu.intToString(cu.getNumAtomBlocks());
    defines["ENERGY_SCALE_FACTOR"] = cu.doubleToString(138.9354558456/innerDielectric);
    if (polarizationType == MMFFMultipoleForce::Direct)
        defines["DIRECT_POLARIZATION"] = "";
    else if (polarizationType == MMFFMultipoleForce::Mutual)
        defines["MUTUAL_POLARIZATION"] = "";
    else if (polarizationType == MMFFMultipoleForce::Extrapolated)
        defines["EXTRAPOLATED_POLARIZATION"] = "";
    if (useShuffle)
        defines["USE_SHUFFLE"] = "";
    if (hasQuadrupoles)
        defines["INCLUDE_QUADRUPOLES"] = "";
    defines["TILE_SIZE"] = cu.intToString(CudaContext::TileSize);
    int numExclusionTiles = tilesWithExclusions.size();
    defines["NUM_TILES_WITH_EXCLUSIONS"] = cu.intToString(numExclusionTiles);
    int numContexts = cu.getPlatformData().contexts.size();
    int startExclusionIndex = cu.getContextIndex()*numExclusionTiles/numContexts;
    int endExclusionIndex = (cu.getContextIndex()+1)*numExclusionTiles/numContexts;
    defines["FIRST_EXCLUSION_TILE"] = cu.intToString(startExclusionIndex);
    defines["LAST_EXCLUSION_TILE"] = cu.intToString(endExclusionIndex);
    maxExtrapolationOrder = force.getExtrapolationCoefficients().size();
    defines["MAX_EXTRAPOLATION_ORDER"] = cu.intToString(maxExtrapolationOrder);
    stringstream coefficients;
    for (int i = 0; i < maxExtrapolationOrder; i++) {
        if (i > 0)
            coefficients << ",";
        double sum = 0;
        for (int j = i; j < maxExtrapolationOrder; j++)
            sum += force.getExtrapolationCoefficients()[j];
        coefficients << cu.doubleToString(sum);
    }
    defines["EXTRAPOLATION_COEFFICIENTS_SUM"] = coefficients.str();
    if (usePME) {
        int nx, ny, nz;
        force.getPMEParameters(alpha, nx, ny, nz);
        if (nx == 0 || alpha == 0.0) {
            NonbondedForce nb;
            nb.setEwaldErrorTolerance(force.getEwaldErrorTolerance());
            nb.setCutoffDistance(force.getCutoffDistance());
            NonbondedForceImpl::calcPMEParameters(system, nb, alpha, gridSizeX, gridSizeY, gridSizeZ, false);
            gridSizeX = CudaFFT3D::findLegalDimension(gridSizeX);
            gridSizeY = CudaFFT3D::findLegalDimension(gridSizeY);
            gridSizeZ = CudaFFT3D::findLegalDimension(gridSizeZ);
        } else {
            gridSizeX = CudaFFT3D::findLegalDimension(nx);
            gridSizeY = CudaFFT3D::findLegalDimension(ny);
            gridSizeZ = CudaFFT3D::findLegalDimension(nz);
        }
        defines["EWALD_ALPHA"] = cu.doubleToString(alpha);
        defines["SQRT_PI"] = cu.doubleToString(sqrt(M_PI));
        defines["USE_EWALD"] = "";
        defines["USE_CUTOFF"] = "";
        defines["USE_PERIODIC"] = "";
        defines["CUTOFF_SQUARED"] = cu.doubleToString(force.getCutoffDistance()*force.getCutoffDistance());
    }
    if (gk != NULL) {
        defines["USE_GK"] = "";
        defines["GK_C"] = cu.doubleToString(2.455);
        double solventDielectric = gk->getSolventDielectric();
        defines["GK_FC"] = cu.doubleToString(1*(1-solventDielectric)/(0+1*solventDielectric));
        defines["GK_FD"] = cu.doubleToString(2*(1-solventDielectric)/(1+2*solventDielectric));
        defines["GK_FQ"] = cu.doubleToString(3*(1-solventDielectric)/(2+3*solventDielectric));
        fixedThreadMemory += 4*elementSize;
        inducedThreadMemory += 13*elementSize;
        if (polarizationType == MMFFMultipoleForce::Mutual) {
            prevDipolesGk = new CudaArray(cu, 3*numMultipoles*MaxPrevDIISDipoles, elementSize, "prevDipolesGk");
            prevDipolesGkPolar = new CudaArray(cu, 3*numMultipoles*MaxPrevDIISDipoles, elementSize, "prevDipolesGkPolar");
        }
        else if (polarizationType == MMFFMultipoleForce::Extrapolated) {
            inducedThreadMemory += 12*elementSize;
            int numOrders = force.getExtrapolationCoefficients().size();
            extrapolatedDipoleGk = new CudaArray(cu, 3*numMultipoles*numOrders, elementSize, "extrapolatedDipoleGk");
            extrapolatedDipoleGkPolar = new CudaArray(cu, 3*numMultipoles*numOrders, elementSize, "extrapolatedDipoleGkPolar");
            inducedDipoleFieldGradientGk = new CudaArray(cu, 6*numMultipoles, elementSize, "inducedDipoleFieldGradientGk");
            inducedDipoleFieldGradientGkPolar = new CudaArray(cu, 6*numMultipoles, elementSize, "inducedDipoleFieldGradientGkPolar");
            extrapolatedDipoleFieldGradientGk = new CudaArray(cu, 6*numMultipoles*(numOrders-1), elementSize, "extrapolatedDipoleFieldGradientGk");
            extrapolatedDipoleFieldGradientGkPolar = new CudaArray(cu, 6*numMultipoles*(numOrders-1), elementSize, "extrapolatedDipoleFieldGradientGkPolar");
        }
    }
    int maxThreads = cu.getNonbondedUtilities().getForceThreadBlockSize();
    fixedFieldThreads = min(maxThreads, cu.computeThreadBlockSize(fixedThreadMemory));
    inducedFieldThreads = min(maxThreads, cu.computeThreadBlockSize(inducedThreadMemory));
    CUmodule module = cu.createModule(CudaKernelSources::vectorOps+CudaMMFFKernelSources::multipoles, defines);
    computeMomentsKernel = cu.getKernel(module, "computeLabFrameMoments");
    recordInducedDipolesKernel = cu.getKernel(module, "recordInducedDipoles");
    mapTorqueKernel = cu.getKernel(module, "mapTorqueToForce");
    computePotentialKernel = cu.getKernel(module, "computePotentialAtPoints");
    defines["THREAD_BLOCK_SIZE"] = cu.intToString(fixedFieldThreads);
    module = cu.createModule(CudaKernelSources::vectorOps+CudaMMFFKernelSources::multipoleFixedField, defines);
    computeFixedFieldKernel = cu.getKernel(module, "computeFixedField");
    if (polarizationType != MMFFMultipoleForce::Direct) {
        defines["THREAD_BLOCK_SIZE"] = cu.intToString(inducedFieldThreads);
        defines["MAX_PREV_DIIS_DIPOLES"] = cu.intToString(MaxPrevDIISDipoles);
        module = cu.createModule(CudaKernelSources::vectorOps+CudaMMFFKernelSources::multipoleInducedField, defines);
        computeInducedFieldKernel = cu.getKernel(module, "computeInducedField");
        updateInducedFieldKernel = cu.getKernel(module, "updateInducedFieldByDIIS");
        recordDIISDipolesKernel = cu.getKernel(module, "recordInducedDipolesForDIIS");
        buildMatrixKernel = cu.getKernel(module, "computeDIISMatrix");
        solveMatrixKernel = cu.getKernel(module, "solveDIISMatrix");
        initExtrapolatedKernel = cu.getKernel(module, "initExtrapolatedDipoles");
        iterateExtrapolatedKernel = cu.getKernel(module, "iterateExtrapolatedDipoles");
        computeExtrapolatedKernel = cu.getKernel(module, "computeExtrapolatedDipoles");
        addExtrapolatedGradientKernel = cu.getKernel(module, "addExtrapolatedFieldGradientToForce");
    }
    stringstream electrostaticsSource;
    electrostaticsSource << CudaKernelSources::vectorOps;
    electrostaticsSource << CudaMMFFKernelSources::sphericalMultipoles;
    if (usePME)
        electrostaticsSource << CudaMMFFKernelSources::pmeMultipoleElectrostatics;
    else
        electrostaticsSource << CudaMMFFKernelSources::multipoleElectrostatics;
    electrostaticsThreadMemory = 24*elementSize+3*sizeof(float)+3*sizeof(int)/(double) cu.TileSize;
    electrostaticsThreads = min(maxThreads, cu.computeThreadBlockSize(electrostaticsThreadMemory));
    defines["THREAD_BLOCK_SIZE"] = cu.intToString(electrostaticsThreads);
    module = cu.createModule(electrostaticsSource.str(), defines);
    electrostaticsKernel = cu.getKernel(module, "computeElectrostatics");

    // Set up PME.
    
    if (usePME) {
        // Create the PME kernels.

        map<string, string> pmeDefines;
        pmeDefines["EWALD_ALPHA"] = cu.doubleToString(alpha);
        pmeDefines["PME_ORDER"] = cu.intToString(PmeOrder);
        pmeDefines["NUM_ATOMS"] = cu.intToString(numMultipoles);
        pmeDefines["PADDED_NUM_ATOMS"] = cu.intToString(cu.getPaddedNumAtoms());
        pmeDefines["EPSILON_FACTOR"] = cu.doubleToString(138.9354558456);
        pmeDefines["GRID_SIZE_X"] = cu.intToString(gridSizeX);
        pmeDefines["GRID_SIZE_Y"] = cu.intToString(gridSizeY);
        pmeDefines["GRID_SIZE_Z"] = cu.intToString(gridSizeZ);
        pmeDefines["M_PI"] = cu.doubleToString(M_PI);
        pmeDefines["SQRT_PI"] = cu.doubleToString(sqrt(M_PI));
        if (polarizationType == MMFFMultipoleForce::Direct)
            pmeDefines["DIRECT_POLARIZATION"] = "";
        else if (polarizationType == MMFFMultipoleForce::Mutual)
            pmeDefines["MUTUAL_POLARIZATION"] = "";
        else if (polarizationType == MMFFMultipoleForce::Extrapolated)
            pmeDefines["EXTRAPOLATED_POLARIZATION"] = "";
        CUmodule module = cu.createModule(CudaKernelSources::vectorOps+CudaMMFFKernelSources::multipolePme, pmeDefines);
        pmeTransformMultipolesKernel = cu.getKernel(module, "transformMultipolesToFractionalCoordinates");
        pmeTransformPotentialKernel = cu.getKernel(module, "transformPotentialToCartesianCoordinates");
        pmeSpreadFixedMultipolesKernel = cu.getKernel(module, "gridSpreadFixedMultipoles");
        pmeSpreadInducedDipolesKernel = cu.getKernel(module, "gridSpreadInducedDipoles");
        pmeFinishSpreadChargeKernel = cu.getKernel(module, "finishSpreadCharge");
        pmeConvolutionKernel = cu.getKernel(module, "reciprocalConvolution");
        pmeFixedPotentialKernel = cu.getKernel(module, "computeFixedPotentialFromGrid");
        pmeInducedPotentialKernel = cu.getKernel(module, "computeInducedPotentialFromGrid");
        pmeFixedForceKernel = cu.getKernel(module, "computeFixedMultipoleForceAndEnergy");
        pmeInducedForceKernel = cu.getKernel(module, "computeInducedDipoleForceAndEnergy");
        pmeRecordInducedFieldDipolesKernel = cu.getKernel(module, "recordInducedFieldDipoles");
        cuFuncSetCacheConfig(pmeSpreadFixedMultipolesKernel, CU_FUNC_CACHE_PREFER_L1);
        cuFuncSetCacheConfig(pmeSpreadInducedDipolesKernel, CU_FUNC_CACHE_PREFER_L1);
        cuFuncSetCacheConfig(pmeFixedPotentialKernel, CU_FUNC_CACHE_PREFER_L1);
        cuFuncSetCacheConfig(pmeInducedPotentialKernel, CU_FUNC_CACHE_PREFER_L1);

        // Create required data structures.

        int elementSize = (cu.getUseDoublePrecision() ? sizeof(double) : sizeof(float));
        pmeGrid = new CudaArray(cu, gridSizeX*gridSizeY*gridSizeZ, 2*elementSize, "pmeGrid");
        cu.addAutoclearBuffer(*pmeGrid);
        pmeBsplineModuliX = new CudaArray(cu, gridSizeX, elementSize, "pmeBsplineModuliX");
        pmeBsplineModuliY = new CudaArray(cu, gridSizeY, elementSize, "pmeBsplineModuliY");
        pmeBsplineModuliZ = new CudaArray(cu, gridSizeZ, elementSize, "pmeBsplineModuliZ");
        pmeIgrid = CudaArray::create<int4>(cu, numMultipoles, "pmeIgrid");
        pmePhi = new CudaArray(cu, 20*numMultipoles, elementSize, "pmePhi");
        pmePhid = new CudaArray(cu, 10*numMultipoles, elementSize, "pmePhid");
        pmePhip = new CudaArray(cu, 10*numMultipoles, elementSize, "pmePhip");
        pmePhidp = new CudaArray(cu, 20*numMultipoles, elementSize, "pmePhidp");
        pmeCphi = new CudaArray(cu, 10*numMultipoles, elementSize, "pmeCphi");
        pmeAtomRange = CudaArray::create<int>(cu, gridSizeX*gridSizeY*gridSizeZ+1, "pmeAtomRange");
        sort = new CudaSort(cu, new SortTrait(), cu.getNumAtoms());
        cufftResult result = cufftPlan3d(&fft, gridSizeX, gridSizeY, gridSizeZ, cu.getUseDoublePrecision() ? CUFFT_Z2Z : CUFFT_C2C);
        if (result != CUFFT_SUCCESS)
            throw OpenMMException("Error initializing FFT: "+cu.intToString(result));
        hasInitializedFFT = true;

        // Initialize the b-spline moduli.

        double data[PmeOrder];
        double x = 0.0;
        data[0] = 1.0 - x;
        data[1] = x;
        for (int i = 2; i < PmeOrder; i++) {
            double denom = 1.0/i;
            data[i] = x*data[i-1]*denom;
            for (int j = 1; j < i; j++)
                data[i-j] = ((x+j)*data[i-j-1] + ((i-j+1)-x)*data[i-j])*denom;
            data[0] = (1.0-x)*data[0]*denom;
        }
        int maxSize = max(max(gridSizeX, gridSizeY), gridSizeZ);
        vector<double> bsplines_data(maxSize+1, 0.0);
        for (int i = 2; i <= PmeOrder+1; i++)
            bsplines_data[i] = data[i-2];
        for (int dim = 0; dim < 3; dim++) {
            int ndata = (dim == 0 ? gridSizeX : dim == 1 ? gridSizeY : gridSizeZ);
            vector<double> moduli(ndata);

            // get the modulus of the discrete Fourier transform

            double factor = 2.0*M_PI/ndata;
            for (int i = 0; i < ndata; i++) {
                double sc = 0.0;
                double ss = 0.0;
                for (int j = 1; j <= ndata; j++) {
                    double arg = factor*i*(j-1);
                    sc += bsplines_data[j]*cos(arg);
                    ss += bsplines_data[j]*sin(arg);
                }
                moduli[i] = sc*sc+ss*ss;
            }

            // Fix for exponential Euler spline interpolation failure.

            double eps = 1.0e-7;
            if (moduli[0] < eps)
                moduli[0] = 0.9*moduli[1];
            for (int i = 1; i < ndata-1; i++)
                if (moduli[i] < eps)
                    moduli[i] = 0.9*(moduli[i-1]+moduli[i+1]);
            if (moduli[ndata-1] < eps)
                moduli[ndata-1] = 0.9*moduli[ndata-2];

            // Compute and apply the optimal zeta coefficient.

            int jcut = 50;
            for (int i = 1; i <= ndata; i++) {
                int k = i - 1;
                if (i > ndata/2)
                    k = k - ndata;
                double zeta;
                if (k == 0)
                    zeta = 1.0;
                else {
                    double sum1 = 1.0;
                    double sum2 = 1.0;
                    factor = M_PI*k/ndata;
                    for (int j = 1; j <= jcut; j++) {
                        double arg = factor/(factor+M_PI*j);
                        sum1 += pow(arg, PmeOrder);
                        sum2 += pow(arg, 2*PmeOrder);
                    }
                    for (int j = 1; j <= jcut; j++) {
                        double arg = factor/(factor-M_PI*j);
                        sum1 += pow(arg, PmeOrder);
                        sum2 += pow(arg, 2*PmeOrder);
                    }
                    zeta = sum2/sum1;
                }
                moduli[i-1] = moduli[i-1]*zeta*zeta;
            }
            if (cu.getUseDoublePrecision()) {
                if (dim == 0)
                    pmeBsplineModuliX->upload(moduli);
                else if (dim == 1)
                    pmeBsplineModuliY->upload(moduli);
                else
                    pmeBsplineModuliZ->upload(moduli);
            }
            else {
                vector<float> modulif(ndata);
                for (int i = 0; i < ndata; i++)
                    modulif[i] = (float) moduli[i];
                if (dim == 0)
                    pmeBsplineModuliX->upload(modulif);
                else if (dim == 1)
                    pmeBsplineModuliY->upload(modulif);
                else
                    pmeBsplineModuliZ->upload(modulif);
            }
        }
    }

    // Add an interaction to the default nonbonded kernel.  This doesn't actually do any calculations.  It's
    // just so that CudaNonbondedUtilities will build the exclusion flags and maintain the neighbor list.
    
    cu.getNonbondedUtilities().addInteraction(usePME, usePME, true, force.getCutoffDistance(), exclusions, "", force.getForceGroup());
    cu.getNonbondedUtilities().setUsePadding(false);
    cu.addForce(new ForceInfo(force));
}

void CudaCalcMMFFMultipoleForceKernel::initializeScaleFactors() {
    hasInitializedScaleFactors = true;
    CudaNonbondedUtilities& nb = cu.getNonbondedUtilities();
    
    // Figure out the covalent flag values to use for each atom pair.

    vector<ushort2> exclusionTiles;
    nb.getExclusionTiles().download(exclusionTiles);
    map<pair<int, int>, int> exclusionTileMap;
    for (int i = 0; i < (int) exclusionTiles.size(); i++) {
        ushort2 tile = exclusionTiles[i];
        exclusionTileMap[make_pair(tile.x, tile.y)] = i;
    }
    covalentFlags = CudaArray::create<uint2>(cu, nb.getExclusions().getSize(), "covalentFlags");
    vector<uint2> covalentFlagsVec(nb.getExclusions().getSize(), make_uint2(0, 0));
    for (int3 values : covalentFlagValues) {
        int atom1 = values.x;
        int atom2 = values.y;
        int value = values.z;
        int x = atom1/CudaContext::TileSize;
        int offset1 = atom1-x*CudaContext::TileSize;
        int y = atom2/CudaContext::TileSize;
        int offset2 = atom2-y*CudaContext::TileSize;
        int f1 = (value == 0 || value == 1 ? 1 : 0);
        int f2 = (value == 0 || value == 2 ? 1 : 0);
        if (x == y) {
            int index = exclusionTileMap[make_pair(x, y)]*CudaContext::TileSize;
            covalentFlagsVec[index+offset1].x |= f1<<offset2;
            covalentFlagsVec[index+offset1].y |= f2<<offset2;
            covalentFlagsVec[index+offset2].x |= f1<<offset1;
            covalentFlagsVec[index+offset2].y |= f2<<offset1;
        }
        else if (x > y) {
            int index = exclusionTileMap[make_pair(x, y)]*CudaContext::TileSize;
            covalentFlagsVec[index+offset1].x |= f1<<offset2;
            covalentFlagsVec[index+offset1].y |= f2<<offset2;
        }
        else {
            int index = exclusionTileMap[make_pair(y, x)]*CudaContext::TileSize;
            covalentFlagsVec[index+offset2].x |= f1<<offset1;
            covalentFlagsVec[index+offset2].y |= f2<<offset1;
        }
    }
    covalentFlags->upload(covalentFlagsVec);
    
    // Do the same for the polarization flags.
    
    polarizationGroupFlags = CudaArray::create<unsigned int>(cu, nb.getExclusions().getSize(), "polarizationGroupFlags");
    vector<unsigned int> polarizationGroupFlagsVec(nb.getExclusions().getSize(), 0);
    for (int2 values : polarizationFlagValues) {
        int atom1 = values.x;
        int atom2 = values.y;
        int x = atom1/CudaContext::TileSize;
        int offset1 = atom1-x*CudaContext::TileSize;
        int y = atom2/CudaContext::TileSize;
        int offset2 = atom2-y*CudaContext::TileSize;
        if (x == y) {
            int index = exclusionTileMap[make_pair(x, y)]*CudaContext::TileSize;
            polarizationGroupFlagsVec[index+offset1] |= 1<<offset2;
            polarizationGroupFlagsVec[index+offset2] |= 1<<offset1;
        }
        else if (x > y) {
            int index = exclusionTileMap[make_pair(x, y)]*CudaContext::TileSize;
            polarizationGroupFlagsVec[index+offset1] |= 1<<offset2;
        }
        else {
            int index = exclusionTileMap[make_pair(y, x)]*CudaContext::TileSize;
            polarizationGroupFlagsVec[index+offset2] |= 1<<offset1;
        }
    }
    polarizationGroupFlags->upload(polarizationGroupFlagsVec);
}

double CudaCalcMMFFMultipoleForceKernel::execute(ContextImpl& context, bool includeForces, bool includeEnergy) {
    if (!hasInitializedScaleFactors) {
        initializeScaleFactors();
        for (auto impl : context.getForceImpls()) {
            MMFFGeneralizedKirkwoodForceImpl* gkImpl = dynamic_cast<MMFFGeneralizedKirkwoodForceImpl*>(impl);
            if (gkImpl != NULL) {
                gkKernel = dynamic_cast<CudaCalcMMFFGeneralizedKirkwoodForceKernel*>(&gkImpl->getKernel().getImpl());
                break;
            }
        }
    }
    CudaNonbondedUtilities& nb = cu.getNonbondedUtilities();
    
    // Compute the lab frame moments.

    void* computeMomentsArgs[] = {&cu.getPosq().getDevicePointer(), &multipoleParticles->getDevicePointer(),
        &molecularDipoles->getDevicePointer(), &molecularQuadrupoles->getDevicePointer(),
        &labFrameDipoles->getDevicePointer(), &labFrameQuadrupoles->getDevicePointer(),
        &sphericalDipoles->getDevicePointer(), &sphericalQuadrupoles->getDevicePointer()};
    cu.executeKernel(computeMomentsKernel, computeMomentsArgs, cu.getNumAtoms());
    int startTileIndex = nb.getStartTileIndex();
    int numTileIndices = nb.getNumTiles();
    int numForceThreadBlocks = nb.getNumForceThreadBlocks();
    int elementSize = (cu.getUseDoublePrecision() ? sizeof(double) : sizeof(float));
    if (pmeGrid == NULL) {
        // Compute induced dipoles.
        
        if (gkKernel == NULL) {
            void* computeFixedFieldArgs[] = {&field->getDevicePointer(), &fieldPolar->getDevicePointer(), &cu.getPosq().getDevicePointer(),
                &covalentFlags->getDevicePointer(), &polarizationGroupFlags->getDevicePointer(), &nb.getExclusionTiles().getDevicePointer(), &startTileIndex, &numTileIndices,
                &labFrameDipoles->getDevicePointer(), &labFrameQuadrupoles->getDevicePointer(), &dampingAndThole->getDevicePointer()};
            cu.executeKernel(computeFixedFieldKernel, computeFixedFieldArgs, numForceThreadBlocks*fixedFieldThreads, fixedFieldThreads);
            void* recordInducedDipolesArgs[] = {&field->getDevicePointer(), &fieldPolar->getDevicePointer(),
                &inducedDipole->getDevicePointer(), &inducedDipolePolar->getDevicePointer(), &polarizability->getDevicePointer()};
            cu.executeKernel(recordInducedDipolesKernel, recordInducedDipolesArgs, cu.getNumAtoms());
        }
        else {
            gkKernel->computeBornRadii();
            void* computeFixedFieldArgs[] = {&field->getDevicePointer(), &fieldPolar->getDevicePointer(), &cu.getPosq().getDevicePointer(),
                &covalentFlags->getDevicePointer(), &polarizationGroupFlags->getDevicePointer(), &nb.getExclusionTiles().getDevicePointer(), &startTileIndex, &numTileIndices,
                &gkKernel->getBornRadii()->getDevicePointer(), &gkKernel->getField()->getDevicePointer(),
                &labFrameDipoles->getDevicePointer(), &labFrameQuadrupoles->getDevicePointer(), &dampingAndThole->getDevicePointer()};
            cu.executeKernel(computeFixedFieldKernel, computeFixedFieldArgs, numForceThreadBlocks*fixedFieldThreads, fixedFieldThreads);
            void* recordInducedDipolesArgs[] = {&field->getDevicePointer(), &fieldPolar->getDevicePointer(),
                &gkKernel->getField()->getDevicePointer(), &gkKernel->getInducedDipoles()->getDevicePointer(),
                &gkKernel->getInducedDipolesPolar()->getDevicePointer(), &inducedDipole->getDevicePointer(),
                &inducedDipolePolar->getDevicePointer(), &polarizability->getDevicePointer()};
            cu.executeKernel(recordInducedDipolesKernel, recordInducedDipolesArgs, cu.getNumAtoms());
        }
        
        // Iterate until the dipoles converge.
        
        if (polarizationType == MMFFMultipoleForce::Extrapolated)
            computeExtrapolatedDipoles(NULL);
        for (int i = 0; i < maxInducedIterations; i++) {
            computeInducedField(NULL);
            bool converged = iterateDipolesByDIIS(i);
            if (converged)
                break;
        }
        
        // Compute electrostatic force.
        
        void* electrostaticsArgs[] = {&cu.getForce().getDevicePointer(), &torque->getDevicePointer(), &cu.getEnergyBuffer().getDevicePointer(),
            &cu.getPosq().getDevicePointer(), &covalentFlags->getDevicePointer(), &polarizationGroupFlags->getDevicePointer(),
            &nb.getExclusionTiles().getDevicePointer(), &startTileIndex, &numTileIndices,
            &sphericalDipoles->getDevicePointer(), &sphericalQuadrupoles->getDevicePointer(),
            &inducedDipole->getDevicePointer(), &inducedDipolePolar->getDevicePointer(), &dampingAndThole->getDevicePointer()};
        cu.executeKernel(electrostaticsKernel, electrostaticsArgs, numForceThreadBlocks*electrostaticsThreads, electrostaticsThreads);
        if (gkKernel != NULL)
            gkKernel->finishComputation(*torque, *labFrameDipoles, *labFrameQuadrupoles, *inducedDipole, *inducedDipolePolar, *dampingAndThole, *covalentFlags, *polarizationGroupFlags);
    }
    else {
        // Compute reciprocal box vectors.
        
        Vec3 boxVectors[3];
        cu.getPeriodicBoxVectors(boxVectors[0], boxVectors[1], boxVectors[2]);
        double determinant = boxVectors[0][0]*boxVectors[1][1]*boxVectors[2][2];
        double scale = 1.0/determinant;
        double3 recipBoxVectors[3];
        recipBoxVectors[0] = make_double3(boxVectors[1][1]*boxVectors[2][2]*scale, 0, 0);
        recipBoxVectors[1] = make_double3(-boxVectors[1][0]*boxVectors[2][2]*scale, boxVectors[0][0]*boxVectors[2][2]*scale, 0);
        recipBoxVectors[2] = make_double3((boxVectors[1][0]*boxVectors[2][1]-boxVectors[1][1]*boxVectors[2][0])*scale, -boxVectors[0][0]*boxVectors[2][1]*scale, boxVectors[0][0]*boxVectors[1][1]*scale);
        float3 recipBoxVectorsFloat[3];
        void* recipBoxVectorPointer[3];
        if (cu.getUseDoublePrecision()) {
            recipBoxVectorPointer[0] = &recipBoxVectors[0];
            recipBoxVectorPointer[1] = &recipBoxVectors[1];
            recipBoxVectorPointer[2] = &recipBoxVectors[2];
        }
        else {
            recipBoxVectorsFloat[0] = make_float3((float) recipBoxVectors[0].x, 0, 0);
            recipBoxVectorsFloat[1] = make_float3((float) recipBoxVectors[1].x, (float) recipBoxVectors[1].y, 0);
            recipBoxVectorsFloat[2] = make_float3((float) recipBoxVectors[2].x, (float) recipBoxVectors[2].y, (float) recipBoxVectors[2].z);
            recipBoxVectorPointer[0] = &recipBoxVectorsFloat[0];
            recipBoxVectorPointer[1] = &recipBoxVectorsFloat[1];
            recipBoxVectorPointer[2] = &recipBoxVectorsFloat[2];
        }

        // Reciprocal space calculation.
        
        unsigned int maxTiles = nb.getInteractingTiles().getSize();
        void* pmeTransformMultipolesArgs[] = {&labFrameDipoles->getDevicePointer(), &labFrameQuadrupoles->getDevicePointer(),
            &fracDipoles->getDevicePointer(), &fracQuadrupoles->getDevicePointer(), recipBoxVectorPointer[0], recipBoxVectorPointer[1], recipBoxVectorPointer[2]};
        cu.executeKernel(pmeTransformMultipolesKernel, pmeTransformMultipolesArgs, cu.getNumAtoms());
        void* pmeSpreadFixedMultipolesArgs[] = {&cu.getPosq().getDevicePointer(), &fracDipoles->getDevicePointer(), &fracQuadrupoles->getDevicePointer(),
            &pmeGrid->getDevicePointer(), cu.getPeriodicBoxVecXPointer(), cu.getPeriodicBoxVecYPointer(), cu.getPeriodicBoxVecZPointer(),
            recipBoxVectorPointer[0], recipBoxVectorPointer[1], recipBoxVectorPointer[2]};
        cu.executeKernel(pmeSpreadFixedMultipolesKernel, pmeSpreadFixedMultipolesArgs, cu.getNumAtoms());
        void* finishSpreadArgs[] = {&pmeGrid->getDevicePointer()};
        if (cu.getUseDoublePrecision())
            cu.executeKernel(pmeFinishSpreadChargeKernel, finishSpreadArgs, pmeGrid->getSize());
        if (cu.getUseDoublePrecision())
            cufftExecZ2Z(fft, (double2*) pmeGrid->getDevicePointer(), (double2*) pmeGrid->getDevicePointer(), CUFFT_FORWARD);
        else
            cufftExecC2C(fft, (float2*) pmeGrid->getDevicePointer(), (float2*) pmeGrid->getDevicePointer(), CUFFT_FORWARD);
        void* pmeConvolutionArgs[] = {&pmeGrid->getDevicePointer(), &pmeBsplineModuliX->getDevicePointer(), &pmeBsplineModuliY->getDevicePointer(),
            &pmeBsplineModuliZ->getDevicePointer(), cu.getPeriodicBoxSizePointer(), recipBoxVectorPointer[0], recipBoxVectorPointer[1], recipBoxVectorPointer[2]};
        cu.executeKernel(pmeConvolutionKernel, pmeConvolutionArgs, gridSizeX*gridSizeY*gridSizeZ, 256);
        if (cu.getUseDoublePrecision())
            cufftExecZ2Z(fft, (double2*) pmeGrid->getDevicePointer(), (double2*) pmeGrid->getDevicePointer(), CUFFT_INVERSE);
        else
            cufftExecC2C(fft, (float2*) pmeGrid->getDevicePointer(), (float2*) pmeGrid->getDevicePointer(), CUFFT_INVERSE);
        void* pmeFixedPotentialArgs[] = {&pmeGrid->getDevicePointer(), &pmePhi->getDevicePointer(), &field->getDevicePointer(),
            &fieldPolar ->getDevicePointer(), &cu.getPosq().getDevicePointer(), &labFrameDipoles->getDevicePointer(),
            cu.getPeriodicBoxVecXPointer(), cu.getPeriodicBoxVecYPointer(), cu.getPeriodicBoxVecZPointer(),
            recipBoxVectorPointer[0], recipBoxVectorPointer[1], recipBoxVectorPointer[2]};
        cu.executeKernel(pmeFixedPotentialKernel, pmeFixedPotentialArgs, cu.getNumAtoms());
        void* pmeTransformFixedPotentialArgs[] = {&pmePhi->getDevicePointer(), &pmeCphi->getDevicePointer(), recipBoxVectorPointer[0], recipBoxVectorPointer[1], recipBoxVectorPointer[2]};
        cu.executeKernel(pmeTransformPotentialKernel, pmeTransformFixedPotentialArgs, cu.getNumAtoms());
        void* pmeFixedForceArgs[] = {&cu.getPosq().getDevicePointer(), &cu.getForce().getDevicePointer(), &torque->getDevicePointer(),
            &cu.getEnergyBuffer().getDevicePointer(), &labFrameDipoles->getDevicePointer(), &labFrameQuadrupoles->getDevicePointer(),
            &fracDipoles->getDevicePointer(), &fracQuadrupoles->getDevicePointer(), &pmePhi->getDevicePointer(), &pmeCphi->getDevicePointer(),
            recipBoxVectorPointer[0], recipBoxVectorPointer[1], recipBoxVectorPointer[2]};
        cu.executeKernel(pmeFixedForceKernel, pmeFixedForceArgs, cu.getNumAtoms());
        
        // Direct space calculation.
        
        void* computeFixedFieldArgs[] = {&field->getDevicePointer(), &fieldPolar->getDevicePointer(), &cu.getPosq().getDevicePointer(),
            &covalentFlags->getDevicePointer(), &polarizationGroupFlags->getDevicePointer(), &nb.getExclusionTiles().getDevicePointer(), &startTileIndex, &numTileIndices,
            &nb.getInteractingTiles().getDevicePointer(), &nb.getInteractionCount().getDevicePointer(), cu.getPeriodicBoxSizePointer(),
            cu.getInvPeriodicBoxSizePointer(), cu.getPeriodicBoxVecXPointer(), cu.getPeriodicBoxVecYPointer(), cu.getPeriodicBoxVecZPointer(),
            &maxTiles, &nb.getBlockCenters().getDevicePointer(), &nb.getInteractingAtoms().getDevicePointer(),
            &labFrameDipoles->getDevicePointer(), &labFrameQuadrupoles->getDevicePointer(), &dampingAndThole->getDevicePointer()};
        cu.executeKernel(computeFixedFieldKernel, computeFixedFieldArgs, numForceThreadBlocks*fixedFieldThreads, fixedFieldThreads);
        void* recordInducedDipolesArgs[] = {&field->getDevicePointer(), &fieldPolar->getDevicePointer(),
            &inducedDipole->getDevicePointer(), &inducedDipolePolar->getDevicePointer(), &polarizability->getDevicePointer()};
        cu.executeKernel(recordInducedDipolesKernel, recordInducedDipolesArgs, cu.getNumAtoms());

        // Reciprocal space calculation for the induced dipoles.

        cu.clearBuffer(*pmeGrid);
        void* pmeSpreadInducedDipolesArgs[] = {&cu.getPosq().getDevicePointer(), &inducedDipole->getDevicePointer(), &inducedDipolePolar->getDevicePointer(),
            &pmeGrid->getDevicePointer(), cu.getPeriodicBoxVecXPointer(), cu.getPeriodicBoxVecYPointer(), cu.getPeriodicBoxVecZPointer(),
            recipBoxVectorPointer[0], recipBoxVectorPointer[1], recipBoxVectorPointer[2]};
        cu.executeKernel(pmeSpreadInducedDipolesKernel, pmeSpreadInducedDipolesArgs, cu.getNumAtoms());
        if (cu.getUseDoublePrecision())
            cu.executeKernel(pmeFinishSpreadChargeKernel, finishSpreadArgs, pmeGrid->getSize());
        if (cu.getUseDoublePrecision())
            cufftExecZ2Z(fft, (double2*) pmeGrid->getDevicePointer(), (double2*) pmeGrid->getDevicePointer(), CUFFT_FORWARD);
        else
            cufftExecC2C(fft, (float2*) pmeGrid->getDevicePointer(), (float2*) pmeGrid->getDevicePointer(), CUFFT_FORWARD);
        cu.executeKernel(pmeConvolutionKernel, pmeConvolutionArgs, gridSizeX*gridSizeY*gridSizeZ, 256);
        if (cu.getUseDoublePrecision())
            cufftExecZ2Z(fft, (double2*) pmeGrid->getDevicePointer(), (double2*) pmeGrid->getDevicePointer(), CUFFT_INVERSE);
        else
            cufftExecC2C(fft, (float2*) pmeGrid->getDevicePointer(), (float2*) pmeGrid->getDevicePointer(), CUFFT_INVERSE);
        void* pmeInducedPotentialArgs[] = {&pmeGrid->getDevicePointer(), &pmePhid->getDevicePointer(), &pmePhip->getDevicePointer(),
            &pmePhidp->getDevicePointer(), &cu.getPosq().getDevicePointer(), cu.getPeriodicBoxVecXPointer(), cu.getPeriodicBoxVecYPointer(),
            cu.getPeriodicBoxVecZPointer(), recipBoxVectorPointer[0], recipBoxVectorPointer[1], recipBoxVectorPointer[2]};
        cu.executeKernel(pmeInducedPotentialKernel, pmeInducedPotentialArgs, cu.getNumAtoms());
        
        // Iterate until the dipoles converge.
        
        if (polarizationType == MMFFMultipoleForce::Extrapolated)
            computeExtrapolatedDipoles(recipBoxVectorPointer);
        for (int i = 0; i < maxInducedIterations; i++) {
            computeInducedField(recipBoxVectorPointer);
            bool converged = iterateDipolesByDIIS(i);
            if (converged)
                break;
        }
        
        // Compute electrostatic force.
        
        void* electrostaticsArgs[] = {&cu.getForce().getDevicePointer(), &torque->getDevicePointer(), &cu.getEnergyBuffer().getDevicePointer(),
            &cu.getPosq().getDevicePointer(), &covalentFlags->getDevicePointer(), &polarizationGroupFlags->getDevicePointer(),
            &nb.getExclusionTiles().getDevicePointer(), &startTileIndex, &numTileIndices,
            &nb.getInteractingTiles().getDevicePointer(), &nb.getInteractionCount().getDevicePointer(),
            cu.getPeriodicBoxSizePointer(), cu.getInvPeriodicBoxSizePointer(), cu.getPeriodicBoxVecXPointer(), cu.getPeriodicBoxVecYPointer(), cu.getPeriodicBoxVecZPointer(),
            &maxTiles, &nb.getBlockCenters().getDevicePointer(), &nb.getInteractingAtoms().getDevicePointer(),
            &sphericalDipoles->getDevicePointer(), &sphericalQuadrupoles->getDevicePointer(),
            &inducedDipole->getDevicePointer(), &inducedDipolePolar->getDevicePointer(), &dampingAndThole->getDevicePointer()};
        cu.executeKernel(electrostaticsKernel, electrostaticsArgs, numForceThreadBlocks*electrostaticsThreads, electrostaticsThreads);
        void* pmeTransformInducedPotentialArgs[] = {&pmePhidp->getDevicePointer(), &pmeCphi->getDevicePointer(), recipBoxVectorPointer[0], recipBoxVectorPointer[1], recipBoxVectorPointer[2]};
        cu.executeKernel(pmeTransformPotentialKernel, pmeTransformInducedPotentialArgs, cu.getNumAtoms());
        void* pmeInducedForceArgs[] = {&cu.getPosq().getDevicePointer(), &cu.getForce().getDevicePointer(), &torque->getDevicePointer(),
            &cu.getEnergyBuffer().getDevicePointer(), &labFrameDipoles->getDevicePointer(), &labFrameQuadrupoles->getDevicePointer(),
            &fracDipoles->getDevicePointer(), &fracQuadrupoles->getDevicePointer(),
            &inducedDipole->getDevicePointer(), &inducedDipolePolar->getDevicePointer(), &pmePhi->getDevicePointer(), &pmePhid->getDevicePointer(),
            &pmePhip->getDevicePointer(), &pmePhidp->getDevicePointer(), &pmeCphi->getDevicePointer(), recipBoxVectorPointer[0], recipBoxVectorPointer[1], recipBoxVectorPointer[2]};
        cu.executeKernel(pmeInducedForceKernel, pmeInducedForceArgs, cu.getNumAtoms());
    }
    
    // If using extrapolated polarization, add in force contributions from µ(m) T µ(n).
    
    if (polarizationType == MMFFMultipoleForce::Extrapolated) {
        if (gkKernel == NULL) {
            void* extrapolatedArgs[] = {&cu.getForce().getDevicePointer(), &extrapolatedDipole->getDevicePointer(),
                &extrapolatedDipolePolar->getDevicePointer(), &extrapolatedDipoleFieldGradient->getDevicePointer(), &extrapolatedDipoleFieldGradientPolar->getDevicePointer()};
            cu.executeKernel(addExtrapolatedGradientKernel, extrapolatedArgs, numMultipoles);
        }
        else {
            void* extrapolatedArgs[] = {&cu.getForce().getDevicePointer(), &extrapolatedDipole->getDevicePointer(),
                &extrapolatedDipolePolar->getDevicePointer(), &extrapolatedDipoleFieldGradient->getDevicePointer(), &extrapolatedDipoleFieldGradientPolar->getDevicePointer(),
                &extrapolatedDipoleGk->getDevicePointer(), &extrapolatedDipoleGkPolar->getDevicePointer(),
                &extrapolatedDipoleFieldGradientGk->getDevicePointer(), &extrapolatedDipoleFieldGradientGkPolar->getDevicePointer()};
            cu.executeKernel(addExtrapolatedGradientKernel, extrapolatedArgs, numMultipoles);
        }
    }

    // Map torques to force.

    void* mapTorqueArgs[] = {&cu.getForce().getDevicePointer(), &torque->getDevicePointer(),
        &cu.getPosq().getDevicePointer(), &multipoleParticles->getDevicePointer()};
    cu.executeKernel(mapTorqueKernel, mapTorqueArgs, cu.getNumAtoms());
    
    // Record the current atom positions so we can tell later if they have changed.
    
    cu.getPosq().copyTo(*lastPositions);
    multipolesAreValid = true;
    return 0.0;
}

void CudaCalcMMFFMultipoleForceKernel::computeInducedField(void** recipBoxVectorPointer) {
    CudaNonbondedUtilities& nb = cu.getNonbondedUtilities();
    int startTileIndex = nb.getStartTileIndex();
    int numTileIndices = nb.getNumTiles();
    int numForceThreadBlocks = nb.getNumForceThreadBlocks();
    unsigned int maxTiles = 0;
    vector<void*> computeInducedFieldArgs;
    computeInducedFieldArgs.push_back(&inducedField->getDevicePointer());
    computeInducedFieldArgs.push_back(&inducedFieldPolar->getDevicePointer());
    computeInducedFieldArgs.push_back(&cu.getPosq().getDevicePointer());
    computeInducedFieldArgs.push_back(&nb.getExclusionTiles().getDevicePointer());
    computeInducedFieldArgs.push_back(&inducedDipole->getDevicePointer());
    computeInducedFieldArgs.push_back(&inducedDipolePolar->getDevicePointer());
    computeInducedFieldArgs.push_back(&startTileIndex);
    computeInducedFieldArgs.push_back(&numTileIndices);
    if (polarizationType == MMFFMultipoleForce::Extrapolated) {
        computeInducedFieldArgs.push_back(&inducedDipoleFieldGradient->getDevicePointer());
        computeInducedFieldArgs.push_back(&inducedDipoleFieldGradientPolar->getDevicePointer());
    }
    if (pmeGrid != NULL) {
        computeInducedFieldArgs.push_back(&nb.getInteractingTiles().getDevicePointer());
        computeInducedFieldArgs.push_back(&nb.getInteractionCount().getDevicePointer());
        computeInducedFieldArgs.push_back(cu.getPeriodicBoxSizePointer());
        computeInducedFieldArgs.push_back(cu.getInvPeriodicBoxSizePointer());
        computeInducedFieldArgs.push_back(cu.getPeriodicBoxVecXPointer());
        computeInducedFieldArgs.push_back(cu.getPeriodicBoxVecYPointer());
        computeInducedFieldArgs.push_back(cu.getPeriodicBoxVecZPointer());
        computeInducedFieldArgs.push_back(&maxTiles);
        computeInducedFieldArgs.push_back(&nb.getBlockCenters().getDevicePointer());
        computeInducedFieldArgs.push_back(&nb.getInteractingAtoms().getDevicePointer());
    }
    if (gkKernel != NULL) {
        computeInducedFieldArgs.push_back(&gkKernel->getInducedField()->getDevicePointer());
        computeInducedFieldArgs.push_back(&gkKernel->getInducedFieldPolar()->getDevicePointer());
        computeInducedFieldArgs.push_back(&gkKernel->getInducedDipoles()->getDevicePointer());
        computeInducedFieldArgs.push_back(&gkKernel->getInducedDipolesPolar()->getDevicePointer());
        computeInducedFieldArgs.push_back(&gkKernel->getBornRadii()->getDevicePointer());
        if (polarizationType == MMFFMultipoleForce::Extrapolated) {
            computeInducedFieldArgs.push_back(&inducedDipoleFieldGradientGk->getDevicePointer());
            computeInducedFieldArgs.push_back(&inducedDipoleFieldGradientGkPolar->getDevicePointer());
        }
    }
    computeInducedFieldArgs.push_back(&dampingAndThole->getDevicePointer());
    cu.clearBuffer(*inducedField);
    cu.clearBuffer(*inducedFieldPolar);
    if (polarizationType == MMFFMultipoleForce::Extrapolated) {
        cu.clearBuffer(*inducedDipoleFieldGradient);
        cu.clearBuffer(*inducedDipoleFieldGradientPolar);
    }
    if (gkKernel != NULL) {
        cu.clearBuffer(*gkKernel->getInducedField());
        cu.clearBuffer(*gkKernel->getInducedFieldPolar());
        if (polarizationType == MMFFMultipoleForce::Extrapolated) {
            cu.clearBuffer(*inducedDipoleFieldGradientGk);
            cu.clearBuffer(*inducedDipoleFieldGradientGkPolar);
        }
    }
    if (pmeGrid == NULL)
        cu.executeKernel(computeInducedFieldKernel, &computeInducedFieldArgs[0], numForceThreadBlocks*inducedFieldThreads, inducedFieldThreads);
    else {
        maxTiles = nb.getInteractingTiles().getSize();
        cu.executeKernel(computeInducedFieldKernel, &computeInducedFieldArgs[0], numForceThreadBlocks*inducedFieldThreads, inducedFieldThreads);
        cu.clearBuffer(*pmeGrid);
        void* pmeSpreadInducedDipolesArgs[] = {&cu.getPosq().getDevicePointer(), &inducedDipole->getDevicePointer(), &inducedDipolePolar->getDevicePointer(),
            &pmeGrid->getDevicePointer(), cu.getPeriodicBoxVecXPointer(), cu.getPeriodicBoxVecYPointer(), cu.getPeriodicBoxVecZPointer(),
            recipBoxVectorPointer[0], recipBoxVectorPointer[1], recipBoxVectorPointer[2]};
        cu.executeKernel(pmeSpreadInducedDipolesKernel, pmeSpreadInducedDipolesArgs, cu.getNumAtoms());
        if (cu.getUseDoublePrecision()) {
            void* finishSpreadArgs[] = {&pmeGrid->getDevicePointer()};
            cu.executeKernel(pmeFinishSpreadChargeKernel, finishSpreadArgs, pmeGrid->getSize());
        }
        if (cu.getUseDoublePrecision())
            cufftExecZ2Z(fft, (double2*) pmeGrid->getDevicePointer(), (double2*) pmeGrid->getDevicePointer(), CUFFT_FORWARD);
        else
            cufftExecC2C(fft, (float2*) pmeGrid->getDevicePointer(), (float2*) pmeGrid->getDevicePointer(), CUFFT_FORWARD);
        void* pmeConvolutionArgs[] = {&pmeGrid->getDevicePointer(), &pmeBsplineModuliX->getDevicePointer(), &pmeBsplineModuliY->getDevicePointer(),
            &pmeBsplineModuliZ->getDevicePointer(), cu.getPeriodicBoxSizePointer(), recipBoxVectorPointer[0], recipBoxVectorPointer[1], recipBoxVectorPointer[2]};
        cu.executeKernel(pmeConvolutionKernel, pmeConvolutionArgs, gridSizeX*gridSizeY*gridSizeZ, 256);
        if (cu.getUseDoublePrecision())
            cufftExecZ2Z(fft, (double2*) pmeGrid->getDevicePointer(), (double2*) pmeGrid->getDevicePointer(), CUFFT_INVERSE);
        else
            cufftExecC2C(fft, (float2*) pmeGrid->getDevicePointer(), (float2*) pmeGrid->getDevicePointer(), CUFFT_INVERSE);
        void* pmeInducedPotentialArgs[] = {&pmeGrid->getDevicePointer(), &pmePhid->getDevicePointer(), &pmePhip->getDevicePointer(),
            &pmePhidp->getDevicePointer(), &cu.getPosq().getDevicePointer(), cu.getPeriodicBoxVecXPointer(), cu.getPeriodicBoxVecYPointer(),
            cu.getPeriodicBoxVecZPointer(), recipBoxVectorPointer[0], recipBoxVectorPointer[1], recipBoxVectorPointer[2]};
        cu.executeKernel(pmeInducedPotentialKernel, pmeInducedPotentialArgs, cu.getNumAtoms());
        if (polarizationType == MMFFMultipoleForce::Extrapolated) {
            void* pmeRecordInducedFieldDipolesArgs[] = {&pmePhid->getDevicePointer(), &pmePhip->getDevicePointer(),
                &inducedField->getDevicePointer(), &inducedFieldPolar->getDevicePointer(), &inducedDipole->getDevicePointer(),
                &inducedDipolePolar->getDevicePointer(), &inducedDipoleFieldGradient->getDevicePointer(), &inducedDipoleFieldGradientPolar->getDevicePointer(),
                recipBoxVectorPointer[0], recipBoxVectorPointer[1], recipBoxVectorPointer[2]};
            cu.executeKernel(pmeRecordInducedFieldDipolesKernel, pmeRecordInducedFieldDipolesArgs, cu.getNumAtoms());
        }
        else {
            void* pmeRecordInducedFieldDipolesArgs[] = {&pmePhid->getDevicePointer(), &pmePhip->getDevicePointer(),
                &inducedField->getDevicePointer(), &inducedFieldPolar->getDevicePointer(), &inducedDipole->getDevicePointer(), &inducedDipolePolar->getDevicePointer(),
                recipBoxVectorPointer[0], recipBoxVectorPointer[1], recipBoxVectorPointer[2]};
            cu.executeKernel(pmeRecordInducedFieldDipolesKernel, pmeRecordInducedFieldDipolesArgs, cu.getNumAtoms());
        }
    }
}

bool CudaCalcMMFFMultipoleForceKernel::iterateDipolesByDIIS(int iteration) {
    void* npt = NULL;
    bool trueValue = true, falseValue = false;
    int elementSize = (cu.getUseDoublePrecision() ? sizeof(double) : sizeof(float));
    
    // Record the dipoles and errors into the lists of previous dipoles.
    
    if (gkKernel != NULL) {
        void* recordDIISDipolesGkArgs[] = {&field->getDevicePointer(), &fieldPolar->getDevicePointer(), &gkKernel->getField()->getDevicePointer(), &gkKernel->getInducedField()->getDevicePointer(),
            &gkKernel->getInducedFieldPolar()->getDevicePointer(), &gkKernel->getInducedDipoles()->getDevicePointer(), &gkKernel->getInducedDipolesPolar()->getDevicePointer(), 
            &polarizability->getDevicePointer(), &inducedDipoleErrors->getDevicePointer(), &prevDipolesGk->getDevicePointer(),
            &prevDipolesGkPolar->getDevicePointer(), &prevErrors->getDevicePointer(), &iteration, &falseValue, &diisMatrix->getDevicePointer()};
        cu.executeKernel(recordDIISDipolesKernel, recordDIISDipolesGkArgs, cu.getNumThreadBlocks()*cu.ThreadBlockSize, cu.ThreadBlockSize, cu.ThreadBlockSize*elementSize*2);
    }
    void* recordDIISDipolesArgs[] = {&field->getDevicePointer(), &fieldPolar->getDevicePointer(), &npt, &inducedField->getDevicePointer(),
        &inducedFieldPolar->getDevicePointer(), &inducedDipole->getDevicePointer(), &inducedDipolePolar->getDevicePointer(),
        &polarizability->getDevicePointer(), &inducedDipoleErrors->getDevicePointer(), &prevDipoles->getDevicePointer(),
        &prevDipolesPolar->getDevicePointer(), &prevErrors->getDevicePointer(), &iteration, &trueValue, &diisMatrix->getDevicePointer()};
    cu.executeKernel(recordDIISDipolesKernel, recordDIISDipolesArgs, cu.getNumThreadBlocks()*cu.ThreadBlockSize, cu.ThreadBlockSize, cu.ThreadBlockSize*elementSize*2);
    float2* errors = (float2*) cu.getPinnedBuffer();
    inducedDipoleErrors->download(errors, false);
    cuEventRecord(syncEvent, cu.getCurrentStream());
    
    // Build the DIIS matrix.
    
    int numPrev = (iteration+1 < MaxPrevDIISDipoles ? iteration+1 : MaxPrevDIISDipoles);
    void* buildMatrixArgs[] = {&prevErrors->getDevicePointer(), &iteration, &diisMatrix->getDevicePointer()};
    int threadBlocks = min(numPrev, cu.getNumThreadBlocks());
    int blockSize = 512;
    cu.executeKernel(buildMatrixKernel, buildMatrixArgs, threadBlocks*blockSize, blockSize, blockSize*elementSize);
    
    // Solve the matrix.

    void* solveMatrixArgs[] = {&iteration, &diisMatrix->getDevicePointer(), &diisCoefficients->getDevicePointer()};
    cu.executeKernel(solveMatrixKernel, solveMatrixArgs, 32, 32);
    
    // Determine whether the iteration has converged.
    
    cuEventSynchronize(syncEvent);
    double total1 = 0.0, total2 = 0.0;
    for (int j = 0; j < inducedDipoleErrors->getSize(); j++) {
        total1 += errors[j].x;
        total2 += errors[j].y;
    }
    if (48.033324*sqrt(max(total1, total2)/cu.getNumAtoms()) < inducedEpsilon)
        return true;
    
    // Compute the dipoles.
    
    void* updateInducedFieldArgs[] = {&inducedDipole->getDevicePointer(), &inducedDipolePolar->getDevicePointer(),
        &prevDipoles->getDevicePointer(), &prevDipolesPolar->getDevicePointer(), &diisCoefficients->getDevicePointer(), &numPrev};
    cu.executeKernel(updateInducedFieldKernel, updateInducedFieldArgs, 3*cu.getNumAtoms(), 256);
    if (gkKernel != NULL) {
        void* updateInducedFieldGkArgs[] = {&gkKernel->getInducedDipoles()->getDevicePointer(), &gkKernel->getInducedDipolesPolar()->getDevicePointer(),
            &prevDipolesGk->getDevicePointer(), &prevDipolesGkPolar->getDevicePointer(), &diisCoefficients->getDevicePointer(), &numPrev};
        cu.executeKernel(updateInducedFieldKernel, updateInducedFieldGkArgs, 3*cu.getNumAtoms(), 256);
    }
    return false;
}

void CudaCalcMMFFMultipoleForceKernel::computeExtrapolatedDipoles(void** recipBoxVectorPointer) {
    // Start by storing the direct dipoles as PT0

    if (gkKernel == NULL) {
        void* initArgs[] = {&inducedDipole->getDevicePointer(), &inducedDipolePolar->getDevicePointer(), &extrapolatedDipole->getDevicePointer(),
            &extrapolatedDipolePolar->getDevicePointer(), &inducedDipoleFieldGradient->getDevicePointer(), &inducedDipoleFieldGradientPolar->getDevicePointer()};
        cu.executeKernel(initExtrapolatedKernel, initArgs, extrapolatedDipole->getSize());
    }
    else {
        void* initArgs[] = {&inducedDipole->getDevicePointer(), &inducedDipolePolar->getDevicePointer(), &extrapolatedDipole->getDevicePointer(),
            &extrapolatedDipolePolar->getDevicePointer(), &inducedDipoleFieldGradient->getDevicePointer(), &inducedDipoleFieldGradientPolar->getDevicePointer(),
            &gkKernel->getInducedDipoles()->getDevicePointer(), &gkKernel->getInducedDipolesPolar()->getDevicePointer(), &extrapolatedDipoleGk->getDevicePointer(),
            &extrapolatedDipoleGkPolar->getDevicePointer(), &inducedDipoleFieldGradientGk->getDevicePointer(), &inducedDipoleFieldGradientGkPolar->getDevicePointer()};
        cu.executeKernel(initExtrapolatedKernel, initArgs, extrapolatedDipole->getSize());
    }

    // Recursively apply alpha.Tau to the µ_(n) components to generate µ_(n+1), and store the result

    for (int order = 1; order < maxExtrapolationOrder; ++order) {
        computeInducedField(recipBoxVectorPointer);
        if (gkKernel == NULL) {
            void* iterateArgs[] = {&order, &inducedDipole->getDevicePointer(), &inducedDipolePolar->getDevicePointer(), &extrapolatedDipole->getDevicePointer(),
                &extrapolatedDipolePolar->getDevicePointer(), &inducedDipoleFieldGradient->getDevicePointer(), &inducedDipoleFieldGradientPolar->getDevicePointer(),
                &inducedField->getDevicePointer(), &inducedFieldPolar->getDevicePointer(), &extrapolatedDipoleFieldGradient->getDevicePointer(), &extrapolatedDipoleFieldGradientPolar->getDevicePointer(),
                &polarizability->getDevicePointer()};
            cu.executeKernel(iterateExtrapolatedKernel, iterateArgs, extrapolatedDipole->getSize());
        }
        else {
            void* iterateArgs[] = {&order, &inducedDipole->getDevicePointer(), &inducedDipolePolar->getDevicePointer(), &extrapolatedDipole->getDevicePointer(),
                &extrapolatedDipolePolar->getDevicePointer(), &inducedDipoleFieldGradient->getDevicePointer(), &inducedDipoleFieldGradientPolar->getDevicePointer(),
                &inducedField->getDevicePointer(), &inducedFieldPolar->getDevicePointer(), &extrapolatedDipoleFieldGradient->getDevicePointer(), &extrapolatedDipoleFieldGradientPolar->getDevicePointer(),
                &gkKernel->getInducedDipoles()->getDevicePointer(), &gkKernel->getInducedDipolesPolar()->getDevicePointer(), &extrapolatedDipoleGk->getDevicePointer(),
                &extrapolatedDipoleGkPolar->getDevicePointer(), &inducedDipoleFieldGradientGk->getDevicePointer(), &inducedDipoleFieldGradientGkPolar->getDevicePointer(),
                &gkKernel->getInducedField()->getDevicePointer(), &gkKernel->getInducedFieldPolar()->getDevicePointer(),
                &extrapolatedDipoleFieldGradientGk->getDevicePointer(), &extrapolatedDipoleFieldGradientGkPolar->getDevicePointer(),
                &polarizability->getDevicePointer()};
            cu.executeKernel(iterateExtrapolatedKernel, iterateArgs, extrapolatedDipole->getSize());
        }
    }
    
    // Take a linear combination of the µ_(n) components to form the total dipole

    if (gkKernel == NULL) {
        void* computeArgs[] = {&inducedDipole->getDevicePointer(), &inducedDipolePolar->getDevicePointer(), &extrapolatedDipole->getDevicePointer(),
                &extrapolatedDipolePolar->getDevicePointer()};
        cu.executeKernel(computeExtrapolatedKernel, computeArgs, extrapolatedDipole->getSize());
    }
    else {
        void* computeArgs[] = {&inducedDipole->getDevicePointer(), &inducedDipolePolar->getDevicePointer(), &extrapolatedDipole->getDevicePointer(),
                &extrapolatedDipolePolar->getDevicePointer(), &gkKernel->getInducedDipoles()->getDevicePointer(), &gkKernel->getInducedDipolesPolar()->getDevicePointer(),
                &extrapolatedDipoleGk->getDevicePointer(), &extrapolatedDipoleGkPolar->getDevicePointer()};
        cu.executeKernel(computeExtrapolatedKernel, computeArgs, extrapolatedDipole->getSize());
    }
    computeInducedField(recipBoxVectorPointer);
}

void CudaCalcMMFFMultipoleForceKernel::ensureMultipolesValid(ContextImpl& context) {
    if (multipolesAreValid) {
        int numParticles = cu.getNumAtoms();
        if (cu.getUseDoublePrecision()) {
            vector<double4> pos1, pos2;
            cu.getPosq().download(pos1);
            lastPositions->download(pos2);
            for (int i = 0; i < numParticles; i++)
                if (pos1[i].x != pos2[i].x || pos1[i].y != pos2[i].y || pos1[i].z != pos2[i].z) {
                    multipolesAreValid = false;
                    break;
                }
        }
        else {
            vector<float4> pos1, pos2;
            cu.getPosq().download(pos1);
            lastPositions->download(pos2);
            for (int i = 0; i < numParticles; i++)
                if (pos1[i].x != pos2[i].x || pos1[i].y != pos2[i].y || pos1[i].z != pos2[i].z) {
                    multipolesAreValid = false;
                    break;
                }
        }
    }
    if (!multipolesAreValid)
        context.calcForcesAndEnergy(false, false, -1);
}


void CudaCalcMMFFMultipoleForceKernel::getLabFramePermanentDipoles(ContextImpl& context, vector<Vec3>& dipoles) {
    ensureMultipolesValid(context);
    int numParticles = cu.getNumAtoms();
    dipoles.resize(numParticles);
    const vector<int>& order = cu.getAtomIndex();
    if (cu.getUseDoublePrecision()) {
        vector<double> labDipoleVec;
        labFrameDipoles->download(labDipoleVec);
        for (int i = 0; i < numParticles; i++)
            dipoles[order[i]] = Vec3(labDipoleVec[3*i], labDipoleVec[3*i+1], labDipoleVec[3*i+2]);
    }
    else {
        vector<float> labDipoleVec;
        labFrameDipoles->download(labDipoleVec);
        for (int i = 0; i < numParticles; i++)
            dipoles[order[i]] = Vec3(labDipoleVec[3*i], labDipoleVec[3*i+1], labDipoleVec[3*i+2]);
    }
}


void CudaCalcMMFFMultipoleForceKernel::getInducedDipoles(ContextImpl& context, vector<Vec3>& dipoles) {
    ensureMultipolesValid(context);
    int numParticles = cu.getNumAtoms();
    dipoles.resize(numParticles);
    const vector<int>& order = cu.getAtomIndex();
    if (cu.getUseDoublePrecision()) {
        vector<double> d;
        inducedDipole->download(d);
        for (int i = 0; i < numParticles; i++)
            dipoles[order[i]] = Vec3(d[3*i], d[3*i+1], d[3*i+2]);
    }
    else {
        vector<float> d;
        inducedDipole->download(d);
        for (int i = 0; i < numParticles; i++)
            dipoles[order[i]] = Vec3(d[3*i], d[3*i+1], d[3*i+2]);
    }
}


void CudaCalcMMFFMultipoleForceKernel::getTotalDipoles(ContextImpl& context, vector<Vec3>& dipoles) {
    ensureMultipolesValid(context);
    int numParticles = cu.getNumAtoms();
    dipoles.resize(numParticles);
    const vector<int>& order = cu.getAtomIndex();
    if (cu.getUseDoublePrecision()) {
        vector<double4> posqVec;
        vector<double> labDipoleVec;
        vector<double> inducedDipoleVec;
        double totalDipoleVecX;
        double totalDipoleVecY;
        double totalDipoleVecZ;
        inducedDipole->download(inducedDipoleVec);
        labFrameDipoles->download(labDipoleVec);
        cu.getPosq().download(posqVec);
        for (int i = 0; i < numParticles; i++) {
            totalDipoleVecX = labDipoleVec[3*i] + inducedDipoleVec[3*i];
            totalDipoleVecY = labDipoleVec[3*i+1] + inducedDipoleVec[3*i+1];
            totalDipoleVecZ = labDipoleVec[3*i+2] + inducedDipoleVec[3*i+2];
            dipoles[order[i]] = Vec3(totalDipoleVecX, totalDipoleVecY, totalDipoleVecZ);
        }
    }
    else {
        vector<float4> posqVec;
        vector<float> labDipoleVec;
        vector<float> inducedDipoleVec;
        float totalDipoleVecX;
        float totalDipoleVecY;
        float totalDipoleVecZ;
        inducedDipole->download(inducedDipoleVec);
        labFrameDipoles->download(labDipoleVec);
        cu.getPosq().download(posqVec);
        for (int i = 0; i < numParticles; i++) {
            totalDipoleVecX = labDipoleVec[3*i] + inducedDipoleVec[3*i];
            totalDipoleVecY = labDipoleVec[3*i+1] + inducedDipoleVec[3*i+1];
            totalDipoleVecZ = labDipoleVec[3*i+2] + inducedDipoleVec[3*i+2];
            dipoles[order[i]] = Vec3(totalDipoleVecX, totalDipoleVecY, totalDipoleVecZ);
        }
    }
}

void CudaCalcMMFFMultipoleForceKernel::getElectrostaticPotential(ContextImpl& context, const vector<Vec3>& inputGrid, vector<double>& outputElectrostaticPotential) {
    ensureMultipolesValid(context);
    int numPoints = inputGrid.size();
    int elementSize = (cu.getUseDoublePrecision() ? sizeof(double) : sizeof(float));
    CudaArray points(cu, numPoints, 4*elementSize, "points");
    CudaArray potential(cu, numPoints, elementSize, "potential");
    
    // Copy the grid points to the GPU.
    
    if (cu.getUseDoublePrecision()) {
        vector<double4> p(numPoints);
        for (int i = 0; i < numPoints; i++)
            p[i] = make_double4(inputGrid[i][0], inputGrid[i][1], inputGrid[i][2], 0);
        points.upload(p);
    }
    else {
        vector<float4> p(numPoints);
        for (int i = 0; i < numPoints; i++)
            p[i] = make_float4((float) inputGrid[i][0], (float) inputGrid[i][1], (float) inputGrid[i][2], 0);
        points.upload(p);
    }
    
    // Compute the potential.
    
    void* computePotentialArgs[] = {&cu.getPosq().getDevicePointer(), &labFrameDipoles->getDevicePointer(),
        &labFrameQuadrupoles->getDevicePointer(), &inducedDipole->getDevicePointer(), &points.getDevicePointer(),
        &potential.getDevicePointer(), &numPoints, cu.getPeriodicBoxSizePointer(), cu.getInvPeriodicBoxSizePointer(),
        cu.getPeriodicBoxVecXPointer(), cu.getPeriodicBoxVecYPointer(), cu.getPeriodicBoxVecZPointer()};
    int blockSize = 128;
    cu.executeKernel(computePotentialKernel, computePotentialArgs, numPoints, blockSize, blockSize*15*elementSize);
    outputElectrostaticPotential.resize(numPoints);
    if (cu.getUseDoublePrecision())
        potential.download(outputElectrostaticPotential);
    else {
        vector<float> p(numPoints);
        potential.download(p);
        for (int i = 0; i < numPoints; i++)
            outputElectrostaticPotential[i] = p[i];
    }
}

template <class T, class T4, class M4>
void CudaCalcMMFFMultipoleForceKernel::computeSystemMultipoleMoments(ContextImpl& context, vector<double>& outputMultipoleMoments) {
    // Compute the local coordinates relative to the center of mass.
    int numAtoms = cu.getNumAtoms();
    vector<T4> posq;
    vector<M4> velm;
    cu.getPosq().download(posq);
    cu.getVelm().download(velm);
    double totalMass = 0.0;
    Vec3 centerOfMass(0, 0, 0);
    for (int i = 0; i < numAtoms; i++) {
        double mass = (velm[i].w > 0 ? 1.0/velm[i].w : 0.0);
        totalMass += mass;
        centerOfMass[0] += mass*posq[i].x;
        centerOfMass[1] += mass*posq[i].y;
        centerOfMass[2] += mass*posq[i].z;
    }
    if (totalMass > 0.0) {
        centerOfMass[0] /= totalMass;
        centerOfMass[1] /= totalMass;
        centerOfMass[2] /= totalMass;
    }
    vector<double4> posqLocal(numAtoms);
    for (int i = 0; i < numAtoms; i++) {
        posqLocal[i].x = posq[i].x - centerOfMass[0];
        posqLocal[i].y = posq[i].y - centerOfMass[1];
        posqLocal[i].z = posq[i].z - centerOfMass[2];
        posqLocal[i].w = posq[i].w;
    }

    // Compute the multipole moments.
    
    double totalCharge = 0.0;
    double xdpl = 0.0;
    double ydpl = 0.0;
    double zdpl = 0.0;
    double xxqdp = 0.0;
    double xyqdp = 0.0;
    double xzqdp = 0.0;
    double yxqdp = 0.0;
    double yyqdp = 0.0;
    double yzqdp = 0.0;
    double zxqdp = 0.0;
    double zyqdp = 0.0;
    double zzqdp = 0.0;
    vector<T> labDipoleVec, inducedDipoleVec, quadrupoleVec;
    labFrameDipoles->download(labDipoleVec);
    inducedDipole->download(inducedDipoleVec);
    labFrameQuadrupoles->download(quadrupoleVec);
    for (int i = 0; i < numAtoms; i++) {
        totalCharge += posqLocal[i].w;
        double netDipoleX = (labDipoleVec[3*i]    + inducedDipoleVec[3*i]);
        double netDipoleY = (labDipoleVec[3*i+1]  + inducedDipoleVec[3*i+1]);
        double netDipoleZ = (labDipoleVec[3*i+2]  + inducedDipoleVec[3*i+2]);
        xdpl += posqLocal[i].x*posqLocal[i].w + netDipoleX;
        ydpl += posqLocal[i].y*posqLocal[i].w + netDipoleY;
        zdpl += posqLocal[i].z*posqLocal[i].w + netDipoleZ;
        xxqdp += posqLocal[i].x*posqLocal[i].x*posqLocal[i].w + 2*posqLocal[i].x*netDipoleX;
        xyqdp += posqLocal[i].x*posqLocal[i].y*posqLocal[i].w + posqLocal[i].x*netDipoleY + posqLocal[i].y*netDipoleX;
        xzqdp += posqLocal[i].x*posqLocal[i].z*posqLocal[i].w + posqLocal[i].x*netDipoleZ + posqLocal[i].z*netDipoleX;
        yxqdp += posqLocal[i].y*posqLocal[i].x*posqLocal[i].w + posqLocal[i].y*netDipoleX + posqLocal[i].x*netDipoleY;
        yyqdp += posqLocal[i].y*posqLocal[i].y*posqLocal[i].w + 2*posqLocal[i].y*netDipoleY;
        yzqdp += posqLocal[i].y*posqLocal[i].z*posqLocal[i].w + posqLocal[i].y*netDipoleZ + posqLocal[i].z*netDipoleY;
        zxqdp += posqLocal[i].z*posqLocal[i].x*posqLocal[i].w + posqLocal[i].z*netDipoleX + posqLocal[i].x*netDipoleZ;
        zyqdp += posqLocal[i].z*posqLocal[i].y*posqLocal[i].w + posqLocal[i].z*netDipoleY + posqLocal[i].y*netDipoleZ;
        zzqdp += posqLocal[i].z*posqLocal[i].z*posqLocal[i].w + 2*posqLocal[i].z*netDipoleZ;
    }

    // Convert the quadrupole from traced to traceless form.
 
    double qave = (xxqdp + yyqdp + zzqdp)/3;
    xxqdp = 1.5*(xxqdp-qave);
    xyqdp = 1.5*xyqdp;
    xzqdp = 1.5*xzqdp;
    yxqdp = 1.5*yxqdp;
    yyqdp = 1.5*(yyqdp-qave);
    yzqdp = 1.5*yzqdp;
    zxqdp = 1.5*zxqdp;
    zyqdp = 1.5*zyqdp;
    zzqdp = 1.5*(zzqdp-qave);

    // Add the traceless atomic quadrupoles to the total quadrupole moment.

    for (int i = 0; i < numAtoms; i++) {
        xxqdp = xxqdp + 3*quadrupoleVec[5*i];
        xyqdp = xyqdp + 3*quadrupoleVec[5*i+1];
        xzqdp = xzqdp + 3*quadrupoleVec[5*i+2];
        yxqdp = yxqdp + 3*quadrupoleVec[5*i+1];
        yyqdp = yyqdp + 3*quadrupoleVec[5*i+3];
        yzqdp = yzqdp + 3*quadrupoleVec[5*i+4];
        zxqdp = zxqdp + 3*quadrupoleVec[5*i+2];
        zyqdp = zyqdp + 3*quadrupoleVec[5*i+4];
        zzqdp = zzqdp + -3*(quadrupoleVec[5*i]+quadrupoleVec[5*i+3]);
    }
 
    double debye = 4.80321;
    outputMultipoleMoments.resize(13);
    outputMultipoleMoments[0] = totalCharge;
    outputMultipoleMoments[1] = 10.0*xdpl*debye;
    outputMultipoleMoments[2] = 10.0*ydpl*debye;
    outputMultipoleMoments[3] = 10.0*zdpl*debye;
    outputMultipoleMoments[4] = 100.0*xxqdp*debye;
    outputMultipoleMoments[5] = 100.0*xyqdp*debye;
    outputMultipoleMoments[6] = 100.0*xzqdp*debye;
    outputMultipoleMoments[7] = 100.0*yxqdp*debye;
    outputMultipoleMoments[8] = 100.0*yyqdp*debye;
    outputMultipoleMoments[9] = 100.0*yzqdp*debye;
    outputMultipoleMoments[10] = 100.0*zxqdp*debye;
    outputMultipoleMoments[11] = 100.0*zyqdp*debye;
    outputMultipoleMoments[12] = 100.0*zzqdp*debye;
}


void CudaCalcMMFFMultipoleForceKernel::getSystemMultipoleMoments(ContextImpl& context, vector<double>& outputMultipoleMoments) {
    ensureMultipolesValid(context);
    if (cu.getUseDoublePrecision())
        computeSystemMultipoleMoments<double, double4, double4>(context, outputMultipoleMoments);
    else if (cu.getUseMixedPrecision())
        computeSystemMultipoleMoments<float, float4, double4>(context, outputMultipoleMoments);
    else
        computeSystemMultipoleMoments<float, float4, float4>(context, outputMultipoleMoments);
}

void CudaCalcMMFFMultipoleForceKernel::copyParametersToContext(ContextImpl& context, const MMFFMultipoleForce& force) {
    // Make sure the new parameters are acceptable.
    
    cu.setAsCurrent();
    if (force.getNumMultipoles() != cu.getNumAtoms())
        throw OpenMMException("updateParametersInContext: The number of multipoles has changed");
    
    // Record the per-multipole parameters.
    
    cu.getPosq().download(cu.getPinnedBuffer());
    float4* posqf = (float4*) cu.getPinnedBuffer();
    double4* posqd = (double4*) cu.getPinnedBuffer();
    vector<float2> dampingAndTholeVec;
    vector<float> polarizabilityVec;
    vector<float> molecularDipolesVec;
    vector<float> molecularQuadrupolesVec;
    vector<int4> multipoleParticlesVec;
    for (int i = 0; i < force.getNumMultipoles(); i++) {
        double charge, thole, damping, polarity;
        int axisType, atomX, atomY, atomZ;
        vector<double> dipole, quadrupole;
        force.getMultipoleParameters(i, charge, dipole, quadrupole, axisType, atomZ, atomX, atomY, thole, damping, polarity);
        if (cu.getUseDoublePrecision())
            posqd[i].w = charge;
        else
            posqf[i].w = (float) charge;
        dampingAndTholeVec.push_back(make_float2((float) damping, (float) thole));
        polarizabilityVec.push_back((float) polarity);
        multipoleParticlesVec.push_back(make_int4(atomX, atomY, atomZ, axisType));
        for (int j = 0; j < 3; j++)
            molecularDipolesVec.push_back((float) dipole[j]);
        molecularQuadrupolesVec.push_back((float) quadrupole[0]);
        molecularQuadrupolesVec.push_back((float) quadrupole[1]);
        molecularQuadrupolesVec.push_back((float) quadrupole[2]);
        molecularQuadrupolesVec.push_back((float) quadrupole[4]);
        molecularQuadrupolesVec.push_back((float) quadrupole[5]);
    }
    if (!hasQuadrupoles) {
        for (auto q : molecularQuadrupolesVec)
            if (q != 0.0)
                throw OpenMMException("updateParametersInContext: Cannot set a non-zero quadrupole moment, because quadrupoles were excluded from the kernel");
    }
    for (int i = force.getNumMultipoles(); i < cu.getPaddedNumAtoms(); i++) {
        dampingAndTholeVec.push_back(make_float2(0, 0));
        polarizabilityVec.push_back(0);
        multipoleParticlesVec.push_back(make_int4(0, 0, 0, 0));
        for (int j = 0; j < 3; j++)
            molecularDipolesVec.push_back(0);
        for (int j = 0; j < 5; j++)
            molecularQuadrupolesVec.push_back(0);
    }
    dampingAndThole->upload(dampingAndTholeVec);
    polarizability->upload(polarizabilityVec);
    multipoleParticles->upload(multipoleParticlesVec);
    molecularDipoles->upload(molecularDipolesVec);
    molecularQuadrupoles->upload(molecularQuadrupolesVec);
    cu.getPosq().upload(cu.getPinnedBuffer());
    cu.invalidateMolecules();
    multipolesAreValid = false;
}

void CudaCalcMMFFMultipoleForceKernel::getPMEParameters(double& alpha, int& nx, int& ny, int& nz) const {
    if (!usePME)
        throw OpenMMException("getPMEParametersInContext: This Context is not using PME");
    alpha = this->alpha;
    nx = gridSizeX;
    ny = gridSizeY;
    nz = gridSizeZ;
}

/* -------------------------------------------------------------------------- *
 *                       MMFFGeneralizedKirkwood                            *
 * -------------------------------------------------------------------------- */

class CudaCalcMMFFGeneralizedKirkwoodForceKernel::ForceInfo : public CudaForceInfo {
public:
    ForceInfo(const MMFFGeneralizedKirkwoodForce& force) : force(force) {
    }
    bool areParticlesIdentical(int particle1, int particle2) {
        double charge1, charge2, radius1, radius2, scale1, scale2;
        force.getParticleParameters(particle1, charge1, radius1, scale1);
        force.getParticleParameters(particle2, charge2, radius2, scale2);
        return (charge1 == charge2 && radius1 == radius2 && scale1 == scale2);
    }
private:
    const MMFFGeneralizedKirkwoodForce& force;
};

CudaCalcMMFFGeneralizedKirkwoodForceKernel::CudaCalcMMFFGeneralizedKirkwoodForceKernel(std::string name, const Platform& platform, CudaContext& cu, const System& system) : 
           CalcMMFFGeneralizedKirkwoodForceKernel(name, platform), cu(cu), system(system), hasInitializedKernels(false), params(NULL), bornRadii(NULL), field(NULL),
           inducedField(NULL), inducedFieldPolar(NULL), inducedDipoleS(NULL), inducedDipolePolarS(NULL), bornSum(NULL), bornForce(NULL) {
}

CudaCalcMMFFGeneralizedKirkwoodForceKernel::~CudaCalcMMFFGeneralizedKirkwoodForceKernel() {
    cu.setAsCurrent();
    if (params != NULL)
        delete params;
    if (bornRadii != NULL)
        delete bornRadii;
    if (field != NULL)
        delete field;
    if (inducedField != NULL)
        delete inducedField;
    if (inducedFieldPolar != NULL)
        delete inducedFieldPolar;
    if (inducedDipoleS != NULL)
        delete inducedDipoleS;
    if (inducedDipolePolarS != NULL)
        delete inducedDipolePolarS;
    if (bornSum != NULL)
        delete bornSum;
    if (bornForce != NULL)
        delete bornForce;
}

void CudaCalcMMFFGeneralizedKirkwoodForceKernel::initialize(const System& system, const MMFFGeneralizedKirkwoodForce& force) {
    cu.setAsCurrent();
    if (cu.getPlatformData().contexts.size() > 1)
        throw OpenMMException("MMFFGeneralizedKirkwoodForce does not support using multiple CUDA devices");
    const MMFFMultipoleForce* multipoles = NULL;
    for (int i = 0; i < system.getNumForces() && multipoles == NULL; i++)
        multipoles = dynamic_cast<const MMFFMultipoleForce*>(&system.getForce(i));
    if (multipoles == NULL)
        throw OpenMMException("MMFFGeneralizedKirkwoodForce requires the System to also contain an MMFFMultipoleForce");
    CudaNonbondedUtilities& nb = cu.getNonbondedUtilities();
    int paddedNumAtoms = cu.getPaddedNumAtoms();
    int elementSize = (cu.getUseDoublePrecision() ? sizeof(double) : sizeof(float));
    params = CudaArray::create<float2>(cu, paddedNumAtoms, "mmffGkParams");
    bornRadii = new CudaArray(cu, paddedNumAtoms, elementSize, "bornRadii");
    field = new CudaArray(cu, 3*paddedNumAtoms, sizeof(long long), "gkField");
    bornSum = CudaArray::create<long long>(cu, paddedNumAtoms, "bornSum");
    bornForce = CudaArray::create<long long>(cu, paddedNumAtoms, "bornForce");
    inducedDipoleS = new CudaArray(cu, 3*paddedNumAtoms, elementSize, "inducedDipoleS");
    inducedDipolePolarS = new CudaArray(cu, 3*paddedNumAtoms, elementSize, "inducedDipolePolarS");
    polarizationType = multipoles->getPolarizationType();
    if (polarizationType != MMFFMultipoleForce::Direct) {
        inducedField = new CudaArray(cu, 3*paddedNumAtoms, sizeof(long long), "gkInducedField");
        inducedFieldPolar = new CudaArray(cu, 3*paddedNumAtoms, sizeof(long long), "gkInducedFieldPolar");
    }
    cu.addAutoclearBuffer(*field);
    cu.addAutoclearBuffer(*bornSum);
    cu.addAutoclearBuffer(*bornForce);
    vector<float2> paramsVector(paddedNumAtoms);
    for (int i = 0; i < force.getNumParticles(); i++) {
        double charge, radius, scalingFactor;
        force.getParticleParameters(i, charge, radius, scalingFactor);
        paramsVector[i] = make_float2((float) radius, (float) (scalingFactor*radius));
        
        // Make sure the charge matches the one specified by the MMFFMultipoleForce.
        
        double charge2, thole, damping, polarity;
        int axisType, atomX, atomY, atomZ;
        vector<double> dipole, quadrupole;
        multipoles->getMultipoleParameters(i, charge2, dipole, quadrupole, axisType, atomZ, atomX, atomY, thole, damping, polarity);
        if (charge != charge2)
            throw OpenMMException("MMFFGeneralizedKirkwoodForce and MMFFMultipoleForce must specify the same charge for every atom");
    }
    params->upload(paramsVector);
    
    // Select the number of threads for each kernel.
    
    double computeBornSumThreadMemory = 4*elementSize+3*sizeof(float);
    double gkForceThreadMemory = 24*elementSize;
    double chainRuleThreadMemory = 10*elementSize;
    double ediffThreadMemory = 28*elementSize+2*sizeof(float)+3*sizeof(int)/(double) cu.TileSize;
    int maxThreads = cu.getNonbondedUtilities().getForceThreadBlockSize();
    computeBornSumThreads = min(maxThreads, cu.computeThreadBlockSize(computeBornSumThreadMemory));
    gkForceThreads = min(maxThreads, cu.computeThreadBlockSize(gkForceThreadMemory));
    chainRuleThreads = min(maxThreads, cu.computeThreadBlockSize(chainRuleThreadMemory));
    ediffThreads = min(maxThreads, cu.computeThreadBlockSize(ediffThreadMemory));
    
    // Set preprocessor macros we will use when we create the kernels.
    
    defines["NUM_ATOMS"] = cu.intToString(cu.getNumAtoms());
    defines["PADDED_NUM_ATOMS"] = cu.intToString(paddedNumAtoms);
    defines["BORN_SUM_THREAD_BLOCK_SIZE"] = cu.intToString(computeBornSumThreads);
    defines["GK_FORCE_THREAD_BLOCK_SIZE"] = cu.intToString(gkForceThreads);
    defines["CHAIN_RULE_THREAD_BLOCK_SIZE"] = cu.intToString(chainRuleThreads);
    defines["EDIFF_THREAD_BLOCK_SIZE"] = cu.intToString(ediffThreads);
    defines["NUM_BLOCKS"] = cu.intToString(cu.getNumAtomBlocks());
    defines["GK_C"] = cu.doubleToString(2.455);
    double solventDielectric = force.getSolventDielectric();
    defines["GK_FC"] = cu.doubleToString(1*(1-solventDielectric)/(0+1*solventDielectric));
    defines["GK_FD"] = cu.doubleToString(2*(1-solventDielectric)/(1+2*solventDielectric));
    defines["GK_FQ"] = cu.doubleToString(3*(1-solventDielectric)/(2+3*solventDielectric));
    defines["EPSILON_FACTOR"] = cu.doubleToString(138.9354558456);
    defines["M_PI"] = cu.doubleToString(M_PI);
    defines["ENERGY_SCALE_FACTOR"] = cu.doubleToString(138.9354558456/force.getSoluteDielectric());
    if (polarizationType == MMFFMultipoleForce::Direct)
        defines["DIRECT_POLARIZATION"] = "";
    else if (polarizationType == MMFFMultipoleForce::Mutual)
        defines["MUTUAL_POLARIZATION"] = "";
    else if (polarizationType == MMFFMultipoleForce::Extrapolated)
        defines["EXTRAPOLATED_POLARIZATION"] = "";
    includeSurfaceArea = force.getIncludeCavityTerm();
    if (includeSurfaceArea) {
        defines["SURFACE_AREA_FACTOR"] = cu.doubleToString(force.getSurfaceAreaFactor());
        defines["PROBE_RADIUS"] = cu.doubleToString(force.getProbeRadius());
        defines["DIELECTRIC_OFFSET"] = cu.doubleToString(0.009);
    }
    cu.addForce(new ForceInfo(force));
}

double CudaCalcMMFFGeneralizedKirkwoodForceKernel::execute(ContextImpl& context, bool includeForces, bool includeEnergy) {
    // Since GK is so tightly entwined with the electrostatics, this method does nothing, and the force calculation
    // is driven by MMFFMultipoleForce.
    return 0.0;
}

void CudaCalcMMFFGeneralizedKirkwoodForceKernel::computeBornRadii() {
    if (!hasInitializedKernels) {
        hasInitializedKernels = true;
        
        // Create the kernels.
        
        int numExclusionTiles = cu.getNonbondedUtilities().getExclusionTiles().getSize();
        defines["NUM_TILES_WITH_EXCLUSIONS"] = cu.intToString(numExclusionTiles);
        int numContexts = cu.getPlatformData().contexts.size();
        int startExclusionIndex = cu.getContextIndex()*numExclusionTiles/numContexts;
        int endExclusionIndex = (cu.getContextIndex()+1)*numExclusionTiles/numContexts;
        defines["FIRST_EXCLUSION_TILE"] = cu.intToString(startExclusionIndex);
        defines["LAST_EXCLUSION_TILE"] = cu.intToString(endExclusionIndex);
        stringstream forceSource;
        forceSource << CudaKernelSources::vectorOps;
        forceSource << CudaMMFFKernelSources::mmffGk;
        forceSource << "#define F1\n";
        forceSource << CudaMMFFKernelSources::gkPairForce1;
        forceSource << CudaMMFFKernelSources::gkPairForce2;
        forceSource << CudaMMFFKernelSources::gkEDiffPairForce;
        forceSource << "#undef F1\n";
        forceSource << "#define F2\n";
        forceSource << CudaMMFFKernelSources::gkPairForce1;
        forceSource << CudaMMFFKernelSources::gkPairForce2;
        forceSource << "#undef F2\n";
        forceSource << "#define T1\n";
        forceSource << CudaMMFFKernelSources::gkPairForce1;
        forceSource << CudaMMFFKernelSources::gkPairForce2;
        forceSource << CudaMMFFKernelSources::gkEDiffPairForce;
        forceSource << "#undef T1\n";
        forceSource << "#define T2\n";
        forceSource << CudaMMFFKernelSources::gkPairForce1;
        forceSource << CudaMMFFKernelSources::gkPairForce2;
        forceSource << "#undef T2\n";
        forceSource << "#define T3\n";
        forceSource << CudaMMFFKernelSources::gkEDiffPairForce;
        forceSource << "#undef T3\n";
        forceSource << "#define B1\n";
        forceSource << "#define B2\n";
        forceSource << CudaMMFFKernelSources::gkPairForce1;
        forceSource << CudaMMFFKernelSources::gkPairForce2;
        CUmodule module = cu.createModule(forceSource.str(), defines);
        computeBornSumKernel = cu.getKernel(module, "computeBornSum");
        reduceBornSumKernel = cu.getKernel(module, "reduceBornSum");
        gkForceKernel = cu.getKernel(module, "computeGKForces");
        chainRuleKernel = cu.getKernel(module, "computeChainRuleForce");
        ediffKernel = cu.getKernel(module, "computeEDiffForce");
        if (includeSurfaceArea)
            surfaceAreaKernel = cu.getKernel(module, "computeSurfaceAreaForce");
    }
    CudaNonbondedUtilities& nb = cu.getNonbondedUtilities();
    int numTiles = nb.getNumTiles();
    int numForceThreadBlocks = nb.getNumForceThreadBlocks();
    void* computeBornSumArgs[] = {&bornSum->getDevicePointer(), &cu.getPosq().getDevicePointer(),
        &params->getDevicePointer(), &numTiles};
    cu.executeKernel(computeBornSumKernel, computeBornSumArgs, numForceThreadBlocks*computeBornSumThreads, computeBornSumThreads);
    void* reduceBornSumArgs[] = {&bornSum->getDevicePointer(), &params->getDevicePointer(), &bornRadii->getDevicePointer()};
    cu.executeKernel(reduceBornSumKernel, reduceBornSumArgs, cu.getNumAtoms());
}

void CudaCalcMMFFGeneralizedKirkwoodForceKernel::finishComputation(CudaArray& torque, CudaArray& labFrameDipoles, CudaArray& labFrameQuadrupoles,
            CudaArray& inducedDipole, CudaArray& inducedDipolePolar, CudaArray& dampingAndThole, CudaArray& covalentFlags, CudaArray& polarizationGroupFlags) {
    CudaNonbondedUtilities& nb = cu.getNonbondedUtilities();
    int startTileIndex = nb.getStartTileIndex();
    int numTileIndices = nb.getNumTiles();
    int numForceThreadBlocks = nb.getNumForceThreadBlocks();
    
    // Compute the GK force.
    
    void* gkForceArgs[] = {&cu.getForce().getDevicePointer(), &torque.getDevicePointer(), &cu.getEnergyBuffer().getDevicePointer(),
        &cu.getPosq().getDevicePointer(), &startTileIndex, &numTileIndices, &labFrameDipoles.getDevicePointer(),
        &labFrameQuadrupoles.getDevicePointer(), &inducedDipoleS->getDevicePointer(), &inducedDipolePolarS->getDevicePointer(),
        &bornRadii->getDevicePointer(), &bornForce->getDevicePointer()};
    cu.executeKernel(gkForceKernel, gkForceArgs, numForceThreadBlocks*gkForceThreads, gkForceThreads);

    // Compute the surface area force.
    
    if (includeSurfaceArea) {
        void* surfaceAreaArgs[] = {&bornForce->getDevicePointer(), &cu.getEnergyBuffer().getDevicePointer(), &params->getDevicePointer(), &bornRadii->getDevicePointer()};
        cu.executeKernel(surfaceAreaKernel, surfaceAreaArgs, cu.getNumAtoms());
    }
    
    // Apply the remaining terms.
    
    void* chainRuleArgs[] = {&cu.getForce().getDevicePointer(), &cu.getPosq().getDevicePointer(), &startTileIndex, &numTileIndices,
        &params->getDevicePointer(), &bornRadii->getDevicePointer(), &bornForce->getDevicePointer()};
    cu.executeKernel(chainRuleKernel, chainRuleArgs, numForceThreadBlocks*chainRuleThreads, chainRuleThreads);    
    void* ediffArgs[] = {&cu.getForce().getDevicePointer(), &torque.getDevicePointer(), &cu.getEnergyBuffer().getDevicePointer(),
        &cu.getPosq().getDevicePointer(), &covalentFlags.getDevicePointer(), &polarizationGroupFlags.getDevicePointer(),
        &nb.getExclusionTiles().getDevicePointer(), &startTileIndex, &numTileIndices,
        &labFrameDipoles.getDevicePointer(), &labFrameQuadrupoles.getDevicePointer(), &inducedDipole.getDevicePointer(),
        &inducedDipolePolar.getDevicePointer(), &inducedDipoleS->getDevicePointer(), &inducedDipolePolarS->getDevicePointer(),
        &dampingAndThole.getDevicePointer()};
    cu.executeKernel(ediffKernel, ediffArgs, numForceThreadBlocks*ediffThreads, ediffThreads);
}

void CudaCalcMMFFGeneralizedKirkwoodForceKernel::copyParametersToContext(ContextImpl& context, const MMFFGeneralizedKirkwoodForce& force) {
    // Make sure the new parameters are acceptable.
    
    cu.setAsCurrent();
    if (force.getNumParticles() != cu.getNumAtoms())
        throw OpenMMException("updateParametersInContext: The number of particles has changed");
    
    // Record the per-particle parameters.
    
    vector<float2> paramsVector(cu.getPaddedNumAtoms());
    for (int i = 0; i < force.getNumParticles(); i++) {
        double charge, radius, scalingFactor;
        force.getParticleParameters(i, charge, radius, scalingFactor);
        paramsVector[i] = make_float2((float) radius, (float) (scalingFactor*radius));
    }
    params->upload(paramsVector);
    cu.invalidateMolecules();
}

/* -------------------------------------------------------------------------- *
 *                           MMFFVdw                                        *
 * -------------------------------------------------------------------------- */

class CudaCalcMMFFVdwForceKernel::ForceInfo : public CudaForceInfo {
public:
    ForceInfo(const MMFFVdwForce& force) : force(force) {
    }
    bool areParticlesIdentical(int particle1, int particle2) {
        double sigma1, sigma2, G_t_alpha1, G_t_alpha2, alpha_d_N1, alpha_d_N2;
        char vdwDA1, vdwDA2;
        force.getParticleParameters(particle1, sigma1, G_t_alpha1, alpha_d_N1, vdwDA1);
        force.getParticleParameters(particle2, sigma2, G_t_alpha2, alpha_d_N2, vdwDA2);
        return (sigma1 == sigma2 && G_t_alpha1 == G_t_alpha2 && alpha_d_N1 == alpha_d_N2 && vdwDA1 == vdwDA2);
    }
private:
    const MMFFVdwForce& force;
};

CudaCalcMMFFVdwForceKernel::CudaCalcMMFFVdwForceKernel(std::string name, const Platform& platform, CudaContext& cu, const System& system) :
        CalcMMFFVdwForceKernel(name, platform), cu(cu), system(system), hasInitializedNonbonded(false), params(NULL), nonbonded(NULL) {
}

CudaCalcMMFFVdwForceKernel::~CudaCalcMMFFVdwForceKernel() {
    cu.setAsCurrent();
    if (params != NULL)
        delete params;
    if (nonbonded != NULL)
        delete nonbonded;
}

void CudaCalcMMFFVdwForceKernel::initialize(const System& system, const MMFFVdwForce& force) {
    cu.setAsCurrent();
    params = CudaArray::create<float3>(cu, cu.getPaddedNumAtoms(), "params");
    
    // Record atom parameters.
    
    vector<float3> paramsVec(cu.getPaddedNumAtoms());
    vector<vector<int> > exclusions(cu.getNumAtoms());
    for (int i = 0; i < force.getNumParticles(); i++) {
        double sigma, G_t_alpha, alpha_d_N;
        char vdwDA;
        force.getParticleParameters(i, sigma, G_t_alpha, alpha_d_N, vdwDA);
        // if vdwDA == 'D', G_t_alpha is negative
        // if vdwDA == 'A', alpha_d_N is negative
        paramsVec[i] = make_float3((float) sigma, (float) (vdwDA == 'D' ? -G_t_alpha : G_t_alpha),
            (float) (vdwDA == 'A' ? -alpha_d_N : alpha_d_N));
        force.getParticleExclusions(i, exclusions[i]);
        exclusions[i].push_back(i);
    }
    params->upload(paramsVec);
    if (force.getUseDispersionCorrection())
        dispersionCoefficient = MMFFVdwForceImpl::calcDispersionCorrection(system, force);
    else
        dispersionCoefficient = 0.0;               
 
    // This force is applied based on modified atom positions, where hydrogens have been moved slightly
    // closer to their parent atoms.  We therefore create a separate CudaNonbondedUtilities just for
    // this force, so it will have its own neighbor list and interaction kernel.
    
    nonbonded = new CudaNonbondedUtilities(cu);
    nonbonded->addParameter(CudaNonbondedUtilities::ParameterInfo("params", "float", 3, sizeof(float3), params->getDevicePointer()));
    
    // Create the interaction kernel.
    
    map<string, string> replacements;
    double cutoff = force.getCutoffDistance();
    double taperCutoff = cutoff*0.9;
    replacements["CUTOFF_DISTANCE"] = cu.doubleToString(force.getCutoffDistance());
    replacements["TAPER_CUTOFF"] = cu.doubleToString(taperCutoff);
    replacements["TAPER_C3"] = cu.doubleToString(10/pow(taperCutoff-cutoff, 3.0));
    replacements["TAPER_C4"] = cu.doubleToString(15/pow(taperCutoff-cutoff, 4.0));
    replacements["TAPER_C5"] = cu.doubleToString(6/pow(taperCutoff-cutoff, 5.0));
    bool useCutoff = (force.getNonbondedMethod() != MMFFVdwForce::NoCutoff);
    nonbonded->addInteraction(useCutoff, useCutoff, true, force.getCutoffDistance(), exclusions,
        cu.replaceStrings(CudaMMFFKernelSources::mmffVdwForce, replacements), 0);
    
    cu.addForce(new ForceInfo(force));
}

double CudaCalcMMFFVdwForceKernel::execute(ContextImpl& context, bool includeForces, bool includeEnergy) {
    if (!hasInitializedNonbonded) {
        hasInitializedNonbonded = true;
        nonbonded->initialize(system);
    }
    nonbonded->prepareInteractions(1);
    nonbonded->computeInteractions(1, includeForces, includeEnergy);
    double4 box = cu.getPeriodicBoxSize();
    return dispersionCoefficient/(box.x*box.y*box.z);
}

void CudaCalcMMFFVdwForceKernel::copyParametersToContext(ContextImpl& context, const MMFFVdwForce& force) {
    // Make sure the new parameters are acceptable.
    
    cu.setAsCurrent();
    if (force.getNumParticles() != cu.getNumAtoms())
        throw OpenMMException("updateParametersInContext: The number of particles has changed");
    
    // Record the per-particle parameters.
    
    vector<float3> paramsVec(cu.getPaddedNumAtoms());
    for (int i = 0; i < force.getNumParticles(); i++) {
        double sigma, G_t_alpha, alpha_d_N;
        char vdwDA;
        force.getParticleParameters(i, sigma, G_t_alpha, alpha_d_N, vdwDA);
        // if vdwDA == 'D', G_t_alpha is negative
        // if vdwDA == 'A', alpha_d_N is negative
        paramsVec[i] = make_float3((float) sigma, (float) (vdwDA == 'D' ? -G_t_alpha : G_t_alpha),
            (float) (vdwDA == 'A' ? -alpha_d_N : alpha_d_N));
    }
    params->upload(paramsVec);
    if (force.getUseDispersionCorrection())
        dispersionCoefficient = MMFFVdwForceImpl::calcDispersionCorrection(system, force);
    else
        dispersionCoefficient = 0.0;               
    cu.invalidateMolecules();
}

/* -------------------------------------------------------------------------- *
 *                           MMFFWcaDispersion                              *
 * -------------------------------------------------------------------------- */

class CudaCalcMMFFWcaDispersionForceKernel::ForceInfo : public CudaForceInfo {
public:
    ForceInfo(const MMFFWcaDispersionForce& force) : force(force) {
    }
    bool areParticlesIdentical(int particle1, int particle2) {
        double radius1, radius2, epsilon1, epsilon2;
        force.getParticleParameters(particle1, radius1, epsilon1);
        force.getParticleParameters(particle2, radius2, epsilon2);
        return (radius1 == radius2 && epsilon1 == epsilon2);
    }
private:
    const MMFFWcaDispersionForce& force;
};

CudaCalcMMFFWcaDispersionForceKernel::CudaCalcMMFFWcaDispersionForceKernel(std::string name, const Platform& platform, CudaContext& cu, const System& system) : 
           CalcMMFFWcaDispersionForceKernel(name, platform), cu(cu), system(system), radiusEpsilon(NULL) {
}

CudaCalcMMFFWcaDispersionForceKernel::~CudaCalcMMFFWcaDispersionForceKernel() {
    cu.setAsCurrent();
    if (radiusEpsilon != NULL)
        delete radiusEpsilon;
}

void CudaCalcMMFFWcaDispersionForceKernel::initialize(const System& system, const MMFFWcaDispersionForce& force) {
    int numParticles = system.getNumParticles();
    int paddedNumAtoms = cu.getPaddedNumAtoms();
    
    // Record parameters.
    
    vector<float2> radiusEpsilonVec(paddedNumAtoms, make_float2(0, 0));
    for (int i = 0; i < numParticles; i++) {
        double radius, epsilon;
        force.getParticleParameters(i, radius, epsilon);
        radiusEpsilonVec[i] = make_float2((float) radius, (float) epsilon);
    }
    radiusEpsilon = CudaArray::create<float2>(cu, paddedNumAtoms, "radiusEpsilon");
    radiusEpsilon->upload(radiusEpsilonVec);
    
    // Create the kernel.
    
    map<string, string> defines;
    defines["NUM_ATOMS"] = cu.intToString(numParticles);
    defines["PADDED_NUM_ATOMS"] = cu.intToString(cu.getPaddedNumAtoms());
    defines["THREAD_BLOCK_SIZE"] = cu.intToString(cu.getNonbondedUtilities().getForceThreadBlockSize());
    defines["NUM_BLOCKS"] = cu.intToString(cu.getNumAtomBlocks());
    defines["EPSO"] = cu.doubleToString(force.getEpso());
    defines["EPSH"] = cu.doubleToString(force.getEpsh());
    defines["RMINO"] = cu.doubleToString(force.getRmino());
    defines["RMINH"] = cu.doubleToString(force.getRminh());
    defines["AWATER"] = cu.doubleToString(force.getAwater());
    defines["SHCTD"] = cu.doubleToString(force.getShctd());
    defines["M_PI"] = cu.doubleToString(M_PI);
    CUmodule module = cu.createModule(CudaKernelSources::vectorOps+CudaMMFFKernelSources::mmffWcaForce, defines);
    forceKernel = cu.getKernel(module, "computeWCAForce");
    totalMaximumDispersionEnergy = MMFFWcaDispersionForceImpl::getTotalMaximumDispersionEnergy(force);

    // Add an interaction to the default nonbonded kernel.  This doesn't actually do any calculations.  It's
    // just so that CudaNonbondedUtilities will keep track of the tiles.
    
    vector<vector<int> > exclusions;
    cu.getNonbondedUtilities().addInteraction(false, false, false, 1.0, exclusions, "", force.getForceGroup());
    cu.addForce(new ForceInfo(force));
}

double CudaCalcMMFFWcaDispersionForceKernel::execute(ContextImpl& context, bool includeForces, bool includeEnergy) {
    CudaNonbondedUtilities& nb = cu.getNonbondedUtilities();
    int startTileIndex = nb.getStartTileIndex();
    int numTileIndices = nb.getNumTiles();
    int numForceThreadBlocks = nb.getNumForceThreadBlocks();
    int forceThreadBlockSize = nb.getForceThreadBlockSize();
    void* forceArgs[] = {&cu.getForce().getDevicePointer(), &cu.getEnergyBuffer().getDevicePointer(),
        &cu.getPosq().getDevicePointer(), &startTileIndex, &numTileIndices, &radiusEpsilon->getDevicePointer()};
    cu.executeKernel(forceKernel, forceArgs, numForceThreadBlocks*forceThreadBlockSize, forceThreadBlockSize);
    return totalMaximumDispersionEnergy;
}

void CudaCalcMMFFWcaDispersionForceKernel::copyParametersToContext(ContextImpl& context, const MMFFWcaDispersionForce& force) {
    // Make sure the new parameters are acceptable.
    
    cu.setAsCurrent();
    if (force.getNumParticles() != cu.getNumAtoms())
        throw OpenMMException("updateParametersInContext: The number of particles has changed");
    
    // Record the per-particle parameters.
    
    vector<float2> radiusEpsilonVec(cu.getPaddedNumAtoms(), make_float2(0, 0));
    for (int i = 0; i < cu.getNumAtoms(); i++) {
        double radius, epsilon;
        force.getParticleParameters(i, radius, epsilon);
        radiusEpsilonVec[i] = make_float2((float) radius, (float) epsilon);
    }
    radiusEpsilon->upload(radiusEpsilonVec);
    totalMaximumDispersionEnergy = MMFFWcaDispersionForceImpl::getTotalMaximumDispersionEnergy(force);
    cu.invalidateMolecules();
}
