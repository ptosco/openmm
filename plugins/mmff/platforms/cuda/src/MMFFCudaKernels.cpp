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
#include "openmm/internal/MMFFVdwForceImpl.h"
#include "openmm/internal/MMFFNonbondedForceImpl.h"
#include "CudaBondedUtilities.h"
#include "CudaFFT3D.h"
#include "CudaForceInfo.h"
#include "CudaKernelSources.h"
#include "CudaMMFFKernelSources.h"
#include "CudaNonbondedUtilities.h"
#include "jama_lu.h"
#include "SimTKOpenMMRealType.h"

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

class CudaCalcMMFFNonbondedForceKernel::ForceInfo : public CudaForceInfo {
public:
    ForceInfo(const MMFFNonbondedForce& force) : force(force) {
    }
    bool areParticlesIdentical(int particle1, int particle2) {
        double charge1, charge2, sigma1, sigma2, G_t_alpha1, G_t_alpha2, alpha_d_N1, alpha_d_N2;
        char vdwDA1, vdwDA2;
        force.getParticleParameters(particle1, charge1, sigma1, G_t_alpha1, alpha_d_N1, vdwDA1);
        force.getParticleParameters(particle2, charge2, sigma2, G_t_alpha2, alpha_d_N2, vdwDA2);
        return (charge1 == charge2 && sigma1 == sigma2 && G_t_alpha1 == G_t_alpha2 && alpha_d_N1 == alpha_d_N2 && vdwDA1 == vdwDA2);
    }
    int getNumParticleGroups() {
        return force.getNumExceptions();
    }
    void getParticlesInGroup(int index, vector<int>& particles) {
        int particle1, particle2;
        double chargeProd, sigma, epsilon;
        force.getExceptionParameters(index, particle1, particle2, chargeProd, sigma, epsilon);
        particles.resize(2);
        particles[0] = particle1;
        particles[1] = particle2;
    }
    bool areGroupsIdentical(int group1, int group2) {
        int particle1, particle2;
        double chargeProd1, chargeProd2, sigma1, sigma2, epsilon1, epsilon2;
        force.getExceptionParameters(group1, particle1, particle2, chargeProd1, sigma1, epsilon1);
        force.getExceptionParameters(group2, particle1, particle2, chargeProd2, sigma2, epsilon2);
        return (chargeProd1 == chargeProd2 && sigma1 == sigma2 && epsilon1 == epsilon2);
    }
private:
    const MMFFNonbondedForce& force;
};

class CudaCalcMMFFNonbondedForceKernel::PmeIO : public CalcPmeReciprocalForceKernel::IO {
public:
    PmeIO(CudaContext& cu, CUfunction addForcesKernel) : cu(cu), addForcesKernel(addForcesKernel), forceTemp(NULL) {
        forceTemp = CudaArray::create<float4>(cu, cu.getNumAtoms(), "PmeForce");
    }
    ~PmeIO() {
        if (forceTemp != NULL)
            delete forceTemp;
    }
    float* getPosq() {
        cu.setAsCurrent();
        cu.getPosq().download(posq);
        return (float*) &posq[0];
    }
    void setForce(float* force) {
        forceTemp->upload(force);
        void* args[] = {&forceTemp->getDevicePointer(), &cu.getForce().getDevicePointer()};
        cu.executeKernel(addForcesKernel, args, cu.getNumAtoms());
    }
private:
    CudaContext& cu;
    vector<float4> posq;
    CudaArray* forceTemp;
    CUfunction addForcesKernel;
};

class CudaCalcMMFFNonbondedForceKernel::PmePreComputation : public CudaContext::ForcePreComputation {
public:
    PmePreComputation(CudaContext& cu, Kernel& pme, CalcPmeReciprocalForceKernel::IO& io) : cu(cu), pme(pme), io(io) {
    }
    void computeForceAndEnergy(bool includeForces, bool includeEnergy, int groups) {
        Vec3 boxVectors[3] = {Vec3(cu.getPeriodicBoxSize().x, 0, 0), Vec3(0, cu.getPeriodicBoxSize().y, 0), Vec3(0, 0, cu.getPeriodicBoxSize().z)};
        pme.getAs<CalcPmeReciprocalForceKernel>().beginComputation(io, boxVectors, includeEnergy);
    }
private:
    CudaContext& cu;
    Kernel pme;
    CalcPmeReciprocalForceKernel::IO& io;
};

class CudaCalcMMFFNonbondedForceKernel::PmePostComputation : public CudaContext::ForcePostComputation {
public:
    PmePostComputation(Kernel& pme, CalcPmeReciprocalForceKernel::IO& io) : pme(pme), io(io) {
    }
    double computeForceAndEnergy(bool includeForces, bool includeEnergy, int groups) {
        return pme.getAs<CalcPmeReciprocalForceKernel>().finishComputation(io);
    }
private:
    Kernel pme;
    CalcPmeReciprocalForceKernel::IO& io;
};

class CudaCalcMMFFNonbondedForceKernel::SyncStreamPreComputation : public CudaContext::ForcePreComputation {
public:
    SyncStreamPreComputation(CudaContext& cu, CUstream stream, CUevent event, int forceGroup) : cu(cu), stream(stream), event(event), forceGroup(forceGroup) {
    }
    void computeForceAndEnergy(bool includeForces, bool includeEnergy, int groups) {
        if ((groups&(1<<forceGroup)) != 0) {
            cuEventRecord(event, cu.getCurrentStream());
            cuStreamWaitEvent(stream, event, 0);
        }
    }
private:
    CudaContext& cu;
    CUstream stream;
    CUevent event;
    int forceGroup;
};

class CudaCalcMMFFNonbondedForceKernel::SyncStreamPostComputation : public CudaContext::ForcePostComputation {
public:
    SyncStreamPostComputation(CudaContext& cu, CUevent event, CUfunction addEnergyKernel, CudaArray& pmeEnergyBuffer, int forceGroup) : cu(cu), event(event),
            addEnergyKernel(addEnergyKernel), pmeEnergyBuffer(pmeEnergyBuffer), forceGroup(forceGroup) {
    }
    double computeForceAndEnergy(bool includeForces, bool includeEnergy, int groups) {
        if ((groups&(1<<forceGroup)) != 0) {
            cuStreamWaitEvent(cu.getCurrentStream(), event, 0);
            if (includeEnergy) {
                int bufferSize = pmeEnergyBuffer.getSize();
                void* args[] = {&pmeEnergyBuffer.getDevicePointer(), &cu.getEnergyBuffer().getDevicePointer(), &bufferSize};
                cu.executeKernel(addEnergyKernel, args, bufferSize);
            }
        }
        return 0.0;
    }
private:
    CudaContext& cu;
    CUevent event;
    CUfunction addEnergyKernel;
    CudaArray& pmeEnergyBuffer;
    int forceGroup;
};

CudaCalcMMFFNonbondedForceKernel::CudaCalcMMFFNonbondedForceKernel(std::string name, const Platform& platform, CudaContext& cu, const System& system) :
        CalcMMFFNonbondedForceKernel(name, platform),
        cu(cu), hasInitializedFFT(false), params(NULL), exceptionParams(NULL), cosSinSums(NULL), directPmeGrid(NULL), reciprocalPmeGrid(NULL),
        pmeBsplineModuliX(NULL), pmeBsplineModuliY(NULL), pmeBsplineModuliZ(NULL),
        pmeAtomRange(NULL), pmeAtomGridIndex(NULL), pmeEnergyBuffer(NULL), sort(NULL), fft(NULL), pmeio(NULL) {
}

CudaCalcMMFFNonbondedForceKernel::~CudaCalcMMFFNonbondedForceKernel() {
    cu.setAsCurrent();
    if (params != NULL)
        delete params;
    if (exceptionParams != NULL)
        delete exceptionParams;
    if (cosSinSums != NULL)
        delete cosSinSums;
    if (directPmeGrid != NULL)
        delete directPmeGrid;
    if (reciprocalPmeGrid != NULL)
        delete reciprocalPmeGrid;
    if (pmeBsplineModuliX != NULL)
        delete pmeBsplineModuliX;
    if (pmeBsplineModuliY != NULL)
        delete pmeBsplineModuliY;
    if (pmeBsplineModuliZ != NULL)
        delete pmeBsplineModuliZ;
    if (pmeAtomRange != NULL)
        delete pmeAtomRange;
    if (pmeAtomGridIndex != NULL)
        delete pmeAtomGridIndex;
    if (pmeEnergyBuffer != NULL)
        delete pmeEnergyBuffer;
    if (sort != NULL)
        delete sort;
    if (fft != NULL)
        delete fft;
    if (pmeio != NULL)
        delete pmeio;
    if (hasInitializedFFT) {
        if (useCudaFFT) {
            cufftDestroy(fftForward);
            cufftDestroy(fftBackward);
        }
        if (usePmeStream) {
            cuStreamDestroy(pmeStream);
            cuEventDestroy(pmeSyncEvent);
        }
    }
}

void CudaCalcMMFFNonbondedForceKernel::initialize(const System& system, const MMFFNonbondedForce& force) {
    cu.setAsCurrent();

    // Identify which exceptions are 1-4 interactions.

    vector<pair<int, int> > exclusions;
    vector<int> exceptions;
    for (int i = 0; i < force.getNumExceptions(); i++) {
        int particle1, particle2;
        double chargeProd, sigma, epsilon;
        force.getExceptionParameters(i, particle1, particle2, chargeProd, sigma, epsilon);
        exclusions.push_back(pair<int, int>(particle1, particle2));
        if (chargeProd != 0.0 || epsilon != 0.0)
            exceptions.push_back(i);
    }

    // Initialize nonbonded interactions.

    int numParticles = force.getNumParticles();
    params = CudaArray::create<float3>(cu, cu.getPaddedNumAtoms(), "params");
    CudaArray& posq = cu.getPosq();
    vector<double4> temp(posq.getSize());
    float4* posqf = (float4*) &temp[0];
    double4* posqd = (double4*) &temp[0];
    vector<float3> paramsVector(cu.getPaddedNumAtoms());
    vector<vector<int> > exclusionList(numParticles);
    double sumSquaredCharges = 0.0;
    hasCoulomb = false;
    hasVdw = false;
    for (int i = 0; i < numParticles; i++) {
        double charge, sigma, G_t_alpha, alpha_d_N;
        char vdwDA;
        force.getParticleParameters(i, charge, sigma, G_t_alpha, alpha_d_N, vdwDA);
        if (cu.getUseDoublePrecision())
            posqd[i] = make_double4(0, 0, 0, charge);
        else
            posqf[i] = make_float4(0, 0, 0, (float) charge);
        paramsVector[i] = make_float3((float) sigma, (float) (vdwDA == 'D' ? -G_t_alpha : G_t_alpha),
            (float) (vdwDA == 'A' ? -alpha_d_N : alpha_d_N));
        exclusionList[i].push_back(i);
        sumSquaredCharges += charge*charge;
        if (charge != 0.0)
            hasCoulomb = true;
        if (G_t_alpha != 0.0)
            hasVdw = true;
    }
    for (auto exclusion : exclusions) {
        exclusionList[exclusion.first].push_back(exclusion.second);
        exclusionList[exclusion.second].push_back(exclusion.first);
    }
    posq.upload(&temp[0]);
    params->upload(paramsVector);
    nonbondedMethod = CalcMMFFNonbondedForceKernel::NonbondedMethod(force.getNonbondedMethod());
    bool useCutoff = (nonbondedMethod != NoCutoff);
    bool usePeriodic = (nonbondedMethod != NoCutoff && nonbondedMethod != CutoffNonPeriodic);
    map<string, string> defines;
    defines["HAS_COULOMB"] = (hasCoulomb ? "1" : "0");
    defines["HAS_VDW"] = (hasVdw ? "1" : "0");
    defines["USE_VDW_SWITCH"] = (useCutoff ? "1" : "0");
    if (useCutoff) {
        // Compute the reaction field constants.

        double reactionFieldK = pow(force.getCutoffDistance(), -3.0)*(force.getReactionFieldDielectric()-1.0)/(2.0*force.getReactionFieldDielectric()+1.0);
        double reactionFieldC = (1.0 / force.getCutoffDistance())*(3.0*force.getReactionFieldDielectric())/(2.0*force.getReactionFieldDielectric()+1.0);
        defines["REACTION_FIELD_K"] = cu.doubleToString(reactionFieldK);
        defines["REACTION_FIELD_C"] = cu.doubleToString(reactionFieldC);
        
        // Compute the switching coefficients.
        
        double switchingDistance = force.getCutoffDistance()*0.9;
        defines["VDW_SWITCH_CUTOFF"] = cu.doubleToString(switchingDistance);
        defines["VDW_SWITCH_C3"] = cu.doubleToString(10/pow(switchingDistance-force.getCutoffDistance(), 3.0));
        defines["VDW_SWITCH_C4"] = cu.doubleToString(15/pow(switchingDistance-force.getCutoffDistance(), 4.0));
        defines["VDW_SWITCH_C5"] = cu.doubleToString(6/pow(switchingDistance-force.getCutoffDistance(), 5.0));
    }
    if (force.getUseDispersionCorrection() && cu.getContextIndex() == 0)
        dispersionCoefficient = MMFFNonbondedForceImpl::calcDispersionCorrection(system, force);
    else
        dispersionCoefficient = 0.0;
    alpha = 0;
    ewaldSelfEnergy = 0.0;
    if (nonbondedMethod == Ewald) {
        // Compute the Ewald parameters.

        int kmaxx, kmaxy, kmaxz;
        MMFFNonbondedForceImpl::calcEwaldParameters(system, force, alpha, kmaxx, kmaxy, kmaxz);
        defines["EWALD_ALPHA"] = cu.doubleToString(alpha);
        defines["TWO_OVER_SQRT_PI"] = cu.doubleToString(2.0/sqrt(M_PI));
        defines["USE_EWALD"] = "1";
        if (cu.getContextIndex() == 0) {
            ewaldSelfEnergy = -ONE_4PI_EPS0*alpha*sumSquaredCharges/sqrt(M_PI);

            // Create the reciprocal space kernels.

            map<string, string> replacements;
            replacements["NUM_ATOMS"] = cu.intToString(numParticles);
            replacements["PADDED_NUM_ATOMS"] = cu.intToString(cu.getPaddedNumAtoms());
            replacements["KMAX_X"] = cu.intToString(kmaxx);
            replacements["KMAX_Y"] = cu.intToString(kmaxy);
            replacements["KMAX_Z"] = cu.intToString(kmaxz);
            replacements["EXP_COEFFICIENT"] = cu.doubleToString(-1.0/(4.0*alpha*alpha));
            replacements["ONE_4PI_EPS0"] = cu.doubleToString(ONE_4PI_EPS0);
            replacements["M_PI"] = cu.doubleToString(M_PI);
            CUmodule module = cu.createModule(CudaKernelSources::vectorOps+CudaKernelSources::ewald, replacements);
            ewaldSumsKernel = cu.getKernel(module, "calculateEwaldCosSinSums");
            ewaldForcesKernel = cu.getKernel(module, "calculateEwaldForces");
            int elementSize = (cu.getUseDoublePrecision() ? sizeof(double2) : sizeof(float2));
            cosSinSums = new CudaArray(cu, (2*kmaxx-1)*(2*kmaxy-1)*(2*kmaxz-1), elementSize, "cosSinSums");
        }
    }
    else if (nonbondedMethod == PME) {
        // Compute the PME parameters.

        MMFFNonbondedForceImpl::calcPMEParameters(system, force, alpha, gridSizeX, gridSizeY, gridSizeZ, false);
        gridSizeX = CudaFFT3D::findLegalDimension(gridSizeX);
        gridSizeY = CudaFFT3D::findLegalDimension(gridSizeY);
        gridSizeZ = CudaFFT3D::findLegalDimension(gridSizeZ);

        defines["EWALD_ALPHA"] = cu.doubleToString(alpha);
        defines["TWO_OVER_SQRT_PI"] = cu.doubleToString(2.0/sqrt(M_PI));
        defines["USE_EWALD"] = "1";
        if (cu.getContextIndex() == 0) {
            ewaldSelfEnergy = -ONE_4PI_EPS0*alpha*sumSquaredCharges/sqrt(M_PI);

            char deviceName[100];
            cuDeviceGetName(deviceName, 100, cu.getDevice());
            usePmeStream = (!cu.getPlatformData().disablePmeStream && string(deviceName) != "GeForce GTX 980"); // Using a separate stream is slower on GTX 980
            map<string, string> pmeDefines;
            pmeDefines["PME_ORDER"] = cu.intToString(PmeOrder);
            pmeDefines["NUM_ATOMS"] = cu.intToString(numParticles);
            pmeDefines["PADDED_NUM_ATOMS"] = cu.intToString(cu.getPaddedNumAtoms());
            pmeDefines["RECIP_EXP_FACTOR"] = cu.doubleToString(M_PI*M_PI/(alpha*alpha));
            pmeDefines["GRID_SIZE_X"] = cu.intToString(gridSizeX);
            pmeDefines["GRID_SIZE_Y"] = cu.intToString(gridSizeY);
            pmeDefines["GRID_SIZE_Z"] = cu.intToString(gridSizeZ);
            pmeDefines["EPSILON_FACTOR"] = cu.doubleToString(sqrt(ONE_4PI_EPS0));
            pmeDefines["M_PI"] = cu.doubleToString(M_PI);
            if (cu.getUseDoublePrecision())
                pmeDefines["USE_DOUBLE_PRECISION"] = "1";
            if (usePmeStream)
                pmeDefines["USE_PME_STREAM"] = "1";
            if (cu.getPlatformData().deterministicForces)
                pmeDefines["USE_DETERMINISTIC_FORCES"] = "1";
            CUmodule module = cu.createModule(CudaKernelSources::vectorOps+CudaKernelSources::pme, pmeDefines);
            if (cu.getPlatformData().useCpuPme) {
                // Create the CPU PME kernel.

                try {
                    cpuPme = getPlatform().createKernel(CalcPmeReciprocalForceKernel::Name(), *cu.getPlatformData().context);
                    cpuPme.getAs<CalcPmeReciprocalForceKernel>().initialize(gridSizeX, gridSizeY, gridSizeZ, numParticles, alpha, cu.getPlatformData().deterministicForces);
                    CUfunction addForcesKernel = cu.getKernel(module, "addForces");
                    pmeio = new PmeIO(cu, addForcesKernel);
                    cu.addPreComputation(new PmePreComputation(cu, cpuPme, *pmeio));
                    cu.addPostComputation(new PmePostComputation(cpuPme, *pmeio));
                }
                catch (OpenMMException& ex) {
                    // The CPU PME plugin isn't available.
                }
            }
            if (pmeio == NULL) {
                pmeGridIndexKernel = cu.getKernel(module, "findAtomGridIndex");
                pmeSpreadChargeKernel = cu.getKernel(module, "gridSpreadCharge");
                pmeConvolutionKernel = cu.getKernel(module, "reciprocalConvolution");
                pmeInterpolateForceKernel = cu.getKernel(module, "gridInterpolateForce");
                pmeEvalEnergyKernel = cu.getKernel(module, "gridEvaluateEnergy");
                pmeFinishSpreadChargeKernel = cu.getKernel(module, "finishSpreadCharge");
                cuFuncSetCacheConfig(pmeSpreadChargeKernel, CU_FUNC_CACHE_PREFER_L1);
                cuFuncSetCacheConfig(pmeInterpolateForceKernel, CU_FUNC_CACHE_PREFER_L1);

                // Create required data structures.

                int elementSize = (cu.getUseDoublePrecision() ? sizeof(double) : sizeof(float));
                int gridElements = gridSizeX*gridSizeY*gridSizeZ;
                directPmeGrid = new CudaArray(cu, gridElements, cu.getComputeCapability() >= 2.0 ? 2*elementSize : 2*sizeof(long long), "originalPmeGrid");
                reciprocalPmeGrid = new CudaArray(cu, gridElements, 2*elementSize, "reciprocalPmeGrid");
                cu.addAutoclearBuffer(*directPmeGrid);
                pmeBsplineModuliX = new CudaArray(cu, gridSizeX, elementSize, "pmeBsplineModuliX");
                pmeBsplineModuliY = new CudaArray(cu, gridSizeY, elementSize, "pmeBsplineModuliY");
                pmeBsplineModuliZ = new CudaArray(cu, gridSizeZ, elementSize, "pmeBsplineModuliZ");
                pmeAtomRange = CudaArray::create<int>(cu, gridSizeX*gridSizeY*gridSizeZ+1, "pmeAtomRange");
                pmeAtomGridIndex = CudaArray::create<int2>(cu, numParticles, "pmeAtomGridIndex");
                int energyElementSize = (cu.getUseDoublePrecision() || cu.getUseMixedPrecision() ? sizeof(double) : sizeof(float));
                pmeEnergyBuffer = new CudaArray(cu, cu.getNumThreadBlocks()*CudaContext::ThreadBlockSize, energyElementSize, "pmeEnergyBuffer");
                cu.clearBuffer(*pmeEnergyBuffer);
                sort = new CudaSort(cu, new SortTrait(), cu.getNumAtoms());
                int cufftVersion;
                cufftGetVersion(&cufftVersion);
                useCudaFFT = (cufftVersion >= 7050); // There was a critical bug in version 7.0
                if (useCudaFFT) {
                    cufftResult result = cufftPlan3d(&fftForward, gridSizeX, gridSizeY, gridSizeZ, cu.getUseDoublePrecision() ? CUFFT_D2Z : CUFFT_R2C);
                    if (result != CUFFT_SUCCESS)
                        throw OpenMMException("Error initializing FFT: "+cu.intToString(result));
                    result = cufftPlan3d(&fftBackward, gridSizeX, gridSizeY, gridSizeZ, cu.getUseDoublePrecision() ? CUFFT_Z2D : CUFFT_C2R);
                    if (result != CUFFT_SUCCESS)
                        throw OpenMMException("Error initializing FFT: "+cu.intToString(result));
                }
                else {
                    fft = new CudaFFT3D(cu, gridSizeX, gridSizeY, gridSizeZ, true);
                }

                // Prepare for doing PME on its own stream.

                if (usePmeStream) {
                    cuStreamCreate(&pmeStream, CU_STREAM_NON_BLOCKING);
                    if (useCudaFFT) {
                        cufftSetStream(fftForward, pmeStream);
                        cufftSetStream(fftBackward, pmeStream);
                    }
                    CHECK_RESULT(cuEventCreate(&pmeSyncEvent, CU_EVENT_DISABLE_TIMING), "Error creating event for MMFFNonbondedForce");
                    int recipForceGroup = force.getReciprocalSpaceForceGroup();
                    if (recipForceGroup < 0)
                        recipForceGroup = force.getForceGroup();
                    cu.addPreComputation(new SyncStreamPreComputation(cu, pmeStream, pmeSyncEvent, recipForceGroup));
                    cu.addPostComputation(new SyncStreamPostComputation(cu, pmeSyncEvent, cu.getKernel(module, "addEnergy"), *pmeEnergyBuffer, recipForceGroup));
                }
                hasInitializedFFT = true;

                // Initialize the b-spline moduli.

                for (int grid = 0; grid < 2; grid++) {
                    int xsize, ysize, zsize;
                    CudaArray *xmoduli, *ymoduli, *zmoduli;
                    if (grid == 0) {
                        xsize = gridSizeX;
                        ysize = gridSizeY;
                        zsize = gridSizeZ;
                        xmoduli = pmeBsplineModuliX;
                        ymoduli = pmeBsplineModuliY;
                        zmoduli = pmeBsplineModuliZ;
                    }
                    else {
                        continue;
                    }
                    int maxSize = max(max(xsize, ysize), zsize);
                    vector<double> data(PmeOrder);
                    vector<double> ddata(PmeOrder);
                    vector<double> bsplines_data(maxSize);
                    data[PmeOrder-1] = 0.0;
                    data[1] = 0.0;
                    data[0] = 1.0;
                    for (int i = 3; i < PmeOrder; i++) {
                        double div = 1.0/(i-1.0);
                        data[i-1] = 0.0;
                        for (int j = 1; j < (i-1); j++)
                            data[i-j-1] = div*(j*data[i-j-2]+(i-j)*data[i-j-1]);
                        data[0] = div*data[0];
                    }

                    // Differentiate.

                    ddata[0] = -data[0];
                    for (int i = 1; i < PmeOrder; i++)
                        ddata[i] = data[i-1]-data[i];
                    double div = 1.0/(PmeOrder-1);
                    data[PmeOrder-1] = 0.0;
                    for (int i = 1; i < (PmeOrder-1); i++)
                        data[PmeOrder-i-1] = div*(i*data[PmeOrder-i-2]+(PmeOrder-i)*data[PmeOrder-i-1]);
                    data[0] = div*data[0];
                    for (int i = 0; i < maxSize; i++)
                        bsplines_data[i] = 0.0;
                    for (int i = 1; i <= PmeOrder; i++)
                        bsplines_data[i] = data[i-1];

                    // Evaluate the actual bspline moduli for X/Y/Z.

                    for(int dim = 0; dim < 3; dim++) {
                        int ndata = (dim == 0 ? xsize : dim == 1 ? ysize : zsize);
                        vector<double> moduli(ndata);
                        for (int i = 0; i < ndata; i++) {
                            double sc = 0.0;
                            double ss = 0.0;
                            for (int j = 0; j < ndata; j++) {
                                double arg = (2.0*M_PI*i*j)/ndata;
                                sc += bsplines_data[j]*cos(arg);
                                ss += bsplines_data[j]*sin(arg);
                            }
                            moduli[i] = sc*sc+ss*ss;
                        }
                        for (int i = 0; i < ndata; i++)
                            if (moduli[i] < 1.0e-7)
                                moduli[i] = (moduli[i-1]+moduli[i+1])*0.5;
                        if (cu.getUseDoublePrecision()) {
                            if (dim == 0)
                                xmoduli->upload(moduli);
                            else if (dim == 1)
                                ymoduli->upload(moduli);
                            else
                                zmoduli->upload(moduli);
                        }
                        else {
                            vector<float> modulif(ndata);
                            for (int i = 0; i < ndata; i++)
                                modulif[i] = (float) moduli[i];
                            if (dim == 0)
                                xmoduli->upload(modulif);
                            else if (dim == 1)
                                ymoduli->upload(modulif);
                            else
                                zmoduli->upload(modulif);
                        }
                    }
                }
            }
        }
    }
    // Add the interaction to the default nonbonded kernel.

    string source = cu.replaceStrings(CudaMMFFKernelSources::mmffNonbonded, defines);
    cu.getNonbondedUtilities().addInteraction(useCutoff, usePeriodic, true, force.getCutoffDistance(), exclusionList, source, force.getForceGroup(), true);
    if (hasVdw)
        cu.getNonbondedUtilities().addParameter(CudaNonbondedUtilities::ParameterInfo("params", "float", 3,
                                                sizeof(float3), params->getDevicePointer()));

    // Initialize the exceptions.

    int numContexts = cu.getPlatformData().contexts.size();
    int startIndex = cu.getContextIndex()*exceptions.size()/numContexts;
    int endIndex = (cu.getContextIndex()+1)*exceptions.size()/numContexts;
    int numExceptions = endIndex-startIndex;
    if (numExceptions > 0) {
        exceptionAtoms.resize(numExceptions);
        vector<vector<int> > atoms(numExceptions, vector<int>(2));
        exceptionParams = CudaArray::create<float4>(cu, numExceptions, "exceptionParams");
        vector<float3> exceptionParamsVector(numExceptions);
        for (int i = 0; i < numExceptions; i++) {
            double chargeProd, sigma, epsilon;
            force.getExceptionParameters(exceptions[startIndex+i], atoms[i][0], atoms[i][1], chargeProd, sigma, epsilon);
            exceptionParamsVector[i] = make_float3((float) (ONE_4PI_EPS0*chargeProd), (float) sigma, (float) epsilon);
            exceptionAtoms[i] = make_pair(atoms[i][0], atoms[i][1]);
        }
        exceptionParams->upload(exceptionParamsVector);
        map<string, string> replacements;
        replacements["PARAMS"] = cu.getBondedUtilities().addArgument(exceptionParams->getDevicePointer(), "float3");
        cu.getBondedUtilities().addInteraction(atoms, cu.replaceStrings(CudaMMFFKernelSources::mmffNonbondedExceptions, replacements), force.getForceGroup());
    }
    info = new ForceInfo(force);
    cu.addForce(info);
}

double CudaCalcMMFFNonbondedForceKernel::execute(ContextImpl& context, bool includeForces, bool includeEnergy, bool includeDirect, bool includeReciprocal) {
    if (cosSinSums != NULL && includeReciprocal) {
        void* sumsArgs[] = {&cu.getEnergyBuffer().getDevicePointer(), &cu.getPosq().getDevicePointer(), &cosSinSums->getDevicePointer(), cu.getPeriodicBoxSizePointer()};
        cu.executeKernel(ewaldSumsKernel, sumsArgs, cosSinSums->getSize());
        void* forcesArgs[] = {&cu.getForce().getDevicePointer(), &cu.getPosq().getDevicePointer(), &cosSinSums->getDevicePointer(), cu.getPeriodicBoxSizePointer()};
        cu.executeKernel(ewaldForcesKernel, forcesArgs, cu.getNumAtoms());
    }
    if (directPmeGrid != NULL && includeReciprocal) {
        if (usePmeStream)
            cu.setCurrentStream(pmeStream);

        // Invert the periodic box vectors.

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

        // Execute the reciprocal space kernels.

        void* gridIndexArgs[] = {&cu.getPosq().getDevicePointer(), &pmeAtomGridIndex->getDevicePointer(), cu.getPeriodicBoxSizePointer(),
                cu.getInvPeriodicBoxSizePointer(), cu.getPeriodicBoxVecXPointer(), cu.getPeriodicBoxVecYPointer(), cu.getPeriodicBoxVecZPointer(),
                recipBoxVectorPointer[0], recipBoxVectorPointer[1], recipBoxVectorPointer[2]};
        cu.executeKernel(pmeGridIndexKernel, gridIndexArgs, cu.getNumAtoms());

        sort->sort(*pmeAtomGridIndex);

        void* spreadArgs[] = {&cu.getPosq().getDevicePointer(), &directPmeGrid->getDevicePointer(), cu.getPeriodicBoxSizePointer(),
                cu.getInvPeriodicBoxSizePointer(), cu.getPeriodicBoxVecXPointer(), cu.getPeriodicBoxVecYPointer(), cu.getPeriodicBoxVecZPointer(),
                recipBoxVectorPointer[0], recipBoxVectorPointer[1], recipBoxVectorPointer[2], &pmeAtomGridIndex->getDevicePointer()};
        cu.executeKernel(pmeSpreadChargeKernel, spreadArgs, cu.getNumAtoms(), 128);

        if (cu.getUseDoublePrecision() || cu.getComputeCapability() < 2.0 || cu.getPlatformData().deterministicForces) {
            void* finishSpreadArgs[] = {&directPmeGrid->getDevicePointer()};
            cu.executeKernel(pmeFinishSpreadChargeKernel, finishSpreadArgs, gridSizeX*gridSizeY*gridSizeZ, 256);
        }

        if (useCudaFFT) {
            if (cu.getUseDoublePrecision())
                cufftExecD2Z(fftForward, (double*) directPmeGrid->getDevicePointer(), (double2*) reciprocalPmeGrid->getDevicePointer());
            else
                cufftExecR2C(fftForward, (float*) directPmeGrid->getDevicePointer(), (float2*) reciprocalPmeGrid->getDevicePointer());
        }
        else {
            fft->execFFT(*directPmeGrid, *reciprocalPmeGrid, true);
        }

        if (includeEnergy) {
            void* computeEnergyArgs[] = {&reciprocalPmeGrid->getDevicePointer(), usePmeStream ? &pmeEnergyBuffer->getDevicePointer() : &cu.getEnergyBuffer().getDevicePointer(),
                    &pmeBsplineModuliX->getDevicePointer(), &pmeBsplineModuliY->getDevicePointer(), &pmeBsplineModuliZ->getDevicePointer(),
                    cu.getPeriodicBoxSizePointer(), recipBoxVectorPointer[0], recipBoxVectorPointer[1], recipBoxVectorPointer[2]};
            cu.executeKernel(pmeEvalEnergyKernel, computeEnergyArgs, gridSizeX*gridSizeY*gridSizeZ);
        }

        void* convolutionArgs[] = {&reciprocalPmeGrid->getDevicePointer(), &cu.getEnergyBuffer().getDevicePointer(),
                &pmeBsplineModuliX->getDevicePointer(), &pmeBsplineModuliY->getDevicePointer(), &pmeBsplineModuliZ->getDevicePointer(),
                cu.getPeriodicBoxSizePointer(), recipBoxVectorPointer[0], recipBoxVectorPointer[1], recipBoxVectorPointer[2]};
        cu.executeKernel(pmeConvolutionKernel, convolutionArgs, gridSizeX*gridSizeY*gridSizeZ, 256);

        if (useCudaFFT) {
            if (cu.getUseDoublePrecision())
                cufftExecZ2D(fftBackward, (double2*) reciprocalPmeGrid->getDevicePointer(), (double*) directPmeGrid->getDevicePointer());
            else
                cufftExecC2R(fftBackward, (float2*) reciprocalPmeGrid->getDevicePointer(), (float*)  directPmeGrid->getDevicePointer());
        }
        else {
            fft->execFFT(*reciprocalPmeGrid, *directPmeGrid, false);
        }

        void* interpolateArgs[] = {&cu.getPosq().getDevicePointer(), &cu.getForce().getDevicePointer(), &directPmeGrid->getDevicePointer(), cu.getPeriodicBoxSizePointer(),
                cu.getInvPeriodicBoxSizePointer(), cu.getPeriodicBoxVecXPointer(), cu.getPeriodicBoxVecYPointer(), cu.getPeriodicBoxVecZPointer(),
                recipBoxVectorPointer[0], recipBoxVectorPointer[1], recipBoxVectorPointer[2], &pmeAtomGridIndex->getDevicePointer()};
        cu.executeKernel(pmeInterpolateForceKernel, interpolateArgs, cu.getNumAtoms(), 128);

        if (usePmeStream) {
            cuEventRecord(pmeSyncEvent, pmeStream);
            cu.restoreDefaultStream();
        }
    }

    double energy = (includeReciprocal ? ewaldSelfEnergy : 0.0);
    if (dispersionCoefficient != 0.0 && includeDirect) {
        double4 boxSize = cu.getPeriodicBoxSize();
        energy += dispersionCoefficient/(boxSize.x*boxSize.y*boxSize.z);
    }
    return energy;
}

void CudaCalcMMFFNonbondedForceKernel::copyParametersToContext(ContextImpl& context, const MMFFNonbondedForce& force) {
    // Make sure the new parameters are acceptable.
    
    cu.setAsCurrent();
    if (force.getNumParticles() != cu.getNumAtoms())
        throw OpenMMException("updateParametersInContext: The number of particles has changed");
    if (!hasCoulomb || !hasVdw) {
        for (int i = 0; i < force.getNumParticles(); i++) {
            double charge, sigma, G_t_alpha, alpha_d_N;
            char vdwDA;
            force.getParticleParameters(i, charge, sigma, G_t_alpha, alpha_d_N, vdwDA);
            if (!hasCoulomb && charge != 0.0)
                throw OpenMMException("updateParametersInContext: The nonbonded force kernel does not include Coulomb interactions, because all charges were originally 0");
            if (!hasVdw && G_t_alpha != 0.0)
                throw OpenMMException("updateParametersInContext: The nonbonded force kernel does not include van der Waals interactions, because all G_t_alphas were originally 0");
        }
    }
    vector<int> exceptions;
    for (int i = 0; i < force.getNumExceptions(); i++) {
        int particle1, particle2;
        double chargeProd, sigma, epsilon;
        force.getExceptionParameters(i, particle1, particle2, chargeProd, sigma, epsilon);
        if (exceptionAtoms.size() > exceptions.size() && make_pair(particle1, particle2) == exceptionAtoms[exceptions.size()])
            exceptions.push_back(i);
        else if (chargeProd != 0.0 || epsilon != 0.0)
            throw OpenMMException("updateParametersInContext: The set of non-excluded exceptions has changed");
    }
    int numContexts = cu.getPlatformData().contexts.size();
    int startIndex = cu.getContextIndex()*exceptions.size()/numContexts;
    int endIndex = (cu.getContextIndex()+1)*exceptions.size()/numContexts;
    int numExceptions = endIndex-startIndex;
    
    // Record the per-particle parameters.
    
    vector<double> chargeVector(cu.getNumAtoms());
    vector<float3> paramsVector(cu.getPaddedNumAtoms());
    double sumSquaredCharges = 0.0;
    const vector<int>& order = cu.getAtomIndex();
    for (int i = 0; i < force.getNumParticles(); i++) {
        double charge, sigma, G_t_alpha, alpha_d_N;
        char vdwDA;
        force.getParticleParameters(i, charge, sigma, G_t_alpha, alpha_d_N, vdwDA);
        chargeVector[i] = charge;
        paramsVector[i] = make_float3((float) sigma, (float) (vdwDA == 'D' ? -G_t_alpha : G_t_alpha),
            (float) (vdwDA == 'A' ? -alpha_d_N : alpha_d_N));
        sumSquaredCharges += charge*charge;
    }
    cu.setCharges(chargeVector);
    params->upload(paramsVector);
    
    // Record the exceptions.
    
    if (numExceptions > 0) {
        vector<vector<int> > atoms(numExceptions, vector<int>(2));
        vector<float3> exceptionParamsVector(numExceptions);
        for (int i = 0; i < numExceptions; i++) {
            double chargeProd, sigma, epsilon;
            force.getExceptionParameters(exceptions[startIndex+i], atoms[i][0], atoms[i][1], chargeProd, sigma, epsilon);
            exceptionParamsVector[i] = make_float3((float) (ONE_4PI_EPS0*chargeProd), (float) sigma, (float) (4.0*epsilon));
        }
        exceptionParams->upload(exceptionParamsVector);
    }
    
    // Compute other values.
    
    if (nonbondedMethod == Ewald || nonbondedMethod == PME)
        ewaldSelfEnergy = (cu.getContextIndex() == 0 ? -ONE_4PI_EPS0*alpha*sumSquaredCharges/sqrt(M_PI) : 0.0);
    if (force.getUseDispersionCorrection() && cu.getContextIndex() == 0 && (nonbondedMethod == CutoffPeriodic || nonbondedMethod == Ewald || nonbondedMethod == PME))
        dispersionCoefficient = MMFFNonbondedForceImpl::calcDispersionCorrection(context.getSystem(), force);
    cu.invalidateMolecules();
}

void CudaCalcMMFFNonbondedForceKernel::getPMEParameters(double& alpha, int& nx, int& ny, int& nz) const {
    if (nonbondedMethod != PME)
        throw OpenMMException("getPMEParametersInContext: This Context is not using PME");
    if (cu.getPlatformData().useCpuPme)
        cpuPme.getAs<CalcPmeReciprocalForceKernel>().getPMEParameters(alpha, nx, ny, nz);
    else {
        alpha = this->alpha;
        nx = gridSizeX;
        ny = gridSizeY;
        nz = gridSizeZ;
    }
}
