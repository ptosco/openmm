#ifndef MMFF_OPENMM_REFERENCE_KERNELS_H_
#define MMFF_OPENMM_REFERENCE_KERNELS_H_

/* -------------------------------------------------------------------------- *
 *                              OpenMMMMFF                                  *
 * -------------------------------------------------------------------------- *
 * This is part of the OpenMM molecular simulation toolkit originating from   *
 * Simbios, the NIH National Center for Physics-Based Simulation of           *
 * Biological Structures at Stanford, funded under the NIH Roadmap for        *
 * Medical Research, grant U54 GM072970. See https://simtk.org.               *
 *                                                                            *
 * Portions copyright (c) 2008-2015 Stanford University and the Authors.      *
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

#include "openmm/System.h"
#include "openmm/mmffKernels.h"
#include "ReferenceNeighborList.h"
#include "SimTKOpenMMRealType.h"

namespace OpenMM {

/**
 * This kernel is invoked by MMFFBondForce to calculate the forces acting on the system and the energy of the system.
 */
class ReferenceCalcMMFFBondForceKernel : public CalcMMFFBondForceKernel {
public:
    ReferenceCalcMMFFBondForceKernel(std::string name, 
                                               const Platform& platform,
                                               const System& system);
    ~ReferenceCalcMMFFBondForceKernel();
    /**
     * Initialize the kernel.
     * 
     * @param system     the System this kernel will be applied to
     * @param force      the MMFFBondForce this kernel will be used for
     */
    void initialize(const System& system, const MMFFBondForce& force);
    /**
     * Execute the kernel to calculate the forces and/or energy.
     *
     * @param context        the context in which to execute this kernel
     * @param includeForces  true if forces should be calculated
     * @param includeEnergy  true if the energy should be calculated
     * @return the potential energy due to the force
     */
    double execute(ContextImpl& context, bool includeForces, bool includeEnergy);
    /**
     * Copy changed parameters over to a context.
     *
     * @param context    the context to copy parameters to
     * @param force      the MMFFBondForce to copy the parameters from
     */
    void copyParametersToContext(ContextImpl& context, const MMFFBondForce& force);
private:
    int numBonds;
    std::vector<int>   particle1;
    std::vector<int>   particle2;
    std::vector<double> length;
    std::vector<double> kQuadratic;
    double globalBondCubic;
    double globalBondQuartic;
    const System& system;
    bool usePeriodic;
};

/**
 * This kernel is invoked by MMFFAngleForce to calculate the forces acting on the system and the energy of the system.
 */
class ReferenceCalcMMFFAngleForceKernel : public CalcMMFFAngleForceKernel {
public:
    ReferenceCalcMMFFAngleForceKernel(std::string name, const Platform& platform, const System& system);
    ~ReferenceCalcMMFFAngleForceKernel();
    /**
     * Initialize the kernel.
     * 
     * @param system     the System this kernel will be applied to
     * @param force      the MMFFAngleForce this kernel will be used for
     */
    void initialize(const System& system, const MMFFAngleForce& force);
    /**
     * Execute the kernel to calculate the forces and/or energy.
     *
     * @param context        the context in which to execute this kernel
     * @param includeForces  true if forces should be calculated
     * @param includeEnergy  true if the energy should be calculated
     * @return the potential energy due to the force
     */
    double execute(ContextImpl& context, bool includeForces, bool includeEnergy);
    /**
     * Copy changed parameters over to a context.
     *
     * @param context    the context to copy parameters to
     * @param force      the MMFFAngleForce to copy the parameters from
     */
    void copyParametersToContext(ContextImpl& context, const MMFFAngleForce& force);
private:
    int numAngles;
    std::vector<int>   particle1;
    std::vector<int>   particle2;
    std::vector<int>   particle3;
    std::vector<double> angle;
    std::vector<double> kQuadratic;
    std::vector<bool> linear;
    double globalAngleCubic;
    const System& system;
    bool usePeriodic;
};

/**
 * This kernel is invoked by MMFFTorsionForce to calculate the forces acting on the system and the energy of the system.
 */
class ReferenceCalcMMFFTorsionForceKernel : public CalcMMFFTorsionForceKernel {
public:
    ReferenceCalcMMFFTorsionForceKernel(std::string name, const Platform& platform, const System& system);
    ~ReferenceCalcMMFFTorsionForceKernel();
    /**
     * Initialize the kernel.
     * 
     * @param system     the System this kernel will be applied to
     * @param force      the MMFFTorsionForce this kernel will be used for
     */
    void initialize(const System& system, const MMFFTorsionForce& force);
    /**
     * Execute the kernel to calculate the forces and/or energy.
     *
     * @param context        the context in which to execute this kernel
     * @param includeForces  true if forces should be calculated
     * @param includeEnergy  true if the energy should be calculated
     * @return the potential energy due to the force
     */
    double execute(ContextImpl& context, bool includeForces, bool includeEnergy);
    /**
     * Copy changed parameters over to a context.
     *
     * @param context    the context to copy parameters to
     * @param force      the MMFFTorsionForce to copy the parameters from
     */
    void copyParametersToContext(ContextImpl& context, const MMFFTorsionForce& force);
private:
    int numTorsions;
    std::vector<int>   particle1;
    std::vector<int>   particle2;
    std::vector<int>   particle3;
    std::vector<int>   particle4;
    std::vector<double> k1Torsion;
    std::vector<double> k2Torsion;
    std::vector<double> k3Torsion;
    const System& system;
    bool usePeriodic;
};

/**
 * This kernel is invoked by MMFFStretchBendForce to calculate the forces acting on the system and the energy of the system.
 */
class ReferenceCalcMMFFStretchBendForceKernel : public CalcMMFFStretchBendForceKernel {
public:
    ReferenceCalcMMFFStretchBendForceKernel(std::string name, const Platform& platform, const System& system);
    ~ReferenceCalcMMFFStretchBendForceKernel();
    /**
     * Initialize the kernel.
     * 
     * @param system     the System this kernel will be applied to
     * @param force      the MMFFStretchBendForce this kernel will be used for
     */
    void initialize(const System& system, const MMFFStretchBendForce& force);
    /**
     * Execute the kernel to calculate the forces and/or energy.
     *
     * @param context        the context in which to execute this kernel
     * @param includeForces  true if forces should be calculated
     * @param includeEnergy  true if the energy should be calculated
     * @return the potential energy due to the force
     */
    double execute(ContextImpl& context, bool includeForces, bool includeEnergy);
    /**
     * Copy changed parameters over to a context.
     *
     * @param context    the context to copy parameters to
     * @param force      the MMFFStretchBendForce to copy the parameters from
     */
    void copyParametersToContext(ContextImpl& context, const MMFFStretchBendForce& force);
private:
    int numStretchBends;
    std::vector<int>   particle1;
    std::vector<int>   particle2;
    std::vector<int>   particle3;
    std::vector<double> lengthABParameters;
    std::vector<double> lengthCBParameters;
    std::vector<double> angleParameters;
    std::vector<double> k1Parameters;
    std::vector<double> k2Parameters;
    const System& system;
    bool usePeriodic;
};

/**
 * This kernel is invoked by MMFFOutOfPlaneBendForce to calculate the forces acting on the system and the energy of the system.
 */
class ReferenceCalcMMFFOutOfPlaneBendForceKernel : public CalcMMFFOutOfPlaneBendForceKernel {
public:
    ReferenceCalcMMFFOutOfPlaneBendForceKernel(std::string name, const Platform& platform, const System& system);
    ~ReferenceCalcMMFFOutOfPlaneBendForceKernel();
    /**
     * Initialize the kernel.
     * 
     * @param system     the System this kernel will be applied to
     * @param force      the MMFFOutOfPlaneBendForce this kernel will be used for
     */
    void initialize(const System& system, const MMFFOutOfPlaneBendForce& force);
    /**
     * Execute the kernel to calculate the forces and/or energy.
     *
     * @param context        the context in which to execute this kernel
     * @param includeForces  true if forces should be calculated
     * @param includeEnergy  true if the energy should be calculated
     * @return the potential energy due to the force
     */
    double execute(ContextImpl& context, bool includeForces, bool includeEnergy);
    /**
     * Copy changed parameters over to a context.
     *
     * @param context    the context to copy parameters to
     * @param force      the MMFFOutOfPlaneBendForce to copy the parameters from
     */
    void copyParametersToContext(ContextImpl& context, const MMFFOutOfPlaneBendForce& force);
private:
    int numOutOfPlaneBends;
    std::vector<int>   particle1;
    std::vector<int>   particle2;
    std::vector<int>   particle3;
    std::vector<int>   particle4;
    std::vector<double> kParameters;
    double globalOutOfPlaneBendAngleCubic;
    double globalOutOfPlaneBendAngleQuartic;
    double globalOutOfPlaneBendAnglePentic;
    double globalOutOfPlaneBendAngleSextic;
    const System& system;
    bool usePeriodic;
};

/**
 * This kernel is invoked to calculate the vdw forces acting on the system and the energy of the system.
 */
class ReferenceCalcMMFFVdwForceKernel : public CalcMMFFVdwForceKernel {
public:
    ReferenceCalcMMFFVdwForceKernel(std::string name, const Platform& platform, const System& system);
    ~ReferenceCalcMMFFVdwForceKernel();
    /**
     * Initialize the kernel.
     * 
     * @param system     the System this kernel will be applied to
     * @param force      the MMFFVdwForce this kernel will be used for
     */
    void initialize(const System& system, const MMFFVdwForce& force);
    /**
     * Execute the kernel to calculate the forces and/or energy.
     *
     * @param context        the context in which to execute this kernel
     * @param includeForces  true if forces should be calculated
     * @param includeEnergy  true if the energy should be calculated
     * @return the potential energy due to the force
     */
    double execute(ContextImpl& context, bool includeForces, bool includeEnergy);
    /**
     * Copy changed parameters over to a context.
     *
     * @param context    the context to copy parameters to
     * @param force      the MMFFVdwForce to copy the parameters from
     */
    void copyParametersToContext(ContextImpl& context, const MMFFVdwForce& force);
private:
    int numParticles;
    int useCutoff;
    int usePBC;
    double cutoff;
    double dispersionCoefficient;
    std::vector< std::set<int> > allExclusions;
    std::vector<double> sigmas;
    std::vector<double> G_t_alphas;
    std::vector<double> alpha_d_Ns;
    std::vector<char> vdwDAs;
    const System& system;
    NeighborList* neighborList;
};

/**
 * This kernel is invoked by MMFFNonbondedForce to calculate the forces acting on the system.
 */
class ReferenceCalcMMFFNonbondedForceKernel : public CalcMMFFNonbondedForceKernel {
public:
    ReferenceCalcMMFFNonbondedForceKernel(std::string name, const Platform& platform, const System& system);
    ~ReferenceCalcMMFFNonbondedForceKernel();
    /**
     * Initialize the kernel.
     * 
     * @param system     the System this kernel will be applied to
     * @param force      the MMFFNonbondedForce this kernel will be used for
     */
    void initialize(const System& system, const MMFFNonbondedForce& force);
    /**
     * Execute the kernel to calculate the forces and/or energy.
     *
     * @param context        the context in which to execute this kernel
     * @param includeForces  true if forces should be calculated
     * @param includeEnergy  true if the energy should be calculated
     * @param includeReciprocal  true if reciprocal space interactions should be included
     * @return the potential energy due to the force
     */
    double execute(ContextImpl& context, bool includeForces, bool includeEnergy, bool includeDirect, bool includeReciprocal);
    /**
     * Copy changed parameters over to a context.
     *
     * @param context    the context to copy parameters to
     * @param force      the MMFFNonbondedForce to copy the parameters from
     */
    void copyParametersToContext(ContextImpl& context, const MMFFNonbondedForce& force);
    /**
     * Get the parameters being used for PME.
     * 
     * @param alpha   the separation parameter
     * @param nx      the number of grid points along the X axis
     * @param ny      the number of grid points along the Y axis
     * @param nz      the number of grid points along the Z axis
     */
    void getPMEParameters(double& alpha, int& nx, int& ny, int& nz) const;
private:
    int numParticles, num14;
    int **bonded14IndexArray;
    double **particleParamArray, **bonded14ParamArray;
    double nonbondedCutoff, rfDielectric, ewaldAlpha, ewaldDispersionAlpha, dispersionCoefficient;
    int kmax[3], gridSize[3], dispersionGridSize[3];
    std::vector<std::set<int> > exclusions;
    NonbondedMethod nonbondedMethod;
    const System& system;
    NeighborList* neighborList;
};

} // namespace OpenMM

#endif /*MMFF_OPENMM_REFERENCE_KERNELS_H*/
