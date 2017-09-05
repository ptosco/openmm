#ifndef MMFF_OPENMM_CUDAKERNELS_H_
#define MMFF_OPENMM_CUDAKERNELS_H_

/* -------------------------------------------------------------------------- *
 *                              OpenMMMMFF                                  *
 * -------------------------------------------------------------------------- *
 * This is part of the OpenMM molecular simulation toolkit originating from   *
 * Simbios, the NIH National Center for Physics-Based Simulation of           *
 * Biological Structures at Stanford, funded under the NIH Roadmap for        *
 * Medical Research, grant U54 GM072970. See https://simtk.org.               *
 *                                                                            *
 * Portions copyright (c) 2008-2015 Stanford University and the Authors.      *
 * Authors: Mark Friedrichs, Peter Eastman                                    *
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

#include "openmm/mmffKernels.h"
#include "openmm/kernels.h"
#include "openmm/System.h"
#include "CudaArray.h"
#include "CudaContext.h"
#include "CudaSort.h"
#include "CudaFFT3D.h"
#include <cufft.h>

namespace OpenMM {

/**
 * This kernel is invoked by MMFFBondForce to calculate the forces acting on the system and the energy of the system.
 */
class CudaCalcMMFFBondForceKernel : public CalcMMFFBondForceKernel {
public:
    CudaCalcMMFFBondForceKernel(std::string name, 
                                          const Platform& platform,
                                          CudaContext& cu,
                                          const System& system);
    ~CudaCalcMMFFBondForceKernel();
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
    class ForceInfo;
    int numBonds;
    CudaContext& cu;
    const System& system;
    CudaArray* params;
};

/**
 * This kernel is invoked by MMFFAngleForce to calculate the forces acting on the system and the energy of the system.
 */
class CudaCalcMMFFAngleForceKernel : public CalcMMFFAngleForceKernel {
public:
    CudaCalcMMFFAngleForceKernel(std::string name, const Platform& platform, CudaContext& cu, const System& system);
    ~CudaCalcMMFFAngleForceKernel();
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
    class ForceInfo;
    int numAngles;
    CudaContext& cu;
    const System& system;
    CudaArray* params;
};

/**
 * This kernel is invoked by MMFFTorsionForce to calculate the forces acting on the system and the energy of the system.
 */
class CudaCalcMMFFTorsionForceKernel : public CalcMMFFTorsionForceKernel {
public:
    CudaCalcMMFFTorsionForceKernel(std::string name, const Platform& platform, CudaContext& cu, const System& system);
    ~CudaCalcMMFFTorsionForceKernel();
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
    class ForceInfo;
    int numTorsions;
    CudaContext& cu;
    const System& system;
    CudaArray* params;
};

/**
 * This kernel is invoked by MMFFStretchBendForce to calculate the forces acting on the system and the energy of the system.
 */
class CudaCalcMMFFStretchBendForceKernel : public CalcMMFFStretchBendForceKernel {
public:
    CudaCalcMMFFStretchBendForceKernel(std::string name, const Platform& platform, CudaContext& cu, const System& system);
    ~CudaCalcMMFFStretchBendForceKernel();
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
    class ForceInfo;
    int numStretchBends;
    CudaContext& cu;
    const System& system;
    CudaArray* params1; // Equilibrium values
    CudaArray* params2; // force constants
};

/**
 * This kernel is invoked by MMFFOutOfPlaneBendForce to calculate the forces acting on the system and the energy of the system.
 */
class CudaCalcMMFFOutOfPlaneBendForceKernel : public CalcMMFFOutOfPlaneBendForceKernel {
public:
    CudaCalcMMFFOutOfPlaneBendForceKernel(std::string name, const Platform& platform, CudaContext& cu, const System& system);
    ~CudaCalcMMFFOutOfPlaneBendForceKernel();
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
    class ForceInfo;
    int numOutOfPlaneBends;
    CudaContext& cu;
    const System& system;
    CudaArray* params;
};

/**
 * This kernel is invoked by MMFFNonbondedForce to calculate the forces acting on the system.
 */
class CudaCalcMMFFNonbondedForceKernel : public CalcMMFFNonbondedForceKernel {
public:
    CudaCalcMMFFNonbondedForceKernel(std::string name, const Platform& platform, CudaContext& cu, const System& system);
    ~CudaCalcMMFFNonbondedForceKernel();
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
     * @param includeDirect  true if direct space interactions should be included
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
    class SortTrait : public CudaSort::SortTrait {
        int getDataSize() const {return 8;}
        int getKeySize() const {return 4;}
        const char* getDataType() const {return "int2";}
        const char* getKeyType() const {return "int";}
        const char* getMinKey() const {return "(-2147483647-1)";}
        const char* getMaxKey() const {return "2147483647";}
        const char* getMaxValue() const {return "make_int2(2147483647, 2147483647)";}
        const char* getSortKey() const {return "value.y";}
    };
    class ForceInfo;
    class PmeIO;
    class PmePreComputation;
    class PmePostComputation;
    class SyncStreamPreComputation;
    class SyncStreamPostComputation;
    CudaContext& cu;
    ForceInfo* info;
    bool hasInitializedFFT;
    CudaArray* params;
    CudaArray* exceptionParams;
    CudaArray* cosSinSums;
    CudaArray* directPmeGrid;
    CudaArray* reciprocalPmeGrid;
    CudaArray* pmeBsplineModuliX;
    CudaArray* pmeBsplineModuliY;
    CudaArray* pmeBsplineModuliZ;
    CudaArray* pmeAtomRange;
    CudaArray* pmeAtomGridIndex;
    CudaArray* pmeEnergyBuffer;
    CudaSort* sort;
    Kernel cpuPme;
    PmeIO* pmeio;
    CUstream pmeStream;
    CUevent pmeSyncEvent;
    CudaFFT3D* fft;
    cufftHandle fftForward;
    cufftHandle fftBackward;
    CUfunction ewaldSumsKernel;
    CUfunction ewaldForcesKernel;
    CUfunction pmeGridIndexKernel;
    CUfunction pmeSpreadChargeKernel;
    CUfunction pmeFinishSpreadChargeKernel;
    CUfunction pmeEvalEnergyKernel;
    CUfunction pmeConvolutionKernel;
    CUfunction pmeInterpolateForceKernel;
    std::vector<std::pair<int, int> > exceptionAtoms;
    double ewaldSelfEnergy, dispersionCoefficient, alpha;
    int interpolateForceThreads;
    int gridSizeX, gridSizeY, gridSizeZ;
    bool hasCoulomb, hasVdw, usePmeStream, useCudaFFT;
    NonbondedMethod nonbondedMethod;
    static const int PmeOrder = 5;
};

} // namespace OpenMM

#endif /*MMFF_OPENMM_CUDAKERNELS_H*/
