#ifndef MMFF_OPENMM_KERNELS_H_
#define MMFF_OPENMM_KERNELS_H_

/* -------------------------------------------------------------------------- *
 *                             OpenMMMMFF                                   *
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
 * Permission is hereby granted, free of charge, to any person obtaining a    *
 * copy of this software and associated documentation files (the "Software"), *
 * to deal in the Software without restriction, including without limitation  *
 * the rights to use, copy, modify, merge, publish, distribute, sublicense,   *
 * and/or sell copies of the Software, and to permit persons to whom the      *
 * Software is furnished to do so, subject to the following conditions:       *
 *                                                                            *
 * The above copyright notice and this permission notice shall be included in *
 * all copies or substantial portions of the Software.                        *
 *                                                                            *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR *
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,   *
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL    *
 * THE AUTHORS, CONTRIBUTORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM,    *
 * DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR      *
 * OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE  *
 * USE OR OTHER DEALINGS IN THE SOFTWARE.                                     *
 * -------------------------------------------------------------------------- */

#include "OpenMMMMFF.h"
#include "openmm/KernelImpl.h"
#include "openmm/System.h"
#include "openmm/Platform.h"

#include <set>
#include <string>
#include <vector>

namespace OpenMM {

/**
 * This kernel is invoked by MMFFBondForce to calculate the forces acting on the system and the energy of the system.
 */
class CalcMMFFBondForceKernel : public KernelImpl {

public:

    static std::string Name() {
        return "CalcMMFFBondForce";
    }

    CalcMMFFBondForceKernel(std::string name, const Platform& platform) : KernelImpl(name, platform) {
    }

    /**
     * Initialize the kernel.
     *
     * @param system     the System this kernel will be applied to
     * @param force      the MMFFBondForce this kernel will be used for
     */
    virtual void initialize(const System& system, const MMFFBondForce& force) = 0;

    /**
     * Execute the kernel to calculate the forces and/or energy.
     *
     * @param context        the context in which to execute this kernel
     * @param includeForces  true if forces should be calculated
     * @param includeEnergy  true if the energy should be calculated
     * @return the potential energy due to the force
     */
    virtual double execute(ContextImpl& context, bool includeForces, bool includeEnergy) = 0;
    /**
     * Copy changed parameters over to a context.
     *
     * @param context    the context to copy parameters to
     * @param force      the MMFFBondForce to copy the parameters from
     */
    virtual void copyParametersToContext(ContextImpl& context, const MMFFBondForce& force) = 0;
};

/**
 * This kernel is invoked by MMFFAngleForce to calculate the forces acting on the system and the energy of the system.
 */
class CalcMMFFAngleForceKernel : public KernelImpl {

public:

    static std::string Name() {
        return "CalcMMFFAngleForce";
    }

    CalcMMFFAngleForceKernel(std::string name, const Platform& platform) : KernelImpl(name, platform) {
    }

    /**
     * Initialize the kernel.
     *
     * @param system     the System this kernel will be applied to
     * @param force      the MMFFAngleForce this kernel will be used for
     */
    virtual void initialize(const System& system, const MMFFAngleForce& force) = 0;

    /**
     * Execute the kernel to calculate the forces and/or energy.
     *
     * @param context        the context in which to execute this kernel
     * @param includeForces  true if forces should be calculated
     * @param includeEnergy  true if the energy should be calculated
     * @return the potential energy due to the force
     */
    virtual double execute(ContextImpl& context, bool includeForces, bool includeEnergy) = 0;
    /**
     * Copy changed parameters over to a context.
     *
     * @param context    the context to copy parameters to
     * @param force      the MMFFAngleForce to copy the parameters from
     */
    virtual void copyParametersToContext(ContextImpl& context, const MMFFAngleForce& force) = 0;
};

/**
 * This kernel is invoked by MMFFInPlaneAngleForce to calculate the forces acting on the system and the energy of the system.
 */
class CalcMMFFInPlaneAngleForceKernel : public KernelImpl {

public:

    static std::string Name() {
        return "CalcMMFFInPlaneAngleForce";
    }

    CalcMMFFInPlaneAngleForceKernel(std::string name, const Platform& platform) : KernelImpl(name, platform) {
    }

    /**
     * Initialize the kernel.
     *
     * @param system     the System this kernel will be applied to
     * @param force      the MMFFInPlaneAngleForce this kernel will be used for
     */
    virtual void initialize(const System& system, const MMFFInPlaneAngleForce& force) = 0;

    /**
     * Execute the kernel to calculate the forces and/or energy.
     *
     * @param context        the context in which to execute this kernel
     * @param includeForces  true if forces should be calculated
     * @param includeEnergy  true if the energy should be calculated
     * @return the potential energy due to the force
     */
    virtual double execute(ContextImpl& context, bool includeForces, bool includeEnergy) = 0;
    /**
     * Copy changed parameters over to a context.
     *
     * @param context    the context to copy parameters to
     * @param force      the MMFFInPlaneAngleForce to copy the parameters from
     */
    virtual void copyParametersToContext(ContextImpl& context, const MMFFInPlaneAngleForce& force) = 0;
};

/**
 * This kernel is invoked by MMFFTorsionForce to calculate the forces acting on the system and the energy of the system.
 */
class CalcMMFFPiTorsionForceKernel : public KernelImpl {

public:

    static std::string Name() {
        return "CalcMMFFPiTorsionForce";
    }

    CalcMMFFPiTorsionForceKernel(std::string name, const Platform& platform) : KernelImpl(name, platform) {
    }

    /**
     * Initialize the kernel.
     *
     * @param system     the System this kernel will be applied to
     * @param force      the PiTorsionForce this kernel will be used for
     */
    virtual void initialize(const System& system, const MMFFPiTorsionForce& force) = 0;

    /**
     * Execute the kernel to calculate the forces and/or energy.
     *
     * @param context        the context in which to execute this kernel
     * @param includeForces  true if forces should be calculated
     * @param includeEnergy  true if the energy should be calculated
     * @return the potential energy due to the force
     */
    virtual double execute(ContextImpl& context, bool includeForces, bool includeEnergy) = 0;
    /**
     * Copy changed parameters over to a context.
     *
     * @param context    the context to copy parameters to
     * @param force      the MMFFPiTorsionForce to copy the parameters from
     */
    virtual void copyParametersToContext(ContextImpl& context, const MMFFPiTorsionForce& force) = 0;
};

/**
 * This kernel is invoked by MMFFTorsionForce to calculate the forces acting on the system and the energy of the system.
 */
class CalcMMFFStretchBendForceKernel : public KernelImpl {

public:

    static std::string Name() {
        return "CalcMMFFStretchBendForce";
    }

    CalcMMFFStretchBendForceKernel(std::string name, const Platform& platform) : KernelImpl(name, platform) {
    }

    /**
     * Initialize the kernel.
     *
     * @param system     the System this kernel will be applied to
     * @param force      the StretchBendForce this kernel will be used for
     */
    virtual void initialize(const System& system, const MMFFStretchBendForce& force) = 0;

    /**
     * Execute the kernel to calculate the forces and/or energy.
     *
     * @param context        the context in which to execute this kernel
     * @param includeForces  true if forces should be calculated
     * @param includeEnergy  true if the energy should be calculated
     * @return the potential energy due to the force
     */
    virtual double execute(ContextImpl& context, bool includeForces, bool includeEnergy) = 0;
    /**
     * Copy changed parameters over to a context.
     *
     * @param context    the context to copy parameters to
     * @param force      the MMFFStretchBendForce to copy the parameters from
     */
    virtual void copyParametersToContext(ContextImpl& context, const MMFFStretchBendForce& force) = 0;
};

/**
 * This kernel is invoked by MMFFTorsionForce to calculate the forces acting on the system and the energy of the system.
 */
class CalcMMFFOutOfPlaneBendForceKernel : public KernelImpl {

public:

    static std::string Name() {
        return "CalcMMFFOutOfPlaneBendForce";
    }

    CalcMMFFOutOfPlaneBendForceKernel(std::string name, const Platform& platform) : KernelImpl(name, platform) {
    }

    /**
     * Initialize the kernel.
     *
     * @param system     the System this kernel will be applied to
     * @param force      the OutOfPlaneBendForce this kernel will be used for
     */
    virtual void initialize(const System& system, const MMFFOutOfPlaneBendForce& force) = 0;

    /**
     * Execute the kernel to calculate the forces and/or energy.
     *
     * @param context        the context in which to execute this kernel
     * @param includeForces  true if forces should be calculated
     * @param includeEnergy  true if the energy should be calculated
     * @return the potential energy due to the force
     */
    virtual double execute(ContextImpl& context, bool includeForces, bool includeEnergy) = 0;
    /**
     * Copy changed parameters over to a context.
     *
     * @param context    the context to copy parameters to
     * @param force      the MMFFOutOfPlaneBendForce to copy the parameters from
     */
    virtual void copyParametersToContext(ContextImpl& context, const MMFFOutOfPlaneBendForce& force) = 0;
};

/**
 * This kernel is invoked by MMFFTorsionTorsionForce to calculate the forces acting on the system and the energy of the system.
 */
class CalcMMFFTorsionTorsionForceKernel : public KernelImpl {

public:

    static std::string Name() {
        return "CalcMMFFTorsionTorsionForce";
    }

    CalcMMFFTorsionTorsionForceKernel(std::string name, const Platform& platform) : KernelImpl(name, platform) {
    }

    /**
     * Initialize the kernel.
     *
     * @param system     the System this kernel will be applied to
     * @param force      the TorsionTorsionForce this kernel will be used for
     */
    virtual void initialize(const System& system, const MMFFTorsionTorsionForce& force) = 0;

    /**
     * Execute the kernel to calculate the forces and/or energy.
     *
     * @param context        the context in which to execute this kernel
     * @param includeForces  true if forces should be calculated
     * @param includeEnergy  true if the energy should be calculated
     * @return the potential energy due to the force
     */
    virtual double execute(ContextImpl& context, bool includeForces, bool includeEnergy) = 0;
};

/**
 * This kernel is invoked by MMFFMultipoleForce to calculate the forces acting on the system and the energy of the system.
 */
class CalcMMFFMultipoleForceKernel : public KernelImpl {

public:

    static std::string Name() {
        return "CalcMMFFMultipoleForce";
    }

    CalcMMFFMultipoleForceKernel(std::string name, const Platform& platform) : KernelImpl(name, platform) {
    }

    /**
     * Initialize the kernel.
     *
     * @param system     the System this kernel will be applied to
     * @param force      the MultipoleForce this kernel will be used for
     */
    virtual void initialize(const System& system, const MMFFMultipoleForce& force) = 0;

    /**
     * Execute the kernel to calculate the forces and/or energy.
     *
     * @param context        the context in which to execute this kernel
     * @param includeForces  true if forces should be calculated
     * @param includeEnergy  true if the energy should be calculated
     * @return the potential energy due to the force
     */
    virtual double execute(ContextImpl& context, bool includeForces, bool includeEnergy) = 0;

    virtual void getLabFramePermanentDipoles(ContextImpl& context, std::vector<Vec3>& dipoles) = 0;
    virtual void getInducedDipoles(ContextImpl& context, std::vector<Vec3>& dipoles) = 0;
    virtual void getTotalDipoles(ContextImpl& context, std::vector<Vec3>& dipoles) = 0;

    virtual void getElectrostaticPotential(ContextImpl& context, const std::vector< Vec3 >& inputGrid,
                                           std::vector< double >& outputElectrostaticPotential) = 0;

    virtual void getSystemMultipoleMoments(ContextImpl& context, std::vector< double >& outputMultipoleMoments) = 0;
    /**
     * Copy changed parameters over to a context.
     *
     * @param context    the context to copy parameters to
     * @param force      the MMFFMultipoleForce to copy the parameters from
     */
    virtual void copyParametersToContext(ContextImpl& context, const MMFFMultipoleForce& force) = 0;

    /**
     * Get the parameters being used for PME.
     *
     * @param alpha   the separation parameter
     * @param nx      the number of grid points along the X axis
     * @param ny      the number of grid points along the Y axis
     * @param nz      the number of grid points along the Z axis
     */
    virtual void getPMEParameters(double& alpha, int& nx, int& ny, int& nz) const = 0;
};

/**
 * This kernel is invoked by MMFFGeneralizedKirkwoodForce to calculate the forces acting on the system and the energy of the system.
 */
class CalcMMFFGeneralizedKirkwoodForceKernel : public KernelImpl {

public:

    static std::string Name() {
        return "CalcMMFFGeneralizedKirkwoodForce";
    }

    CalcMMFFGeneralizedKirkwoodForceKernel(std::string name, const Platform& platform) : KernelImpl(name, platform) {
    }

    /**
     * Initialize the kernel.
     *
     * @param system     the System this kernel will be applied to
     * @param force      the GBSAOBCForce this kernel will be used for
     */
    virtual void initialize(const System& system, const MMFFGeneralizedKirkwoodForce& force) = 0;

    /**
     * Execute the kernel to calculate the forces and/or energy.
     *
     * @param context        the context in which to execute this kernel
     * @param includeForces  true if forces should be calculated
     * @param includeEnergy  true if the energy should be calculated
     * @return the potential energy due to the force
     */
    virtual double execute(ContextImpl& context, bool includeForces, bool includeEnergy) = 0;
    /**
     * Copy changed parameters over to a context.
     *
     * @param context    the context to copy parameters to
     * @param force      the MMFFGeneralizedKirkwoodForce to copy the parameters from
     */
    virtual void copyParametersToContext(ContextImpl& context, const MMFFGeneralizedKirkwoodForce& force) = 0;
};


/**
 * This kernel is invoked by MMFFVdwForce to calculate the vdw forces acting on the system and the vdw energy of the system.
 */
class CalcMMFFVdwForceKernel : public KernelImpl {
public:

    static std::string Name() {
        return "CalcMMFFVdwForce";
    }

    CalcMMFFVdwForceKernel(std::string name, const Platform& platform) : KernelImpl(name, platform) {
    }

    /**
     * Initialize the kernel.
     *
     * @param system     the System this kernel will be applied to
     * @param force      the GBSAOBCForce this kernel will be used for
     */
    virtual void initialize(const System& system, const MMFFVdwForce& force) = 0;

    /**
     * Execute the kernel to calculate the forces and/or energy.
     *
     * @param context        the context in which to execute this kernel
     * @param includeForces  true if forces should be calculated
     * @param includeEnergy  true if the energy should be calculated
     * @return the potential energy due to the force
     */
    virtual double execute(ContextImpl& context, bool includeForces, bool includeEnergy) = 0;
    /**
     * Copy changed parameters over to a context.
     *
     * @param context    the context to copy parameters to
     * @param force      the MMFFVdwForce to copy the parameters from
     */
    virtual void copyParametersToContext(ContextImpl& context, const MMFFVdwForce& force) = 0;
};

/**
 * This kernel is invoked by MMFFWcaDispersionForce to calculate the WCA dispersion forces acting on the system and the WCA dispersion energy of the system.
 */
class CalcMMFFWcaDispersionForceKernel : public KernelImpl {

public:

    static std::string Name() {
        return "CalcMMFFWcaDispersionForce";
    }

    CalcMMFFWcaDispersionForceKernel(std::string name, const Platform& platform) : KernelImpl(name, platform) {
    }

    /**
     * Initialize the kernel.
     *
     * @param system     the System this kernel will be applied to
     * @param force      the GBSAOBCForce this kernel will be used for
     */
    virtual void initialize(const System& system, const MMFFWcaDispersionForce& force) = 0;

    /**
     * Execute the kernel to calculate the forces and/or energy.
     *
     * @param context        the context in which to execute this kernel
     * @param includeForces  true if forces should be calculated
     * @param includeEnergy  true if the energy should be calculated
     * @return the potential energy due to the force
     */
    virtual double execute(ContextImpl& context, bool includeForces, bool includeEnergy) = 0;
    /**
     * Copy changed parameters over to a context.
     *
     * @param context    the context to copy parameters to
     * @param force      the MMFFWcaDispersionForce to copy the parameters from
     */
    virtual void copyParametersToContext(ContextImpl& context, const MMFFWcaDispersionForce& force) = 0;
};

} // namespace OpenMM

#endif /*MMFF_OPENMM_KERNELS_H*/
