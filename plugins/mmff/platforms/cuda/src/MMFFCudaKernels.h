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
#include <cufft.h>

namespace OpenMM {

class CudaCalcMMFFGeneralizedKirkwoodForceKernel;

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
 * This kernel is invoked by MMFFInPlaneAngleForce to calculate the forces acting on the system and the energy of the system.
 */
class CudaCalcMMFFInPlaneAngleForceKernel : public CalcMMFFInPlaneAngleForceKernel {
public:
    CudaCalcMMFFInPlaneAngleForceKernel(std::string name, const Platform& platform, CudaContext& cu, const System& system);
    ~CudaCalcMMFFInPlaneAngleForceKernel();
    /**
     * Initialize the kernel.
     * 
     * @param system     the System this kernel will be applied to
     * @param force      the MMFFInPlaneAngleForce this kernel will be used for
     */
    void initialize(const System& system, const MMFFInPlaneAngleForce& force);
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
     * @param force      the MMFFInPlaneAngleForce to copy the parameters from
     */
    void copyParametersToContext(ContextImpl& context, const MMFFInPlaneAngleForce& force);
private:
    class ForceInfo;
    int numAngles;
    CudaContext& cu;
    const System& system;
    CudaArray* params;
};

/**
 * This kernel is invoked by MMFFPiTorsionForce to calculate the forces acting on the system and the energy of the system.
 */
class CudaCalcMMFFPiTorsionForceKernel : public CalcMMFFPiTorsionForceKernel {
public:
    CudaCalcMMFFPiTorsionForceKernel(std::string name, const Platform& platform, CudaContext& cu, const System& system);
    ~CudaCalcMMFFPiTorsionForceKernel();
    /**
     * Initialize the kernel.
     * 
     * @param system     the System this kernel will be applied to
     * @param force      the MMFFPiTorsionForce this kernel will be used for
     */
    void initialize(const System& system, const MMFFPiTorsionForce& force);
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
     * @param force      the MMFFPiTorsionForce to copy the parameters from
     */
    void copyParametersToContext(ContextImpl& context, const MMFFPiTorsionForce& force);
private:
    class ForceInfo;
    int numPiTorsions;
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
 * This kernel is invoked by MMFFTorsionTorsionForce to calculate the forces acting on the system and the energy of the system.
 */
class CudaCalcMMFFTorsionTorsionForceKernel : public CalcMMFFTorsionTorsionForceKernel {
public:
    CudaCalcMMFFTorsionTorsionForceKernel(std::string name, const Platform& platform, CudaContext& cu, const System& system);
    ~CudaCalcMMFFTorsionTorsionForceKernel();
    /**
     * Initialize the kernel.
     * 
     * @param system     the System this kernel will be applied to
     * @param force      the MMFFTorsionTorsionForce this kernel will be used for
     */
    void initialize(const System& system, const MMFFTorsionTorsionForce& force);
    /**
     * Execute the kernel to calculate the forces and/or energy.
     *
     * @param context        the context in which to execute this kernel
     * @param includeForces  true if forces should be calculated
     * @param includeEnergy  true if the energy should be calculated
     * @return the potential energy due to the force
     */
    double execute(ContextImpl& context, bool includeForces, bool includeEnergy);
private:
    class ForceInfo;
    int numTorsionTorsions;
    int numTorsionTorsionGrids;
    CudaContext& cu;
    const System& system;
    CudaArray* gridValues;
    CudaArray* gridParams;
    CudaArray* torsionParams;
};

/**
 * This kernel is invoked by MMFFMultipoleForce to calculate the forces acting on the system and the energy of the system.
 */
class CudaCalcMMFFMultipoleForceKernel : public CalcMMFFMultipoleForceKernel {
public:
    CudaCalcMMFFMultipoleForceKernel(std::string name, const Platform& platform, CudaContext& cu, const System& system);
    ~CudaCalcMMFFMultipoleForceKernel();
    /**
     * Initialize the kernel.
     * 
     * @param system     the System this kernel will be applied to
     * @param force      the MMFFMultipoleForce this kernel will be used for
     */
    void initialize(const System& system, const MMFFMultipoleForce& force);
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
     * Get the LabFrame dipole moments of all particles.
     * 
     * @param context    the Context for which to get the induced dipoles
     * @param dipoles    the induced dipole moment of particle i is stored into the i'th element
     */
    void getLabFramePermanentDipoles(ContextImpl& context, std::vector<Vec3>& dipoles);
    /**
     * Get the induced dipole moments of all particles.
     * 
     * @param context    the Context for which to get the induced dipoles
     * @param dipoles    the induced dipole moment of particle i is stored into the i'th element
     */
    void getInducedDipoles(ContextImpl& context, std::vector<Vec3>& dipoles);
    /**
     * Get the total dipole moments of all particles.
     * 
     * @param context    the Context for which to get the induced dipoles
     * @param dipoles    the induced dipole moment of particle i is stored into the i'th element
     */
    void getTotalDipoles(ContextImpl& context, std::vector<Vec3>& dipoles);
    /**
     * Execute the kernel to calculate the electrostatic potential
     *
     * @param context        the context in which to execute this kernel
     * @param inputGrid      input grid coordinates
     * @param outputElectrostaticPotential output potential 
     */
    void getElectrostaticPotential(ContextImpl& context, const std::vector< Vec3 >& inputGrid,
                                   std::vector< double >& outputElectrostaticPotential);

   /** 
     * Get the system multipole moments
     *
     * @param context      context
     * @param outputMultipoleMoments (charge,
     *                                dipole_x, dipole_y, dipole_z,
     *                                quadrupole_xx, quadrupole_xy, quadrupole_xz,
     *                                quadrupole_yx, quadrupole_yy, quadrupole_yz,
     *                                quadrupole_zx, quadrupole_zy, quadrupole_zz)
     */
    void getSystemMultipoleMoments(ContextImpl& context, std::vector<double>& outputMultipoleMoments);
    /**
     * Copy changed parameters over to a context.
     *
     * @param context    the context to copy parameters to
     * @param force      the MMFFMultipoleForce to copy the parameters from
     */
    void copyParametersToContext(ContextImpl& context, const MMFFMultipoleForce& force);
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
    class ForceInfo;
    class SortTrait : public CudaSort::SortTrait {
        int getDataSize() const {return 8;}
        int getKeySize() const {return 4;}
        const char* getDataType() const {return "int2";}
        const char* getKeyType() const {return "int";}
        const char* getMinKey() const {return "(-2147483647 - 1)";}
        const char* getMaxKey() const {return "2147483647";}
        const char* getMaxValue() const {return "make_int2(2147483647, 2147483647)";}
        const char* getSortKey() const {return "value.y";}
    };
    void initializeScaleFactors();
    void computeInducedField(void** recipBoxVectorPointer);
    bool iterateDipolesByDIIS(int iteration);
    void computeExtrapolatedDipoles(void** recipBoxVectorPointer);
    void ensureMultipolesValid(ContextImpl& context);
    template <class T, class T4, class M4> void computeSystemMultipoleMoments(ContextImpl& context, std::vector<double>& outputMultipoleMoments);
    int numMultipoles, maxInducedIterations, maxExtrapolationOrder;
    int fixedFieldThreads, inducedFieldThreads, electrostaticsThreads;
    int gridSizeX, gridSizeY, gridSizeZ;
    double alpha, inducedEpsilon;
    bool usePME, hasQuadrupoles, hasInitializedScaleFactors, hasInitializedFFT, multipolesAreValid, hasCreatedEvent;
    MMFFMultipoleForce::PolarizationType polarizationType;
    CudaContext& cu;
    const System& system;
    std::vector<int3> covalentFlagValues;
    std::vector<int2> polarizationFlagValues;
    CudaArray* multipoleParticles;
    CudaArray* molecularDipoles;
    CudaArray* molecularQuadrupoles;
    CudaArray* labFrameDipoles;
    CudaArray* labFrameQuadrupoles;
    CudaArray* sphericalDipoles;
    CudaArray* sphericalQuadrupoles;
    CudaArray* fracDipoles;
    CudaArray* fracQuadrupoles;
    CudaArray* field;
    CudaArray* fieldPolar;
    CudaArray* inducedField;
    CudaArray* inducedFieldPolar;
    CudaArray* torque;
    CudaArray* dampingAndThole;
    CudaArray* inducedDipole;
    CudaArray* inducedDipolePolar;
    CudaArray* inducedDipoleErrors;
    CudaArray* prevDipoles;
    CudaArray* prevDipolesPolar;
    CudaArray* prevDipolesGk;
    CudaArray* prevDipolesGkPolar;
    CudaArray* prevErrors;
    CudaArray* diisMatrix;
    CudaArray* diisCoefficients;
    CudaArray* extrapolatedDipole;
    CudaArray* extrapolatedDipolePolar;
    CudaArray* extrapolatedDipoleGk;
    CudaArray* extrapolatedDipoleGkPolar;
    CudaArray* inducedDipoleFieldGradient;
    CudaArray* inducedDipoleFieldGradientPolar;
    CudaArray* inducedDipoleFieldGradientGk;
    CudaArray* inducedDipoleFieldGradientGkPolar;
    CudaArray* extrapolatedDipoleFieldGradient;
    CudaArray* extrapolatedDipoleFieldGradientPolar;
    CudaArray* extrapolatedDipoleFieldGradientGk;
    CudaArray* extrapolatedDipoleFieldGradientGkPolar;
    CudaArray* polarizability;
    CudaArray* covalentFlags;
    CudaArray* polarizationGroupFlags;
    CudaArray* pmeGrid;
    CudaArray* pmeBsplineModuliX;
    CudaArray* pmeBsplineModuliY;
    CudaArray* pmeBsplineModuliZ;
    CudaArray* pmeIgrid;
    CudaArray* pmePhi;
    CudaArray* pmePhid;
    CudaArray* pmePhip;
    CudaArray* pmePhidp;
    CudaArray* pmeCphi;
    CudaArray* pmeAtomRange;
    CudaArray* lastPositions;
    CudaSort* sort;
    cufftHandle fft;
    CUfunction computeMomentsKernel, recordInducedDipolesKernel, computeFixedFieldKernel, computeInducedFieldKernel, updateInducedFieldKernel, electrostaticsKernel, mapTorqueKernel;
    CUfunction pmeSpreadFixedMultipolesKernel, pmeSpreadInducedDipolesKernel, pmeFinishSpreadChargeKernel, pmeConvolutionKernel;
    CUfunction pmeFixedPotentialKernel, pmeInducedPotentialKernel, pmeFixedForceKernel, pmeInducedForceKernel, pmeRecordInducedFieldDipolesKernel, computePotentialKernel;
    CUfunction recordDIISDipolesKernel, buildMatrixKernel, solveMatrixKernel;
    CUfunction initExtrapolatedKernel, iterateExtrapolatedKernel, computeExtrapolatedKernel, addExtrapolatedGradientKernel;
    CUfunction pmeTransformMultipolesKernel, pmeTransformPotentialKernel;
    CUevent syncEvent;
    CudaCalcMMFFGeneralizedKirkwoodForceKernel* gkKernel;
    static const int PmeOrder = 5;
    static const int MaxPrevDIISDipoles = 20;
};

/**
 * This kernel is invoked by MMFFMultipoleForce to calculate the forces acting on the system and the energy of the system.
 */
class CudaCalcMMFFGeneralizedKirkwoodForceKernel : public CalcMMFFGeneralizedKirkwoodForceKernel {
public:
    CudaCalcMMFFGeneralizedKirkwoodForceKernel(std::string name, const Platform& platform, CudaContext& cu, const System& system);
    ~CudaCalcMMFFGeneralizedKirkwoodForceKernel();
    /**
     * Initialize the kernel.
     * 
     * @param system     the System this kernel will be applied to
     * @param force      the MMFFMultipoleForce this kernel will be used for
     */
    void initialize(const System& system, const MMFFGeneralizedKirkwoodForce& force);
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
     * Perform the computation of Born radii.
     */
    void computeBornRadii();
    /**
     * Perform the final parts of the force/energy computation.
     */
    void finishComputation(CudaArray& torque, CudaArray& labFrameDipoles, CudaArray& labFrameQuadrupoles, CudaArray& inducedDipole, CudaArray& inducedDipolePolar, CudaArray& dampingAndThole, CudaArray& covalentFlags, CudaArray& polarizationGroupFlags);
    CudaArray* getBornRadii() {
        return bornRadii;
    }
    CudaArray* getField() {
        return field;
    }
    CudaArray* getInducedField() {
        return inducedField;
    }
    CudaArray* getInducedFieldPolar() {
        return inducedFieldPolar;
    }
    CudaArray* getInducedDipoles() {
        return inducedDipoleS;
    }
    CudaArray* getInducedDipolesPolar() {
        return inducedDipolePolarS;
    }
    /**
     * Copy changed parameters over to a context.
     *
     * @param context    the context to copy parameters to
     * @param force      the MMFFGeneralizedKirkwoodForce to copy the parameters from
     */
    void copyParametersToContext(ContextImpl& context, const MMFFGeneralizedKirkwoodForce& force);
private:
    class ForceInfo;
    CudaContext& cu;
    const System& system;
    bool includeSurfaceArea, hasInitializedKernels;
    int computeBornSumThreads, gkForceThreads, chainRuleThreads, ediffThreads;
    MMFFMultipoleForce::PolarizationType polarizationType;
    std::map<std::string, std::string> defines;
    CudaArray* params;
    CudaArray* bornSum;
    CudaArray* bornRadii;
    CudaArray* bornForce;
    CudaArray* field;
    CudaArray* inducedField;
    CudaArray* inducedFieldPolar;
    CudaArray* inducedDipoleS;
    CudaArray* inducedDipolePolarS;
    CUfunction computeBornSumKernel, reduceBornSumKernel, surfaceAreaKernel, gkForceKernel, chainRuleKernel, ediffKernel;
};

/**
 * This kernel is invoked to calculate the vdw forces acting on the system and the energy of the system.
 */
class CudaCalcMMFFVdwForceKernel : public CalcMMFFVdwForceKernel {
public:
    CudaCalcMMFFVdwForceKernel(std::string name, const Platform& platform, CudaContext& cu, const System& system);
    ~CudaCalcMMFFVdwForceKernel();
    /**
     * Initialize the kernel.
     * 
     * @param system     the System this kernel will be applied to
     * @param force      the MMFFMultipoleForce this kernel will be used for
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
    class ForceInfo;
    CudaContext& cu;
    const System& system;
    bool hasInitializedNonbonded;
    double dispersionCoefficient;
    CudaArray* sigmaEpsilon;
    CudaArray* bondReductionAtoms;
    CudaArray* bondReductionFactors;
    CudaArray* tempPosq;
    CudaArray* tempForces;
    CudaNonbondedUtilities* nonbonded;
    CUfunction prepareKernel, spreadKernel;
};

/**
 * This kernel is invoked to calculate the WCA dispersion forces acting on the system and the energy of the system.
 */
class CudaCalcMMFFWcaDispersionForceKernel : public CalcMMFFWcaDispersionForceKernel {
public:
    CudaCalcMMFFWcaDispersionForceKernel(std::string name, const Platform& platform, CudaContext& cu, const System& system);
    ~CudaCalcMMFFWcaDispersionForceKernel();
    /**
     * Initialize the kernel.
     * 
     * @param system     the System this kernel will be applied to
     * @param force      the MMFFMultipoleForce this kernel will be used for
     */
    void initialize(const System& system, const MMFFWcaDispersionForce& force);
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
     * @param force      the MMFFWcaDispersionForce to copy the parameters from
     */
    void copyParametersToContext(ContextImpl& context, const MMFFWcaDispersionForce& force);
private:
    class ForceInfo;
    CudaContext& cu;
    const System& system;
    double totalMaximumDispersionEnergy;
    CudaArray* radiusEpsilon;
    CUfunction forceKernel;
};

} // namespace OpenMM

#endif /*MMFF_OPENMM_CUDAKERNELS_H*/
