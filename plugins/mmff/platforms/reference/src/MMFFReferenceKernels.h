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
#include "openmm/MMFFMultipoleForce.h"
#include "MMFFReferenceMultipoleForce.h"
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
 * This kernel is invoked by MMFFMultipoleForce to calculate the forces acting on the system and the energy of the system.
 */
class ReferenceCalcMMFFMultipoleForceKernel : public CalcMMFFMultipoleForceKernel {
public:
    ReferenceCalcMMFFMultipoleForceKernel(std::string name, const Platform& platform, const System& system);
    ~ReferenceCalcMMFFMultipoleForceKernel();
    /**
     * Initialize the kernel.
     * 
     * @param system     the System this kernel will be applied to
     * @param force      the MMFFMultipoleForce this kernel will be used for
     */
    void initialize(const System& system, const MMFFMultipoleForce& force);
    /**
     * Setup for MMFFReferenceMultipoleForce instance. 
     *
     * @param context        the current context
     *
     * @return pointer to initialized instance of MMFFReferenceMultipoleForce
     */
    MMFFReferenceMultipoleForce* setupMMFFReferenceMultipoleForce(ContextImpl& context);
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
     * Get the induced dipole moments of all particles.
     * 
     * @param context    the Context for which to get the induced dipoles
     * @param dipoles    the induced dipole moment of particle i is stored into the i'th element
     */
    void getInducedDipoles(ContextImpl& context, std::vector<Vec3>& dipoles);
    /**
     * Get the fixed dipole moments of all particles in the global reference frame.
     * 
     * @param context    the Context for which to get the fixed dipoles
     * @param dipoles    the fixed dipole moment of particle i is stored into the i'th element
     */
    void getLabFramePermanentDipoles(ContextImpl& context, std::vector<Vec3>& dipoles);
    /**
     * Get the total dipole moments of all particles in the global reference frame.
     * 
     * @param context    the Context for which to get the fixed dipoles
     * @param dipoles    the fixed dipole moment of particle i is stored into the i'th element
     */
    void getTotalDipoles(ContextImpl& context, std::vector<Vec3>& dipoles);
    /** 
     * Calculate the electrostatic potential given vector of grid coordinates.
     *
     * @param context                      context
     * @param inputGrid                    input grid coordinates
     * @param outputElectrostaticPotential output potential 
     */
    void getElectrostaticPotential(ContextImpl& context, const std::vector< Vec3 >& inputGrid,
                                   std::vector< double >& outputElectrostaticPotential);

    /**
     * Get the system multipole moments.
     *
     * @param context                context 
     * @param outputMultipoleMoments vector of multipole moments:
                                     (charge,
                                      dipole_x, dipole_y, dipole_z,
                                      quadrupole_xx, quadrupole_xy, quadrupole_xz,
                                      quadrupole_yx, quadrupole_yy, quadrupole_yz,
                                      quadrupole_zx, quadrupole_zy, quadrupole_zz)
     */
    void getSystemMultipoleMoments(ContextImpl& context, std::vector< double >& outputMultipoleMoments);
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

    int numMultipoles;
    MMFFMultipoleForce::NonbondedMethod nonbondedMethod;
    MMFFMultipoleForce::PolarizationType polarizationType;
    std::vector<double> charges;
    std::vector<double> dipoles;
    std::vector<double> quadrupoles;
    std::vector<double> tholes;
    std::vector<double> dampingFactors;
    std::vector<double> polarity;
    std::vector<int>   axisTypes;
    std::vector<int>   multipoleAtomZs;
    std::vector<int>   multipoleAtomXs;
    std::vector<int>   multipoleAtomYs;
    std::vector< std::vector< std::vector<int> > > multipoleAtomCovalentInfo;

    int mutualInducedMaxIterations;
    double mutualInducedTargetEpsilon;
    std::vector<double> extrapolationCoefficients;

    bool usePme;
    double alphaEwald;
    double cutoffDistance;
    std::vector<int> pmeGridDimension;

    const System& system;
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
    std::vector<int> indexIVs;
    std::vector< std::set<int> > allExclusions;
    std::vector<double> sigmas;
    std::vector<double> epsilons;
    std::vector<double> reductions;
    std::string sigmaCombiningRule;
    std::string epsilonCombiningRule;
    const System& system;
    NeighborList* neighborList;
};

/**
 * This kernel is invoked to calculate the WCA dispersion forces acting on the system and the energy of the system.
 */
class ReferenceCalcMMFFWcaDispersionForceKernel : public CalcMMFFWcaDispersionForceKernel {
public:
    ReferenceCalcMMFFWcaDispersionForceKernel(std::string name, const Platform& platform, const System& system);
    ~ReferenceCalcMMFFWcaDispersionForceKernel();
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

    int numParticles;
    std::vector<double> radii;
    std::vector<double> epsilons;
    double epso; 
    double epsh; 
    double rmino; 
    double rminh; 
    double awater; 
    double shctd; 
    double dispoff;
    double slevy;
    double totalMaximumDispersionEnergy;
    const System& system;
};

/**
 * This kernel is invoked to calculate the Gerneralized Kirkwood forces acting on the system and the energy of the system.
 */
class ReferenceCalcMMFFGeneralizedKirkwoodForceKernel : public CalcMMFFGeneralizedKirkwoodForceKernel {
public:
    ReferenceCalcMMFFGeneralizedKirkwoodForceKernel(std::string name, const Platform& platform, const System& system);
    ~ReferenceCalcMMFFGeneralizedKirkwoodForceKernel();
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
     *  Get the 'include cavity term' flag.
     *
     *  @return includeCavityTerm
     */
    int getIncludeCavityTerm() const;

    /**
     *  Get the number of particles.
     *
     *  @return number of particles
     */
    int getNumParticles() const;

    /**
     *  Get Direct Polarization flag.
     *
     *  @return directPolarization
     *
     */
    int getDirectPolarization() const;

    /**
     *  Get the solute dielectric.
     *
     *  @return soluteDielectric
     *
     */
    double getSoluteDielectric() const;

    /**
     *  Get the solvent dielectric.
     *
     *  @return solventDielectric
     *
     */
    double getSolventDielectric() const;

    /**
     *  Get the dielectric offset.
     *
     *  @return dielectricOffset
     *
     */
    double getDielectricOffset() const;

    /**
     *  Get the probe radius.
     *
     *  @return probeRadius
     *
     */
    double getProbeRadius() const;

    /**
     *  Get the surface area factor.
     *
     *  @return surfaceAreaFactor
     *
     */
    double getSurfaceAreaFactor() const;

    /**
     *  Get the vector of particle radii.
     *
     *  @param atomicRadii vector of atomic radii
     *
     */
    void getAtomicRadii(std::vector<double>& atomicRadii) const;

    /**
     *  Get the vector of scale factors.
     *
     *  @param scaleFactors vector of scale factors
     *
     */
    void getScaleFactors(std::vector<double>& scaleFactors) const;

    /**
     *  Get the vector of charges.
     *
     *  @param charges vector of charges
     *
     */
    void getCharges(std::vector<double>& charges) const;

    /**
     * Copy changed parameters over to a context.
     *
     * @param context    the context to copy parameters to
     * @param force      the MMFFGeneralizedKirkwoodForce to copy the parameters from
     */
    void copyParametersToContext(ContextImpl& context, const MMFFGeneralizedKirkwoodForce& force);

private:

    int numParticles;
    std::vector<double> atomicRadii;
    std::vector<double> scaleFactors;
    std::vector<double> charges;
    double soluteDielectric;
    double solventDielectric;
    double dielectricOffset;
    double probeRadius;
    double surfaceAreaFactor;
    int includeCavityTerm;
    int directPolarization;
    const System& system;
};

} // namespace OpenMM

#endif /*MMFF_OPENMM_REFERENCE_KERNELS_H*/
