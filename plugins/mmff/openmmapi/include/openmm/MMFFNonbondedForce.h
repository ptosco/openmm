#ifndef OPENMM_MMFF_NONBONDEDFORCE_H_
#define OPENMM_MMFF_NONBONDEDFORCE_H_

/* -------------------------------------------------------------------------- *
 *                                   OpenMM                                   *
 * -------------------------------------------------------------------------- *
 * This is part of the OpenMM molecular simulation toolkit originating from   *
 * Simbios, the NIH National Center for Physics-Based Simulation of           *
 * Biological Structures at Stanford, funded under the NIH Roadmap for        *
 * Medical Research, grant U54 GM072970. See https://simtk.org.               *
 *                                                                            *
 * Portions copyright (c) 2008-2014 Stanford University and the Authors.      *
 * Authors: Peter Eastman                                                     *
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

#include "openmm/Force.h"
#include <map>
#include <set>
#include <utility>
#include <vector>
#include "internal/windowsExportMMFF.h"

namespace OpenMM {

/**
 * This class implements nonbonded interactions between particles, including a Coulomb force to represent
 * electrostatics and a 14-7 potential to represent van der Waals interactions.  It optionally supports
 * periodic boundary conditions and cutoffs for long range interactions.
 *
 * To use this class, create a MMFFNonbondedForce object, then call addParticle() once for each particle in the
 * System to define its parameters.  The number of particles for which you define nonbonded parameters must
 * be exactly equal to the number of particles in the System, or else an exception will be thrown when you
 * try to create a Context.  After a particle has been added, you can modify its force field parameters
 * by calling setParticleParameters().  This will have no effect on Contexts that already exist unless you
 * call updateParametersInContext().
 *
 * MMFFNonbondedForce also lets you specify "exceptions", particular pairs of particles whose interactions should be
 * computed based on different parameters than those defined for the individual particles.  This can be used to
 * completely exclude certain interactions from the force calculation, or to alter how they interact with each other.
 *
 * MMFF omits Coulomb and van der Waals interactions between particles separated by one or two bonds, while damping
 * coulombic interactions for those separated by three bonds (known as "1-4 interactions").
 * This class provides a convenience method for this case called createExceptionsFromBonds().  You pass to it
 * a list of bonds and the Coulomb scale factor to use for 1-4 interactions.  It identifies all pairs of particles which
 * are separated by 1, 2, or 3 bonds, then automatically creates exceptions for them.
 *
 * When using a cutoff, van der Waals interactions are sharply truncated at the cutoff distance.
 *
 * An optional feature of this class (enabled by default) is to add a contribution to the energy which approximates
 * the effect of all van der Waals interactions beyond the cutoff in a periodic system.  When running a simulation
 * at constant pressure, this can improve the quality of the result.  Call setUseDispersionCorrection() to set whether
 * this should be used.
 */

class OPENMM_EXPORT_MMFF MMFFNonbondedForce : public Force {
public:
    /**
     * This is an enumeration of the different methods that may be used for handling long range nonbonded forces.
     */
    enum NonbondedMethod {
        /**
         * No cutoff is applied to nonbonded interactions.  The full set of N^2 interactions is computed exactly.
         * This necessarily means that periodic boundary conditions cannot be used.  This is the default.
         */
        NoCutoff = 0,
        /**
         * Interactions beyond the cutoff distance are ignored.  Coulomb interactions closer than the cutoff distance
         * are modified using the reaction field method.
         */
        CutoffNonPeriodic = 1,
        /**
         * Periodic boundary conditions are used, so that each particle interacts only with the nearest periodic copy of
         * each other particle.  Interactions beyond the cutoff distance are ignored.  Coulomb interactions closer than the
         * cutoff distance are modified using the reaction field method.
         */
        CutoffPeriodic = 2,
        /**
         * Periodic boundary conditions are used, and Ewald summation is used to compute the Coulomb interaction of each particle
         * with all periodic copies of every other particle.
         */
        Ewald = 3,
        /**
         * Periodic boundary conditions are used, and Particle-Mesh Ewald (PME) summation is used to compute the Coulomb interaction of each particle
         * with all periodic copies of every other particle.
         */
        PME = 4,
    };
    /**
     * Create a MMFFNonbondedForce.
     */
    MMFFNonbondedForce();
    /**
     * Get the number of particles for which force field parameters have been defined.
     */
    int getNumParticles() const {
        return particles.size();
    }
    /**
     * Get the number of special interactions that should be calculated differently from other interactions.
     */
    int getNumExceptions() const {
        return exceptions.size();
    }
    /**
     * Get the method used for handling long range nonbonded interactions.
     */
    NonbondedMethod getNonbondedMethod() const;
    /**
     * Set the method used for handling long range nonbonded interactions.
     */
    void setNonbondedMethod(NonbondedMethod method);
    /**
     * Get the cutoff distance (in nm) being used for nonbonded interactions.  If the NonbondedMethod in use
     * is NoCutoff, this value will have no effect.
     *
     * @return the cutoff distance, measured in nm
     */
    double getCutoffDistance() const;
    /**
     * Set the cutoff distance (in nm) being used for nonbonded interactions.  If the NonbondedMethod in use
     * is NoCutoff, this value will have no effect.
     *
     * @param distance    the cutoff distance, measured in nm
     */
    void setCutoffDistance(double distance);
    /**
     * Get the dielectric constant to use for the solvent in the reaction field approximation.
     */
    double getReactionFieldDielectric() const;
    /**
     * Set the dielectric constant to use for the solvent in the reaction field approximation.
     */
    void setReactionFieldDielectric(double dielectric);
    /**
     * Get the error tolerance for Ewald summation.  This corresponds to the fractional error in the forces
     * which is acceptable.  This value is used to select the reciprocal space cutoff and separation
     * parameter so that the average error level will be less than the tolerance.  There is not a
     * rigorous guarantee that all forces on all atoms will be less than the tolerance, however.
     *
     * For PME calculations, if setPMEParameters() is used to set alpha to something other than 0,
     * this value is ignored.
     */
    double getEwaldErrorTolerance() const;
    /**
     * Set the error tolerance for Ewald summation.  This corresponds to the fractional error in the forces
     * which is acceptable.  This value is used to select the reciprocal space cutoff and separation
     * parameter so that the average error level will be less than the tolerance.  There is not a
     * rigorous guarantee that all forces on all atoms will be less than the tolerance, however.
     *
     * For PME calculations, if setPMEParameters() is used to set alpha to something other than 0,
     * this value is ignored.
     */
    void setEwaldErrorTolerance(double tol);
    /**
     * Get the parameters to use for PME calculations.  If alpha is 0 (the default), these parameters are
     * ignored and instead their values are chosen based on the Ewald error tolerance.
     *
     * @param[out] alpha   the separation parameter
     * @param[out] nx      the number of grid points along the X axis
     * @param[out] ny      the number of grid points along the Y axis
     * @param[out] nz      the number of grid points along the Z axis
     */
    void getPMEParameters(double& alpha, int& nx, int& ny, int& nz) const;
    /**
     * Set the parameters to use for PME calculations.  If alpha is 0 (the default), these parameters are
     * ignored and instead their values are chosen based on the Ewald error tolerance.
     *
     * @param alpha   the separation parameter
     * @param nx      the number of grid points along the X axis
     * @param ny      the number of grid points along the Y axis
     * @param nz      the number of grid points along the Z axis
     */
    void setPMEParameters(double alpha, int nx, int ny, int nz);
    /**
     * Get the parameters being used for PME in a particular Context.  Because some platforms have restrictions
     * on the allowed grid sizes, the values that are actually used may be slightly different from those
     * specified with setPMEParameters(), or the standard values calculated based on the Ewald error tolerance.
     * See the manual for details.
     *
     * @param context      the Context for which to get the parameters
     * @param[out] alpha   the separation parameter
     * @param[out] nx      the number of grid points along the X axis
     * @param[out] ny      the number of grid points along the Y axis
     * @param[out] nz      the number of grid points along the Z axis
     */
    void getPMEParametersInContext(const Context& context, double& alpha, int& nx, int& ny, int& nz) const;
    /**
     * Add the nonbonded force parameters for a particle.  This should be called once for each particle
     * in the System.  When it is called for the i'th time, it specifies the parameters for the i'th particle.
     * For calculating the van der Waals interaction between two particles, the per-particle constants are combined
     * using the MMFF combination rules.
     *
     * @param charge          the charge of the particle, measured in units of the proton charge
     * @param sigma           vdw sigma
     * @param G_t_alpha       vdw G*alpha
     * @param alpha_d_N       vdw alpha/N
     * @param vdwDA           vdw DA status ('-', 'D', 'A')
     * @return the index of the particle that was added
     */
    int addParticle(double charge, double sigma, double G_t_alpha, double alpha_d_N, char vdwDA);
    /**
     * Get the nonbonded force parameters for a particle.
     *
     * @param index                the index of the particle for which to get parameters
     * @param[out] charge          the charge of the particle, measured in units of the proton charge
     * @param[out] sigma           vdw sigma
     * @param[out] G_t_alpha       vdw G*alpha
     * @param[out] alpha_d_N       vdw alpha/N
     * @param[out] vdwDA           vdw DA status ('-', 'D', 'A')
     */
    void getParticleParameters(int index, double& charge, double& sigma, double& G_t_alpha, double& alpha_d_N, char& vdwDA) const;
    /**
     * Set the nonbonded force parameters for a particle.  When calculating the van der Waals interaction between
     * two particles, the per-particle constants are combined using the MMFF combination rules.
     *
     * @param index           the index of the particle for which to set parameters
     * @param charge          the charge of the particle, measured in units of the proton charge
     * @param sigma           vdw sigma
     * @param G_t_alpha       vdw G*alpha
     * @param alpha_d_N       vdw alpha/N
     * @param vdwDA           vdw DA status ('-', 'D', 'A')
     */
    void setParticleParameters(int index, double charge, double sigma, double G_t_alpha, double alpha_d_N, char vdwDA);
    /**
     * Add an interaction to the list of exceptions that should be calculated differently from other interactions.
     * If chargeProd and epsilon are both equal to 0, this will cause the interaction to be completely omitted from
     * force and energy calculations.
     *
     * In many cases, you can use createExceptionsFromBonds() rather than adding each exception explicitly.
     *
     * @param particle1    the index of the first particle involved in the interaction
     * @param particle2    the index of the second particle involved in the interaction
     * @param chargeProd   the product of particle charges
     * @param sigma        the combined sigma of the two particles (already scaled if appropriate)
     * @param epsilon      the combined epsilon of the two particles (already scaled if appropriate)
     * @param replace      determines the behavior if there is already an exception for the same two particles.  If true, the existing one is replaced.
     *                     If false, an exception is thrown.
     * @return the index of the exception that was added
     */
    int addException(int particle1, int particle2, double chargeProd, double sigma, double epsilon, bool replace = false);
    /**
     * Get the scaling factors for an interaction that should be calculated differently from others.
     *
     * @param index             the index of the interaction for which to get parameters
     * @param[out] particle1    the index of the first particle involved in the interaction
     * @param[out] particle2    the index of the second particle involved in the interaction
     * @param[out] chargeProd   the product of particle charges
     * @param[out] sigma        the combined sigma of the two particles (already scaled if appropriate)
     * @param[out] epsilon      the combined epsilon of the two particles (already scaled if appropriate)
     */
    void getExceptionParameters(int index, int& particle1, int& particle2, double& chargeProd, double& sigma, double& epsilon) const;
    /**
     * Set the scaling factors for an interaction that should be calculated differently from others.
     * If chargeProd and epsilon are both equal to 0, this will cause the interaction to be completely omitted from
     * force and energy calculations.
     *
     * @param index        the index of the interaction for which to get parameters
     * @param particle1    the index of the first particle involved in the interaction
     * @param particle2    the index of the second particle involved in the interaction
     * @param chargeProd   the product of particle charges
     * @param sigma        the combined sigma of the two particles (already scaled if appropriate)
     * @param epsilon      the combined epsilon of the two particles (already scaled if appropriate)
     */
    void setExceptionParameters(int index, int particle1, int particle2, double chargeProd, double sigma, double epsilon);
    /**
     * Identify exceptions based on the molecular topology.  Particles which are separated by one or two bonds are set
     * to not interact at all, while pairs of particles separated by three bonds (known as "1-4 interactions") have
     * their Coulomb and van der Waals interactions reduced by a fixed factor.
     *
     * @param bonds           the set of bonds based on which to construct exceptions.  Each element specifies the indices of
     *                        two particles that are bonded to each other.
     * @param coulomb14Scale  pairs of particles separated by three bonds will have the strength of their Coulomb interaction
     *                        multiplied by this factor (defaults to 0.75)
     * @param vdw14Scale      pairs of particles separated by three bonds will have the strength of their 14-7 interaction
     *                        multiplied by this factor (defaults to 1.0)
     */
    void createExceptionsFromBonds(const std::vector<std::pair<int, int> >& bonds, double coulomb14Scale = 0.75, double vdw14Scale = 1.0);
    /**
     * Get whether to add a contribution to the energy that approximately represents the effect of van der Waals
     * interactions beyond the cutoff distance.  The energy depends on the volume of the periodic box, and is only
     * applicable when periodic boundary conditions are used.  When running simulations at constant pressure, adding
     * this contribution can improve the quality of results.
     */
    bool getUseDispersionCorrection() const {
        return useDispersionCorrection;
    }
    /**
     * Set whether to add a contribution to the energy that approximately represents the effect of van der Waals
     * interactions beyond the cutoff distance.  The energy depends on the volume of the periodic box, and is only
     * applicable when periodic boundary conditions are used.  When running simulations at constant pressure, adding
     * this contribution can improve the quality of results.
     */
    void setUseDispersionCorrection(bool useCorrection) {
        useDispersionCorrection = useCorrection;
    }
    /**
     * Get the force group that reciprocal space interactions for Ewald or PME are included in.  This allows multiple
     * time step integrators to evaluate direct and reciprocal space interactions at different intervals: getForceGroup()
     * specifies the group for direct space, and getReciprocalSpaceForceGroup() specifies the group for reciprocal space.
     * If this is -1 (the default value), the same force group is used for reciprocal space as for direct space.
     */
    int getReciprocalSpaceForceGroup() const;
    /**
     * Set the force group that reciprocal space interactions for Ewald or PME are included in.  This allows multiple
     * time step integrators to evaluate direct and reciprocal space interactions at different intervals: setForceGroup()
     * specifies the group for direct space, and setReciprocalSpaceForceGroup() specifies the group for reciprocal space.
     * If this is -1 (the default value), the same force group is used for reciprocal space as for direct space.
     *
     * @param group    the group index.  Legal values are between 0 and 31 (inclusive), or -1 to use the same force group
     *                 that is specified for direct space.
     */
    void setReciprocalSpaceForceGroup(int group);
    /**
     * Update the particle and exception parameters in a Context to match those stored in this Force object.  This method
     * provides an efficient method to update certain parameters in an existing Context without needing to reinitialize it.
     * Simply call setParticleParameters() and setExceptionParameters() to modify this object's parameters, then call
     * updateParametersInContext() to copy them over to the Context.
     *
     * This method has several limitations.  The only information it updates is the parameters of particles and exceptions.
     * All other aspects of the Force (the nonbonded method, the cutoff distance, etc.) are unaffected and can only be
     * changed by reinitializing the Context.  Furthermore, only the chargeProd, sigma, and epsilon values of an exception
     * can be changed; the pair of particles involved in the exception cannot change.  Finally, this method cannot be used
     * to add new particles or exceptions, only to change the parameters of existing ones.
     */
    void updateParametersInContext(Context& context);
    /**
     * Returns whether or not this force makes use of periodic boundary
     * conditions.
     *
     * @returns true if force uses PBC and false otherwise
     */
    bool usesPeriodicBoundaryConditions() const {
        return nonbondedMethod == MMFFNonbondedForce::CutoffPeriodic ||
               nonbondedMethod == MMFFNonbondedForce::Ewald ||
               nonbondedMethod == MMFFNonbondedForce::PME;
    }
    static double sigmaCombiningRule(double sigmaI, double sigmaJ, bool haveDonor);
    static double epsilonCombiningRule(double combinedSigma,
        double alphaI_d_NI, double alphaJ_d_NJ, double GI_t_alphaI, double GJ_t_alphaJ);
    static void scaleSigmaEpsilon(double& combinedSigma, double& combinedEpsilon);
protected:
    ForceImpl* createImpl() const;
private:
    class ParticleInfo;
    class ExceptionInfo;
    NonbondedMethod nonbondedMethod;
    double cutoffDistance, rfDielectric, ewaldErrorTol, alpha, dalpha;
    bool useDispersionCorrection;
    int recipForceGroup, nx, ny, nz, dnx, dny, dnz;
void addExclusionsToSet(const std::vector<std::set<int> >& bonded12, std::set<int>& exclusions, int baseParticle, int fromParticle, int currentLevel) const;
    std::vector<ParticleInfo> particles;
    std::vector<ExceptionInfo> exceptions;
    std::map<std::pair<int, int>, int> exceptionMap;
};

/**
 * This is an internal class used to record information about a particle.
 * @private
 */
class MMFFNonbondedForce::ParticleInfo {
public:
    double charge, sigma, G_t_alpha, alpha_d_N;
    char vdwDA;
    ParticleInfo() {
        charge = G_t_alpha = 0.0;
        sigma = alpha_d_N = 1.0;
        vdwDA = '-';
    }
    ParticleInfo(double charge, double sigma, double G_t_alpha, double alpha_d_N, char vdwDA) :
        charge(charge), sigma(sigma), G_t_alpha(G_t_alpha), alpha_d_N(alpha_d_N), vdwDA(vdwDA) {
    }
};

/**
 * This is an internal class used to record information about an exception.
 * @private
 */
class MMFFNonbondedForce::ExceptionInfo {
public:
    int particle1, particle2;
    double chargeProd, sigma, epsilon;
    ExceptionInfo() {
        particle1 = particle2 = -1;
        sigma = 1.0;
        chargeProd = epsilon = 0.0;
    }
    ExceptionInfo(int particle1, int particle2, double chargeProd, double sigma, double epsilon) :
        particle1(particle1), particle2(particle2), chargeProd(chargeProd), sigma(sigma), epsilon(epsilon) {
    }
};

} // namespace OpenMM

#endif /*OPENMM_MMFF_NONBONDEDFORCE_H_*/
