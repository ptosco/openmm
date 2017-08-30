#ifndef OPENMM_MMFFTORSIONFORCE_H_
#define OPENMM_MMFFTORSIONFORCE_H_

/* -------------------------------------------------------------------------- *
 *                                   OpenMM                                   *
 * -------------------------------------------------------------------------- *
 * This is part of the OpenMM molecular simulation toolkit originating from   *
 * Simbios, the NIH National Center for Physics-Based Simulation of           *
 * Biological Structures at Stanford, funded under the NIH Roadmap for        *
 * Medical Research, grant U54 GM072970. See https://simtk.org.               *
 *                                                                            *
 * Portions copyright (c) 2008-2016 Stanford University and the Authors.      *
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
#include "openmm/Vec3.h"
#include <map>
#include <vector>
#include "openmm/internal/windowsExport.h"

namespace OpenMM {

/**
 * This class implements an interaction between groups of four particles that varies with the torsion angle between them
 * according to the Ryckaert-Bellemans potential.  To use it, create an MMFFTorsionForce object then call addTorsion() once
 * for each torsion.  After a torsion has been added, you can modify its force field parameters by calling setTorsionParameters().
 * This will have no effect on Contexts that already exist unless you call updateParametersInContext().
 */

class OPENMM_EXPORT MMFFTorsionForce : public Force {
public:
    /**
     * Create a MMFFTorsionForce.
     */
    MMFFTorsionForce();
    /**
     * Get the number of MMFF torsion terms in the potential function
     */
    int getNumTorsions() const {
        return mmffTorsions.size();
    }
    /**
     * Add a MMFF torsion term to the force field.
     *
     * @param particle1    the index of the first particle forming the torsion
     * @param particle2    the index of the second particle forming the torsion
     * @param particle3    the index of the third particle forming the torsion
     * @param particle4    the index of the fourth particle forming the torsion
     * @param c1           the coefficient of the 1st order term, measured in kJ/mol
     * @param c2           the coefficient of the 2nd order term, measured in kJ/mol
     * @param c3           the coefficient of the 3rd order term, measured in kJ/mol
     * @return the index of the torsion that was added
     */
    int addTorsion(int particle1, int particle2, int particle3, int particle4, double c1, double c2, double c3);
    /**
     * Get the force field parameters for a MMFF torsion term.
     *
     * @param index             the index of the torsion for which to get parameters
     * @param[out] particle1    the index of the first particle forming the torsion
     * @param[out] particle2    the index of the second particle forming the torsion
     * @param[out] particle3    the index of the third particle forming the torsion
     * @param[out] particle4    the index of the fourth particle forming the torsion
     * @param[out] c1           the coefficient of the 1st order term, measured in kJ/mol
     * @param[out] c2           the coefficient of the 2nd order term, measured in kJ/mol
     * @param[out] c3           the coefficient of the 3rd order term, measured in kJ/mol
     */
    void getTorsionParameters(int index, int& particle1, int& particle2, int& particle3, int& particle4, double& c1, double& c2, double& c3) const;
    /**
     * Set the force field parameters for a MMFF torsion term.
     *
     * @param index        the index of the torsion for which to set parameters
     * @param particle1    the index of the first particle forming the torsion
     * @param particle2    the index of the second particle forming the torsion
     * @param particle3    the index of the third particle forming the torsion
     * @param particle4    the index of the fourth particle forming the torsion
     * @param c1           the coefficient of the 1st order term, measured in kJ/mol
     * @param c2           the coefficient of the 2nd order term, measured in kJ/mol
     * @param c3           the coefficient of the 3rd order term, measured in kJ/mol
     */
    void setTorsionParameters(int index, int particle1, int particle2, int particle3, int particle4, double c1, double c2, double c3);
    /**
     * Update the per-torsion parameters in a Context to match those stored in this Force object.  This method provides
     * an efficient method to update certain parameters in an existing Context without needing to reinitialize it.
     * Simply call setTorsionParameters() to modify this object's parameters, then call updateParametersInContext()
     * to copy them over to the Context.
     *
     * The only information this method updates is the values of per-torsion parameters.  The set of particles involved
     * in a torsion cannot be changed, nor can new torsions be added.
     */
    void updateParametersInContext(Context& context);
    /**
     * Set whether this force should apply periodic boundary conditions when calculating displacements.
     * Usually this is not appropriate for bonded forces, but there are situations when it can be useful.
     */
    void setUsesPeriodicBoundaryConditions(bool periodic);
    /**
     * Returns whether or not this force makes use of periodic boundary
     * conditions.
     *
     * @returns true if force uses PBC and false otherwise
     */
    bool usesPeriodicBoundaryConditions() const;
protected:
    ForceImpl* createImpl() const;
private:
    class MMFFTorsionInfo;
    std::vector<MMFFTorsionInfo> mmffTorsions;
    bool usePeriodic;
};

/**
 * This is an internal class used to record information about a torsion.
 * @private
 */
class MMFFTorsionForce::MMFFTorsionInfo {
public:
    int particle1, particle2, particle3, particle4;
    double c[3];
    MMFFTorsionInfo() {
        particle1 = particle2 = particle3 = particle4 = -1;
        c[0] = c[1] = c[2] = 0.0;
    }
    MMFFTorsionInfo(int particle1, int particle2, int particle3, int particle4, double c1, double c2, double c3) :
            particle1(particle1), particle2(particle2), particle3(particle3), particle4(particle4) {
        c[0] = c1;
        c[1] = c2;
        c[2] = c3;
    }
};

} // namespace OpenMM

#endif /*OPENMM_MMFFTORSIONFORCE_H_*/
