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
#include "openmm/OpenMMException.h"
#include "openmm/MMFFTorsionForce.h"
#include "openmm/internal/AssertionUtilities.h"
#include "openmm/internal/MMFFTorsionForceImpl.h"

using namespace OpenMM;

MMFFTorsionForce::MMFFTorsionForce() : usePeriodic(false) {
}

int MMFFTorsionForce::addTorsion(int particle1, int particle2, int particle3, int particle4, double c1, double c2, double c3) {
    mmffTorsions.push_back(MMFFTorsionInfo(particle1, particle2, particle3, particle4, c1, c2, c3));
    return mmffTorsions.size()-1;
}

void MMFFTorsionForce::getTorsionParameters(int index, int& particle1, int& particle2, int& particle3, int& particle4, double& c1, double& c2, double& c3) const {
    ASSERT_VALID_INDEX(index, mmffTorsions);
    particle1 = mmffTorsions[index].particle1;
    particle2 = mmffTorsions[index].particle2;
    particle3 = mmffTorsions[index].particle3;
    particle4 = mmffTorsions[index].particle4;
    c1 = mmffTorsions[index].c[0];
    c2 = mmffTorsions[index].c[1];
    c3 = mmffTorsions[index].c[2];
}

void MMFFTorsionForce::setTorsionParameters(int index, int particle1, int particle2, int particle3, int particle4, double c1, double c2, double c3) {
    ASSERT_VALID_INDEX(index, mmffTorsions);
    mmffTorsions[index].particle1 = particle1;
    mmffTorsions[index].particle2 = particle2;
    mmffTorsions[index].particle3 = particle3;
    mmffTorsions[index].particle4 = particle4;
    mmffTorsions[index].c[0] = c1;
    mmffTorsions[index].c[1] = c2;
    mmffTorsions[index].c[2] = c3;
}

ForceImpl* MMFFTorsionForce::createImpl() const {
    return new MMFFTorsionForceImpl(*this);
}

void MMFFTorsionForce::updateParametersInContext(Context& context) {
    dynamic_cast<MMFFTorsionForceImpl&>(getImplInContext(context)).updateParametersInContext(getContextImpl(context));
}

void MMFFTorsionForce::setUsesPeriodicBoundaryConditions(bool periodic) {
    usePeriodic = periodic;
}

bool MMFFTorsionForce::usesPeriodicBoundaryConditions() const {
    return usePeriodic;
}
