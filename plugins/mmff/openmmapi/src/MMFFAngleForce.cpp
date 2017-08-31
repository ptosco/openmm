/* -------------------------------------------------------------------------- *
 *                                OpenMMMMFF                                *
 * -------------------------------------------------------------------------- *
 * This is part of the OpenMM molecular simulation toolkit originating from   *
 * Simbios, the NIH National Center for Physics-Based Simulation of           *
 * Biological Structures at Stanford, funded under the NIH Roadmap for        *
 * Medical Research, grant U54 GM072970. See https://simtk.org.               *
 *                                                                            *
 * Portions copyright (c) 2008-2016 Stanford University and the Authors.      *
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

#include "openmm/Force.h"
#include "openmm/OpenMMException.h"
#include "openmm/MMFFAngleForce.h"
#include "openmm/internal/MMFFAngleForceImpl.h"
#include "openmm/Units.h"

using namespace OpenMM;

MMFFAngleForce::MMFFAngleForce() : usePeriodic(false) {
    static const double MMFF_ANGLE_CUBIC_K = -0.006981317 * DegreesPerRadian;
    _globalCubicK = MMFF_ANGLE_CUBIC_K;
}

int MMFFAngleForce::addAngle(int particle1, int particle2, int particle3,  double length, double quadraticK, bool isLinear) {
    angles.push_back(AngleInfo(particle1, particle2, particle3, length, quadraticK, isLinear));
    return angles.size()-1;
}

void MMFFAngleForce::getAngleParameters(int index, int& particle1, int& particle2, int& particle3,
                                                  double& length, double& quadraticK, bool& isLinear) const {
    particle1       = angles[index].particle1;
    particle2       = angles[index].particle2;
    particle3       = angles[index].particle3;
    length          = angles[index].length;
    quadraticK      = angles[index].quadraticK;
    isLinear        = angles[index].isLinear;
}

void MMFFAngleForce::setAngleParameters(int index, int particle1, int particle2, int particle3, 
                                                  double length, double quadraticK, bool isLinear) {
    angles[index].particle1  = particle1;
    angles[index].particle2  = particle2;
    angles[index].particle3  = particle3;
    angles[index].length     = length;
    angles[index].quadraticK = quadraticK;
    angles[index].isLinear   = isLinear;
}

double MMFFAngleForce::getMMFFGlobalAngleCubic() const {
    return _globalCubicK;
}

void MMFFAngleForce::setMMFFGlobalAngleCubic(double cubicK) {
    _globalCubicK           = cubicK;
}

ForceImpl* MMFFAngleForce::createImpl() const {
    return new MMFFAngleForceImpl(*this);
}

void MMFFAngleForce::updateParametersInContext(Context& context) {
    dynamic_cast<MMFFAngleForceImpl&>(getImplInContext(context)).updateParametersInContext(getContextImpl(context));
}

void MMFFAngleForce::setUsesPeriodicBoundaryConditions(bool periodic) {
    usePeriodic = periodic;
}

bool MMFFAngleForce::usesPeriodicBoundaryConditions() const {
    return usePeriodic;
}
