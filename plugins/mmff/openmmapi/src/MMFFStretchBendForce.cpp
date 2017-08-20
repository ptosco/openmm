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
#include "openmm/MMFFStretchBendForce.h"
#include "openmm/internal/MMFFStretchBendForceImpl.h"

using namespace OpenMM;

MMFFStretchBendForce::MMFFStretchBendForce() : usePeriodic(false) {
}

int MMFFStretchBendForce::addStretchBend(int particle1, int particle2, int particle3,
                                           double lengthAB,  double lengthCB, double angle, double k1, double k2) {
    stretchBends.push_back(StretchBendInfo(particle1, particle2, particle3, lengthAB,  lengthCB, angle, k1, k2));
    return stretchBends.size()-1;
}

void MMFFStretchBendForce::getStretchBendParameters(int index, int& particle1, int& particle2, int& particle3,
                                                      double& lengthAB, double& lengthCB, double& angle, double& k1, double& k2) const {
    particle1       = stretchBends[index].particle1;
    particle2       = stretchBends[index].particle2;
    particle3       = stretchBends[index].particle3;
    lengthAB        = stretchBends[index].lengthAB;
    lengthCB        = stretchBends[index].lengthCB;
    angle           = stretchBends[index].angle;
    k1              = stretchBends[index].k1;
    k2              = stretchBends[index].k2;
}

void MMFFStretchBendForce::setStretchBendParameters(int index, int particle1, int particle2, int particle3,
                                                      double lengthAB,  double lengthCB, double angle, double k1, double k2) {
    stretchBends[index].particle1  = particle1;
    stretchBends[index].particle2  = particle2;
    stretchBends[index].particle3  = particle3;
    stretchBends[index].lengthAB   = lengthAB;
    stretchBends[index].lengthCB   = lengthCB;
    stretchBends[index].angle      = angle;
    stretchBends[index].k1         = k1;
    stretchBends[index].k2         = k2;
}

ForceImpl* MMFFStretchBendForce::createImpl() const {
    return new MMFFStretchBendForceImpl(*this);
}

void MMFFStretchBendForce::updateParametersInContext(Context& context) {
    dynamic_cast<MMFFStretchBendForceImpl&>(getImplInContext(context)).updateParametersInContext(getContextImpl(context));
}

void MMFFStretchBendForce::setUsesPeriodicBoundaryConditions(bool periodic) {
    usePeriodic = periodic;
}

bool MMFFStretchBendForce::usesPeriodicBoundaryConditions() const {
    return usePeriodic;
}
