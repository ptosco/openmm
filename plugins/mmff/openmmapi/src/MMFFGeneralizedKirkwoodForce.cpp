/* -------------------------------------------------------------------------- *
 *                                   OpenMM                                   *
 * -------------------------------------------------------------------------- *
 * This is part of the OpenMM molecular simulation toolkit originating from   *
 * Simbios, the NIH National Center for Physics-Based Simulation of           *
 * Biological Structures at Stanford, funded under the NIH Roadmap for        *
 * Medical Research, grant U54 GM072970. See https://simtk.org.               *
 *                                                                            *
 * Portions copyright (c) 2008-2009 Stanford University and the Authors.      *
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
#include "openmm/MMFFGeneralizedKirkwoodForce.h"
#include "openmm/internal/MMFFGeneralizedKirkwoodForceImpl.h"

using namespace OpenMM;

MMFFGeneralizedKirkwoodForce::MMFFGeneralizedKirkwoodForce() : solventDielectric(78.3), soluteDielectric(1.0), dielectricOffset(0.009), includeCavityTerm(1), probeRadius(0.14) {

     surfaceAreaFactor = -6.0* 3.1415926535*0.0216*1000.0*0.4184;
}

int MMFFGeneralizedKirkwoodForce::addParticle(double charge, double radius, double scalingFactor) {
    particles.push_back(ParticleInfo(charge, radius, scalingFactor));
    return particles.size()-1;
}

void MMFFGeneralizedKirkwoodForce::getParticleParameters(int index, double& charge, double& radius, double& scalingFactor) const {
    charge = particles[index].charge;
    radius = particles[index].radius;
    scalingFactor = particles[index].scalingFactor;
}

void MMFFGeneralizedKirkwoodForce::setParticleParameters(int index, double charge, double radius, double scalingFactor) {
    particles[index].charge = charge;
    particles[index].radius = radius;
    particles[index].scalingFactor = scalingFactor;
}
/*
double MMFFGeneralizedKirkwoodForce::getDielectricOffset() const {
    return dielectricOffset;
}

void MMFFGeneralizedKirkwoodForce::setDielectricOffset(double inputDielectricOffset) {
    dielectricOffset = inputDielectricOffset;
} */

int MMFFGeneralizedKirkwoodForce::getIncludeCavityTerm() const {
    return includeCavityTerm;
}

void MMFFGeneralizedKirkwoodForce::setIncludeCavityTerm(int inputIncludeCavityTerm) {
    includeCavityTerm = inputIncludeCavityTerm;
}

double MMFFGeneralizedKirkwoodForce::getProbeRadius() const {
    return probeRadius;
}

void MMFFGeneralizedKirkwoodForce::setProbeRadius(double inputProbeRadius) {
    probeRadius = inputProbeRadius;
}

double MMFFGeneralizedKirkwoodForce::getSurfaceAreaFactor() const {
    return surfaceAreaFactor;
}

void MMFFGeneralizedKirkwoodForce::setSurfaceAreaFactor(double inputSurfaceAreaFactor) {
    surfaceAreaFactor = inputSurfaceAreaFactor;
}

ForceImpl* MMFFGeneralizedKirkwoodForce::createImpl() const {
    return new MMFFGeneralizedKirkwoodForceImpl(*this);
}

void MMFFGeneralizedKirkwoodForce::updateParametersInContext(Context& context) {
    dynamic_cast<MMFFGeneralizedKirkwoodForceImpl&>(getImplInContext(context)).updateParametersInContext(getContextImpl(context));
}
