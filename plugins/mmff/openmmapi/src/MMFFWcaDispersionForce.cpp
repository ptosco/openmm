/* -------------------------------------------------------------------------- *
 *                                 OpenMMMMFF                               *
 * -------------------------------------------------------------------------- *
 * This is part of the OpenMM molecular simulation toolkit originating from   *
 * Simbios, the NIH National Center for Physics-Based Simulation of           *
 * Biological Structures at Stanford, funded under the NIH Roadmap for        *
 * Medical Research, grant U54 GM072970. See https://simtk.org.               *
 *                                                                            *
 * Portions copyright (c) 2008-2009 Stanford University and the Authors.      *
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
#include "openmm/MMFFWcaDispersionForce.h"
#include "openmm/internal/MMFFWcaDispersionForceImpl.h"
#include <cmath>

using namespace OpenMM;

MMFFWcaDispersionForce::MMFFWcaDispersionForce() {
    epso      = 0.1100;
    epsh      = 0.0135;
    rmino     = 1.7025;
    rminh     = 1.3275;
    awater    = 0.033428;
    slevy     = 1.0;
    shctd     = 0.81;
    dispoff   = 0.26;
}

int MMFFWcaDispersionForce::addParticle(double radius, double epsilon) {
    parameters.push_back(WcaDispersionInfo(radius, epsilon));
    return parameters.size()-1;
}

void MMFFWcaDispersionForce::getParticleParameters(int particleIndex, double& radius, double& epsilon) const {
    radius          = parameters[particleIndex].radius;
    epsilon         = parameters[particleIndex].epsilon;
}

void MMFFWcaDispersionForce::setParticleParameters(int particleIndex, double radius, double epsilon) {
    parameters[particleIndex].radius          = radius;
    parameters[particleIndex].epsilon         = epsilon;
}

double MMFFWcaDispersionForce::getEpso() const {
    return epso;
}

double MMFFWcaDispersionForce::getEpsh() const {
    return epsh;
}

double MMFFWcaDispersionForce::getRmino() const {
    return rmino;
}

double MMFFWcaDispersionForce::getRminh() const {
    return rminh;
}

double MMFFWcaDispersionForce::getAwater() const {
    return awater;
}

double MMFFWcaDispersionForce::getShctd() const {
    return shctd;
}

double MMFFWcaDispersionForce::getDispoff() const {
    return dispoff;
}

double MMFFWcaDispersionForce::getSlevy() const {
    return slevy;
}

void MMFFWcaDispersionForce::setEpso(double inputEpso) {
    epso = inputEpso;
}

void MMFFWcaDispersionForce::setEpsh(double inputEpsh) {
    epsh = inputEpsh;
}

void MMFFWcaDispersionForce::setRmino(double inputRmino) {
    rmino = inputRmino;
}

void MMFFWcaDispersionForce::setRminh(double inputRminh) {
    rminh = inputRminh;
}

void MMFFWcaDispersionForce::setAwater(double inputAwater) {
    awater = inputAwater;
}

void MMFFWcaDispersionForce::setShctd(double inputShctd) {
    shctd = inputShctd;
}

void MMFFWcaDispersionForce::setDispoff(double inputDispoff) {
    dispoff = inputDispoff;
}

void MMFFWcaDispersionForce::setSlevy(double inputSlevy) {
    slevy = inputSlevy;
}

ForceImpl* MMFFWcaDispersionForce::createImpl() const {
    return new MMFFWcaDispersionForceImpl(*this);
}

void MMFFWcaDispersionForce::updateParametersInContext(Context& context) {
    dynamic_cast<MMFFWcaDispersionForceImpl&>(getImplInContext(context)).updateParametersInContext(getContextImpl(context));
}
