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
#include "openmm/MMFFVdwForce.h"
#include "openmm/internal/MMFFVdwForceImpl.h"

using namespace OpenMM;
using std::string;
using std::vector;

MMFFVdwForce::MMFFVdwForce() : nonbondedMethod(NoCutoff), cutoff(1.0e+10), useDispersionCorrection(true) {
}

int MMFFVdwForce::addParticle(double sigma, double G_t_alpha, double alpha_d_N, char vdwDA) {
    parameters.push_back(VdwInfo(sigma, G_t_alpha, alpha_d_N, vdwDA));
    return parameters.size()-1;
}

void MMFFVdwForce::getParticleParameters(int particleIndex, double& sigma,
                                        double& G_t_alpha, double& alpha_d_N, char& vdwDA) const {
    sigma           = parameters[particleIndex].sigma;
    G_t_alpha       = parameters[particleIndex].G_t_alpha;
    alpha_d_N       = parameters[particleIndex].alpha_d_N;
    vdwDA           = parameters[particleIndex].vdwDA;
}

void MMFFVdwForce::setParticleParameters(int particleIndex, double sigma,
                                        double G_t_alpha, double alpha_d_N, char vdwDA) {
    parameters[particleIndex].sigma           = sigma;
    parameters[particleIndex].G_t_alpha       = G_t_alpha;
    parameters[particleIndex].alpha_d_N       = alpha_d_N;
    parameters[particleIndex].vdwDA           = vdwDA;
}

void MMFFVdwForce::setParticleExclusions(int particleIndex, const std::vector< int >& inputExclusions) {

   if (exclusions.size() < parameters.size()) {
       exclusions.resize(parameters.size());
   }
   if (static_cast<int>(exclusions.size()) < particleIndex) {
       exclusions.resize(particleIndex + 10);
   }
   for (unsigned int ii = 0; ii < inputExclusions.size(); ii++) {
       exclusions[particleIndex].push_back(inputExclusions[ii]);
   }
}

void MMFFVdwForce::getParticleExclusions(int particleIndex, std::vector< int >& outputExclusions) const {

   if (particleIndex < static_cast<int>(exclusions.size())) {
       outputExclusions.resize(exclusions[particleIndex].size());
       for (unsigned int ii = 0; ii < exclusions[particleIndex].size(); ii++) {
           outputExclusions[ii] = exclusions[particleIndex][ii];
       }
   }

}

double MMFFVdwForce::getCutoffDistance() const {
    return cutoff;
}

void MMFFVdwForce::setCutoffDistance(double inputCutoff) {
    cutoff = inputCutoff;
}

void MMFFVdwForce::setCutoff(double inputCutoff) {
    setCutoffDistance(inputCutoff);
}

double MMFFVdwForce::getCutoff() const {
    return getCutoffDistance();
}

MMFFVdwForce::NonbondedMethod MMFFVdwForce::getNonbondedMethod() const {
    return nonbondedMethod;
}

void MMFFVdwForce::setNonbondedMethod(NonbondedMethod method) {
    if (method < 0 || method > 1)
        throw OpenMMException("MMFFVdwForce: Illegal value for nonbonded method");
    nonbondedMethod = method;
}

ForceImpl* MMFFVdwForce::createImpl() const {
    return new MMFFVdwForceImpl(*this);
}

void MMFFVdwForce::updateParametersInContext(Context& context) {
    dynamic_cast<MMFFVdwForceImpl&>(getImplInContext(context)).updateParametersInContext(getContextImpl(context));
}
