/* -------------------------------------------------------------------------- *
 *                                OpenMMMMFF                                *
 * -------------------------------------------------------------------------- *
 * This is part of the OpenMM molecular simulation toolkit originating from   *
 * Simbios, the NIH National Center for Physics-Based Simulation of           *
 * Biological Structures at Stanford, funded under the NIH Roadmap for        *
 * Medical Research, grant U54 GM072970. See https://simtk.org.               *
 *                                                                            *
 * Portions copyright (c) 2010-2016 Stanford University and the Authors.      *
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

#include "openmm/serialization/MMFFVdwForceProxy.h"
#include "openmm/serialization/SerializationNode.h"
#include "openmm/Force.h"
#include "openmm/MMFFVdwForce.h"
#include <sstream>

using namespace OpenMM;
using namespace std;

MMFFVdwForceProxy::MMFFVdwForceProxy() : SerializationProxy("MMFFVdwForce") {
}

void MMFFVdwForceProxy::serialize(const void* object, SerializationNode& node) const {
    node.setIntProperty("version", 2);
    const MMFFVdwForce& force = *reinterpret_cast<const MMFFVdwForce*>(object);

    node.setIntProperty("forceGroup", force.getForceGroup());
    node.setDoubleProperty("VdwCutoff", force.getCutoffDistance());

    node.setIntProperty("method", (int) force.getNonbondedMethod());

    SerializationNode& particles = node.createChildNode("VdwParticles");
    for (unsigned int ii = 0; ii < static_cast<unsigned int>(force.getNumParticles()); ii++) {

        double sigma, G_t_alpha, alpha_d_N;
        char vdwDA;
        force.getParticleParameters(ii, sigma, G_t_alpha, alpha_d_N, vdwDA);

        SerializationNode& particle = particles.createChildNode("Particle");
        particle.setDoubleProperty("sigma", sigma).setDoubleProperty("G_t_alpha", G_t_alpha).setDoubleProperty("alpha_d_N", alpha_d_N).setCharProperty("vdwDA", vdwDA);

        std::vector< int > exclusions;
        force.getParticleExclusions(ii,  exclusions);

        SerializationNode& particleExclusions = particle.createChildNode("ParticleExclusions");
        for (unsigned int jj = 0; jj < exclusions.size(); jj++) {
            particleExclusions.createChildNode("excl").setIntProperty("index", exclusions[jj]);
        }
    }
}

void* MMFFVdwForceProxy::deserialize(const SerializationNode& node) const {
    int version = node.getIntProperty("version");
    if (version < 1 || version > 2)
        throw OpenMMException("Unsupported version number");
    MMFFVdwForce* force = new MMFFVdwForce();
    try {
        if (version > 1)
            force->setForceGroup(node.getIntProperty("forceGroup", 0));
        force->setCutoffDistance(node.getDoubleProperty("VdwCutoff"));
        force->setNonbondedMethod((MMFFVdwForce::NonbondedMethod) node.getIntProperty("method"));

        const SerializationNode& particles = node.getChildNode("VdwParticles");
        for (unsigned int ii = 0; ii < particles.getChildren().size(); ii++) {
            const SerializationNode& particle = particles.getChildren()[ii];
            force->addParticle(particle.getDoubleProperty("sigma"), particle.getDoubleProperty("G_t_alpha"), particle.getDoubleProperty("alpha_d_N"), particle.getCharProperty("vdwDA"));

            // exclusions

            const SerializationNode& particleExclusions = particle.getChildNode("ParticleExclusions");
            std::vector< int > exclusions;
            for (unsigned int jj = 0; jj < particleExclusions.getChildren().size(); jj++) {
                exclusions.push_back(particleExclusions.getChildren()[jj].getIntProperty("index"));
            }
            force->setParticleExclusions(ii, exclusions);
        }

    }
    catch (...) {
        delete force;
        throw;
    }
    return force;
}
