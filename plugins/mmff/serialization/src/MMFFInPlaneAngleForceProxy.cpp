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

#include "openmm/serialization/MMFFInPlaneAngleForceProxy.h"
#include "openmm/serialization/SerializationNode.h"
#include "openmm/Force.h"
#include "openmm/MMFFInPlaneAngleForce.h"
#include <sstream>

using namespace OpenMM;
using namespace std;

MMFFInPlaneAngleForceProxy::MMFFInPlaneAngleForceProxy() : SerializationProxy("MMFFInPlaneAngleForce") {
}

void MMFFInPlaneAngleForceProxy::serialize(const void* object, SerializationNode& node) const {

    node.setIntProperty("version", 3);
    const MMFFInPlaneAngleForce& force = *reinterpret_cast<const MMFFInPlaneAngleForce*>(object);

    node.setIntProperty("forceGroup", force.getForceGroup());
    node.setBoolProperty("usesPeriodic", force.usesPeriodicBoundaryConditions());
    node.setDoubleProperty("cubic",      force.getMMFFGlobalInPlaneAngleCubic());
    node.setDoubleProperty("quartic",    force.getMMFFGlobalInPlaneAngleQuartic());
    node.setDoubleProperty("pentic",     force.getMMFFGlobalInPlaneAnglePentic());
    node.setDoubleProperty("sextic",     force.getMMFFGlobalInPlaneAngleSextic());

    SerializationNode& bonds = node.createChildNode("InPlaneAngles");
    for (unsigned int ii = 0; ii < static_cast<unsigned int>(force.getNumAngles()); ii++) {
        int particle1, particle2, particle3, particle4;
        double distance, k;
        force.getAngleParameters(ii, particle1, particle2, particle3, particle4, distance, k);
        bonds.createChildNode("InPlaneAngle").setIntProperty("p1", particle1).setIntProperty("p2", particle2).setIntProperty("p3", particle3).setIntProperty("p4", particle4).setDoubleProperty("d", distance).setDoubleProperty("k", k);
    }
}

void* MMFFInPlaneAngleForceProxy::deserialize(const SerializationNode& node) const {
    int version = node.getIntProperty("version");
    if (version < 1 || version > 3)
        throw OpenMMException("Unsupported version number");
    MMFFInPlaneAngleForce* force = new MMFFInPlaneAngleForce();
    try {
        if (version > 1)
            force->setForceGroup(node.getIntProperty("forceGroup", 0));
        if (version > 2)
            force->setUsesPeriodicBoundaryConditions(node.getBoolProperty("usesPeriodic"));
        force->setMMFFGlobalInPlaneAngleCubic( node.getDoubleProperty("cubic"));
        force->setMMFFGlobalInPlaneAngleQuartic(node.getDoubleProperty("quartic"));
        force->setMMFFGlobalInPlaneAnglePentic(node.getDoubleProperty("pentic"));
        force->setMMFFGlobalInPlaneAngleSextic(node.getDoubleProperty("sextic"));

        const SerializationNode& bonds = node.getChildNode("InPlaneAngles");
        for (unsigned int ii = 0; ii < bonds.getChildren().size(); ii++) {
            const SerializationNode& bond = bonds.getChildren()[ii];
            force->addAngle(bond.getIntProperty("p1"), bond.getIntProperty("p2"), bond.getIntProperty("p3"),  bond.getIntProperty("p4"), bond.getDoubleProperty("d"), bond.getDoubleProperty("k"));
        }
    }
    catch (...) {
        delete force;
        throw;
    }
    return force;
}
