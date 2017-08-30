/* -------------------------------------------------------------------------- *
 *                               OpenMMMMFF                                 *
 * -------------------------------------------------------------------------- *
 * This is part of the OpenMM molecular simulation toolkit originating from   *
 * Simbios, the NIH National Center for Physics-Based Simulation of           *
 * Biological Structures at Stanford, funded under the NIH Roadmap for        *
 * Medical Research, grant U54 GM072970. See https://simtk.org.               *
 *                                                                            *
 * Portions copyright (c) 2008 Stanford University and the Authors.           *
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

#include "openmm/internal/ContextImpl.h"
#include "openmm/internal/MMFFMultipoleForceImpl.h"
#include "openmm/mmffKernels.h"
#include <cmath>
#include <stdio.h>

using namespace OpenMM;

using std::vector;

bool MMFFMultipoleForceImpl::initializedCovalentDegrees = false;
int MMFFMultipoleForceImpl::CovalentDegrees[]           = { 1,2,3,4,0,1,2,3};

MMFFMultipoleForceImpl::MMFFMultipoleForceImpl(const MMFFMultipoleForce& owner) : owner(owner) {
}

MMFFMultipoleForceImpl::~MMFFMultipoleForceImpl() {
}

void MMFFMultipoleForceImpl::initialize(ContextImpl& context) {

    const System& system = context.getSystem();
    int numParticles = system.getNumParticles();
    if (owner.getNumMultipoles() != numParticles)
        throw OpenMMException("MMFFMultipoleForce must have exactly as many particles as the System it belongs to.");

    // check cutoff < 0.5*boxSize

    if (owner.getNonbondedMethod() == MMFFMultipoleForce::PME) {
        Vec3 boxVectors[3];
        system.getDefaultPeriodicBoxVectors(boxVectors[0], boxVectors[1], boxVectors[2]);
        double cutoff = owner.getCutoffDistance();
        if (cutoff > 0.5*boxVectors[0][0] || cutoff > 0.5*boxVectors[1][1] || cutoff > 0.5*boxVectors[2][2])
            throw OpenMMException("MMFFMultipoleForce: The cutoff distance cannot be greater than half the periodic box size.");
    }

    double quadrupoleValidationTolerance = 1.0e-05;
    for (int ii = 0; ii < numParticles; ii++) {

        int axisType, multipoleAtomZ, multipoleAtomX, multipoleAtomY;
        double charge, thole, dampingFactor, polarity ;
        std::vector<double> molecularDipole;
        std::vector<double> molecularQuadrupole;

        owner.getMultipoleParameters(ii, charge, molecularDipole, molecularQuadrupole, axisType, multipoleAtomZ, multipoleAtomX, multipoleAtomY,
                                     thole, dampingFactor, polarity);

       // check quadrupole is traceless and symmetric

       double trace = fabs(molecularQuadrupole[0] + molecularQuadrupole[4] + molecularQuadrupole[8]);
       if (trace > quadrupoleValidationTolerance) {
             std::stringstream buffer;
             buffer << "MMFFMultipoleForce: qudarupole for particle=" << ii;
             buffer << " has nonzero trace: " << trace << "; MMFF plugin assumes traceless quadrupole.";
             throw OpenMMException(buffer.str());
       }
       if (fabs(molecularQuadrupole[1] - molecularQuadrupole[3]) > quadrupoleValidationTolerance ) {
             std::stringstream buffer;
             buffer << "MMFFMultipoleForce: XY and YX components of quadrupole for particle=" << ii;
             buffer << "  are not equal: [" << molecularQuadrupole[1] << " " << molecularQuadrupole[3] << "];";
             buffer << " MMFF plugin assumes symmetric quadrupole tensor.";
             throw OpenMMException(buffer.str());
       }

       if (fabs(molecularQuadrupole[2] - molecularQuadrupole[6]) > quadrupoleValidationTolerance ) {
             std::stringstream buffer;
             buffer << "MMFFMultipoleForce: XZ and ZX components of quadrupole for particle=" << ii;
             buffer << "  are not equal: [" << molecularQuadrupole[2] << " " << molecularQuadrupole[6] << "];";
             buffer << " MMFF plugin assumes symmetric quadrupole tensor.";
             throw OpenMMException(buffer.str());
       }

       if (fabs(molecularQuadrupole[5] - molecularQuadrupole[7]) > quadrupoleValidationTolerance ) {
             std::stringstream buffer;
             buffer << "MMFFMultipoleForce: YZ and ZY components of quadrupole for particle=" << ii;
             buffer << "  are not equal: [" << molecularQuadrupole[5] << " " << molecularQuadrupole[7] << "];";
             buffer << " MMFF plugin assumes symmetric quadrupole tensor.";
             throw OpenMMException(buffer.str());
       }

       // only 'Z-then-X', 'Bisector', Z-Bisect, ThreeFold  currently handled

        if (axisType != MMFFMultipoleForce::ZThenX     && axisType != MMFFMultipoleForce::Bisector &&
            axisType != MMFFMultipoleForce::ZBisect    && axisType != MMFFMultipoleForce::ThreeFold &&
            axisType != MMFFMultipoleForce::ZOnly      && axisType != MMFFMultipoleForce::NoAxisType) {
             std::stringstream buffer;
             buffer << "MMFFMultipoleForce: axis type=" << axisType;
             buffer << " not currently handled - only axisTypes[ ";
             buffer << MMFFMultipoleForce::ZThenX   << ", " << MMFFMultipoleForce::Bisector  << ", ";
             buffer << MMFFMultipoleForce::ZBisect  << ", " << MMFFMultipoleForce::ThreeFold << ", ";
             buffer << MMFFMultipoleForce::NoAxisType;
             buffer << "] (ZThenX, Bisector, Z-Bisect, ThreeFold, NoAxisType) currently handled .";
             throw OpenMMException(buffer.str());
        }
        if (axisType != MMFFMultipoleForce::NoAxisType && (multipoleAtomZ < 0 || multipoleAtomZ >= numParticles)) {
            std::stringstream buffer;
            buffer << "MMFFMultipoleForce: invalid z axis particle: " << multipoleAtomZ;
            throw OpenMMException(buffer.str());
        }
        if (axisType != MMFFMultipoleForce::NoAxisType && axisType != MMFFMultipoleForce::ZOnly &&
                (multipoleAtomX < 0 || multipoleAtomX >= numParticles)) {
            std::stringstream buffer;
            buffer << "MMFFMultipoleForce: invalid x axis particle: " << multipoleAtomX;
            throw OpenMMException(buffer.str());
        }
        if ((axisType == MMFFMultipoleForce::ZBisect || axisType == MMFFMultipoleForce::ThreeFold) &&
                (multipoleAtomY < 0 || multipoleAtomY >= numParticles)) {
            std::stringstream buffer;
            buffer << "MMFFMultipoleForce: invalid y axis particle: " << multipoleAtomY;
            throw OpenMMException(buffer.str());
        }
    }
    kernel = context.getPlatform().createKernel(CalcMMFFMultipoleForceKernel::Name(), context);
    kernel.getAs<CalcMMFFMultipoleForceKernel>().initialize(context.getSystem(), owner);
}

double MMFFMultipoleForceImpl::calcForcesAndEnergy(ContextImpl& context, bool includeForces, bool includeEnergy, int groups) {
    if ((groups&(1<<owner.getForceGroup())) != 0)
        return kernel.getAs<CalcMMFFMultipoleForceKernel>().execute(context, includeForces, includeEnergy);
    return 0.0;
}

std::vector<std::string> MMFFMultipoleForceImpl::getKernelNames() {
    std::vector<std::string> names;
    names.push_back(CalcMMFFMultipoleForceKernel::Name());
    return names;
}

const int* MMFFMultipoleForceImpl::getCovalentDegrees() {
    if (!initializedCovalentDegrees) {
        initializedCovalentDegrees                                      = true;
        CovalentDegrees[MMFFMultipoleForce::Covalent12]               = 1;
        CovalentDegrees[MMFFMultipoleForce::Covalent13]               = 2;
        CovalentDegrees[MMFFMultipoleForce::Covalent14]               = 3;
        CovalentDegrees[MMFFMultipoleForce::Covalent15]               = 4;
        CovalentDegrees[MMFFMultipoleForce::PolarizationCovalent11]   = 0;
        CovalentDegrees[MMFFMultipoleForce::PolarizationCovalent12]   = 1;
        CovalentDegrees[MMFFMultipoleForce::PolarizationCovalent13]   = 2;
        CovalentDegrees[MMFFMultipoleForce::PolarizationCovalent14]   = 3;
    }
    return CovalentDegrees;
}

void MMFFMultipoleForceImpl::getCovalentRange(const MMFFMultipoleForce& force, int atomIndex, const std::vector<MMFFMultipoleForce::CovalentType>& lists,
                                                int* minCovalentIndex, int* maxCovalentIndex) {

    *minCovalentIndex =  999999999;
    *maxCovalentIndex = -999999999;
    for (unsigned int kk = 0; kk < lists.size(); kk++) {
        MMFFMultipoleForce::CovalentType jj = lists[kk];
        std::vector<int> covalentList;
        force.getCovalentMap(atomIndex, jj, covalentList);
        for (unsigned int ii = 0; ii < covalentList.size(); ii++) {
            if (*minCovalentIndex > covalentList[ii]) {
               *minCovalentIndex = covalentList[ii];
            }
            if (*maxCovalentIndex < covalentList[ii]) {
               *maxCovalentIndex = covalentList[ii];
            }
        }
    }
    return;
}

void MMFFMultipoleForceImpl::getCovalentDegree(const MMFFMultipoleForce& force, std::vector<int>& covalentDegree) {
    covalentDegree.resize(MMFFMultipoleForce::CovalentEnd);
    const int* CovalentDegrees = MMFFMultipoleForceImpl::getCovalentDegrees();
    for (unsigned int kk = 0; kk < MMFFMultipoleForce::CovalentEnd; kk++) {
        covalentDegree[kk] = CovalentDegrees[kk];
    }
    return;
}

void MMFFMultipoleForceImpl::getLabFramePermanentDipoles(ContextImpl& context, vector<Vec3>& dipoles) {
    kernel.getAs<CalcMMFFMultipoleForceKernel>().getLabFramePermanentDipoles(context, dipoles);
}

void MMFFMultipoleForceImpl::getInducedDipoles(ContextImpl& context, vector<Vec3>& dipoles) {
    kernel.getAs<CalcMMFFMultipoleForceKernel>().getInducedDipoles(context, dipoles);
}

void MMFFMultipoleForceImpl::getTotalDipoles(ContextImpl& context, vector<Vec3>& dipoles) {
    kernel.getAs<CalcMMFFMultipoleForceKernel>().getTotalDipoles(context, dipoles);
}

void MMFFMultipoleForceImpl::getElectrostaticPotential(ContextImpl& context, const std::vector< Vec3 >& inputGrid,
                                                         std::vector< double >& outputElectrostaticPotential) {
    kernel.getAs<CalcMMFFMultipoleForceKernel>().getElectrostaticPotential(context, inputGrid, outputElectrostaticPotential);
}

void MMFFMultipoleForceImpl::getSystemMultipoleMoments(ContextImpl& context, std::vector< double >& outputMultipoleMoments) {
    kernel.getAs<CalcMMFFMultipoleForceKernel>().getSystemMultipoleMoments(context, outputMultipoleMoments);
}

void MMFFMultipoleForceImpl::updateParametersInContext(ContextImpl& context) {
    kernel.getAs<CalcMMFFMultipoleForceKernel>().copyParametersToContext(context, owner);
    context.systemChanged();
}

void MMFFMultipoleForceImpl::getPMEParameters(double& alpha, int& nx, int& ny, int& nz) const {
    kernel.getAs<CalcMMFFMultipoleForceKernel>().getPMEParameters(alpha, nx, ny, nz);
}
