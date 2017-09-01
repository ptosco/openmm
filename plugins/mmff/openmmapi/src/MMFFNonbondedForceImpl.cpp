/* -------------------------------------------------------------------------- *
 *                                   OpenMM                                   *
 * -------------------------------------------------------------------------- *
 * This is part of the OpenMM molecular simulation toolkit originating from   *
 * Simbios, the NIH National Center for Physics-Based Simulation of           *
 * Biological Structures at Stanford, funded under the NIH Roadmap for        *
 * Medical Research, grant U54 GM072970. See https://simtk.org.               *
 *                                                                            *
 * Portions copyright (c) 2008-2015 Stanford University and the Authors.      *
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

#ifdef WIN32
  #define _USE_MATH_DEFINES // Needed to get M_PI
#endif
#include "openmm/OpenMMException.h"
#include "openmm/internal/ContextImpl.h"
#include "openmm/internal/MMFFNonbondedForceImpl.h"
#include "openmm/kernels.h"
#include <cmath>
#include <map>
#include <sstream>
#include <algorithm>

using namespace OpenMM;
using namespace std;

MMFFNonbondedForceImpl::MMFFNonbondedForceImpl(const MMFFNonbondedForce& owner) : owner(owner) {
}

MMFFNonbondedForceImpl::~MMFFNonbondedForceImpl() {
}

void MMFFNonbondedForceImpl::initialize(ContextImpl& context) {
    kernel = context.getPlatform().createKernel(CalcMMFFNonbondedForceKernel::Name(), context);

    // Check for errors in the specification of exceptions.

    const System& system = context.getSystem();
    if (owner.getNumParticles() != system.getNumParticles())
        throw OpenMMException("MMFFNonbondedForce must have exactly as many particles as the System it belongs to.");
    if (owner.getUseSwitchingFunction()) {
        if (owner.getSwitchingDistance() < 0 || owner.getSwitchingDistance() >= owner.getCutoffDistance())
            throw OpenMMException("MMFFNonbondedForce: Switching distance must satisfy 0 <= r_switch < r_cutoff");
    }
    vector<set<int> > exceptions(owner.getNumParticles());
    for (int i = 0; i < owner.getNumExceptions(); i++) {
        int particle1, particle2;
        double chargeProd, sigma, epsilon;
        owner.getExceptionParameters(i, particle1, particle2, chargeProd, sigma, epsilon);
        if (particle1 < 0 || particle1 >= owner.getNumParticles()) {
            stringstream msg;
            msg << "MMFFNonbondedForce: Illegal particle index for an exception: ";
            msg << particle1;
            throw OpenMMException(msg.str());
        }
        if (particle2 < 0 || particle2 >= owner.getNumParticles()) {
            stringstream msg;
            msg << "MMFFNonbondedForce: Illegal particle index for an exception: ";
            msg << particle2;
            throw OpenMMException(msg.str());
        }
        if (exceptions[particle1].count(particle2) > 0 || exceptions[particle2].count(particle1) > 0) {
            stringstream msg;
            msg << "MMFFNonbondedForce: Multiple exceptions are specified for particles ";
            msg << particle1;
            msg << " and ";
            msg << particle2;
            throw OpenMMException(msg.str());
        }
        exceptions[particle1].insert(particle2);
        exceptions[particle2].insert(particle1);
    }
    if (owner.getNonbondedMethod() != MMFFNonbondedForce::NoCutoff && owner.getNonbondedMethod() != MMFFNonbondedForce::CutoffNonPeriodic) {
        Vec3 boxVectors[3];
        system.getDefaultPeriodicBoxVectors(boxVectors[0], boxVectors[1], boxVectors[2]);
        double cutoff = owner.getCutoffDistance();
        if (cutoff > 0.5*boxVectors[0][0] || cutoff > 0.5*boxVectors[1][1] || cutoff > 0.5*boxVectors[2][2])
            throw OpenMMException("MMFFNonbondedForce: The cutoff distance cannot be greater than half the periodic box size.");
        if (owner.getNonbondedMethod() == MMFFNonbondedForce::Ewald && (boxVectors[1][0] != 0.0 || boxVectors[2][0] != 0.0 || boxVectors[2][1] != 0))
            throw OpenMMException("MMFFNonbondedForce: Ewald is not supported with non-rectangular boxes.  Use PME instead.");
    }
    kernel.getAs<CalcMMFFNonbondedForceKernel>().initialize(context.getSystem(), owner);
}

double MMFFNonbondedForceImpl::calcForcesAndEnergy(ContextImpl& context, bool includeForces, bool includeEnergy, int groups) {
    bool includeDirect = ((groups&(1<<owner.getForceGroup())) != 0);
    bool includeReciprocal = includeDirect;
    if (owner.getReciprocalSpaceForceGroup() >= 0)
        includeReciprocal = ((groups&(1<<owner.getReciprocalSpaceForceGroup())) != 0);
    return kernel.getAs<CalcMMFFNonbondedForceKernel>().execute(context, includeForces, includeEnergy, includeDirect, includeReciprocal);
}

std::vector<std::string> MMFFNonbondedForceImpl::getKernelNames() {
    std::vector<std::string> names;
    names.push_back(CalcMMFFNonbondedForceKernel::Name());
    return names;
}

class MMFFNonbondedForceImpl::ErrorFunction {
public:
    virtual double getValue(int arg) const = 0;
};

class MMFFNonbondedForceImpl::EwaldErrorFunction : public ErrorFunction {
public:
    EwaldErrorFunction(double width, double alpha, double target) : width(width), alpha(alpha), target(target) {
    }
    double getValue(int arg) const {
        double temp = arg*M_PI/(width*alpha);
        return target-0.05*sqrt(width*alpha)*arg*exp(-temp*temp);
    }
private:
    double width, alpha, target;
};

void MMFFNonbondedForceImpl::calcEwaldParameters(const System& system, const MMFFNonbondedForce& force, double& alpha, int& kmaxx, int& kmaxy, int& kmaxz) {
    Vec3 boxVectors[3];
    system.getDefaultPeriodicBoxVectors(boxVectors[0], boxVectors[1], boxVectors[2]);
    double tol = force.getEwaldErrorTolerance();
    alpha = (1.0/force.getCutoffDistance())*std::sqrt(-log(2.0*tol));
    kmaxx = findZero(EwaldErrorFunction(boxVectors[0][0], alpha, tol), 10);
    kmaxy = findZero(EwaldErrorFunction(boxVectors[1][1], alpha, tol), 10);
    kmaxz = findZero(EwaldErrorFunction(boxVectors[2][2], alpha, tol), 10);
    if (kmaxx%2 == 0)
        kmaxx++;
    if (kmaxy%2 == 0)
        kmaxy++;
    if (kmaxz%2 == 0)
        kmaxz++;
}

void MMFFNonbondedForceImpl::calcPMEParameters(const System& system, const MMFFNonbondedForce& force, double& alpha, int& xsize, int& ysize, int& zsize, bool lj) {
    force.getPMEParameters(alpha, xsize, ysize, zsize);
    if (alpha == 0.0) {
        Vec3 boxVectors[3];
        system.getDefaultPeriodicBoxVectors(boxVectors[0], boxVectors[1], boxVectors[2]);
        double tol = force.getEwaldErrorTolerance();
        alpha = (1.0/force.getCutoffDistance())*std::sqrt(-log(2.0*tol));
        if (lj) {
            xsize = (int) ceil(alpha*boxVectors[0][0]/(3*pow(tol, 0.2)));
            ysize = (int) ceil(alpha*boxVectors[1][1]/(3*pow(tol, 0.2)));
            zsize = (int) ceil(alpha*boxVectors[2][2]/(3*pow(tol, 0.2)));
        }
        else {
            xsize = (int) ceil(2*alpha*boxVectors[0][0]/(3*pow(tol, 0.2)));
            ysize = (int) ceil(2*alpha*boxVectors[1][1]/(3*pow(tol, 0.2)));
            zsize = (int) ceil(2*alpha*boxVectors[2][2]/(3*pow(tol, 0.2)));
        }
        xsize = max(xsize, 6);
        ysize = max(ysize, 6);
        zsize = max(zsize, 6);
    }
}

int MMFFNonbondedForceImpl::findZero(const MMFFNonbondedForceImpl::ErrorFunction& f, int initialGuess) {
    int arg = initialGuess;
    double value = f.getValue(arg);
    if (value > 0.0) {
        while (value > 0.0 && arg > 0)
            value = f.getValue(--arg);
        return arg+1;
    }
    while (value < 0.0)
        value = f.getValue(++arg);
    return arg;
}

double MMFFNonbondedForceImpl::calcDispersionCorrection(const System& system, const MMFFNonbondedForce& force) {
    // MMFF vdW dispersion correction implemented by LPW
    // There is no dispersion correction if PBC is off or the cutoff is set to the default value of ten billion (MMFFNonbondedForce.cpp)
    if (force.getNonbondedMethod() == MMFFNonbondedForce::NoCutoff || force.getNonbondedMethod() == MMFFNonbondedForce::CutoffNonPeriodic)
        return 0.0;
    
    // Identify all particle classes (defined by MMFFVdwParams), and count the number of
    // particles in each class.

    map<MMFFVdwParams, int> classCounts;
    for (int i = 0; i < force.getNumParticles(); i++) {
        MMFFVdwParams key;
        // charge is unused
        double charge;
        // Get the sigma, G*alpha, alpha/N and vdwDA parameters.
        force.getParticleParameters(i, charge, key.sigma, key.G_t_alpha, key.alpha_d_N, key.vdwDA);
        map<MMFFVdwParams, int>::iterator entry = classCounts.find(key);
        if (entry == classCounts.end())
            classCounts[key] = 1;
        else
            entry->second++;
    }

    // Compute the VdW tapering coefficients.
    double cutoff = force.getCutoffDistance();
    double vdwTaper = 0.90; // vdwTaper is a scaling factor, it is not a distance.
    double c0 = 0.0;
    double c1 = 0.0;
    double c2 = 0.0;
    double c3 = 0.0;
    double c4 = 0.0;
    double c5 = 0.0;

    double vdwCut = cutoff;
    double vdwTaperCut = vdwTaper*cutoff;

    double vdwCut2 = vdwCut*vdwCut;
    double vdwCut3 = vdwCut2*vdwCut;
    double vdwCut4 = vdwCut2*vdwCut2;
    double vdwCut5 = vdwCut2*vdwCut3;
    double vdwCut6 = vdwCut3*vdwCut3;
    double vdwCut7 = vdwCut3*vdwCut4;

    double vdwTaperCut2 = vdwTaperCut*vdwTaperCut;
    double vdwTaperCut3 = vdwTaperCut2*vdwTaperCut;
    double vdwTaperCut4 = vdwTaperCut2*vdwTaperCut2;
    double vdwTaperCut5 = vdwTaperCut2*vdwTaperCut3;
    double vdwTaperCut6 = vdwTaperCut3*vdwTaperCut3;
    double vdwTaperCut7 = vdwTaperCut3*vdwTaperCut4;

    // get 5th degree multiplicative switching function coefficients;

    double denom = 1.0 / (vdwCut - vdwTaperCut);
    double denom2 = denom*denom;
    denom = denom * denom2*denom2;

    c0 = vdwCut * vdwCut2 * (vdwCut2 - 5.0 * vdwCut * vdwTaperCut + 10.0 * vdwTaperCut2) * denom;
    c1 = -30.0 * vdwCut2 * vdwTaperCut2*denom;
    c2 = 30.0 * (vdwCut2 * vdwTaperCut + vdwCut * vdwTaperCut2) * denom;
    c3 = -10.0 * (vdwCut2 + 4.0 * vdwCut * vdwTaperCut + vdwTaperCut2) * denom;
    c4 = 15.0 * (vdwCut + vdwTaperCut) * denom;
    c5 = -6.0 * denom;

    // Loop over all pairs of classes to compute the coefficient.
    // Copied over from TINKER - numerical integration.
    double range = 20.0;
    double cut = vdwTaperCut; // This is where tapering BEGINS
    double off = vdwCut; // This is where tapering ENDS
    int nstep = 200;
    int ndelta = int(double(nstep) * (range - cut));
    double rdelta = (range - cut) / double(ndelta);
    double offset = cut - 0.5 * rdelta;
    double dhal = 0.07; // This magic number also appears in mmffVdwForce.cu
    double ghal = 0.12; // This magic number also appears in mmffVdwForce.cu
    double elrc = 0.0; // This number is incremented and passed out at the end
    double e = 0.0;
    double sigma, epsilon; // The pairwise sigma and epsilon parameters.
    int i = 0, k = 0; // Loop counters.
    static const double B = 0.2;
    static const double Beta = 12.0;
    static const double C4 = 7.5797344e-4;
    static const double DARAD = 0.8;
    static const double DAEPS = 0.5;

    // Double loop over different atom types.
    for (auto& class1 : classCounts) {
        k = 0;
        for (auto& class2 : classCounts) { 
            // MMFF combining rules, copied over from the CUDA code.
            bool haveDAPair = (class1.first.vdwDA == 'D' && class2.first.vdwDA == 'A')
                || (class1.first.vdwDA == 'A' && class2.first.vdwDA == 'D');
            bool haveDonor = (class1.first.vdwDA == 'D' || class2.first.vdwDA == 'D');
            double gamma = (class1.first.sigma - class2.first.sigma) / (class1.first.sigma + class2.first.sigma);
            sigma = 0.5 * (class1.first.sigma + class2.first.sigma) * (1.0 + (haveDonor
                ? 0.0 : B * (1.0 - exp(-Beta * gamma * gamma))));
            double sigmaSq = sigma * sigma;
            epsilon = C4 * class1.first.G_t_alpha * class2.first.G_t_alpha
                / ((sqrt(class1.first.alpha_d_N) + sqrt(class2.first.alpha_d_N))
                * sigmaSq * sigmaSq * sigmaSq);
            if (haveDAPair) {
              sigma *= DARAD;
              epsilon *= DAEPS;
            }
            int count = class1.second * class2.second;
            // Below is an exact copy of stuff from the previous block.
            double rv = sigma;
            double termik = 2.0 * M_PI * count; // termik is equivalent to 2 * pi * count.
            double rv2 = rv * rv;
            double rv6 = rv2 * rv2 * rv2;
            double rv7 = rv6 * rv;
            double etot = 0.0;
            double r2 = 0.0;
            for (int j = 1; j <= ndelta; j++) {
                double r = offset + double(j) * rdelta;
                r2 = r*r;
                double r3 = r2 * r;
                double r6 = r3 * r3;
                double r7 = r6 * r;
                // The following is for buffered 14-7 only.
                /*
                double rho = r/rv;
                double term1 = pow(((dhal + 1.0) / (dhal + rho)),7);
                double term2 = ((ghal + 1.0) / (ghal + pow(rho,7))) - 2.0;
                e = epsilon * term1 * term2;
                */
                double rho = r7 + ghal*rv7;
                double tau = (dhal + 1.0) / (r + dhal * rv);
                double tau7 = pow(tau, 7);
                e = epsilon * rv7 * tau7 * ((ghal + 1.0) * rv7 / rho - 2.0);
                double taper = 0.0;
                if (r < off) {
                    double r4 = r2 * r2;
                    double r5 = r2 * r3;
                    taper = c5 * r5 + c4 * r4 + c3 * r3 + c2 * r2 + c1 * r + c0;
                    e = e * (1.0 - taper);
                }
                etot = etot + e * rdelta * r2;
            }
            elrc = elrc + termik * etot;
            k++;
        }
        i++;
    }
    return elrc;
}

void MMFFNonbondedForceImpl::updateParametersInContext(ContextImpl& context) {
    kernel.getAs<CalcMMFFNonbondedForceKernel>().copyParametersToContext(context, owner);
    context.systemChanged();
}

void MMFFNonbondedForceImpl::getPMEParameters(double& alpha, int& nx, int& ny, int& nz) const {
    kernel.getAs<CalcMMFFNonbondedForceKernel>().getPMEParameters(alpha, nx, ny, nz);
}
