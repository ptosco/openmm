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
#include "openmm/MMFFNonbondedForce.h"
#include "openmm/internal/AssertionUtilities.h"
#include "openmm/internal/MMFFNonbondedForceImpl.h"
#include <cmath>
#include <map>
#include <sstream>
#include <utility>
#include <iostream>

using namespace OpenMM;
using std::map;
using std::pair;
using std::set;
using std::string;
using std::stringstream;
using std::vector;

double MMFFNonbondedForce::sigmaCombiningRule(double sigmaI, double sigmaJ, bool haveDonor) {
    static const double B = 0.2;
    static const double Beta = 12.0;
    double gammaIJ = (sigmaI - sigmaJ) / (sigmaI + sigmaJ);
    double sigma = 0.5 * (sigmaI + sigmaJ) * (1.0 + (haveDonor
        ? 0.0 : B * (1.0 - exp(-Beta * gammaIJ * gammaIJ))));
    return sigma;
}

double MMFFNonbondedForce::epsilonCombiningRule(double combinedSigma,
    double alpha_d_NI, double alpha_d_NJ, double G_t_alphaI, double G_t_alphaJ) {
    double combinedSigma2 = combinedSigma * combinedSigma;
    static const double c = 7.5797344e-4;
    double epsilon = G_t_alphaI * G_t_alphaJ / ((sqrt(alpha_d_NI) + sqrt(alpha_d_NJ)) *
        combinedSigma2 * combinedSigma2 * combinedSigma2);
    return c * epsilon;
}

void MMFFNonbondedForce::scaleSigmaEpsilon(double &combinedSigma, double &combinedEpsilon) {
    static const double DARAD = 0.8;
    static const double DAEPS = 0.5;
    combinedSigma *= DARAD;
    combinedEpsilon *= DAEPS;
}

MMFFNonbondedForce::MMFFNonbondedForce() : nonbondedMethod(NoCutoff), cutoffDistance(1.0), switchingDistance(-1.0), rfDielectric(1.0),
        ewaldErrorTol(5e-4), alpha(0.0), dalpha(0.0), useSwitchingFunction(false), useDispersionCorrection(false), recipForceGroup(-1),
        nx(0), ny(0), nz(0), dnx(0), dny(0), dnz(0) {
}

MMFFNonbondedForce::NonbondedMethod MMFFNonbondedForce::getNonbondedMethod() const {
    return nonbondedMethod;
}

void MMFFNonbondedForce::setNonbondedMethod(NonbondedMethod method) {
    if (method < 0 || method > 4)
        throw OpenMMException("MMFFNonbondedForce: Illegal value for nonbonded method");
    nonbondedMethod = method;
}

double MMFFNonbondedForce::getCutoffDistance() const {
    return cutoffDistance;
}

void MMFFNonbondedForce::setCutoffDistance(double distance) {
    cutoffDistance = distance;
}

bool MMFFNonbondedForce::getUseSwitchingFunction() const {
    return useSwitchingFunction;
}

void MMFFNonbondedForce::setUseSwitchingFunction(bool use) {
    useSwitchingFunction = use;
}

double MMFFNonbondedForce::getSwitchingDistance() const {
    return switchingDistance;
}

void MMFFNonbondedForce::setSwitchingDistance(double distance) {
    switchingDistance = distance;
}

double MMFFNonbondedForce::getReactionFieldDielectric() const {
    return rfDielectric;
}

void MMFFNonbondedForce::setReactionFieldDielectric(double dielectric) {
    rfDielectric = dielectric;
}

double MMFFNonbondedForce::getEwaldErrorTolerance() const {
    return ewaldErrorTol;
}

void MMFFNonbondedForce::setEwaldErrorTolerance(double tol) {
    ewaldErrorTol = tol;
}

void MMFFNonbondedForce::getPMEParameters(double& alpha, int& nx, int& ny, int& nz) const {
    alpha = this->alpha;
    nx = this->nx;
    ny = this->ny;
    nz = this->nz;
}

void MMFFNonbondedForce::setPMEParameters(double alpha, int nx, int ny, int nz) {
    this->alpha = alpha;
    this->nx = nx;
    this->ny = ny;
    this->nz = nz;
}

void MMFFNonbondedForce::getPMEParametersInContext(const Context& context, double& alpha, int& nx, int& ny, int& nz) const {
    dynamic_cast<const MMFFNonbondedForceImpl&>(getImplInContext(context)).getPMEParameters(alpha, nx, ny, nz);
}

int MMFFNonbondedForce::addParticle(double charge, double sigma, double G_t_alpha, double alpha_d_N, char vdwDA) {
    particles.push_back(ParticleInfo(charge, sigma, G_t_alpha, alpha_d_N, vdwDA));
    return particles.size()-1;
}

void MMFFNonbondedForce::getParticleParameters(int index, double& charge, double& sigma, double& G_t_alpha, double& alpha_d_N, char& vdwDA) const {
    ASSERT_VALID_INDEX(index, particles);
    charge = particles[index].charge;
    sigma = particles[index].sigma;
    G_t_alpha = particles[index].G_t_alpha;
    alpha_d_N = particles[index].alpha_d_N;
    vdwDA = particles[index].vdwDA;
}

void MMFFNonbondedForce::setParticleParameters(int index, double charge, double sigma, double G_t_alpha, double alpha_d_N, char vdwDA) {
    ASSERT_VALID_INDEX(index, particles);
    particles[index].charge = charge;
    particles[index].sigma = sigma;
    particles[index].G_t_alpha = G_t_alpha;
    particles[index].alpha_d_N = alpha_d_N;
    particles[index].vdwDA = vdwDA;
}

int MMFFNonbondedForce::addException(int particle1, int particle2, double chargeProd, double sigma, double epsilon, bool replace) {
    map<pair<int, int>, int>::iterator iter = exceptionMap.find(pair<int, int>(particle1, particle2));
    int newIndex;
    if (iter == exceptionMap.end())
        iter = exceptionMap.find(pair<int, int>(particle2, particle1));
    if (iter != exceptionMap.end()) {
        if (!replace) {
            stringstream msg;
            msg << "MMFFNonbondedForce: There is already an exception for particles ";
            msg << particle1;
            msg << " and ";
            msg << particle2;
            throw OpenMMException(msg.str());
        }
        exceptions[iter->second] = ExceptionInfo(particle1, particle2, chargeProd, sigma, epsilon);
        newIndex = iter->second;
        exceptionMap.erase(iter->first);
    }
    else {
        exceptions.push_back(ExceptionInfo(particle1, particle2, chargeProd, sigma, epsilon));
        newIndex = exceptions.size()-1;
    }
    exceptionMap[pair<int, int>(particle1, particle2)] = newIndex;
    return newIndex;
}
void MMFFNonbondedForce::getExceptionParameters(int index, int& particle1, int& particle2, double& chargeProd, double& sigma, double& epsilon) const {
    ASSERT_VALID_INDEX(index, exceptions);
    particle1 = exceptions[index].particle1;
    particle2 = exceptions[index].particle2;
    chargeProd = exceptions[index].chargeProd;
    sigma = exceptions[index].sigma;
    epsilon = exceptions[index].epsilon;
}

void MMFFNonbondedForce::setExceptionParameters(int index, int particle1, int particle2, double chargeProd, double sigma, double epsilon) {
    ASSERT_VALID_INDEX(index, exceptions);
    exceptions[index].particle1 = particle1;
    exceptions[index].particle2 = particle2;
    exceptions[index].chargeProd = chargeProd;
    exceptions[index].sigma = sigma;
    exceptions[index].epsilon = epsilon;
}

ForceImpl* MMFFNonbondedForce::createImpl() const {
    return new MMFFNonbondedForceImpl(*this);
}

void MMFFNonbondedForce::createExceptionsFromBonds(const vector<pair<int, int> >& bonds, double coulomb14Scale, double vdw14Scale) {
    for (auto& bond : bonds)
        if (bond.first < 0 || bond.second < 0 || bond.first >= particles.size() || bond.second >= particles.size())
            throw OpenMMException("createExceptionsFromBonds: Illegal particle index in list of bonds");

    // Find particles separated by 1, 2, or 3 bonds.

    vector<set<int> > exclusions(particles.size());
    vector<set<int> > bonded12(exclusions.size());
    for (auto& bond : bonds) {
        bonded12[bond.first].insert(bond.second);
        bonded12[bond.second].insert(bond.first);
    }
    for (int i = 0; i < (int) exclusions.size(); ++i)
        addExclusionsToSet(bonded12, exclusions[i], i, i, 2);

    // Find particles separated by 1 or 2 bonds and create the exceptions.

    for (int i = 0; i < (int) exclusions.size(); ++i) {
        set<int> bonded13;
        addExclusionsToSet(bonded12, bonded13, i, i, 1);
        for (int j : exclusions[i]) {
            if (j < i) {
                if (bonded13.find(j) == bonded13.end()) {
                    // This is a 1-4 interaction.
                    //std::cerr << "1-4 interaction between particles " << i << " and " << j << std::endl;
                    const ParticleInfo& particle1 = particles[j];
                    const ParticleInfo& particle2 = particles[i];
                    const double chargeProd = coulomb14Scale*particle1.charge*particle2.charge;
                    bool haveDonor = (particle1.vdwDA == 'D' || particle2.vdwDA == 'D');
                    bool haveDAPair = (particle1.vdwDA == 'D' && particle2.vdwDA == 'A')
                        || (particle1.vdwDA == 'A' && particle2.vdwDA == 'D');
                    double combinedSigma   = sigmaCombiningRule(particle1.sigma, particle2.sigma, haveDonor);
                    double combinedEpsilon = vdw14Scale*epsilonCombiningRule(combinedSigma, particle1.alpha_d_N,
                        particle2.alpha_d_N, particle1.G_t_alpha, particle2.G_t_alpha);
                    if (haveDAPair)
                        scaleSigmaEpsilon(combinedSigma, combinedEpsilon);
                    addException(j, i, chargeProd, combinedSigma, combinedEpsilon);
                }
                else {
                    // This interaction should be completely excluded.

                    addException(j, i, 0.0, 1.0, 0.0);
                }
            }
        }
    }
}

void MMFFNonbondedForce::addExclusionsToSet(const vector<set<int> >& bonded12, set<int>& exclusions, int baseParticle, int fromParticle, int currentLevel) const {
    for (int i : bonded12[fromParticle]) {
        if (i != baseParticle)
            exclusions.insert(i);
        if (currentLevel > 0)
            addExclusionsToSet(bonded12, exclusions, baseParticle, i, currentLevel-1);
    }
}

int MMFFNonbondedForce::getReciprocalSpaceForceGroup() const {
    return recipForceGroup;
}

void MMFFNonbondedForce::setReciprocalSpaceForceGroup(int group) {
    if (group < -1 || group > 31)
        throw OpenMMException("Force group must be between -1 and 31");
    recipForceGroup = group;
}

void MMFFNonbondedForce::updateParametersInContext(Context& context) {
    dynamic_cast<MMFFNonbondedForceImpl&>(getImplInContext(context)).updateParametersInContext(getContextImpl(context));
}
