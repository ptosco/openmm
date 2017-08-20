
/* Portions copyright (c) 2006 Stanford University and Simbios.
 * Contributors: Pande Group
 *
 * Permission is hereby granted, free of charge, to any person obtaining
 * a copy of this software and associated documentation files (the
 * "Software"), to deal in the Software without restriction, including
 * without limitation the rights to use, copy, modify, merge, publish,
 * distribute, sublicense, and/or sell copies of the Software, and to
 * permit persons to whom the Software is furnished to do so, subject
 * to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included
 * in all copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS
 * OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF
 * MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.
 * IN NO EVENT SHALL THE AUTHORS, CONTRIBUTORS OR COPYRIGHT HOLDERS BE
 * LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION
 * OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION
 * WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
 */

#include "MMFFReferenceForce.h"
#include "MMFFReferenceVdwForce.h"
#include "ReferenceForce.h"
#include <openmm/Units.h>
#include <algorithm>
#include <cctype>
#include <cmath>

using std::vector;
using namespace OpenMM;

MMFFReferenceVdwForce::MMFFReferenceVdwForce() : _nonbondedMethod(NoCutoff), _cutoff(1.0e+10), _taperCutoffFactor(0.9) {

    setTaperCoefficients(_cutoff);
    setSigmaCombiningRule("ARITHMETIC");
    setEpsilonCombiningRule("GEOMETRIC");
}


MMFFReferenceVdwForce::MMFFReferenceVdwForce(const std::string& sigmaCombiningRule, const std::string& epsilonCombiningRule) : _nonbondedMethod(NoCutoff), _cutoff(1.0e+10), _taperCutoffFactor(0.9) {

    setTaperCoefficients(_cutoff);
    setSigmaCombiningRule(sigmaCombiningRule);
    setEpsilonCombiningRule(epsilonCombiningRule);
}

MMFFReferenceVdwForce::NonbondedMethod MMFFReferenceVdwForce::getNonbondedMethod() const {
    return _nonbondedMethod;
}

void MMFFReferenceVdwForce::setNonbondedMethod(MMFFReferenceVdwForce::NonbondedMethod nonbondedMethod) {
    _nonbondedMethod = nonbondedMethod;
}

void MMFFReferenceVdwForce::setTaperCoefficients(double cutoff) {
    _taperCutoff = cutoff*_taperCutoffFactor;
    if (_taperCutoff != cutoff) {
        _taperCoefficients[C3] = 10.0/pow(_taperCutoff - cutoff, 3.0);
        _taperCoefficients[C4] = 15.0/pow(_taperCutoff - cutoff, 4.0);
        _taperCoefficients[C5] =  6.0/pow(_taperCutoff - cutoff, 5.0);
    } else {
        _taperCoefficients[C3] = 0.0;
        _taperCoefficients[C4] = 0.0;
        _taperCoefficients[C5] = 0.0;
    }
}

void MMFFReferenceVdwForce::setCutoff(double cutoff) {
    _cutoff  = cutoff;
    setTaperCoefficients(_cutoff);
}

double MMFFReferenceVdwForce::getCutoff() const {
    return _cutoff;
}

void MMFFReferenceVdwForce::setPeriodicBox(OpenMM::Vec3* vectors) {
    _periodicBoxVectors[0] = vectors[0];
    _periodicBoxVectors[1] = vectors[1];
    _periodicBoxVectors[2] = vectors[2];
}

void MMFFReferenceVdwForce::setSigmaCombiningRule(const std::string& sigmaCombiningRule) {

    _sigmaCombiningRule = sigmaCombiningRule;

    // convert to upper case and set combining function

    std::transform(_sigmaCombiningRule.begin(), _sigmaCombiningRule.end(), _sigmaCombiningRule.begin(),  (int(*)(int)) std::toupper);
    if (_sigmaCombiningRule == "GEOMETRIC") {
        _combineSigmas = &MMFFReferenceVdwForce::geometricSigmaCombiningRule;
    } else if (_sigmaCombiningRule == "CUBIC-MEAN") {
        _combineSigmas = &MMFFReferenceVdwForce::cubicMeanSigmaCombiningRule;
    } else if (_sigmaCombiningRule == "MMFF") {
        _combineSigmas = &MMFFReferenceVdwForce::mmffSigmaCombiningRule;
    } else {
        _combineSigmas = &MMFFReferenceVdwForce::arithmeticSigmaCombiningRule;
    }
}

std::string MMFFReferenceVdwForce::getSigmaCombiningRule() const {
    return _sigmaCombiningRule;
}

double MMFFReferenceVdwForce::arithmeticSigmaCombiningRule(double sigmaI, double sigmaJ) const {
    return (sigmaI + sigmaJ);
}

double MMFFReferenceVdwForce::geometricSigmaCombiningRule(double sigmaI, double sigmaJ) const {
    return 2.0*sqrt(sigmaI*sigmaJ);
}

double MMFFReferenceVdwForce::cubicMeanSigmaCombiningRule(double sigmaI, double sigmaJ) const {
    double sigmaI2 = sigmaI*sigmaI;
    double sigmaJ2 = sigmaJ*sigmaJ;

    return sigmaI != 0.0 && sigmaJ != 0.0 ? 2.0*(sigmaI2*sigmaI + sigmaJ2*sigmaJ)/(sigmaI2 + sigmaJ2) : 0.0;
}

double MMFFReferenceVdwForce::mmffSigmaCombiningRule(double sigmaI, double sigmaJ) const {

    static const double B = 0.2;
    static const double Beta = 12.0;
    bool haveDonor = false;
    if (sigmaI < 0.0) {
        sigmaI = -sigmaI;
        haveDonor = true;
    }
    if (sigmaJ < 0.0) {
        sigmaJ = -sigmaJ;
        haveDonor = true;
    }
    double gammaIJ = (sigmaI - sigmaJ) / (sigmaI + sigmaJ);
    double sigma = 0.5 * (sigmaI + sigmaJ) * (1.0 + (haveDonor
        ? 0.0 : B * (1.0 - exp(-Beta * gammaIJ * gammaIJ))));
    return sigma;
}

void MMFFReferenceVdwForce::setEpsilonCombiningRule(const std::string& epsilonCombiningRule) {

    _epsilonCombiningRule = epsilonCombiningRule;
    std::transform(_epsilonCombiningRule.begin(), _epsilonCombiningRule.end(), _epsilonCombiningRule.begin(),  (int(*)(int)) std::toupper);

    // convert to upper case and set combining function

    if (_epsilonCombiningRule == "ARITHMETIC") {
         _combineEpsilons = &MMFFReferenceVdwForce::arithmeticEpsilonCombiningRule;
    } else if (_epsilonCombiningRule == "HARMONIC") {
         _combineEpsilons = &MMFFReferenceVdwForce::harmonicEpsilonCombiningRule;
    } else if (_epsilonCombiningRule == "HHG") {
         _combineEpsilons = &MMFFReferenceVdwForce::hhgEpsilonCombiningRule;
    } else if (_epsilonCombiningRule == "MMFF") {
         _combineEpsilons = &MMFFReferenceVdwForce::mmffEpsilonCombiningRule;
    } else {
         _combineEpsilons = &MMFFReferenceVdwForce::geometricEpsilonCombiningRule;
    }
}

std::string MMFFReferenceVdwForce::getEpsilonCombiningRule() const {
    return _epsilonCombiningRule;
}

double MMFFReferenceVdwForce::arithmeticEpsilonCombiningRule(double epsilonI, double epsilonJ) const {
    return 0.5*(epsilonI + epsilonJ);
}

double MMFFReferenceVdwForce::geometricEpsilonCombiningRule(double epsilonI, double epsilonJ) const {
    return sqrt(epsilonI*epsilonJ);
}

double MMFFReferenceVdwForce::harmonicEpsilonCombiningRule(double epsilonI, double epsilonJ) const {
    return (epsilonI != 0.0 && epsilonJ != 0.0) ? 2.0*(epsilonI*epsilonJ)/(epsilonI + epsilonJ) : 0.0;
}

double MMFFReferenceVdwForce::hhgEpsilonCombiningRule(double epsilonI, double epsilonJ) const {
    double denominator = sqrt(epsilonI) + sqrt(epsilonJ);
    return (epsilonI != 0.0 && epsilonJ != 0.0) ? 4.0*(epsilonI*epsilonJ)/(denominator*denominator) : 0.0;
}

double MMFFReferenceVdwForce::mmffEpsilonCombiningRule(double epsilonI, double epsilonJ) const {
}

double MMFFReferenceVdwForce::mmffEpsilonCombiningRuleHelper(double combinedSigma,
    double alphaI_d_NI, double alphaJ_d_NJ, double GI_t_alphaI, double GJ_t_alphaJ) {
    double combinedSigma2 = combinedSigma * combinedSigma;
    static const double NmPerAngstrom2 = NmPerAngstrom * NmPerAngstrom;
    if (alphaI_d_NI < 0.0)
        alphaI_d_NI = -alphaI_d_NI;
    if (alphaJ_d_NJ < 0.0)
        alphaJ_d_NJ = -alphaJ_d_NJ;
    static const double c4 = 181.16 * KJPerKcal * NmPerAngstrom2 * NmPerAngstrom2 * NmPerAngstrom2;
    double epsilon = GI_t_alphaI * GJ_t_alphaJ / ((sqrt(alphaI_d_NI) + sqrt(alphaJ_d_NJ)) *
        combinedSigma2 * combinedSigma2 * combinedSigma2);
    return c4 * epsilon;
}

void MMFFReferenceVdwForce::addReducedForce(unsigned int particleI, unsigned int particleIV,
                                              double reduction, double sign,
                                              Vec3& force, vector<Vec3>& forces) const {

    forces[particleI][0]  += sign*force[0]*reduction;
    forces[particleI][1]  += sign*force[1]*reduction;
    forces[particleI][2]  += sign*force[2]*reduction;

    forces[particleIV][0] += sign*force[0]*(1.0 - reduction);
    forces[particleIV][1] += sign*force[1]*(1.0 - reduction);
    forces[particleIV][2] += sign*force[2]*(1.0 - reduction);
}

double MMFFReferenceVdwForce::calculatePairIxn(double combinedSigma, double combinedEpsilon,
                                                 const Vec3& particleIPosition,
                                                 const Vec3& particleJPosition,
                                                 Vec3& force) const {
    
    static const double dhal = 0.07;
    static const double ghal = 0.12;

    // get deltaR, R2, and R between 2 atoms

    double deltaR[ReferenceForce::LastDeltaRIndex];
    if (_nonbondedMethod == CutoffPeriodic)
        ReferenceForce::getDeltaRPeriodic(particleJPosition, particleIPosition, _periodicBoxVectors, deltaR);
    else
        ReferenceForce::getDeltaR(particleJPosition, particleIPosition, deltaR);

    double r_ij_2       = deltaR[ReferenceForce::R2Index];
    double r_ij         = deltaR[ReferenceForce::RIndex];
    double sigma_7      = combinedSigma*combinedSigma*combinedSigma;
           sigma_7      = sigma_7*sigma_7*combinedSigma;

    double r_ij_6       = r_ij_2*r_ij_2*r_ij_2;
    double r_ij_7       = r_ij_6*r_ij;

    double rho          = r_ij_7 + ghal*sigma_7;

    double tau          = (dhal + 1.0)/(r_ij + dhal*combinedSigma);
    double tau_7        = tau*tau*tau;
           tau_7        = tau_7*tau_7*tau;

    double dtau         = tau/(dhal + 1.0);

    double ratio        = (sigma_7/rho);
    double gtau         = combinedEpsilon*tau_7*r_ij_6*(ghal+1.0)*ratio*ratio;

    double energy       = combinedEpsilon*tau_7*sigma_7*((ghal+1.0)*sigma_7/rho - 2.0);

    double dEdR         = -7.0*(dtau*energy + gtau);

    // tapering

    if ((_nonbondedMethod == CutoffNonPeriodic || _nonbondedMethod == CutoffPeriodic) && r_ij > _taperCutoff) {
        double delta    = r_ij - _taperCutoff;
        double taper    = 1.0 + delta*delta*delta*(_taperCoefficients[C3] + delta*(_taperCoefficients[C4] + delta*_taperCoefficients[C5]));
        double dtaper   = delta*delta*(3.0*_taperCoefficients[C3] + delta*(4.0*_taperCoefficients[C4] + delta*5.0*_taperCoefficients[C5]));
        dEdR            = energy*dtaper + dEdR*taper;
        energy         *= taper;
    }

    dEdR                   /= r_ij;

    force[0]                = dEdR*deltaR[0];
    force[1]                = dEdR*deltaR[1];
    force[2]                = dEdR*deltaR[2];

    return energy;

}

void MMFFReferenceVdwForce::setReducedPositions(int numParticles,
                                                  const vector<Vec3>& particlePositions,
                                                  const std::vector<int>& indexIVs, 
                                                  const std::vector<double>& reductions,
                                                  std::vector<Vec3>& reducedPositions) const {

    reducedPositions.resize(numParticles);
    for (unsigned int ii = 0; ii <  static_cast<unsigned int>(numParticles); ii++) {
        if (reductions[ii] != 0.0) {
            int reductionIndex     = indexIVs[ii];
            reducedPositions[ii]   = Vec3(reductions[ii]*(particlePositions[ii][0] - particlePositions[reductionIndex][0]) + particlePositions[reductionIndex][0], 
                                          reductions[ii]*(particlePositions[ii][1] - particlePositions[reductionIndex][1]) + particlePositions[reductionIndex][1], 
                                          reductions[ii]*(particlePositions[ii][2] - particlePositions[reductionIndex][2]) + particlePositions[reductionIndex][2]); 
        } else {
            reducedPositions[ii]   = Vec3(particlePositions[ii][0], particlePositions[ii][1], particlePositions[ii][2]); 
        }
    }
}

double MMFFReferenceVdwForce::calculateForceAndEnergy(int numParticles,
                                                        const vector<OpenMM::Vec3>& particlePositions,
                                                        const std::vector<int>& indexIVs, 
                                                        const std::vector<double>& sigmas,
                                                        const std::vector<double>& epsilons,
                                                        const std::vector<double>& reductions,
                                                        const std::vector< std::set<int> >& allExclusions,
                                                        vector<OpenMM::Vec3>& forces) const {

    // MMFF rules:
    // combinedSigma is equivalent to R_star_ij
    // the sigmas array should be initialized with R_star for each particle
    // the epsilons array should be initialized with alpha_i/N_i for each particle
    // the reductions array should be initialized with gamma_i*alpha_i for each particle
    // the signs matter:
    // - if sigma > 0.0 and epsilon > 0.0, then the particle is neither a donor nor an acceptor
    // - if sigma > 0.0 and epsilon < 0.0, then the particle is an acceptor
    // - if sigma < 0.0, then the particle is a donor
    static const double DARAD = 0.8;
    static const double DAEPS = 0.5;

    // set reduced coordinates

    std::vector<Vec3> reducedPositions;
    setReducedPositions(numParticles, particlePositions, indexIVs, reductions, reducedPositions);

    // loop over all particle pairs

    //    (1) initialize exclusion vector
    //    (2) calculate pair ixn, if not excluded
    //    (3) accumulate forces: if particle is a site where interaction position != particle position,
    //        then call addReducedForce() to apportion force to particle and its covalent partner
    //        based on reduction factor
    //    (4) reset exclusion vector

    double energy = 0.0;
    std::vector<unsigned int> exclusions(numParticles, 0);
    for (unsigned int ii = 0; ii < static_cast<unsigned int>(numParticles); ii++) {
 
        double sigmaI      = sigmas[ii];
        double epsilonI    = epsilons[ii];
        for (int jj : allExclusions[ii])
            exclusions[jj] = 1;

        for (unsigned int jj = ii+1; jj < static_cast<unsigned int>(numParticles); jj++) {
            if (exclusions[jj] == 0) {

                double sigmaJ      = sigmas[jj];
                double epsilonJ    = epsilons[jj];
                double combinedSigma   = (this->*_combineSigmas)(sigmaI, sigmaJ);
                double combinedEpsilon = (this->_combineEpsilons == &MMFFReferenceVdwForce::mmffEpsilonCombiningRule)
                                         ? mmffEpsilonCombiningRuleHelper(combinedSigma, epsilonI, epsilonJ, reductions[ii], reductions[jj])
                                         : (this->*_combineEpsilons)(epsilonI, epsilonJ);
                
                // in MMFF, if one of the particles is an acceptor and the other one is a donor,
                // then we want to scale sigma and epsilon
                if ((this->_combineSigmas == &MMFFReferenceVdwForce::mmffSigmaCombiningRule)
                    && (((sigmas[ii] < 0.0) && (sigmas[jj] > 0.0) && (epsilons[jj] < 0.0))
                    || ((sigmas[jj] < 0.0) && (sigmas[ii] > 0.0) && (epsilons[ii] < 0.0)))) {
                    combinedSigma *= DARAD;
                    combinedEpsilon *= DAEPS;
                }

                Vec3 force;
                energy += calculatePairIxn(combinedSigma, combinedEpsilon,
                                           reducedPositions[ii], reducedPositions[jj], force);
                
                if (indexIVs[ii] == ii) {
                    forces[ii][0] -= force[0];
                    forces[ii][1] -= force[1];
                    forces[ii][2] -= force[2];
                } else {
                    addReducedForce(ii, indexIVs[ii], reductions[ii], -1.0, force, forces);
                }
                if (indexIVs[jj] == jj) {
                    forces[jj][0] += force[0];
                    forces[jj][1] += force[1];
                    forces[jj][2] += force[2];
                } else {
                    addReducedForce(jj, indexIVs[jj], reductions[jj], 1.0, force, forces);
                }

            }
        }

        for (int jj : allExclusions[ii])
            exclusions[jj] = 0;
    }

    return energy;
}

double MMFFReferenceVdwForce::calculateForceAndEnergy(int numParticles,
                                                        const vector<Vec3>& particlePositions,
                                                        const std::vector<int>& indexIVs, 
                                                        const std::vector<double>& sigmas,
                                                        const std::vector<double>& epsilons,
                                                        const std::vector<double>& reductions,
                                                        const NeighborList& neighborList,
                                                        vector<Vec3>& forces) const {

    // set reduced coordinates

    std::vector<Vec3> reducedPositions;
    setReducedPositions(numParticles, particlePositions, indexIVs, reductions, reducedPositions);
 
    // loop over neighbor list
    //    (1) calculate pair vdw ixn
    //    (2) accumulate forces: if particle is a site where interaction position != particle position,
    //        then call addReducedForce() to apportion force to particle and its covalent partner
    //        based on reduction factor

    double energy = 0.0;
    for (unsigned int ii = 0; ii < neighborList.size(); ii++) {

        OpenMM::AtomPair pair       = neighborList[ii];
        int siteI                   = pair.first;
        int siteJ                   = pair.second;

        double combinedSigma   = (this->*_combineSigmas)(sigmas[siteI], sigmas[siteJ]);
        double combinedEpsilon = (this->*_combineEpsilons)(epsilons[siteI], epsilons[siteJ]);

        Vec3 force;
        energy                     += calculatePairIxn(combinedSigma, combinedEpsilon,
                                                       reducedPositions[siteI], reducedPositions[siteJ], force);
                
        if (indexIVs[siteI] == siteI) {
            forces[siteI][0] -= force[0];
            forces[siteI][1] -= force[1];
            forces[siteI][2] -= force[2];
        } else {
            addReducedForce(siteI, indexIVs[siteI], reductions[siteI], -1.0, force, forces);
        }
        if (indexIVs[siteJ] == siteJ) {
            forces[siteJ][0] += force[0];
            forces[siteJ][1] += force[1];
            forces[siteJ][2] += force[2];
        } else {
            addReducedForce(siteJ, indexIVs[siteJ], reductions[siteJ], 1.0, force, forces);
        }

    }

    return energy;
}
