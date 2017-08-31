
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

double MMFFReferenceVdwForce::_mmffSigmaCombiningRule(double sigmaI, double sigmaJ, char vdwDAI, char vdwDAJ) {
    static const double B = 0.2;
    static const double Beta = 12.0;
    bool haveDonor = (vdwDAI == 'D' || vdwDAJ == 'D');
    double gammaIJ = (sigmaI - sigmaJ) / (sigmaI + sigmaJ);
    double sigma = 0.5 * (sigmaI + sigmaJ) * (1.0 + (haveDonor
        ? 0.0 : B * (1.0 - exp(-Beta * gammaIJ * gammaIJ))));
    return sigma;
}

double MMFFReferenceVdwForce::_mmffEpsilonCombiningRule(double combinedSigma,
    double alpha_d_NI, double alpha_d_NJ, double G_t_alphaI, double G_t_alphaJ) {
    double combinedSigma2 = combinedSigma * combinedSigma;
    static const double NmPerAngstrom2 = NmPerAngstrom * NmPerAngstrom;
    static const double C4 = 7.5797344e-4;
    double epsilon = G_t_alphaI * G_t_alphaJ / ((sqrt(alpha_d_NI) + sqrt(alpha_d_NJ)) *
        combinedSigma2 * combinedSigma2 * combinedSigma2);
    return C4 * epsilon;
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

double MMFFReferenceVdwForce::calculateForceAndEnergy(int numParticles,
                                                        const vector<OpenMM::Vec3>& particlePositions,
                                                        const std::vector<double>& sigmas,
                                                        const std::vector<double>& G_t_alphas,
                                                        const std::vector<double>& alpha_d_Ns,
                                                        const std::vector<char>& vdwDAs,
                                                        const std::vector< std::set<int> >& allExclusions,
                                                        vector<OpenMM::Vec3>& forces) const {

    // MMFF rules:
    // combinedSigma is equivalent to R_star_ij
    // the sigmas array should be initialized with R_star for each particle
    // the G_t_alphas array should be initialized with gamma_i*alpha_i for each particle
    // the alpha_d_Ns array should be initialized with alpha_i/N_i for each particle
    // the vdwDAs array should be initialized with vdwDA for each particle
    static const double DARAD = 0.8;
    static const double DAEPS = 0.5;

    // loop over all particle pairs

    //    (1) initialize exclusion vector
    //    (2) calculate pair ixn, if not excluded
    //    (3) accumulate forces
    //    (4) reset exclusion vector

    double energy = 0.0;
    std::vector<unsigned int> exclusions(numParticles, 0);
    for (unsigned int ii = 0; ii < static_cast<unsigned int>(numParticles); ii++) {
 
        double sigmaI     = sigmas[ii];
        double G_t_alphaI = G_t_alphas[ii];
        double alpha_d_NI = alpha_d_Ns[ii];
        char vdwDAI       = vdwDAs[ii];
        for (int jj : allExclusions[ii])
            exclusions[jj] = 1;

        for (unsigned int jj = ii+1; jj < static_cast<unsigned int>(numParticles); jj++) {
            if (exclusions[jj] == 0) {

                double sigmaJ     = sigmas[jj];
                double G_t_alphaJ = G_t_alphas[jj];
                double alpha_d_NJ = alpha_d_Ns[jj];
                char vdwDAJ       = vdwDAs[jj];
                double combinedSigma   = _mmffSigmaCombiningRule(sigmaI, sigmaJ, vdwDAI, vdwDAJ);
                double combinedEpsilon = _mmffEpsilonCombiningRule(combinedSigma, alpha_d_NI, alpha_d_NJ, G_t_alphaI, G_t_alphaJ);
                
                // in MMFF, if one of the particles is an acceptor and the other one is a donor,
                // then we want to scale sigma and epsilon
                if ((vdwDAI == 'A' && vdwDAJ == 'D') || (vdwDAI == 'D' && vdwDAJ == 'A')) {
                    combinedSigma *= DARAD;
                    combinedEpsilon *= DAEPS;
                }

                Vec3 force;
                energy += calculatePairIxn(combinedSigma, combinedEpsilon, particlePositions[ii], particlePositions[jj], force);
                
                forces[ii][0] -= force[0];
                forces[ii][1] -= force[1];
                forces[ii][2] -= force[2];
                forces[jj][0] += force[0];
                forces[jj][1] += force[1];
                forces[jj][2] += force[2];
            }
        }

        for (int jj : allExclusions[ii])
            exclusions[jj] = 0;
    }

    return energy;
}

double MMFFReferenceVdwForce::calculateForceAndEnergy(int numParticles,
                                                        const vector<Vec3>& particlePositions,
                                                        const std::vector<double>& sigmas,
                                                        const std::vector<double>& G_t_alphas,
                                                        const std::vector<double>& alpha_d_Ns,
                                                        const std::vector<char>& vdwDAs,
                                                        const NeighborList& neighborList,
                                                        vector<Vec3>& forces) const {

    // loop over neighbor list
    //    (1) calculate pair vdw ixn
    //    (2) accumulate forces

    double energy = 0.0;
    for (unsigned int ii = 0; ii < neighborList.size(); ii++) {

        OpenMM::AtomPair pair       = neighborList[ii];
        int siteI                   = pair.first;
        int siteJ                   = pair.second;

        double combinedSigma   = _mmffSigmaCombiningRule(sigmas[siteI], sigmas[siteJ], vdwDAs[siteI], vdwDAs[siteJ]);
        double combinedEpsilon = _mmffEpsilonCombiningRule(combinedSigma, alpha_d_Ns[siteI], alpha_d_Ns[siteJ],
                                                            G_t_alphas[siteI], G_t_alphas[siteJ]);

        Vec3 force;
        energy                     += calculatePairIxn(combinedSigma, combinedEpsilon,
                                                        particlePositions[siteI], particlePositions[siteJ], force);
                
        forces[siteI][0] -= force[0];
        forces[siteI][1] -= force[1];
        forces[siteI][2] -= force[2];
        forces[siteJ][0] += force[0];
        forces[siteJ][1] += force[1];
        forces[siteJ][2] += force[2];
    }

    return energy;
}
