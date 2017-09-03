
/* Portions copyright (c) 2006-2013 Stanford University and Simbios.
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

#include <string.h>
#include <sstream>
#include <complex>
#include <algorithm>
#include <iostream>

#include "SimTKOpenMMUtilities.h"
#include "MMFFReferenceNonbondedForce.h"
#include "openmm/MMFFNonbondedForce.h"
#include "ReferenceForce.h"
#include "ReferencePME.h"
#include "openmm/OpenMMException.h"

// In case we're using some primitive version of Visual Studio this will
// make sure that erf() and erfc() are defined.
#include "openmm/internal/MSVC_erfc.h"

using std::set;
using std::vector;
using namespace OpenMM;

const int MMFFReferenceNonbondedForce::SigIndex = 0;
const int MMFFReferenceNonbondedForce::GtaIndex = 1;
const int MMFFReferenceNonbondedForce::adNIndex = 2;
const int   MMFFReferenceNonbondedForce::QIndex = 3;

// MMFF vdW constants

const double MMFFReferenceNonbondedForce::dhal  = 0.07;
const double MMFFReferenceNonbondedForce::ghal  = 0.12;

// taper coefficient indices

const int MMFFReferenceNonbondedForce::C3 =       0;
const int MMFFReferenceNonbondedForce::C4 =       1;
const int MMFFReferenceNonbondedForce::C5 =       2;

/**---------------------------------------------------------------------------------------

   MMFFReferenceNonbondedForce constructor

   --------------------------------------------------------------------------------------- */

MMFFReferenceNonbondedForce::MMFFReferenceNonbondedForce() : cutoff(false), periodic(false), ewald(false), pme(false) {
}

/**---------------------------------------------------------------------------------------

   MMFFReferenceNonbondedForce destructor

   --------------------------------------------------------------------------------------- */

MMFFReferenceNonbondedForce::~MMFFReferenceNonbondedForce() {
}

/**---------------------------------------------------------------------------------------

     Set the force to use a cutoff.

     @param distance            the cutoff distance
     @param neighbors           the neighbor list to use
     @param solventDielectric   the dielectric constant of the bulk solvent

     --------------------------------------------------------------------------------------- */

void MMFFReferenceNonbondedForce::setTaperCoefficients(double distance) {
    static const double taperCutoffFactor = 0.9;
    taperCutoff = distance*taperCutoffFactor;
    if (taperCutoff != distance) {
        taperCoefficients[C3] = 10.0/pow(taperCutoff - distance, 3.0);
        taperCoefficients[C4] = 15.0/pow(taperCutoff - distance, 4.0);
        taperCoefficients[C5] =  6.0/pow(taperCutoff - distance, 5.0);
    } else {
        taperCoefficients[C3] = 0.0;
        taperCoefficients[C4] = 0.0;
        taperCoefficients[C5] = 0.0;
    }
}

void MMFFReferenceNonbondedForce::setUseCutoff(double distance, const OpenMM::NeighborList& neighbors, double solventDielectric) {

    cutoff = true;
    cutoffDistance = distance;
    setTaperCoefficients(distance);
    neighborList = &neighbors;
    krf = pow(cutoffDistance, -3.0)*(solventDielectric-1.0)/(2.0*solventDielectric+1.0);
    crf = (1.0/cutoffDistance)*(3.0*solventDielectric)/(2.0*solventDielectric+1.0);
}

/**---------------------------------------------------------------------------------------

     Set the force to use periodic boundary conditions.  This requires that a cutoff has
     also been set, and the smallest side of the periodic box is at least twice the cutoff
     distance.

     @param vectors    the vectors defining the periodic box

     --------------------------------------------------------------------------------------- */

void MMFFReferenceNonbondedForce::setPeriodic(OpenMM::Vec3* vectors) {

    assert(cutoff);
    assert(vectors[0][0] >= 2.0*cutoffDistance);
    assert(vectors[1][1] >= 2.0*cutoffDistance);
    assert(vectors[2][2] >= 2.0*cutoffDistance);
    periodic = true;
    periodicBoxVectors[0] = vectors[0];
    periodicBoxVectors[1] = vectors[1];
    periodicBoxVectors[2] = vectors[2];
}

/**---------------------------------------------------------------------------------------

     Set the force to use Ewald summation.

     @param alpha  the Ewald separation parameter
     @param kmaxx  the largest wave vector in the x direction
     @param kmaxy  the largest wave vector in the y direction
     @param kmaxz  the largest wave vector in the z direction

     --------------------------------------------------------------------------------------- */

void MMFFReferenceNonbondedForce::setUseEwald(double alpha, int kmaxx, int kmaxy, int kmaxz) {
    alphaEwald = alpha;
    numRx = kmaxx;
    numRy = kmaxy;
    numRz = kmaxz;
    ewald = true;
}

/**---------------------------------------------------------------------------------------

     Set the force to use Particle-Mesh Ewald (PME) summation.

     @param alpha  the Ewald separation parameter
     @param gridSize the dimensions of the mesh

     --------------------------------------------------------------------------------------- */

void MMFFReferenceNonbondedForce::setUsePME(double alpha, int meshSize[3]) {
    alphaEwald = alpha;
    meshDim[0] = meshSize[0];
    meshDim[1] = meshSize[1];
    meshDim[2] = meshSize[2];
    pme = true;
}

/**---------------------------------------------------------------------------------------

   Calculate Ewald ixn

   @param numberOfAtoms    number of atoms
   @param atomCoordinates  atom coordinates
   @param atomParameters   atom parameters                             atomParameters[atomIndex][paramterIndex]
   @param exclusions       atom exclusion indices
                           exclusions[atomIndex] contains the list of exclusions for that atom
   @param fixedParameters  non atom parameters (not currently used)
   @param forces           force array (forces added)
   @param energyByAtom     atom energy
   @param totalEnergy      total energy
   @param includeDirect      true if direct space interactions should be included
   @param includeReciprocal  true if reciprocal space interactions should be included

   --------------------------------------------------------------------------------------- */

void MMFFReferenceNonbondedForce::calculateEwaldIxn(int numberOfAtoms, vector<Vec3>& atomCoordinates,
                                              double** atomParameters, vector<set<int> >& exclusions,
                                              double* fixedParameters, vector<Vec3>& forces,
                                              double* energyByAtom, double* totalEnergy, bool includeDirect, bool includeReciprocal) const {
    typedef std::complex<double> d_complex;

    static const double epsilon     =  1.0;

    int kmax                            = (ewald ? std::max(numRx, std::max(numRy,numRz)) : 0);
    double factorEwald              = -1 / (4*alphaEwald*alphaEwald);
    double SQRT_PI                  = sqrt(PI_M);
    double TWO_PI                   = 2.0 * PI_M;
    double recipCoeff               = ONE_4PI_EPS0*4*PI_M/(periodicBoxVectors[0][0] * periodicBoxVectors[1][1] * periodicBoxVectors[2][2]) /epsilon;

    double totalSelfEwaldEnergy     = 0.0;
    double realSpaceEwaldEnergy     = 0.0;
    double recipEnergy              = 0.0;
    double recipDispersionEnergy    = 0.0;
    double totalRecipEnergy         = 0.0;
    double vdwEnergy                = 0.0;

    // **************************************************************************************
    // SELF ENERGY
    // **************************************************************************************

    if (includeReciprocal) {
        for (int atomID = 0; atomID < numberOfAtoms; atomID++) {
            double selfEwaldEnergy       = ONE_4PI_EPS0*atomParameters[atomID][QIndex]*atomParameters[atomID][QIndex] * alphaEwald/SQRT_PI;

            totalSelfEwaldEnergy            -= selfEwaldEnergy;
            if (energyByAtom) {
                energyByAtom[atomID]        -= selfEwaldEnergy;
            }
        }
    }

    if (totalEnergy) {
        *totalEnergy += totalSelfEwaldEnergy;
    }

    // **************************************************************************************
    // RECIPROCAL SPACE EWALD ENERGY AND FORCES
    // **************************************************************************************
    // PME

    if (pme && includeReciprocal) {
        pme_t          pmedata; /* abstract handle for PME data */

        pme_init(&pmedata,alphaEwald,numberOfAtoms,meshDim,5,1);

        vector<double> charges(numberOfAtoms);
        for (int i = 0; i < numberOfAtoms; i++)
            charges[i] = atomParameters[i][QIndex];
        pme_exec(pmedata,atomCoordinates,forces,charges,periodicBoxVectors,&recipEnergy);

        if (totalEnergy)
            *totalEnergy += recipEnergy;

        if (energyByAtom)
            for (int n = 0; n < numberOfAtoms; n++)
                energyByAtom[n] += recipEnergy;

        pme_destroy(pmedata);
    }
    // Ewald method

    else if (ewald && includeReciprocal) {

        // setup reciprocal box

        double recipBoxSize[3] = { TWO_PI / periodicBoxVectors[0][0], TWO_PI / periodicBoxVectors[1][1], TWO_PI / periodicBoxVectors[2][2]};


        // setup K-vectors

#define EIR(x, y, z) eir[(x)*numberOfAtoms*3+(y)*3+z]
        vector<d_complex> eir(kmax*numberOfAtoms*3);
        vector<d_complex> tab_xy(numberOfAtoms);
        vector<d_complex> tab_qxyz(numberOfAtoms);

        if (kmax < 1)
            throw OpenMMException("kmax for Ewald summation < 1");

        for (int i = 0; (i < numberOfAtoms); i++) {
            for (int m = 0; (m < 3); m++)
                EIR(0, i, m) = d_complex(1,0);

            for (int m=0; (m<3); m++)
                EIR(1, i, m) = d_complex(cos(atomCoordinates[i][m]*recipBoxSize[m]),
                                         sin(atomCoordinates[i][m]*recipBoxSize[m]));

            for (int j=2; (j<kmax); j++)
                for (int m=0; (m<3); m++)
                    EIR(j, i, m) = EIR(j-1, i, m) * EIR(1, i, m);
        }

        // calculate reciprocal space energy and forces

        int lowry = 0;
        int lowrz = 1;

        for (int rx = 0; rx < numRx; rx++) {

            double kx = rx * recipBoxSize[0];

            for (int ry = lowry; ry < numRy; ry++) {

                double ky = ry * recipBoxSize[1];

                if (ry >= 0) {
                    for (int n = 0; n < numberOfAtoms; n++)
                        tab_xy[n] = EIR(rx, n, 0) * EIR(ry, n, 1);
                }

                else {
                    for (int n = 0; n < numberOfAtoms; n++)
                        tab_xy[n]= EIR(rx, n, 0) * conj (EIR(-ry, n, 1));
                }

                for (int rz = lowrz; rz < numRz; rz++) {

                    if (rz >= 0) {
                        for (int n = 0; n < numberOfAtoms; n++)
                            tab_qxyz[n] = atomParameters[n][QIndex] * (tab_xy[n] * EIR(rz, n, 2));
                    }

                    else {
                        for (int n = 0; n < numberOfAtoms; n++)
                            tab_qxyz[n] = atomParameters[n][QIndex] * (tab_xy[n] * conj(EIR(-rz, n, 2)));
                    }

                    double cs = 0.0f;
                    double ss = 0.0f;

                    for (int n = 0; n < numberOfAtoms; n++) {
                        cs += tab_qxyz[n].real();
                        ss += tab_qxyz[n].imag();
                    }

                    double kz = rz * recipBoxSize[2];
                    double k2 = kx * kx + ky * ky + kz * kz;
                    double ak = exp(k2*factorEwald) / k2;

                    for (int n = 0; n < numberOfAtoms; n++) {
                        double force = ak * (cs * tab_qxyz[n].imag() - ss * tab_qxyz[n].real());
                        forces[n][0] += 2 * recipCoeff * force * kx ;
                        forces[n][1] += 2 * recipCoeff * force * ky ;
                        forces[n][2] += 2 * recipCoeff * force * kz ;
                    }

                    recipEnergy       = recipCoeff * ak * (cs * cs + ss * ss);
                    totalRecipEnergy += recipEnergy;

                    if (totalEnergy)
                        *totalEnergy += recipEnergy;

                    if (energyByAtom)
                        for (int n = 0; n < numberOfAtoms; n++)
                            energyByAtom[n] += recipEnergy;

                    lowrz = 1 - numRz;
                }
                lowry = 1 - numRy;
            }
        }
    }

    // **************************************************************************************
    // SHORT-RANGE ENERGY AND FORCES
    // **************************************************************************************

    if (!includeDirect)
        return;
    double totalVdwEnergy            = 0.0f;
    double totalRealSpaceEwaldEnergy = 0.0f;

    for (auto& pair : *neighborList) {
        int ii = pair.first;
        int jj = pair.second;

        double deltaR[2][ReferenceForce::LastDeltaRIndex];
        ReferenceForce::getDeltaRPeriodic(atomCoordinates[jj], atomCoordinates[ii], periodicBoxVectors, deltaR[0]);
        double r_ij      = deltaR[0][ReferenceForce::RIndex];
        double inverseR  = 1.0/(deltaR[0][ReferenceForce::RIndex]);
        double alphaR    = alphaEwald * r_ij;


        double dEdR = ONE_4PI_EPS0 * atomParameters[ii][QIndex] * atomParameters[jj][QIndex] * inverseR * inverseR * inverseR;
        dEdR = dEdR * (erfc(alphaR) + 2 * alphaR * exp (- alphaR * alphaR) / SQRT_PI);

        bool haveDonor = (atomParameters[ii][GtaIndex] < 0.0 || atomParameters[jj][GtaIndex] < 0.0);
        bool haveDAPair = (atomParameters[ii][GtaIndex] < 0.0f && atomParameters[jj][adNIndex] < 0.0f)
            || (atomParameters[ii][adNIndex] < 0.0f && atomParameters[jj][GtaIndex] < 0.0f);
        double combinedSigma   = MMFFNonbondedForce::sigmaCombiningRule(atomParameters[ii][SigIndex], atomParameters[jj][SigIndex], haveDonor);
        double combinedEpsilon = MMFFNonbondedForce::epsilonCombiningRule(combinedSigma, fabs(atomParameters[ii][adNIndex]),
            fabs(atomParameters[jj][adNIndex]), fabs(atomParameters[ii][GtaIndex]), fabs(atomParameters[jj][GtaIndex]));
        if (haveDAPair)
            MMFFNonbondedForce::scaleSigmaEpsilon(combinedSigma, combinedEpsilon);
        double r_ij_2       = deltaR[0][ReferenceForce::R2Index];
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

        vdwEnergy           = combinedEpsilon*tau_7*sigma_7*((ghal+1.0)*sigma_7/rho - 2.0);
        double vdw_dEdR     = -7.0*(dtau*vdwEnergy + gtau);

        // tapering

        if (cutoff && r_ij > taperCutoff) {
            double delta    = r_ij - taperCutoff;
            double taper    = 1.0 + delta*delta*delta*(taperCoefficients[C3] + delta*(taperCoefficients[C4] + delta*taperCoefficients[C5]));
            double dtaper   = delta*delta*(3.0*taperCoefficients[C3] + delta*(4.0*taperCoefficients[C4] + delta*5.0*taperCoefficients[C5]));
            vdw_dEdR        = vdwEnergy*dtaper + vdw_dEdR*taper;
            vdwEnergy      *= taper;
        }

        dEdR += vdw_dEdR*inverseR;

        // accumulate forces

        for (int kk = 0; kk < 3; kk++) {
            double force  = dEdR*deltaR[0][kk];
            forces[ii][kk]   += force;
            forces[jj][kk]   -= force;
        }

        // accumulate energies

        realSpaceEwaldEnergy        = ONE_4PI_EPS0*atomParameters[ii][QIndex]*atomParameters[jj][QIndex]*inverseR*erfc(alphaR);

        totalVdwEnergy             += vdwEnergy;
        totalRealSpaceEwaldEnergy  += realSpaceEwaldEnergy;

        if (energyByAtom) {
            energyByAtom[ii] += realSpaceEwaldEnergy + vdwEnergy;
            energyByAtom[jj] += realSpaceEwaldEnergy + vdwEnergy;
        }

    }

    if (totalEnergy)
        *totalEnergy += totalRealSpaceEwaldEnergy + totalVdwEnergy;

    // Now subtract off the exclusions, since they were implicitly included in the reciprocal space sum.

    double totalExclusionEnergy = 0.0f;
    const double TWO_OVER_SQRT_PI = 2/sqrt(PI_M);
    for (int i = 0; i < numberOfAtoms; i++)
        for (int exclusion : exclusions[i]) {
            if (exclusion > i) {
                int ii = i;
                int jj = exclusion;

                double deltaR[2][ReferenceForce::LastDeltaRIndex];
                ReferenceForce::getDeltaR(atomCoordinates[jj], atomCoordinates[ii], deltaR[0]);
                double r         = deltaR[0][ReferenceForce::RIndex];
                double inverseR  = 1.0/(deltaR[0][ReferenceForce::RIndex]);
                double alphaR    = alphaEwald * r;
                if (erf(alphaR) > 1e-6) {
                    double dEdR = ONE_4PI_EPS0 * atomParameters[ii][QIndex] * atomParameters[jj][QIndex] * inverseR * inverseR * inverseR;
                    dEdR = dEdR * (erf(alphaR) - 2 * alphaR * exp (- alphaR * alphaR) / SQRT_PI);

                    // accumulate forces

                    for (int kk = 0; kk < 3; kk++) {
                        double force = dEdR*deltaR[0][kk];
                        forces[ii][kk] -= force;
                        forces[jj][kk] += force;
                    }

                    // accumulate energies

                    realSpaceEwaldEnergy = ONE_4PI_EPS0*atomParameters[ii][QIndex]*atomParameters[jj][QIndex]*inverseR*erf(alphaR);
                }
                else {
                    realSpaceEwaldEnergy = alphaEwald*TWO_OVER_SQRT_PI*ONE_4PI_EPS0*atomParameters[ii][QIndex]*atomParameters[jj][QIndex];
                }

                totalExclusionEnergy += realSpaceEwaldEnergy;
                if (energyByAtom) {
                    energyByAtom[ii] -= realSpaceEwaldEnergy;
                    energyByAtom[jj] -= realSpaceEwaldEnergy;
                }
            }
        }

    if (totalEnergy)
        *totalEnergy -= totalExclusionEnergy;
}


/**---------------------------------------------------------------------------------------

   Calculate vdW Coulomb pair ixn

   @param numberOfAtoms    number of atoms
   @param atomCoordinates  atom coordinates
   @param atomParameters   atom parameters                             atomParameters[atomIndex][paramterIndex]
   @param exclusions       atom exclusion indices
                           exclusions[atomIndex] contains the list of exclusions for that atom
   @param fixedParameters  non atom parameters (not currently used)
   @param forces           force array (forces added)
   @param energyByAtom     atom energy
   @param totalEnergy      total energy
   @param includeDirect      true if direct space interactions should be included
   @param includeReciprocal  true if reciprocal space interactions should be included

   --------------------------------------------------------------------------------------- */

void MMFFReferenceNonbondedForce::calculatePairIxn(int numberOfAtoms, vector<Vec3>& atomCoordinates,
                                             double** atomParameters, vector<set<int> >& exclusions,
                                             double* fixedParameters, vector<Vec3>& forces,
                                             double* energyByAtom, double* totalEnergy, bool includeDirect, bool includeReciprocal) const {

    if (ewald || pme) {
        calculateEwaldIxn(numberOfAtoms, atomCoordinates, atomParameters, exclusions, fixedParameters, forces, energyByAtom,
                          totalEnergy, includeDirect, includeReciprocal);
        return;
    }
    if (!includeDirect)
        return;
    if (cutoff) {
        for (auto& pair : *neighborList)
            calculateOneIxn(pair.first, pair.second, atomCoordinates, atomParameters, forces, energyByAtom, totalEnergy);
    }
    else {
        for (int ii = 0; ii < numberOfAtoms; ii++) {
            // loop over atom pairs

            for (int jj = ii+1; jj < numberOfAtoms; jj++)
                if (exclusions[jj].find(ii) == exclusions[jj].end())
                    calculateOneIxn(ii, jj, atomCoordinates, atomParameters, forces, energyByAtom, totalEnergy);
        }
    }
}

/**---------------------------------------------------------------------------------------

     Calculate vdW Coulomb pair ixn between two atoms

     @param ii               the index of the first atom
     @param jj               the index of the second atom
     @param atomCoordinates  atom coordinates
     @param atomParameters   atom parameters (charges, sigma, ...)     atomParameters[atomIndex][paramterIndex]
     @param forces           force array (forces added)
     @param energyByAtom     atom energy
     @param totalEnergy      total energy

     --------------------------------------------------------------------------------------- */

void MMFFReferenceNonbondedForce::calculateOneIxn(int ii, int jj, vector<Vec3>& atomCoordinates,
                                            double** atomParameters, vector<Vec3>& forces,
                                            double* energyByAtom, double* totalEnergy) const {
    double deltaR[2][ReferenceForce::LastDeltaRIndex];

    // get deltaR, R2, and R between 2 atoms

    if (periodic)
        ReferenceForce::getDeltaRPeriodic(atomCoordinates[jj], atomCoordinates[ii], periodicBoxVectors, deltaR[0]);
    else
        ReferenceForce::getDeltaR(atomCoordinates[jj], atomCoordinates[ii], deltaR[0]);

    bool haveDonor = (atomParameters[ii][GtaIndex] < 0.0 || atomParameters[jj][GtaIndex] < 0.0);
    bool haveDAPair = (atomParameters[ii][GtaIndex] < 0.0f && atomParameters[jj][adNIndex] < 0.0f)
        || (atomParameters[ii][adNIndex] < 0.0f && atomParameters[jj][GtaIndex] < 0.0f);
    double combinedSigma   = MMFFNonbondedForce::sigmaCombiningRule(atomParameters[ii][SigIndex], atomParameters[jj][SigIndex], haveDonor);
    double combinedEpsilon = MMFFNonbondedForce::epsilonCombiningRule(combinedSigma, fabs(atomParameters[ii][adNIndex]),
        fabs(atomParameters[jj][adNIndex]), fabs(atomParameters[ii][GtaIndex]), fabs(atomParameters[jj][GtaIndex]));
    if (haveDAPair)
        MMFFNonbondedForce::scaleSigmaEpsilon(combinedSigma, combinedEpsilon);
    double inverseR     = 1.0/(deltaR[0][ReferenceForce::RIndex]);
    double r_ij         = deltaR[0][ReferenceForce::RIndex];
    double r_ij_2       = deltaR[0][ReferenceForce::R2Index];
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

    if (cutoff && r_ij > taperCutoff) {
        double delta    = r_ij - taperCutoff;
        double taper    = 1.0 + delta*delta*delta*(taperCoefficients[C3] + delta*(taperCoefficients[C4] + delta*taperCoefficients[C5]));
        double dtaper   = delta*delta*(3.0*taperCoefficients[C3] + delta*(4.0*taperCoefficients[C4] + delta*5.0*taperCoefficients[C5]));
        dEdR            = energy*dtaper + dEdR*taper;
        energy         *= taper;
    }

    double coulomb_dEdR = cutoff
        ? ONE_4PI_EPS0*atomParameters[ii][QIndex]*atomParameters[jj][QIndex]*(inverseR-2.0f*krf*r_ij_2)
        : ONE_4PI_EPS0*atomParameters[ii][QIndex]*atomParameters[jj][QIndex]*inverseR;
    dEdR = (dEdR + coulomb_dEdR*inverseR)*inverseR;

    // accumulate forces

    for (int kk = 0; kk < 3; kk++) {
        double force  = dEdR*deltaR[0][kk];
        forces[ii][kk]   += force;
        forces[jj][kk]   -= force;
    }

    // accumulate energies

    if (totalEnergy)
        *totalEnergy += energy;
    if (energyByAtom) {
        energyByAtom[ii] += energy;
        energyByAtom[jj] += energy;
    }
}
