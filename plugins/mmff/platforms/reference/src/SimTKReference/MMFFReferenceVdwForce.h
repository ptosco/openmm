
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

#ifndef __MMFFReferenceVdwForce_H__
#define __MMFFReferenceVdwForce_H__

#include "openmm/Vec3.h"
#include "ReferenceNeighborList.h"
#include <string>
#include <vector>

namespace OpenMM {

class MMFFReferenceVdwForce;

// ---------------------------------------------------------------------------------------

class MMFFReferenceVdwForce {

public:

    /** 
     * This is an enumeration of the different methods that may be used for handling long range Vdw forces.
     */
    enum NonbondedMethod {

        /**
         * No cutoff is applied to the interactions.  The full set of N^2 interactions is computed exactly.
         * This necessarily means that periodic boundary conditions cannot be used.  This is the default.
         */

        NoCutoff = 0,

        /**
         * Interactions beyond the cutoff distance are ignored.  
         */
        CutoffNonPeriodic = 1,
        /**
         * Periodic boundary conditions are used, so that each particle interacts only with the nearest periodic copy of
         * each other particle.  Interactions beyond the cutoff distance are ignored.  
         */
        CutoffPeriodic = 2,
    };
 
    /**---------------------------------------------------------------------------------------
       
       Constructor
       
       --------------------------------------------------------------------------------------- */
 
    MMFFReferenceVdwForce();
 
    /**---------------------------------------------------------------------------------------
       
       Destructor
       
       --------------------------------------------------------------------------------------- */
 
    ~MMFFReferenceVdwForce() {};
 
    /**---------------------------------------------------------------------------------------
    
       Get nonbonded method
    
       @return nonbonded method
    
       --------------------------------------------------------------------------------------- */
    
    NonbondedMethod getNonbondedMethod() const;

    /**---------------------------------------------------------------------------------------
    
       Set nonbonded method
    
       @param nonbonded method
    
       --------------------------------------------------------------------------------------- */
    
    void setNonbondedMethod(NonbondedMethod nonbondedMethod);

    /**---------------------------------------------------------------------------------------
    
       Get cutoff
    
       @return cutoff
    
       --------------------------------------------------------------------------------------- */
    
    double getCutoff() const;

    /**---------------------------------------------------------------------------------------
    
       Set cutoff
    
       @param cutoff
    
       --------------------------------------------------------------------------------------- */
    
    void setCutoff(double cutoff);

    /**---------------------------------------------------------------------------------------
    
       Set box dimensions
    
       @param vectors    the vectors defining the periodic box
    
       --------------------------------------------------------------------------------------- */
    
    void setPeriodicBox(OpenMM::Vec3* vectors);

    /**---------------------------------------------------------------------------------------
    
       Calculate MMFF Hal vdw ixns
    
       @param numParticles            number of particles
       @param particlePositions       Cartesian coordinates of particles
       @param sigmas                  particle sigmas 
       @param G_t_alphas              particle G*alpha
       @param alpha_d_Ns              particle alpha/N
       @param vdwDAs                  particle DA status ('-', 'D', 'A')
       @param vdwExclusions           particle exclusions
       @param forces                  add forces to this vector
    
       @return energy
    
       --------------------------------------------------------------------------------------- */
    
    double calculateForceAndEnergy(int numParticles, const std::vector<OpenMM::Vec3>& particlePositions,
                                   const std::vector<double>& sigmas,
                                   const std::vector<double>& G_t_alphas,
                                   const std::vector<double>& alpha_d_Ns,
                                   const std::vector<char>& vdwDAs,
                                   const std::vector< std::set<int> >& vdwExclusions,
                                   std::vector<OpenMM::Vec3>& forces) const;
         
    /**---------------------------------------------------------------------------------------
    
       Calculate Vdw ixn using neighbor list
    
       @param numParticles            number of particles
       @param particlePositions       Cartesian coordinates of particles
       @param sigmas                  particle sigmas 
       @param G_t_alphas              particle G*alpha
       @param alpha_d_Ns              particle alpha/N
       @param vdwDAs                  particle DA status ('-', 'D', 'A')
       @param neighborList            neighbor list
       @param forces                  add forces to this vector
    
       @return energy
    
       --------------------------------------------------------------------------------------- */
    
    double calculateForceAndEnergy(int numParticles, const std::vector<OpenMM::Vec3>& particlePositions, 
                                   const std::vector<double>& sigmas,
                                   const std::vector<double>& G_t_alphas,
                                   const std::vector<double>& alpha_d_Ns,
                                   const std::vector<char>& vdwDAs,
                                   const NeighborList& neighborList,
                                   std::vector<OpenMM::Vec3>& forces) const;
         
private:

    // taper coefficient indices

    static const int C3=0;
    static const int C4=1;
    static const int C5=2;

    NonbondedMethod _nonbondedMethod;
    double _cutoff;
    double _taperCutoffFactor;
    double _taperCutoff;
    double _taperCoefficients[3];
    Vec3 _periodicBoxVectors[3];
    static double _mmffSigmaCombiningRule(double sigmaI, double sigmaJ, char vdwDAI, char vdwDAJ);
    static double _mmffEpsilonCombiningRule(double combinedSigma,
        double alphaI_d_NI, double alphaJ_d_NJ, double GI_t_alphaI, double GJ_t_alphaJ);

    /**---------------------------------------------------------------------------------------
    
       Set taper coefficients
    
       @param  cutoff cutoff

       --------------------------------------------------------------------------------------- */
    
    void setTaperCoefficients(double cutoff);

    /**---------------------------------------------------------------------------------------
    
       Calculate pair ixn
    
       @param  combinedSigma        combined sigmas
       @param  combinedEpsilon      combined epsilons
       @param  particleIPosition    particle I position 
       @param  particleJPosition    particle J position 
       @param  force                output force
    
       @return energy for ixn

       --------------------------------------------------------------------------------------- */
    
    double calculatePairIxn(double combinedSigma, double combinedEpsilon,
                            const Vec3& particleIPosition, const Vec3& particleJPosition,
                            Vec3& force) const;

};

}
// ---------------------------------------------------------------------------------------

#endif // _MMFFReferenceVdwForce___
