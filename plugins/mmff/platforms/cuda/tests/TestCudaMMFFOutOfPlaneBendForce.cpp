/* -------------------------------------------------------------------------- *
 *                                   OpenMMMMFF                             *
 * -------------------------------------------------------------------------- *
 * This is part of the OpenMM molecular simulation toolkit originating from   *
 * Simbios, the NIH National Center for Physics-Based Simulation of           *
 * Biological Structures at Stanford, funded under the NIH Roadmap for        *
 * Medical Research, grant U54 GM072970. See https://simtk.org.               *
 *                                                                            *
 * Portions copyright (c) 2008-2016 Stanford University and the Authors.      *
 * Authors: Mark Friedrichs                                                   *
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

/**
 * This tests the CUDA implementation of CudaMMFFOutOfPlaneBendForce.
 */

#include "openmm/internal/AssertionUtilities.h"
#include "openmm/Context.h"
#include "OpenMMMMFF.h"
#include "openmm/System.h"
#include "openmm/OpenMMException.h"
#include "openmm/LangevinIntegrator.h"
#include <iostream>
#include <vector>

using namespace OpenMM;

extern "C" void registerMMFFCudaKernelFactories();

const double TOL = 1e-3;
#define PI_M               3.141592653589
#define RADIAN            57.29577951308

/* ---------------------------------------------------------------------------------------

   Compute cross product of two 3-vectors and place in 3rd vector

   vectorZ = vectorX x vectorY

   @param vectorX             x-vector
   @param vectorY             y-vector
   @param vectorZ             z-vector

   @return vector is vectorZ

   --------------------------------------------------------------------------------------- */
     
static void crossProductVector3(double* vectorX, double* vectorY, double* vectorZ) {

    vectorZ[0]  = vectorX[1]*vectorY[2] - vectorX[2]*vectorY[1];
    vectorZ[1]  = vectorX[2]*vectorY[0] - vectorX[0]*vectorY[2];
    vectorZ[2]  = vectorX[0]*vectorY[1] - vectorX[1]*vectorY[0];

    return;
}

static double dotVector3(double* vectorX, double* vectorY) {
    return vectorX[0]*vectorY[0] + vectorX[1]*vectorY[1] + vectorX[2]*vectorY[2];
}


static void computeMMFFOutOfPlaneBendForce(int bondIndex,  std::vector<Vec3>& positions, MMFFOutOfPlaneBendForce& mmffOutOfPlaneBendForce,
                                             std::vector<Vec3>& forces, double* energy) {


    int particle1, particle2, particle3, particle4;
    double kAngleQuadratic;
    mmffOutOfPlaneBendForce.getOutOfPlaneBendParameters(bondIndex, particle1, particle2, particle3, particle4, kAngleQuadratic);

    enum { A, B, C, D, LastAtomIndex };
    enum { AB, CB, DB, AD, CD, LastDeltaIndex };
 
    // ---------------------------------------------------------------------------------------
 
    // get deltaR between various combinations of the 4 atoms
    // and various intermediate terms
 
    double deltaR[LastDeltaIndex][6];
    for (int ii = 0; ii < 3; ii++) {
         deltaR[AB][ii] = positions[particle1][ii] - positions[particle2][ii];
         deltaR[CB][ii] = positions[particle3][ii] - positions[particle2][ii];
         deltaR[DB][ii] = positions[particle4][ii] - positions[particle2][ii];
         deltaR[AD][ii] = positions[particle1][ii] - positions[particle4][ii];
         deltaR[CD][ii] = positions[particle3][ii] - positions[particle4][ii];
    }   

    double rDB2  = dotVector3(deltaR[DB], deltaR[DB]);
    double rAD2  = dotVector3(deltaR[AD], deltaR[AD]);
    double rCD2  = dotVector3(deltaR[CD], deltaR[CD]);
 
    double tempVector[3];
    crossProductVector3(deltaR[CB], deltaR[DB], tempVector);
    double   eE  = dotVector3(deltaR[AB], tempVector );
    double  dot  = dotVector3(deltaR[AD],  deltaR[CD]);
    double   cc  = rAD2*rCD2 - dot*dot;
 
    if (rDB2 <= 0.0 || cc == 0.0) {
       return;
    }
    double bkk2   = rDB2 - eE*eE/cc;
    double cosine = sqrt(bkk2/rDB2);
    double angle;
    if (cosine >= 1.0) {
        angle = 0.0;
    } else if (cosine <= -1.0) {
        angle = RADIAN*M_PI;
    } else {
        angle = RADIAN*acos(cosine);
    }

    // chain rule
 
    double dt    = angle;
    double dt2   = dt*dt;
 
    double dEdDt = kAngleQuadratic*dt;
 
    double dEdCos;
    dEdCos       = dEdDt/sqrt(cc*bkk2);
    if (eE > 0.0) {
        dEdCos *= -1.0;
    }
 
    double term = eE/cc;
 
    double dccd[LastAtomIndex][3];
    for (int ii = 0; ii < 3; ii++) {
        dccd[A][ii] = (deltaR[AD][ii]*rCD2 - deltaR[CD][ii]*dot)*term;
        dccd[C][ii] = (deltaR[CD][ii]*rAD2 - deltaR[AD][ii]*dot)*term;
        dccd[D][ii] = -1.0*(dccd[A][ii] + dccd[C][ii]);
    }
 
    double deed[LastAtomIndex][3];
    crossProductVector3(deltaR[DB], deltaR[CB], deed[A]);
    crossProductVector3(deltaR[AB], deltaR[DB], deed[C]);
    crossProductVector3(deltaR[CB], deltaR[AB], deed[D]);
 
    term        = eE/rDB2;
    deed[D][0] += deltaR[DB][0]*term;
    deed[D][1] += deltaR[DB][1]*term;
    deed[D][2] += deltaR[DB][2]*term;
 
    // ---------------------------------------------------------------------------------------
 
    // forces
 
    // calculate forces for atoms a, c, d
    // the force for b is then -(a+ c + d)
 
    double subForce[LastAtomIndex][3];
 
    for (int jj = 0; jj < LastAtomIndex; jj++) {
 
        // A, C, D
  
        for (int ii = 0; ii < 3; ii++) {
            subForce[jj][ii] = dEdCos*(dccd[jj][ii] + deed[jj][ii]);
        }
  
        if (jj == 0)jj++; // skip B
  
        // now compute B
  
        if (jj == 3) {
           for (int ii = 0; ii < 3; ii++) {
               subForce[1][ii] = -1.0*(subForce[0][ii] + subForce[2][ii] + subForce[3][ii]);
           }
        }
    }
 
    // accumulate forces and energy

    forces[particle1][0]       -= subForce[0][0];
    forces[particle1][1]       -= subForce[0][1];
    forces[particle1][2]       -= subForce[0][2];

    forces[particle2][0]       -= subForce[1][0];
    forces[particle2][1]       -= subForce[1][1];
    forces[particle2][2]       -= subForce[1][2];

    forces[particle3][0]       -= subForce[2][0];
    forces[particle3][1]       -= subForce[2][1];
    forces[particle3][2]       -= subForce[2][2];
    
    forces[particle4][0]       -= subForce[3][0];
    forces[particle4][1]       -= subForce[3][1];
    forces[particle4][2]       -= subForce[3][2];
    
    // ---------------------------------------------------------------------------------------
 
    // calculate energy if 'energy' is set
 
    double energyTerm  = 0.5*kAngleQuadratic*dt2;
    *energy           += energyTerm;
    return;
}
 
static void computeMMFFOutOfPlaneBendForces(Context& context, MMFFOutOfPlaneBendForce& mmffOutOfPlaneBendForce,
                                              std::vector<Vec3>& expectedForces, double* expectedEnergy) {

    // get positions and zero forces

    State state                 = context.getState(State::Positions);
    std::vector<Vec3> positions = state.getPositions();
    expectedForces.resize(positions.size());
    
    for (unsigned int ii = 0; ii < expectedForces.size(); ii++) {
        expectedForces[ii][0] = expectedForces[ii][1] = expectedForces[ii][2] = 0.0;
    }

    // calculates forces/energy

    *expectedEnergy = 0.0;
    for (int ii = 0; ii < mmffOutOfPlaneBendForce.getNumOutOfPlaneBends(); ii++) {
        computeMMFFOutOfPlaneBendForce(ii, positions, mmffOutOfPlaneBendForce, expectedForces, expectedEnergy);
    }
}

void compareWithExpectedForceAndEnergy(Context& context, MMFFOutOfPlaneBendForce& mmffOutOfPlaneBendForce,
                                       double tolerance, const std::string& idString) {

    std::vector<Vec3> expectedForces;
    double expectedEnergy;
    computeMMFFOutOfPlaneBendForces(context, mmffOutOfPlaneBendForce, expectedForces, &expectedEnergy);
   
    State state                      = context.getState(State::Forces | State::Energy);
    const std::vector<Vec3> forces   = state.getForces();
    for (unsigned int ii = 0; ii < forces.size(); ii++) {
        ASSERT_EQUAL_VEC(expectedForces[ii], forces[ii], tolerance);
    }
    ASSERT_EQUAL_TOL(expectedEnergy, state.getPotentialEnergy(), tolerance);
}

void testOneOutOfPlaneBend() {

    System system;
    int numberOfParticles = 4;
    for (int ii = 0; ii < numberOfParticles; ii++) {
        system.addParticle(1.0);
    }

    LangevinIntegrator integrator(0.0, 0.1, 0.01);

    MMFFOutOfPlaneBendForce* mmffOutOfPlaneBendForce = new MMFFOutOfPlaneBendForce();

    double kOutOfPlaneBend = 0.328682196E-01;
    mmffOutOfPlaneBendForce->addOutOfPlaneBend(0, 1, 2, 3, kOutOfPlaneBend);

    system.addForce(mmffOutOfPlaneBendForce);
    Context context(system, integrator, Platform::getPlatformByName("CUDA"));

    std::vector<Vec3> positions(numberOfParticles);

    positions[0] = Vec3(0.262660000E+02,  0.254130000E+02,  0.284200000E+01);
    positions[1] = Vec3(0.269130000E+02,  0.266390000E+02,  0.353100000E+01);

    positions[2] = Vec3(0.278860000E+02,  0.264630000E+02,  0.426300000E+01);
    positions[3] = Vec3(0.245568230E+02,  0.250215290E+02,  0.796852800E+01);

    context.setPositions(positions);
    compareWithExpectedForceAndEnergy(context, *mmffOutOfPlaneBendForce, TOL, "testOneOutOfPlaneBend");
    
    // Try changing the bend parameters and make sure it's still correct.
    
    mmffOutOfPlaneBendForce->setOutOfPlaneBendParameters(0, 0, 1, 2, 3, 1.1*kOutOfPlaneBend);
    bool exceptionThrown = false;
    try {
        // This should throw an exception.
        compareWithExpectedForceAndEnergy(context, *mmffOutOfPlaneBendForce, TOL, "testOneOutOfPlaneBend");
    }
    catch (std::exception ex) {
        exceptionThrown = true;
    }
    ASSERT(exceptionThrown);
    mmffOutOfPlaneBendForce->updateParametersInContext(context);
    compareWithExpectedForceAndEnergy(context, *mmffOutOfPlaneBendForce, TOL, "testOneOutOfPlaneBend");
}

void testOneOutOfPlaneBend2(int setId) {

    System system;
    int numberOfParticles = 4;
    for (int ii = 0; ii < numberOfParticles; ii++) {
        system.addParticle(1.0);
    }

    LangevinIntegrator integrator(0.0, 0.1, 0.01);

    MMFFOutOfPlaneBendForce* mmffOutOfPlaneBendForce = new MMFFOutOfPlaneBendForce();
/*
   285    441    442    443    444  0.328682196E-01
   286    441    442    444    443  0.164493407E-01
   287    443    442    444    441  0.636650407E-02
   288    442    444    447    448  0.392956472E-02
   289    442    444    448    447  0.392956472E-02
   290    447    444    448    442  0.214755281E-01
  441  0.893800000E+01  0.439800000E+01  0.343100000E+01
  442  0.779100000E+01  0.614600000E+01  0.390100000E+01
  443  0.915400000E+01  0.683900000E+01  0.389400000E+01
  444  0.101770000E+02  0.619000000E+01  0.379900000E+01
  445  0.921000000E+01  0.813800000E+01  0.398600000E+01
  446  0.708500000E+01  0.672900000E+01  0.332700000E+01
  447  0.744300000E+01  0.605200000E+01  0.491900000E+01
  448  0.100820000E+02  0.859300000E+01  0.398200000E+01
  449  0.838000000E+01  0.866100000E+01  0.406000000E+01
*/

    std::map<int,Vec3> coordinates;
    coordinates[440] = Vec3( 0.893800000E+01,  0.439800000E+01,  0.343100000E+01);
    coordinates[441] = Vec3( 0.779100000E+01,  0.614600000E+01,  0.390100000E+01);
    coordinates[442] = Vec3( 0.915400000E+01,  0.683900000E+01,  0.389400000E+01);
    coordinates[443] = Vec3( 0.101770000E+02,  0.619000000E+01,  0.379900000E+01);
    coordinates[444] = Vec3( 0.921000000E+01,  0.813800000E+01,  0.398600000E+01);
    coordinates[445] = Vec3( 0.708500000E+01,  0.672900000E+01,  0.332700000E+01);
    coordinates[446] = Vec3( 0.744300000E+01,  0.605200000E+01,  0.491900000E+01);
    coordinates[447] = Vec3( 0.100820000E+02,  0.859300000E+01,  0.398200000E+01);
    coordinates[448] = Vec3( 0.838000000E+01,  0.866100000E+01,  0.406000000E+01);
    
    double kOutOfPlaneBend = 0.328682196E-01;
    std::vector<int> particleIndices;
    if (setId == 1) {
        particleIndices.push_back(441); 
        particleIndices.push_back(442); 
        particleIndices.push_back(443); 
        particleIndices.push_back(444); 
        kOutOfPlaneBend = 0.328682196E-01;
    } else if (setId == 2) {
        particleIndices.push_back(441); 
        particleIndices.push_back(442); 
        particleIndices.push_back(444); 
        particleIndices.push_back(443); 
        kOutOfPlaneBend = 0.164493407E-01;
    } else if (setId == 3) {
        particleIndices.push_back(443); 
        particleIndices.push_back(442); 
        particleIndices.push_back(444); 
        particleIndices.push_back(441); 
        kOutOfPlaneBend = 0.636650407E-02;
    } else if (setId == 4) {
        particleIndices.push_back(442); 
        particleIndices.push_back(444); 
        particleIndices.push_back(447); 
        particleIndices.push_back(448); 
        kOutOfPlaneBend = 0.392956472E-02;
    } else if (setId == 5) {
        particleIndices.push_back(442); 
        particleIndices.push_back(444); 
        particleIndices.push_back(448); 
        particleIndices.push_back(447); 
        kOutOfPlaneBend = 0.392956472E-02;
    } else if (setId == 6) {
        particleIndices.push_back(447); 
        particleIndices.push_back(444); 
        particleIndices.push_back(448); 
        particleIndices.push_back(442); 
        kOutOfPlaneBend = 0.214755281E-01;
    } else {
        std::stringstream buffer;
        buffer << "Set id " << setId << " not recognized.";
        throw OpenMMException(buffer.str());
    }
    mmffOutOfPlaneBendForce->addOutOfPlaneBend(0, 1, 2, 3, kOutOfPlaneBend);

    system.addForce(mmffOutOfPlaneBendForce);
    Context context(system, integrator, Platform::getPlatformByName("CUDA"));
    std::vector<Vec3> positions(numberOfParticles);

    for (unsigned int ii = 0; ii < numberOfParticles; ii++) {
        if (coordinates.find(particleIndices[ii]) == coordinates.end()) {
            std::stringstream buffer;
            buffer << "Coordinates " << particleIndices[ii] << " not loaded.";
            throw OpenMMException(buffer.str());
        }
        positions[ii] = coordinates[particleIndices[ii]];
    }

    context.setPositions(positions);
    compareWithExpectedForceAndEnergy(context, *mmffOutOfPlaneBendForce, TOL, "testOneOutOfPlaneBend");

    static int iter = 0;
    static std::map<int,Vec3> totalForces;
    static double totalEnergy;
    if (iter == 0) {

        totalForces[441] = Vec3( 0.0, 0.0, 0.0);
        totalForces[442] = Vec3( 0.0, 0.0, 0.0);
        totalForces[443] = Vec3( 0.0, 0.0, 0.0);
        totalForces[444] = Vec3( 0.0, 0.0, 0.0);
        totalForces[445] = Vec3( 0.0, 0.0, 0.0);
        totalForces[446] = Vec3( 0.0, 0.0, 0.0);
        totalForces[447] = Vec3( 0.0, 0.0, 0.0);
        totalForces[448] = Vec3( 0.0, 0.0, 0.0);
        totalForces[449] = Vec3( 0.0, 0.0, 0.0);
        totalEnergy      = 0.0;
    }
    iter++;

    std::vector<Vec3> forces;
    forces.resize(numberOfParticles);
    double energy;
    computeMMFFOutOfPlaneBendForce(0, positions, *mmffOutOfPlaneBendForce, forces, &energy);

    totalEnergy += energy;
    for (unsigned int ii = 0; ii < numberOfParticles; ii++) {
        for (unsigned int jj = 0; jj < 3; jj++) {
            totalForces[particleIndices[ii]][jj] += forces[ii][jj]; 
        }
    }
}

void testPeriodic() {
    // Create a force that uses periodic boundary conditions.

    System system;
    system.setDefaultPeriodicBoxVectors(Vec3(3, 0, 0), Vec3(0, 3, 0), Vec3(0, 0, 3));
    int numberOfParticles = 4;
    for (int ii = 0; ii < numberOfParticles; ii++)
        system.addParticle(1.0);
    LangevinIntegrator integrator(0.0, 0.1, 0.01);
    MMFFOutOfPlaneBendForce* mmffOutOfPlaneBendForce = new MMFFOutOfPlaneBendForce();
    double kOutOfPlaneBend = 0.328682196E-01;
    mmffOutOfPlaneBendForce->addOutOfPlaneBend(0, 1, 2, 3, kOutOfPlaneBend);
    mmffOutOfPlaneBendForce->setUsesPeriodicBoundaryConditions(true);
    system.addForce(mmffOutOfPlaneBendForce);
    Context context(system, integrator, Platform::getPlatformByName("CUDA"));
    std::vector<Vec3> positions(numberOfParticles);
    positions[0] = Vec3(0, 0, 0);
    positions[1] = Vec3(1, 0, 0);
    positions[2] = Vec3(0, 1, 0);
    positions[3] = Vec3(0, 0, 1);
    context.setPositions(positions);
    State s1 = context.getState(State::Forces | State::Energy);
    
    // Move one atom to a position that should give identical results.

    positions[3] = Vec3(0, 0, -2);
    context.setPositions(positions);
    State s2 = context.getState(State::Forces | State::Energy);
    ASSERT_EQUAL_TOL(s1.getPotentialEnergy(), s2.getPotentialEnergy(), 1e-5);
    for (int i = 0; i < numberOfParticles; i++)
        ASSERT_EQUAL_VEC(s1.getForces()[i], s2.getForces()[i], 1e-5);
}

int main(int argc, char* argv[]) {
    try {
        std::cout << "TestCudaMMFFOutOfPlaneBendForce running test..." << std::endl;
        registerMMFFCudaKernelFactories();
        if (argc > 1)
            Platform::getPlatformByName("CUDA").setPropertyDefaultValue("Precision", std::string(argv[1]));

        testOneOutOfPlaneBend();
        testPeriodic();

    } catch(const std::exception& e) {
        std::cout << "exception: " << e.what() << std::endl;
        std::cout << "FAIL - ERROR.  Test failed." << std::endl;
        return 1;
    }
    std::cout << "Done" << std::endl;
    return 0;
}
