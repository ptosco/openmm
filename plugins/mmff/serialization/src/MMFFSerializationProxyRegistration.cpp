/* -------------------------------------------------------------------------- *
 *                                OpenMMMMFF                                *
 * -------------------------------------------------------------------------- *
 * This is part of the OpenMM molecular simulation toolkit originating from   *
 * Simbios, the NIH National Center for Physics-Based Simulation of           *
 * Biological Structures at Stanford, funded under the NIH Roadmap for        *
 * Medical Research, grant U54 GM072970. See https://simtk.org.               *
 *                                                                            *
 * Portions copyright (c) 2010 Stanford University and the Authors.           *
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
#include <windows.h>
#include <sstream>
#else
#include <dlfcn.h>
#include <dirent.h>
#include <cstdlib>
#endif

#include "openmm/OpenMMException.h"

#include "openmm/MMFFGeneralizedKirkwoodForce.h"
#include "openmm/MMFFBondForce.h"
#include "openmm/MMFFAngleForce.h"
#include "openmm/MMFFMultipoleForce.h"
#include "openmm/MMFFOutOfPlaneBendForce.h"
#include "openmm/MMFFStretchBendForce.h"
#include "openmm/MMFFVdwForce.h"
#include "openmm/MMFFWcaDispersionForce.h"

#include "openmm/serialization/SerializationProxy.h"

#include "openmm/serialization/MMFFGeneralizedKirkwoodForceProxy.h"
#include "openmm/serialization/MMFFBondForceProxy.h"
#include "openmm/serialization/MMFFAngleForceProxy.h"
#include "openmm/serialization/MMFFMultipoleForceProxy.h"
#include "openmm/serialization/MMFFOutOfPlaneBendForceProxy.h"
#include "openmm/serialization/MMFFStretchBendForceProxy.h"
#include "openmm/serialization/MMFFVdwForceProxy.h"
#include "openmm/serialization/MMFFWcaDispersionForceProxy.h"

#if defined(WIN32)
    #include <windows.h>
    extern "C" OPENMM_EXPORT_MMFF void registerMMFFSerializationProxies();
    BOOL WINAPI DllMain(HANDLE hModule, DWORD  ul_reason_for_call, LPVOID lpReserved) {
        if (ul_reason_for_call == DLL_PROCESS_ATTACH)
            registerMMFFSerializationProxies();
        return TRUE;
    }
#else
    extern "C" void __attribute__((constructor)) registerMMFFSerializationProxies();
#endif

using namespace OpenMM;

extern "C" OPENMM_EXPORT_MMFF void registerMMFFSerializationProxies() {
    SerializationProxy::registerProxy(typeid(MMFFGeneralizedKirkwoodForce),         new MMFFGeneralizedKirkwoodForceProxy());
    SerializationProxy::registerProxy(typeid(MMFFBondForce),                new MMFFBondForceProxy());
    SerializationProxy::registerProxy(typeid(MMFFAngleForce),               new MMFFAngleForceProxy());
    SerializationProxy::registerProxy(typeid(MMFFMultipoleForce),                   new MMFFMultipoleForceProxy());
    SerializationProxy::registerProxy(typeid(MMFFOutOfPlaneBendForce),              new MMFFOutOfPlaneBendForceProxy());
    SerializationProxy::registerProxy(typeid(MMFFStretchBendForce),                 new MMFFStretchBendForceProxy());
    SerializationProxy::registerProxy(typeid(MMFFVdwForce),                         new MMFFVdwForceProxy());
    SerializationProxy::registerProxy(typeid(MMFFWcaDispersionForce),               new MMFFWcaDispersionForceProxy());
}
