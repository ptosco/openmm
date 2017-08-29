/* -------------------------------------------------------------------------- *
 *                              OpenMMMMFF                                  *
 * -------------------------------------------------------------------------- *
 * This is part of the OpenMM molecular simulation toolkit originating from   *
 * Simbios, the NIH National Center for Physics-Based Simulation of           *
 * Biological Structures at Stanford, funded under the NIH Roadmap for        *
 * Medical Research, grant U54 GM072970. See https://simtk.org.               *
 *                                                                            *
 * Portions copyright (c) 2008-2016 Stanford University and the Authors.      *
 * Authors: Mark Friedrichs, Peter Eastman                                    *
 * Contributors:                                                              *
 *                                                                            *
 * This program is free software: you can redistribute it and/or modify       *
 * it under the terms of the GNU Lesser General Public License as published   *
 * by the Free Software Foundation, either version 3 of the License, or       *
 * (at your option) any later version.                                        *
 *                                                                            *
 * This program is distributed in the hope that it will be useful,            *
 * but WITHOUT ANY WARRANTY; without even the implied warranty of             *
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the              *
 * GNU Lesser General Public License for more details.                        *
 *                                                                            *
 * You should have received a copy of the GNU Lesser General Public License   *
 * along with this program.  If not, see <http://www.gnu.org/licenses/>.      *
 * -------------------------------------------------------------------------- */

#include "MMFFCudaKernelFactory.h"
#include "MMFFCudaKernels.h"
#include "CudaPlatform.h"
#include "openmm/internal/ContextImpl.h"
#include "openmm/OpenMMException.h"
#include "openmm/internal/windowsExport.h"

using namespace OpenMM;

#ifdef OPENMM_BUILDING_STATIC_LIBRARY
static void registerPlatforms() {
#else
extern "C" OPENMM_EXPORT void registerPlatforms() {
#endif
}

#ifdef OPENMM_BUILDING_STATIC_LIBRARY
static void registerKernelFactories() {
#else
extern "C" OPENMM_EXPORT void registerKernelFactories() {
#endif
    try {
        Platform& platform = Platform::getPlatformByName("CUDA");
        MMFFCudaKernelFactory* factory = new MMFFCudaKernelFactory();
        platform.registerKernelFactory(CalcMMFFBondForceKernel::Name(), factory);
        platform.registerKernelFactory(CalcMMFFAngleForceKernel::Name(), factory);
        platform.registerKernelFactory(CalcMMFFPiTorsionForceKernel::Name(), factory);
        platform.registerKernelFactory(CalcMMFFStretchBendForceKernel::Name(), factory);
        platform.registerKernelFactory(CalcMMFFOutOfPlaneBendForceKernel::Name(), factory);
        platform.registerKernelFactory(CalcMMFFTorsionTorsionForceKernel::Name(), factory);
        platform.registerKernelFactory(CalcMMFFMultipoleForceKernel::Name(), factory);
        platform.registerKernelFactory(CalcMMFFGeneralizedKirkwoodForceKernel::Name(), factory);
        platform.registerKernelFactory(CalcMMFFVdwForceKernel::Name(), factory);
        platform.registerKernelFactory(CalcMMFFWcaDispersionForceKernel::Name(), factory);
    }
    catch (...) {
        // Ignore.  The CUDA platform isn't available.
    }
}

extern "C" OPENMM_EXPORT void registerMMFFCudaKernelFactories() {
    try {
        Platform::getPlatformByName("CUDA");
    }
    catch (...) {
        Platform::registerPlatform(new CudaPlatform());
    }
    registerKernelFactories();
}

KernelImpl* MMFFCudaKernelFactory::createKernelImpl(std::string name, const Platform& platform, ContextImpl& context) const {
    CudaPlatform::PlatformData& data = *static_cast<CudaPlatform::PlatformData*>(context.getPlatformData());
    CudaContext& cu = *data.contexts[0];

    if (name == CalcMMFFBondForceKernel::Name())
        return new CudaCalcMMFFBondForceKernel(name, platform, cu, context.getSystem());

    if (name == CalcMMFFAngleForceKernel::Name())
        return new CudaCalcMMFFAngleForceKernel(name, platform, cu, context.getSystem());

    if (name == CalcMMFFPiTorsionForceKernel::Name())
        return new CudaCalcMMFFPiTorsionForceKernel(name, platform, cu, context.getSystem());

    if (name == CalcMMFFStretchBendForceKernel::Name())
        return new CudaCalcMMFFStretchBendForceKernel(name, platform, cu, context.getSystem());

    if (name == CalcMMFFOutOfPlaneBendForceKernel::Name())
        return new CudaCalcMMFFOutOfPlaneBendForceKernel(name, platform, cu, context.getSystem());

    if (name == CalcMMFFTorsionTorsionForceKernel::Name())
        return new CudaCalcMMFFTorsionTorsionForceKernel(name, platform, cu, context.getSystem());

    if (name == CalcMMFFMultipoleForceKernel::Name())
        return new CudaCalcMMFFMultipoleForceKernel(name, platform, cu, context.getSystem());

    if (name == CalcMMFFGeneralizedKirkwoodForceKernel::Name())
        return new CudaCalcMMFFGeneralizedKirkwoodForceKernel(name, platform, cu, context.getSystem());

    if (name == CalcMMFFVdwForceKernel::Name())
        return new CudaCalcMMFFVdwForceKernel(name, platform, cu, context.getSystem());

    if (name == CalcMMFFWcaDispersionForceKernel::Name())
        return new CudaCalcMMFFWcaDispersionForceKernel(name, platform, cu, context.getSystem());

    throw OpenMMException((std::string("Tried to create kernel with illegal kernel name '")+name+"'").c_str());
}