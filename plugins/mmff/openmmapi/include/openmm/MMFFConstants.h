
/* Portions copyright (c) 2006-2016 Stanford University and Simbios.
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

#ifndef __MMFFConstants_H__
#define __MMFFConstants_H__

#include "openmm/Units.h"

namespace OpenMM {

static const double MDYNE_A_TO_KCAL_MOL = 143.9325;
static const double MMFF_BOND_C1 = MDYNE_A_TO_KCAL_MOL * KJPerKcal * AngstromsPerNm * AngstromsPerNm;
static const double MMFF_BOND_CUBIC_K = -2.0 * AngstromsPerNm;
static const double MMFF_BOND_QUARTIC_K = 7.0 / 12.0 * MMFF_BOND_CUBIC_K * MMFF_BOND_CUBIC_K;
static const double MMFF_ANGLE_C2 = 0.5 * MDYNE_A_TO_KCAL_MOL * DEGREE_TO_RADIAN * DEGREE_TO_RADIAN;
static const double MMFF_ANGLE_CUBIC_K = -0.006981317;

} // namespace OpenMM

#endif // __MMFFConstants_H__
