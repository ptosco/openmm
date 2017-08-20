float2 bondParams = PARAMS[index];
real deltaIdeal = r-bondParams.x;
real deltaIdeal2 = deltaIdeal*deltaIdeal;
energy += 0.5f*MMFF_BOND_C1*bondParams.y*deltaIdeal2*(1.0f + CUBIC_K*deltaIdeal + QUARTIC_K*deltaIdeal2);
real dEdR = MMFF_BOND_C1*bondParams.y*deltaIdeal*(1.0f + 1.5f*CUBIC_K*deltaIdeal + 2.0f*QUARTIC_K*deltaIdeal2);
