{
    const real eleBuf = 0.005f;
    real rBuf = r + eleBuf;
    real rBufInv = RECIP(rBuf);
#if USE_EWALD
    bool needCorrection = hasExclusions && isExcluded && atom1 != atom2 && atom1 < NUM_ATOMS && atom2 < NUM_ATOMS;
    bool includeInteraction = ((!isExcluded && r2 < CUTOFF_SQUARED) || needCorrection);
    const real alphaR = EWALD_ALPHA*rBuf;
    const real expAlphaRSqr = EXP(-alphaR*alphaR);
    const real prefactor = 138.935456f*posq1.w*posq2.w*rBufInv;

#ifdef USE_DOUBLE_PRECISION
    const real erfcAlphaR = erfc(alphaR);
#else
    // This approximation for erfc is from Abramowitz and Stegun (1964) p. 299.  They cite the following as
    // the original source: C. Hastings, Jr., Approximations for Digital Computers (1955).  It has a maximum
    // error of 1.5e-7.

    const real t = RECIP(1.0f+0.3275911f*alphaR);
    const real erfcAlphaR = (0.254829592f+(-0.284496736f+(1.421413741f+(-1.453152027f+1.061405429f*t)*t)*t)*t)*t*expAlphaRSqr;
#endif
    real tempForce = 0.0f;
    if (needCorrection) {
        // Subtract off the part of this interaction that was included in the reciprocal space contribution.

        if (1-erfcAlphaR > 1e-6) {
            real erfAlphaR = ERF(alphaR); // Our erfc approximation is not accurate enough when r is very small, which happens with Drude particles.
            tempForce = -prefactor*(erfAlphaR-alphaR*expAlphaRSqr*TWO_OVER_SQRT_PI);
            tempEnergy += -prefactor*erfAlphaR;
        }
        else {
            includeInteraction = false;
            tempEnergy -= TWO_OVER_SQRT_PI*EWALD_ALPHA*138.935456f*posq1.w*posq2.w;
        }
    }
    else {
#if HAS_VDW
    if (includeInteraction) {
        bool haveDonor = false;
        bool haveDAPair = (params1.y < 0.0f && params2.z < 0.0f)
            || (params1.z < 0.0f && params2.y < 0.0f);
        real G_t_alpha1 = params1.y;
        if (params1.y < 0.0f) {
            haveDonor = true;
            G_t_alpha1 = -G_t_alpha1;
        }
        real G_t_alpha2 = params2.y;
        if (params2.y < 0.0f) {
            haveDonor = true;
            G_t_alpha2 = -G_t_alpha2;
        }
        const real B = 0.2f;
        const real Beta = 12.0f;
        const real c = 7.5797344e-4f;
        const real DARAD = 0.8f;
        const real DAEPS = 0.5f;
        real gamma = (params1.x - params2.x) / (params1.x + params2.x);
        real sigma = 0.5f * (params1.x + params2.x) * (1.0f + (haveDonor
                     ? 0.0f : B * (1.0f - exp(-Beta * gamma * gamma))));
        real alpha_d_N1 = ((params1.z < 0.0f) ? -params1.z : params1.z);
        real alpha_d_N2 = ((params2.z < 0.0f) ? -params2.z : params2.z);
        real sigmaSq = sigma * sigma;
        real epsilon = c * G_t_alpha1 * G_t_alpha2
                       / ((sqrt(alpha_d_N1) + sqrt(alpha_d_N2)) * sigmaSq * sigmaSq * sigmaSq);
        if (haveDAPair) {
            sigma *= DARAD;
            epsilon *= DAEPS;
        }
        real r6 = r2*r2*r2;
        real r7 = r6*r;
        real sigma7 = sigma*sigma;
        sigma7 = sigma7*sigma7*sigma7*sigma;
        real rho = r7 + sigma7*0.12f;
        real invRho = RECIP(rho);
        real tau = 1.07f/(r + 0.07f*sigma);
        real tau7 = tau*tau*tau;
        tau7 = tau7*tau7*tau;
        real dTau = tau/1.07f;
        real tmp = sigma7*invRho;
        real gTau = epsilon*tau7*r6*1.12f*tmp*tmp;
        real vdwEnergy = epsilon*sigma7*tau7*((sigma7*1.12f*invRho)-2.0f);
        real deltaE = -7.0f*(dTau*vdwEnergy+gTau);
    #if USE_VDW_SWITCH
        if (r > VDW_SWITCH_CUTOFF) {
            real x = r-VDW_SWITCH_CUTOFF;
            real taper = 1+x*x*x*(VDW_SWITCH_C3+x*(VDW_SWITCH_C4+x*VDW_SWITCH_C5));
            real dtaper = x*x*(3*VDW_SWITCH_C3+x*(4*VDW_SWITCH_C4+x*5*VDW_SWITCH_C5));
            deltaE = vdwEnergy*dtaper + deltaE*taper;
            vdwEnergy *= taper;
        }
    #endif
        dEdR -= deltaE*invR;
        tempEnergy += vdwEnergy + prefactor*erfcAlphaR;
    }
#else
        tempEnergy += includeInteraction ? prefactor*erfcAlphaR : 0;
#endif
        tempForce = prefactor*(erfcAlphaR+alphaR*expAlphaRSqr*TWO_OVER_SQRT_PI);
    }
    dEdR += includeInteraction ? tempForce*rBufInv*rBufInv : 0;
#else
#ifdef USE_CUTOFF
    bool includeInteraction = (!isExcluded && r2 < CUTOFF_SQUARED);
#else
    bool includeInteraction = (!isExcluded);
#endif
    real tempForce = 0.0f;
  #if HAS_VDW
    if (includeInteraction) {
        bool haveDonor = false;
        bool haveDAPair = (params1.y < 0.0f && params2.z < 0.0f)
            || (params1.z < 0.0f && params2.y < 0.0f);
        real G_t_alpha1 = params1.y;
        if (params1.y < 0.0f) {
            haveDonor = true;
            G_t_alpha1 = -G_t_alpha1;
        }
        real G_t_alpha2 = params2.y;
        if (params2.y < 0.0f) {
            haveDonor = true;
            G_t_alpha2 = -G_t_alpha2;
        }
        const real B = 0.2f;
        const real Beta = 12.0f;
        const real c = 7.5797344e-4f;
        const real DARAD = 0.8f;
        const real DAEPS = 0.5f;
        real gamma = (params1.x - params2.x) / (params1.x + params2.x);
        real sigma = 0.5f * (params1.x + params2.x) * (1.0f + (haveDonor
                     ? 0.0f : B * (1.0f - exp(-Beta * gamma * gamma))));
        real alpha_d_N1 = ((params1.z < 0.0f) ? -params1.z : params1.z);
        real alpha_d_N2 = ((params2.z < 0.0f) ? -params2.z : params2.z);
        real sigmaSq = sigma * sigma;
        real epsilon = c * G_t_alpha1 * G_t_alpha2
                       / ((sqrt(alpha_d_N1) + sqrt(alpha_d_N2)) * sigmaSq * sigmaSq * sigmaSq);
        if (haveDAPair) {
            sigma *= DARAD;
            epsilon *= DAEPS;
        }
        real r6 = r2*r2*r2;
        real r7 = r6*r;
        real sigma7 = sigma*sigma;
        sigma7 = sigma7*sigma7*sigma7*sigma;
        real rho = r7 + sigma7*0.12f;
        real invRho = RECIP(rho);
        real tau = 1.07f/(r + 0.07f*sigma);
        real tau7 = tau*tau*tau;
        tau7 = tau7*tau7*tau;
        real dTau = tau/1.07f;
        real tmp = sigma7*invRho;
        real gTau = epsilon*tau7*r6*1.12f*tmp*tmp;
        real vdwEnergy = epsilon*sigma7*tau7*((sigma7*1.12f*invRho)-2.0f);
        real deltaE = -7.0f*(dTau*vdwEnergy+gTau);
      #if USE_VDW_SWITCH
        if (r > VDW_SWITCH_CUTOFF) {
            real x = r-VDW_SWITCH_CUTOFF;
            real taper = 1+x*x*x*(VDW_SWITCH_C3+x*(VDW_SWITCH_C4+x*VDW_SWITCH_C5));
            real dtaper = x*x*(3*VDW_SWITCH_C3+x*(4*VDW_SWITCH_C4+x*5*VDW_SWITCH_C5));
            deltaE = vdwEnergy*dtaper + deltaE*taper;
            vdwEnergy *= taper;
        }
      #endif
        dEdR -= deltaE*invR;
        tempEnergy += vdwEnergy;
    }
  #endif
#if HAS_COULOMB
  #ifdef USE_CUTOFF
    const real prefactor = 138.935456f*posq1.w*posq2.w;
    tempForce += prefactor*(rBufInv - 2.0f*REACTION_FIELD_K*r2);
    tempEnergy += includeInteraction ? prefactor*(rBufInv + REACTION_FIELD_K*rBuf*rBuf - REACTION_FIELD_C) : 0;
  #else
    const real prefactor = 138.935456f*posq1.w*posq2.w*rBufInv;
    tempForce += prefactor;
    tempEnergy += includeInteraction ? prefactor : 0;
  #endif
#endif
    dEdR += includeInteraction ? tempForce*rBufInv*invR : 0;
#endif
}
