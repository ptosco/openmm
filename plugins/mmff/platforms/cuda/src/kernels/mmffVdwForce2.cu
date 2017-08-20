{
#ifdef USE_CUTOFF
    unsigned int includeInteraction = (!isExcluded && r2 < CUTOFF_SQUARED);
#else
    unsigned int includeInteraction = (!isExcluded);
#endif
    real tempForce = 0.0f;
#if SIGMA_COMBINING_RULE == 1
    real sigma = sigmaEpsilon1.x + sigmaEpsilon2.x;
#elif SIGMA_COMBINING_RULE == 2
    real sigma = 2*SQRT(sigmaEpsilon1.x*sigmaEpsilon2.x);
#elif SIGMA_COMBINING_RULE == 3
    real sigma1_2 = sigmaEpsilon1.x*sigmaEpsilon1.x;
    real sigma2_2 = sigmaEpsilon2.x*sigmaEpsilon2.x;
    real sigmasum = sigma1_2+sigma2_2;
    real sigma = (sigmasum == 0.0f ? (real) 0 : 2*(sigmaEpsilon1.x*sigma1_2 + sigmaEpsilon2.x*sigma2_2)/(sigma1_2+sigma2_2));
#else
    bool haveDonor = false;
    real sigma1 = sigmaEpsilon1.x;
    if (sigma1 < 0.0f) {
        haveDonor = true;
        sigma1 = -sigma1;
    }
    real sigma2 = sigmaEpsilon2.x;
    if (sigma2 < 0.0f) {
        haveDonor = true;
        sigma2 = -sigma2;
    }
    real gamma = (sigma1 - sigma2) / (sigma1 + sigma2);
    real sigma = 0.5f * (sigma1 + sigma2) * (1.0f + (haveDonor
                 ? 0.0f : 0.2f * (1.0f - exp(-12.0 * gamma * gamma))));
#endif
#if EPSILON_COMBINING_RULE == 1
    real epsilon = 0.5f*(sigmaEpsilon1.y + sigmaEpsilon2.y);
#elif EPSILON_COMBINING_RULE == 2
    real epsilon = SQRT(sigmaEpsilon1.y*sigmaEpsilon2.y);
#elif EPSILON_COMBINING_RULE == 3
    real epssum = sigmaEpsilon1.y+sigmaEpsilon2.y;
    real epsilon = (epssum == 0.0f ? (real) 0 : 2*(sigmaEpsilon1.y*sigmaEpsilon2.y)/(sigmaEpsilon1.y+sigmaEpsilon2.y));
#elif EPSILON_COMBINING_RULE == 4
    real epsilon_s = SQRT(sigmaEpsilon1.y) + SQRT(sigmaEpsilon2.y);
    real epsilon = (epsilon_s == 0.0f ? (real) 0 : 4*sigmaEpsilon1.y*sigmaEpsilon2.y/(epsilon_s*epsilon_s));
#else
    real epsilon1 = ((sigmaEpsilon1.y < 0.0f) ? -sigmaEpsilon1.y : sigmaEpsilon1.y);
    real epsilon2 = ((sigmaEpsilon2.y < 0.0f) ? -sigmaEpsilon2.y : sigmaEpsilon2.y);
    real sigmaSq = sigma * sigma;
    const real c4 = 7.5797344e-4f;
    real epsilon = c4 * bondReductionFactors1 * bondReductionFactors2
                   / ((sqrt(epsilon1) + sqrt(epsilon2)) * sigmaSq * sigmaSq * sigmaSq);
#endif
#if SIGMA_COMBINING_RULE == 4
#if EPSILON_COMBINING_RULE == 5
    const real DARAD = 0.8;
    const real DAEPS = 0.5;
    bool sc = false;
    if (((sigmaEpsilon1.x < 0.0f) && (sigmaEpsilon2.x > 0.0f) && (sigmaEpsilon2.y < 0.0f))
        || ((sigmaEpsilon2.x < 0.0f) && (sigmaEpsilon1.x > 0.0f) && (sigmaEpsilon1.y < 0.0f))) {
        sc = true;
        sigma *= DARAD;
        epsilon *= DAEPS;
    }
#if 0
    printf("GIaI=%f,GJaJ=%f,aI/NI=%f,aJ/NJ=%f,s1=%f,s2=%f,s=%f,e=%f,sc=%d\n",bondReductionFactors1,bondReductionFactors2,epsilon1,epsilon2,sigma1,sigma2,sigma,epsilon,sc);
#endif
#endif
#endif
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
    real termEnergy = epsilon*sigma7*tau7*((sigma7*1.12f*invRho)-2.0f);
    real deltaE = -7.0f*(dTau*termEnergy+gTau);
#ifdef USE_CUTOFF
    if (r > TAPER_CUTOFF) {
        real x = r-TAPER_CUTOFF;
        real taper = 1+x*x*x*(TAPER_C3+x*(TAPER_C4+x*TAPER_C5));
        real dtaper = x*x*(3*TAPER_C3+x*(4*TAPER_C4+x*5*TAPER_C5));
        deltaE = termEnergy*dtaper + deltaE*taper;
        termEnergy *= taper;
    }
#endif
    tempEnergy += (includeInteraction ? termEnergy : 0);
    dEdR -= (includeInteraction ? deltaE*invR : 0);
}
