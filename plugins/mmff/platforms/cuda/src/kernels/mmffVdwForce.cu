{
//#define VDWDEBUG
#ifdef USE_CUTOFF
    unsigned int includeInteraction = (!isExcluded && r2 < CUTOFF_SQUARED);
#else
    unsigned int includeInteraction = (!isExcluded);
#endif
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
    const real C4 = 7.5797344e-4f;
    const real DARAD = 0.8f;
    const real DAEPS = 0.5f;
    real gamma = (params1.x - params2.x) / (params1.x + params2.x);
    real sigma = 0.5f * (params1.x + params2.x) * (1.0f + (haveDonor
                 ? 0.0f : B * (1.0f - exp(-Beta * gamma * gamma))));
    real alpha_d_N1 = ((params1.z < 0.0f) ? -params1.z : params1.z);
    real alpha_d_N2 = ((params2.z < 0.0f) ? -params2.z : params2.z);
    real sigmaSq = sigma * sigma;
    real epsilon = C4 * G_t_alpha1 * G_t_alpha2
                   / ((sqrt(alpha_d_N1) + sqrt(alpha_d_N2)) * sigmaSq * sigmaSq * sigmaSq);
#ifdef VDWDEBUG
    bool sc = false;
#endif
    if (haveDAPair) {
#ifdef VDWDEBUG
        sc = true;
#endif
        sigma *= DARAD;
        epsilon *= DAEPS;
    }
#ifdef VDWDEBUG
    printf("GIaI=%f,GJaJ=%f,aI/NI=%f,aJ/NJ=%f,s1=%f,s2=%f,s=%f,e=%f,sc=%d\n",G_t_alpha1,G_t_alpha2,alpha_d_N1,alpha_d_N2,params1.x,params2.x,sigma,epsilon,sc);
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
