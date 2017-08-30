float2 angleParams = PARAMS[index];
real deltaIdeal = theta-angleParams.x;
real dEdAngle = angleParams.y;
if (angleParams.y < 0.0f) {
    const real MIN_SINE = 1.0e-8f;
    real cosine = COS(theta);
    real sine2 = 1.0f - cosine*cosine;
    real sine = (sine2 < 0.0f) ? MIN_SINE : sqrt(sine2);
    if (sine < MIN_SINE) sine = MIN_SINE;
    energy += -dEdAngle*(1.0f + cosine);
    dEdAngle *= -sine;
}
else {
    real p = CUBIC_K*deltaIdeal;
    energy += 0.5f*dEdAngle*deltaIdeal*deltaIdeal*(1.0f + p);
    dEdAngle *= deltaIdeal*(1.0f + 1.5f*p);
}
