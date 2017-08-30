float3 torsionK = PARAMS[index];
const real torsionParams1 = torsionK.y + 0.5f*(torsionK.x + torsionK.z);
const real torsionParams2 = 0.5f*(3.0f*torsionK.z - torsionK.x);
const real torsionParams3 = -torsionK.y;
const real torsionParams4 = -2.0f*torsionK.z;
if (theta < 0)
    theta += PI;
else
    theta -= PI;
cosangle = -cosangle;
real cosFactor = cosangle;
real dEdAngle = -torsionParams2;
real rbEnergy = torsionParams1;
rbEnergy += torsionParams2*cosFactor;
dEdAngle -= 2.0f*torsionParams3*cosFactor;
cosFactor *= cosangle;
dEdAngle -= 3.0f*torsionParams4*cosFactor;
rbEnergy += torsionParams3*cosFactor;
cosFactor *= cosangle;
rbEnergy += torsionParams4*cosFactor;
energy += rbEnergy;
dEdAngle *= SIN(theta);
