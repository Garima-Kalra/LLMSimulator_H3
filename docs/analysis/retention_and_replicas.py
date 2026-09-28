import math
k_B=8.617e-5; Ea=1.04  # eV, HeatWatch value as used by HBFSim
def accel(T1,T2): return math.exp(Ea/k_B*(1/(T1+273.15)-1/(T2+273.15)))
print("Arrhenius retention acceleration (Ea=1.04 eV):")
for T in (55,70,85,95,105):
    af=accel(85,T)  # relative to the 85C spec point
    print(f"  {T} C: retention = 24 h / {af:.2f} = {24/af:.1f} h")
designs={"8H8F":(8,1457,8881),"10H6F":(6,1824,8839),"12H4F":(4,2191,8579),"14H2F":(2,2557,7721),"15H1F":(1,2741,6380)}
S=165.8; SEC_YR=365*86400
print("\nStatic-data read pressure and replica rotation (FLINT-like R=1e6 page reads/blk, k=256, PE=1e5):")
for n,(f,req,tps) in designs.items():
    T=req/tps; C=8*f*48
    reads_per_day=86400/T
    tau=1e6*T/256
    r=math.floor(C/S)
    L0=1e5*tau/SEC_YR
    print(f"  {n}: step {T*1000:.0f} ms, static read BW {S/T:.2f} TB/s-equiv({S/T/1000*1000:.0f} GB/s), block step-reads/day {reads_per_day:,.0f}, over 5 yr {reads_per_day*365*5:.2e}; replicas r=floor(C/S)={r}; life noWL {L0:.2f} yr -> with r replicas {L0*r:.1f} yr")
