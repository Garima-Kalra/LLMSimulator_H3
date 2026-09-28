import math
# From BTP_Report.tex Table (Llama-3.1-405B, 10M tokens, 32 GPUs, cascaded, 8 sites)
designs = {  # name: (hbm_dies, hbf_dies, requests, tok_s)
 "8H8F": (8,8,1457,8881), "10H6F": (10,6,1824,8839), "12H4F": (12,4,2191,8579),
 "14H2F": (14,2,2557,7721), "15H1F": (15,1,2741,6380)}
SITES, GPU_W, HBM_CUBE_W, DIES = 8, 680, 40, 8
HBM_DIE_GB, HBF_DIE_GB = 3, 48
STATIC_PER_DEV_GB = 165.8          # weights + shared KV per device (report)
TOTAL_STATIC_TB = 181.8*32/1000    # flash needed per device x 32 (report table) -> cluster static
print("== power sensitivity (tokens/s/W per GPU, throughput held fixed) ==")
for hbf_cube_w in (160, 80, 30):
    row=[]
    for n,(h,f,req,tps) in designs.items():
        p = GPU_W + SITES*(h*HBM_CUBE_W/DIES + f*hbf_cube_w/DIES)
        row.append((n,p,tps/p/32))
    base=row[0][2]
    print(hbf_cube_w,"W/cube:", ", ".join(f"{n} {p:.0f}W {e:.4f} ({e/base-1:+.1%})" for n,p,e in row))
# crossover HBF cube power where 12H4F == 8H8F efficiency
# P(h,f,x) = 680 + 8*(h*5 + f*x/8)
def P(h,f,x): return GPU_W + SITES*(h*HBM_CUBE_W/DIES + f*x/DIES)
lo,hi=0,400
for _ in range(100):
    m=(lo+hi)/2
    d=designs["12H4F"][3]/P(12,4,m)-designs["8H8F"][3]/P(8,8,m)
    lo,hi=(m,hi) if d<0 else (lo,m)
print(f"12H4F beats 8H8F on tokens/s/W only if HBF cube power > {m:.1f} W")
for other in ("10H6F","14H2F"):
    lo,hi=0,400
    h,f=designs[other][:2]
    for _ in range(100):
        m=(lo+hi)/2
        d=designs[other][3]/P(h,f,m)-designs["8H8F"][3]/P(8,8,m)
        lo,hi=(m,hi) if d<0 else (lo,m)
    print(f"{other} beats 8H8F only if HBF cube power > {m:.1f} W")
print("\n== minimum GPUs to hold weights+shared KV (capacity only) ==")
print(f"cluster static footprint = {TOTAL_STATIC_TB:.2f} TB")
for n,(h,f,req,tps) in designs.items():
    flash_tb = SITES*f*HBF_DIE_GB/1000
    print(n, f"flash/GPU {flash_tb:.3f} TB -> min GPUs {math.ceil(TOTAL_STATIC_TB/flash_tb)}  (pow2: {2**math.ceil(math.log2(TOTAL_STATIC_TB/flash_tb))})")
hbm_only = 192/1000
print("HBM-only min GPUs (static only):", math.ceil((TOTAL_STATIC_TB)/hbm_only))
print("\n== read-disturb reclaim model ==")
SEC_YR=365*86400
for PE,R,k,label in ((1e5,1e6,256,"FLINT-like R=1e6 reads/blk, k=256 page reads/blk/step"),
                     (1e5,1e4,1,"Lincoln-like R=1e4 step-reads (k=1)")):
    print(label, f"PE={PE:.0e}")
    for n,(h,f,req,tps) in designs.items():
        T = req/tps                        # decode step time (s)
        C = SITES*f*HBF_DIE_GB             # flash per device (GB)
        tau = R*T/k                        # per-block reclaim interval (s)
        bw = STATIC_PER_DEV_GB/tau         # refresh write bandwidth GB/s
        L0 = PE*tau/SEC_YR                 # no static WL
        Lwl = PE*tau*C/STATIC_PER_DEV_GB/SEC_YR  # ideal WL over whole device
        print(f"  {n}: T={T*1000:.0f} ms C={C} GB C/S={C/STATIC_PER_DEV_GB:.2f} tau={tau:.0f}s ({86400/tau:.0f}x OCP 24h retention rate) refreshBW={bw*1000:.0f} MB/s  life noWL={L0:.2f} yr  WL={Lwl:.1f} yr")
