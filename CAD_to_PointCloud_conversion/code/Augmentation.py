#!/usr/bin/env python3
# aug_final_roi.py : 회전 + 입사각 가림(ROI 보존) + 드롭아웃, 최소점수≥140k
import os, csv, math, numpy as np, open3d as o3d

# 경로
ROOT="/Users/jiwan/Desktop/marker/CAD_to_PointCloud_conversion/dataset"
SRC=f"{ROOT}/output"; DSTX=f"{ROOT}/augmentation/xyz"; DSTL=f"{ROOT}/augmentation/labels"
FNAME="ID{cls}_aug{i:03d}.xyz"

# 파라미터
K=100
MIN_POINTS=100_000
YAW=(0.0,360.0)          # z-yaw
TILT=(5.0,15.0)          # pitch/roll 최대(±)
OCC =(0.05,0.25)         # 가림 비율(입사각 하위 frac 제거)
DROP=(0.00,0.15)         # 랜덤 누락(불균일 밀도)
TOP_Q=1              # ROI=상면 분할 임계 분위수
OCC_ID_MAX=0.30          # ROI에서 허용하는 최대 가림 비율

# 유틸
def norm(P): P=P-P.mean(0,keepdims=True); return P/(np.max(np.linalg.norm(P,axis=1))+1e-12)
def Rz(d): r=math.radians(d); c,s=math.cos(r),math.sin(r); return np.array([[c,-s,0],[s,c,0],[0,0,1]])
def Rx(d): r=math.radians(d); c,s=math.cos(r),math.sin(r); return np.array([[1,0,0],[0,c,-s],[0,s,c]])
def Ry(d): r=math.radians(d); c,s=math.cos(r),math.sin(r); return np.array([[c,0,s],[0,1,0],[-s,0,c]])
def raw_id(fn): return os.path.splitext(fn)[0].replace("-","_").split("_")[0]
def build_cls_idx(folder):
    ids=sorted({raw_id(e.name) for e in os.scandir(folder) if e.is_file() and e.name.lower().endswith(".xyz")})
    return {rid:i for i,rid in enumerate(ids)}
def ensure_min(P,n,fb):
    if len(P)==0: P=fb
    if len(P)>=n: return P
    return P[np.random.choice(len(P), n, replace=True)]

# 법선(한 번)
def estimate_normals(P, k=30):
    pc=o3d.geometry.PointCloud(o3d.utility.Vector3dVector(P))
    pc.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamKNN(knn=k))
    N=np.asarray(pc.normals,float); N/=np.linalg.norm(N,axis=1,keepdims=True)+1e-12
    return N

# ROI=상면 마스크
def id_roi(P):
    z=P[:,2]; thr=np.quantile(z, TOP_Q)
    return z>=thr

# 입사각 가림(ROI 보존): 시선 V=[0,1,0] 평면파 가정
V_FIX=np.array([0.0,1.0,0.0])
def occlude_incidence_roi(P,N,frac):
    if frac<=0 or len(P)==0: return P
    cos=np.einsum("ij,ij->i", N, np.tile(V_FIX,(len(P),1)))
    roi=id_roi(P)
    k_tot=int(len(P)*frac)
    k_roi=int(roi.sum()*min(OCC_ID_MAX, frac))
    keep=np.ones(len(P), bool)

    out=np.where(~roi)[0]; out_sorted=out[np.argsort(cos[out])]
    rem=min(k_tot, len(out_sorted)); drop=list(out_sorted[:rem])

    need=k_tot-rem
    if need>0 and k_roi>0:
        inn=np.where(roi)[0]; inn_sorted=inn[np.argsort(cos[inn])]
        drop+=list(inn_sorted[:min(need, k_roi)])

    if drop: keep[np.array(drop)]=False
    return P[keep] if keep.any() else P

def augment_one(path, cls, writer):
    P=np.loadtxt(path); P0=norm(P[:,:3]); N0=estimate_normals(P0, k=30)
    for i in range(1,K+1):
        yaw=np.random.uniform(*YAW)
        tmax=np.random.uniform(*TILT); pitch=np.random.uniform(-tmax,tmax); roll=np.random.uniform(-tmax,tmax)
        occ=np.random.uniform(*OCC); drop=np.random.uniform(*DROP)

        R=(Rz(yaw).T @ Ry(pitch).T @ Rx(roll).T)
        A=P0@R; N=N0@R

        A=occlude_incidence_roi(A,N,occ)
        if drop>0 and len(A)>0: A=A[np.random.rand(len(A))>drop]
        A=ensure_min(A, MIN_POINTS, P0)

        name=FNAME.format(cls=cls,i=i)
        np.savetxt(os.path.join(DSTX,name),A,fmt="%.6f")
        writer.writerow([name,cls,round(yaw,2),round(pitch,2),round(roll,2),round(occ,3),round(drop,3)])

if __name__=="__main__":
    os.makedirs(DSTX,exist_ok=True); os.makedirs(DSTL,exist_ok=True)
    CLS=build_cls_idx(SRC)
    with open(os.path.join(DSTL,"labels.csv"),"w",newline="") as f:
        w=csv.writer(f); w.writerow(["file","class","yaw","pitch","roll","occ","drop"])
        for nm in sorted(os.listdir(SRC)):
            if nm.lower().endswith(".xyz"):
                augment_one(os.path.join(SRC,nm), CLS[raw_id(nm)], w)
    print("done:", DSTX)
