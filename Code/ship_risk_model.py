#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Ship Navigation Safety Risk Assessment, South China Sea, 2018-12-27..30

Reproduce:
    pip install numpy scipy matplotlib basemap
    python ship_risk_model.py --data-dir . --out-dir ./ship_risk_results

Inputs: wind.mat, wave.mat, current.mat, tem.mat (must contain matching variables,
        u_new longitude and v_new latitude).
Outputs: AHP/EV weights, 5-grade fuzzy ship vulnerability, 4 daily risk maps,
         risk figures, CSV, JSON and MATLAB .mat risk grid.

IMPORTANT:
* This is a transparent, illustrative course-project risk INDEX, not an
  empirically calibrated collision/grounding casualty probability.
* SST proxy is a model ASSUMPTION, not a validated accident-rate relation.
* Two SST limits and all vessel-size/age thresholds are scenario hypotheses.
"""
import argparse
import csv
import json
from pathlib import Path

import numpy as np
from scipy.io import loadmat, savemat

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import BoundaryNorm
from matplotlib.cm import ScalarMappable

try:
    from mpl_toolkits.basemap import Basemap
except ImportError:
    Basemap = None

INDICATORS = ("wind", "wave", "current", "sst_stress")
WIND_NODES = [0, 4, 8, 12, 16]          # m/s
WAVE_NODES = [0, 0.75, 1.5, 3, 5]      # significant wave height, m
CURRENT_NODES = [0, 0.25, 0.5, 1, 1.8]# current speed, m/s
SST_NODES = [0, 1, 2, 4, 7]            # degrees C outside 20--30 C
TONNAGE_NODES = [10000, 5000, 2500, 1000, 300]  # GT (larger safer proxy)
LENGTH_NODES = [200, 140, 100, 60, 30]          # m (larger safer proxy)
AGE_NODES = [0, 5, 10, 20, 30]                 # years (older vulnerable)
GRADE_CENTERS = np.array([0, .25, .5, .75, 1.0])
GRADE_LABELS = ("Very low", "Low", "Medium", "High", "Very high")
LEVEL_EDGES = np.array([0, .2, .4, .6, .8, 1.00000001])
A_H = np.array([
    [1, .5, 3, 5],
    [2, 1, 4, 7],
    [1/3, 1/4, 1, 2],
    [1/5, 1/7, .5, 1],
], dtype=float)
A_V = np.array([
    [1, 2, .5],
    [.5, 1, 1/3],
    [2, 3, 1],
], dtype=float)
ALPHA_AHP = 0.5
SHIP_GT, SHIP_LENGTH, SHIP_AGE = 2646, 95.68, 11.5


def ahp(A, ri):
    """Principal right eigenvector (Saaty AHP) with consistency test."""
    eigvals, eigvecs = np.linalg.eig(A)
    i = int(np.argmax(eigvals.real))
    e = eigvecs[:, i].real
    if e.sum() < 0:
        e = -e
    assert np.all(e > 0), "Pairwise judgment matrix must be positive"
    e = e/e.sum()
    lam = float(eigvals[i].real)
    ci = (lam - A.shape[0])/(A.shape[0] - 1)
    cr = ci/ri
    if cr >= .1:
        raise ValueError(f"AHP inconsistency: CR={cr:.5f}")
    return e, lam, ci, cr


def indicator_score(x, knots):
    """
    Piecewise-linear grade score [0,1], equivalent to a weighted sum of
    five linear-shoulder/triangular fuzzy memberships.
    For descending knots, smaller raw physical quantity means higher risk.
    """
    knots = np.asarray(knots, dtype=float)
    x = np.asarray(x, dtype=float)
    if knots[0] > knots[-1]:
        return np.interp(x, knots[::-1], GRADE_CENTERS[::-1])
    return np.interp(x, knots, GRADE_CENTERS)


def fuzzy_membership_from_score(score):
    """Five memberships, partition of unity, on grade centers."""
    score = np.asarray(score, dtype=float)
    memberships = np.maximum(
        0.0, 1.0 - np.abs(score[..., None] - GRADE_CENTERS)/0.25
    )
    return memberships


def load_ocean_inputs(data_dir):
    arrays = {}
    lon0 = None
    lat0 = None
    for name in ("wind", "wave", "current", "tem"):
        mat = loadmat(data_dir / (name+".mat"))
        A = np.asarray(mat[name], dtype=float)
        lon = np.asarray(mat["u_new"]).ravel()
        lat = np.asarray(mat["v_new"]).ravel()
        if A.ndim != 3 or A.shape != (len(lat), len(lon), 4):
            raise ValueError(f"Invalid grid for {name}: {A.shape}")
        if lon0 is not None and (not np.allclose(lon,lon0) or
                                 not np.allclose(lat,lat0)):
            raise ValueError("Input grids differ: cannot combine without regridding")
        arrays[name] = A
        lon0,lat0 = lon,lat
    return arrays, lon0, lat0


def area_mean(field, mask, weights):
    vals = field[mask]
    ws = weights[mask]
    return float(np.average(vals, weights=ws))


def area_fraction(field, mask, weights, lower=.6):
    vals = field[mask]
    ws = weights[mask]
    return float(ws[vals >= lower].sum()/ws.sum())


def map_panel(ax, lon, lat, field, date, show_labels=True):
    # Values outside observed marine cells are drawn in pale grey, not blue.
    field = np.ma.masked_invalid(field)
    if Basemap is not None:
        bm = Basemap(
            projection="cyl", llcrnrlon=100,urcrnrlon=135,
            llcrnrlat=0,urcrnrlat=30, resolution="l", ax=ax
        )
        bm.drawmapboundary(fill_color="#eef1f5", linewidth=.65)
        obj = bm.pcolormesh(
            lon, lat, field, latlon=True, shading="auto",
            cmap="turbo", vmin=0, vmax=1, rasterized=True
        )
        bm.fillcontinents(color="#c6c8cc", lake_color="#eef1f5", zorder=3)
        bm.drawcoastlines(linewidth=.42, color="#303a45", zorder=4)
        bm.drawparallels([0,10,20,30],
                         labels=[int(show_labels),0,0,0],
                         linewidth=.25, color="#76818b", fontsize=8,
                         dashes=[2,4])
        bm.drawmeridians([100,110,120,130],
                         labels=[0,0,0,int(show_labels)],
                         linewidth=.25, color="#76818b", fontsize=8,
                         dashes=[2,4])
    else:
        ax.set_facecolor("#eef1f5")
        obj=ax.pcolormesh(lon,lat,field,shading="auto",cmap="turbo",
                          vmin=0,vmax=1,rasterized=True)
        ax.set_xticks([100,110,120,130])
        ax.set_yticks([0,10,20,30])
        ax.set_xlabel("Longitude (°E)")
        ax.set_ylabel("Latitude (°N)")
        ax.grid(alpha=.2)
    ax.set_xlim(100,135); ax.set_ylim(0,30)
    ax.set_title(date, fontsize=12, fontweight="bold", pad=10)
    return obj


def save_map_figures(lon,lat,risk,out_dir):
    days=["2018-12-27","2018-12-28","2018-12-29","2018-12-30"]
    fig, axes=plt.subplots(2,2,figsize=(13,10.4))
    for i, ax in enumerate(axes.flat):
        map_panel(ax,lon,lat,risk[:,:,i],days[i],show_labels=True)
    fig.suptitle("Ship Navigation Safety Risk | South China Sea",
                 fontsize=17,weight="bold",y=.99)
    fig.subplots_adjust(left=.065,right=.89,top=.94,bottom=.085,wspace=.12,hspace=.18)
    cb=fig.colorbar(ScalarMappable(norm=plt.Normalize(0,1),cmap="turbo"),
        ax=axes.ravel().tolist(),orientation="vertical",fraction=.026,pad=.02,
        ticks=np.arange(0,1.01,.2))
    cb.set_label("Risk index (0-1)",size=11)
    fig.text(.5,.025,"Data: 2018-12-27 to 2018-12-30 | Grey: land / invalid sea cells",
             ha="center",fontsize=10,color="#46515c")
    fig.savefig(out_dir/"risk_4day_comparison.png",dpi=205,bbox_inches="tight")
    fig.savefig(out_dir/"risk_4day_comparison.pdf",bbox_inches="tight")
    plt.close(fig)
    for t, day in enumerate(days):
        fig,ax=plt.subplots(figsize=(9.7,7.5))
        im=map_panel(ax,lon,lat,risk[:,:,t],day)
        fig.suptitle("Ship Navigation Safety Risk",fontsize=16,weight="bold",y=.98)
        fig.colorbar(im,ax=ax,shrink=.82,pad=.025,extend="neither",
                     label="Risk index (0-1)")
        fig.subplots_adjust(top=.9,bottom=.1)
        fig.savefig(out_dir/("risk_"+day.replace("-","")+".png"),
                    dpi=210,bbox_inches="tight")
        plt.close(fig)


def save_weights_fig(w_ahp,w_ev,w_mix,out_dir):
    fig,ax=plt.subplots(figsize=(9.1,4.3))
    x=np.arange(4)
    width=.25
    ax.bar(x-width,w_ahp,width,label="AHP (subjective)")
    ax.bar(x,w_ev,width,label="Eigenvector (objective)")
    ax.bar(x+width,w_mix,width,label="Combined 50/50")
    ax.set_xticks(x,["Wind","Wave","Current","SST stress"])
    ax.set_ylim(0,.58)
    ax.set_ylabel("Weight")
    ax.set_title("Hazard indicator weights",pad=11)
    ax.legend(loc="upper right",frameon=False)
    ax.grid(axis="y",alpha=.2)
    fig.tight_layout()
    fig.savefig(out_dir/"hazard_weights.png",dpi=180,bbox_inches="tight")
    plt.close(fig)



def save_sensitivity_fig(daily, sensitivity, out_dir):
    dates=[d["date"][5:] for d in daily]
    baseline=[d["high_or_very_high_share"]*100 for d in daily]
    stricter=[d["high_or_very_high_share"]*100
              for d in sensitivity["stricter_minus20pct"]]
    looser=[d["high_or_very_high_share"]*100
            for d in sensitivity["looser_plus20pct"]]
    fig,ax=plt.subplots(figsize=(9.0,4.0))
    ax.plot(dates,baseline,marker="o",linewidth=2.2,
            label="Baseline")
    ax.plot(dates,stricter,marker="s",linewidth=2.0,
            label="Wind/wave thresholds -20%")
    ax.plot(dates,looser,marker="^",linewidth=2.0,
            label="Wind/wave thresholds +20%")
    ax.set_ylim(0,75)
    ax.set_ylabel("High and very-high risk area (%)")
    ax.set_xlabel("Date in 2018")
    ax.set_title("Sensitivity to wind and wave risk-grade thresholds")
    ax.grid(axis="y",alpha=.2)
    ax.legend(frameon=False,fontsize=9,loc="upper left")
    fig.tight_layout()
    fig.savefig(out_dir/"risk_sensitivity.png",dpi=200,bbox_inches="tight")
    plt.close(fig)


def main(data_dir, out_dir):
    out_dir.mkdir(parents=True,exist_ok=True)
    a,lon,lat=load_ocean_inputs(data_dir)
    mask=np.logical_and.reduce([np.isfinite(a[n]) for n in
                                ("wind","wave","current","tem")])
    # 4-day common geometry: invalid = land and missing measurements.
    sst_stress=np.maximum.reduce([np.zeros_like(a["tem"]),
                                  20.0-a["tem"],a["tem"]-30.0])
    hs=np.stack([
        indicator_score(a["wind"],WIND_NODES),
        indicator_score(a["wave"],WAVE_NODES),
        indicator_score(a["current"],CURRENT_NODES),
        indicator_score(sst_stress,SST_NODES),
    ],axis=-1)
    x=hs[mask]
    geographic_area=np.broadcast_to(
        np.cos(np.deg2rad(lat))[:,None,None],mask.shape
    )
    wcell=geographic_area[mask]
    ahp_h,lam_h,ci_h,cr_h=ahp(A_H,.90)
    ahp_v,lam_v,ci_v,cr_v=ahp(A_V,.58)

    # Objective method: principal eigenvector of NON-CENTERED,
    # area-weighted second-moment matrix of positive risk scores.
    # This does NOT equal centered-covariance PCA.
    M=(x*wcell[:,None]).T @ x / wcell.sum()
    eigvals,eigvecs=np.linalg.eigh(M)
    j=int(np.argmax(eigvals))
    ev=eigvecs[:,j]
    if ev.sum() < 0: ev=-ev
    if np.any(ev<0):
        raise RuntimeError("Nonnegative principal eigenvector expected")
    obj_h=ev/ev.sum()
    blend_h=ALPHA_AHP*ahp_h+(1-ALPHA_AHP)*obj_h

    ship_scores=np.array([
        indicator_score(SHIP_GT,TONNAGE_NODES),
        indicator_score(SHIP_LENGTH,LENGTH_NODES),
        indicator_score(SHIP_AGE,AGE_NODES),
    ])
    ship_R=fuzzy_membership_from_score(ship_scores)
    ship_B=ahp_v @ ship_R
    V=float(ship_B @ GRADE_CENTERS)
    ship_factor=.5+V

    # Fuzzy operator: weighted arithmetic composition, B = w @ R.
    # Scores are exactly equivalent to weighted average of score centers.
    H_AHP=np.einsum("ijkl,l->ijk", hs, ahp_h)
    H_EV=np.einsum("ijkl,l->ijk", hs, obj_h)
    H_MIX=np.einsum("ijkl,l->ijk", hs, blend_h)
    hazards={"ahp":H_AHP,"eigenvector":H_EV,"combined":H_MIX}
    risks={k:np.where(mask,np.clip(h*ship_factor,0,1),np.nan)
            for k,h in hazards.items()}
    for h in hazards.values(): h[~mask]=np.nan

    daily=[]
    for t in range(4):
        dt=f"2018-12-{27+t}"
        m=mask[:,:,t]
        w=geographic_area[:,:,t]
        rr=risks["combined"][:,:,t]
        areas=np.histogram(rr[m],bins=LEVEL_EDGES,weights=w[m])[0]/w[m].sum()
        row={
            "date":dt,"valid_cells":int(m.sum()),
            "mean_risk":area_mean(rr,m,w),
            "max_risk":float(np.max(rr[m])),
            "high_or_very_high_share":float(areas[3]+areas[4]),
            "very_high_share":float(areas[4]),
        }
        for j,label in enumerate(GRADE_LABELS):
            row["area_share_"+label.replace(" ","_").lower()]=float(areas[j])
        for method in ("ahp","eigenvector"):
            row["mean_risk_"+method]=area_mean(risks[method][:,:,t],m,w)
            row["high_share_"+method]=area_fraction(
                risks[method][:,:,t],m,w,.6
            )
        daily.append(row)

    # Threshold sensitivity: hold weights fixed while increasing/decreasing
    # only nonzero wind/wave grade anchors by 20%.
    sensitivity={}
    for change,label in ((.8,"stricter_minus20pct"),
                         (1.2,"looser_plus20pct")):
        wn=np.asarray(WIND_NODES)*change
        wa=np.asarray(WAVE_NODES)*change
        sh=np.stack([
            indicator_score(a["wind"],wn),
            indicator_score(a["wave"],wa),
            hs[...,2], hs[...,3]
        ],axis=-1)
        sr=np.where(mask,np.clip(
            np.einsum("ijkl,l->ijk",sh,blend_h)*ship_factor,0,1),np.nan)
        sensitivity[label]=[{
            "date":daily[t]["date"],
            "mean_risk":area_mean(sr[:,:,t],mask[:,:,t],geographic_area[:,:,t]),
            "high_or_very_high_share":area_fraction(
                sr[:,:,t],mask[:,:,t],geographic_area[:,:,t],.6),
        } for t in range(4)]

    save_map_figures(lon,lat,risks["combined"],out_dir)
    save_weights_fig(ahp_h,obj_h,blend_h,out_dir)
    save_sensitivity_fig(daily,sensitivity,out_dir)
    # MATLAB array compatible with original axes.
    savemat(out_dir/"risk_maps_4days.mat",{
        "risk":risks["combined"], "hazard":hazards["combined"],
        "risk_ahp":risks["ahp"], "risk_eigenvector":risks["eigenvector"],
        "lon":lon,"lat":lat, "w_hazard_ahp":ahp_h,
        "w_hazard_eigenvector":obj_h,"w_hazard_combined":blend_h,
        "w_vulnerability_ahp":ahp_v,"vulnerability_score":V,
        "valid_mask":mask.astype(np.uint8),
    },do_compression=True)
    with (out_dir/"risk_daily_summary.csv").open("w",newline="",encoding="utf-8-sig") as f:
        writer=csv.DictWriter(f,fieldnames=list(daily[0]))
        writer.writeheader()
        writer.writerows(daily)
    with (out_dir/"hazard_weights.csv").open("w",newline="",encoding="utf-8-sig") as f:
        writer=csv.writer(f)
        writer.writerow(["indicator","AHP","Eigenvector","Combined"])
        writer.writerows(zip(INDICATORS,ahp_h,obj_h,blend_h))
    with (out_dir/"vulnerability.csv").open("w",newline="",encoding="utf-8-sig") as f:
        writer=csv.writer(f)
        writer.writerow(["indicator","value","AHP_weight","risk_score"]+
                        ["mu_"+k.replace(" ","_") for k in GRADE_LABELS])
        for name,value,weight,score,m in zip(
                ("tonnage_GT","length_m","age_years"),
                (SHIP_GT,SHIP_LENGTH,SHIP_AGE),
                ahp_v,ship_scores,ship_R):
            writer.writerow([name,value,weight,score,*m])
        writer.writerow(["Composite_vulnerability",V])
    comparison_risk=risks["ahp"][mask]
    comparison_ev=risks["eigenvector"][mask]
    report={
        "grid":{"shape":list(a["wind"].shape),
                "longitude_min":float(lon.min()),"longitude_max":float(lon.max()),
                "latitude_min":float(lat.min()),"latitude_max":float(lat.max()),
                "common_valid_cells_per_day":int(mask[:,:,0].sum())},
        "AHP_hazard":{"matrix":A_H.tolist(),"weights":ahp_h.tolist(),
                      "lambda_max":lam_h,"CI":ci_h,"CR":cr_h},
        "AHP_vulnerability":{"matrix":A_V.tolist(),"weights":ahp_v.tolist(),
                      "lambda_max":lam_v,"CI":ci_v,"CR":cr_v},
        "eigenvector_hazard":{"second_moment":M.tolist(),"eigenvalues":eigvals.tolist(),
                              "weights":obj_h.tolist()},
        "combined_hazard_weights":blend_h.tolist(),
        "ship_scores":ship_scores.tolist(),"ship_R":ship_R.tolist(),
        "ship_B":ship_B.tolist(),"V":V,"vulnerability_multiplier":ship_factor,
        "daily":daily,"threshold_sensitivity":sensitivity,
        "method_comparison":{
           "map_correlation":float(np.corrcoef(comparison_risk,comparison_ev)[0,1]),
           "mean_absolute_risk_difference":float(
               np.mean(np.abs(comparison_risk-comparison_ev)))},
        "reference_thresholds":{
          "wind_mps":WIND_NODES,"wave_m":WAVE_NODES,
          "current_mps":CURRENT_NODES,
          "sst_stress_C":SST_NODES,
          "sst_comfort_interval_C":[20,30],
          "tonnage_GT":TONNAGE_NODES,
          "length_m":LENGTH_NODES,
          "age_years":AGE_NODES
        },
        "risk_definition":"R=clip(H*(0.5+V),0,1); not accident probability",
        "class_edges":[0,.2,.4,.6,.8,1],
    }
    (out_dir/"model_results.json").write_text(
        json.dumps(report,indent=2,ensure_ascii=False),encoding="utf-8")
    print("AHP hazard weights:",np.round(ahp_h,5), "CR",round(cr_h,6))
    print("EV objective weights:",np.round(obj_h,5))
    print("Combined hazard weights:",np.round(blend_h,5))
    print("Vulnerability weights:",np.round(ahp_v,5),"CR",round(cr_v,6))
    print("Ship vulnerability:",round(V,6),"Multiplier:",round(ship_factor,6))
    for row in daily:
        print(row["date"],"area weighted mean:",round(row["mean_risk"],5),
              ">=0.6:",round(row["high_or_very_high_share"]*100,2),"%",
              ">=0.8:",round(row["very_high_share"]*100,2),"%")
    print("Saved files in:",out_dir)


if __name__=="__main__":
    p=argparse.ArgumentParser()
    p.add_argument("--data-dir",type=Path,default=Path("."))
    p.add_argument("--out-dir",type=Path,default=Path("./ship_risk_results"))
    a=p.parse_args()
    main(a.data_dir,a.out_dir)
