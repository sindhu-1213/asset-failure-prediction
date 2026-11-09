"""Streamlit industrial asset-health command center."""

from datetime import datetime
import html
import json
import os
import sys
import textwrap
from urllib.parse import quote

import numpy as np
import pandas as pd
import streamlit as st

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

from src.features.build_features import build_features, get_feature_cols
from src.ingestion.load_data import load_test

ARTIFACTS = os.path.join(PROJECT_ROOT, "models", "artifacts")
SENSOR_COLS = [f"s{i}" for i in range(1, 22)]
SETTING_COLS = ["op1", "op2", "op3"]
STATUS_COLORS = {
    "HEALTHY": "#48d1a0",
    "WARNING": "#f0b35b",
    "CRITICAL": "#ff657a",
}


@st.cache_resource
def load_models():
    from src.models.xgboost_model import XGBClassifier, XGBRULModel

    rul_path = os.path.join(ARTIFACTS, "xgb_rul.pkl")
    clf_path = os.path.join(ARTIFACTS, "xgb_clf.pkl")
    missing_paths = [path for path in (rul_path, clf_path) if not os.path.isfile(path)]
    xgb_rul = XGBRULModel.load(rul_path) if os.path.isfile(rul_path) else None
    xgb_clf = XGBClassifier.load(clf_path) if os.path.isfile(clf_path) else None

    threshold_path = os.path.join(ARTIFACTS, "best_threshold.json")
    threshold = 0.5
    if os.path.isfile(threshold_path):
        with open(threshold_path, encoding="utf-8") as file:
            threshold = json.load(file).get("threshold", 0.5)
    return xgb_rul, xgb_clf, threshold, missing_paths


@st.cache_data
def load_test_data():
    raw_df, rul_df = load_test(data_dir=os.path.join(PROJECT_ROOT, "data", "raw"))
    feature_df = build_features(raw_df, is_train=False)
    return raw_df, feature_df, rul_df


def get_latest_per_unit(df: pd.DataFrame) -> pd.DataFrame:
    return df.sort_values("cycle").groupby("unit_id").last().reset_index()


def risk_color(prob: float) -> str:
    if prob < 0.3:
        return "HEALTHY"
    if prob < 0.6:
        return "WARNING"
    return "CRITICAL"


def score_reading(history, reading, xgb_rul, xgb_clf):
    """Score one new reading using its selected demo unit's real history."""
    row = dict(reading)
    row["unit_id"] = int(history["unit_id"].iloc[0])
    window = pd.concat([history, pd.DataFrame([row])], ignore_index=True)
    featured = build_features(window, is_train=False)

    feature_cols = xgb_rul.feature_cols
    missing_cols = [col for col in feature_cols if col not in featured.columns]
    if missing_cols:
        raise ValueError(f"Feature builder did not produce: {', '.join(missing_cols)}")
    if feature_cols != xgb_clf.feature_cols:
        raise ValueError("RUL and risk models require different feature orders.")

    X = featured.loc[:, feature_cols].tail(1)
    rul_prediction = float(xgb_rul.predict(X)[0])
    failure_probability = float(xgb_clf.predict_proba(X)[0])
    return rul_prediction, failure_probability


def render_digital_twin(unit_id: int, failure_probability: float, predicted_rul: float):
    status = risk_color(failure_probability)
    status_color = STATUS_COLORS[status]
    twin_html = textwrap.dedent(
        """
        <div class="twin-shell">
          <div class="twin-topline">
            <span>ILLUSTRATIVE DIGITAL TWIN</span>
            <span class="twin-status"><i></i> __STATUS__</span>
          </div>
          <div class="twin-scene">
            <div class="twin-fallback" id="fallback">
              <div class="fallback-engine">
                <span class="fallback-cap"></span><span class="fallback-core"></span>
                <span class="fallback-rings"></span><span class="fallback-cap"></span>
              </div>
              <small>3D preview unavailable · use the fleet data below</small>
            </div>
            <canvas id="scene" tabindex="0" aria-label="Interactive illustrative turbine assembly. Use arrow keys to orbit."></canvas>
            <div class="scene-label scene-label-left">TURBINE<br><b>ROTOR ASSEMBLY</b></div>
            <div class="scene-label scene-label-right">UNIT __UNIT__<br><b>SYNTHETIC DEMO</b></div>
            <div class="scene-controls" aria-label="Digital twin controls">
              <button id="zoom-in" aria-label="Zoom in">+</button>
              <button id="zoom-out" aria-label="Zoom out">−</button>
              <button id="pause" aria-label="Pause automatic rotation">Ⅱ</button>
            </div>
          </div>
          <div class="twin-bottomline">
            <span>RISK <b>__RISK__</b></span>
            <span>EST. RUL <b>__RUL__ cycles</b></span>
            <span>DRAG TO ORBIT · SCROLL TO ZOOM</span>
          </div>
        </div>
        <style>
          * { box-sizing: border-box; }
          body { margin: 0; background: transparent; color: #d7e5f4; font-family: Inter, ui-sans-serif, system-ui, sans-serif; }
          .twin-shell { height: 486px; padding: 14px 18px 12px; border: 1px solid rgba(98,174,211,.2); border-radius: 18px; background: linear-gradient(145deg,rgba(13,29,44,.96),rgba(7,15,25,.98)); box-shadow: inset 0 1px rgba(255,255,255,.035),0 20px 55px rgba(0,0,0,.2); }
          .twin-topline,.twin-bottomline { display:flex; justify-content:space-between; align-items:center; gap:10px; color:#7891a9; font-size:10px; letter-spacing:.14em; }
          .twin-status { color:__STATUS_COLOR__; display:flex; align-items:center; gap:7px; }
          .twin-status i { width:7px;height:7px;border-radius:50%;background:__STATUS_COLOR__;box-shadow:0 0 12px __STATUS_COLOR__; }
          .twin-scene { position:relative;height:406px;margin:6px -4px 8px;overflow:hidden;border-radius:12px;background:radial-gradient(ellipse at 50% 57%,rgba(15,76,105,.24),transparent 53%),linear-gradient(180deg,#091521,#07111c 68%,#081521); }
          #scene { position:absolute;inset:0;width:100%;height:100%;touch-action:none;cursor:grab; }
          #scene:active { cursor:grabbing; }
          .scene-label { position:absolute;top:18px;color:#57748d;font-size:9px;line-height:1.55;letter-spacing:.13em;pointer-events:none; }
          .scene-label b { color:#b8d5e8;font-size:10px; }
          .scene-label-left { left:18px; }.scene-label-right { right:18px;text-align:right; }
          .scene-controls { position:absolute;right:13px;bottom:13px;display:flex;gap:6px; }
          .scene-controls button { width:30px;height:30px;border:1px solid rgba(103,185,217,.25);border-radius:8px;color:#b9dbe9;background:rgba(7,19,30,.8);cursor:pointer;font-size:15px; }
          .scene-controls button:hover,.scene-controls button:focus-visible { border-color:#53c8e8;outline:none;box-shadow:0 0 12px rgba(50,191,224,.22); }
          .twin-bottomline { font-size:9px; }
          .twin-bottomline b { color:#d5e8f4;letter-spacing:.04em; }
          .twin-bottomline span:first-child b { color:__STATUS_COLOR__; }
          .twin-bottomline span:last-child { color:#547087; }
          .twin-fallback { display:none;position:absolute;inset:0;z-index:1;place-content:center;justify-items:center;gap:28px;color:#7891a9;font-size:11px; }
          .fallback-engine { display:flex;align-items:center;filter:drop-shadow(0 0 18px rgba(57,188,222,.2)); }
          .fallback-engine span { display:block;height:54px;border:1px solid #47788c;background:linear-gradient(180deg,#38586c,#142c3c); }
          .fallback-cap { width:28px;border-radius:50%; }
          .fallback-core { width:130px;border-radius:12px 0 0 12px; }
          .fallback-rings { width:52px;border-radius:50%;border:5px double #6ca9bd!important; }
          @media (max-width:620px) { .twin-shell { height:430px;padding:12px; }.twin-scene {height:350px}.twin-bottomline span:last-child {display:none;} }
        </style>
        <script src="https://cdnjs.cloudflare.com/ajax/libs/three.js/r128/three.min.js"></script>
        <script>
        (function() {
          const canvas = document.getElementById("scene");
          const fallback = document.getElementById("fallback");
          const statusColor = "__STATUS_COLOR__";
          if (!window.THREE) {
            fallback.style.display = "grid";
            canvas.style.display = "none";
            return;
          }
          const THREE = window.THREE;
          const scene = new THREE.Scene();
          scene.background = new THREE.Color("#08131f");
          const camera = new THREE.PerspectiveCamera(36, 1, 0.1, 100);
          camera.position.set(6.1, 4.1, 7.3);
          camera.lookAt(0, 0, 0);
          let renderer;
          try {
            renderer = new THREE.WebGLRenderer({canvas: canvas, antialias: true, alpha: false});
          } catch (error) {
            fallback.style.display = "grid";
            canvas.style.display = "none";
            return;
          }
          renderer.setPixelRatio(Math.min(window.devicePixelRatio || 1, 2));
          renderer.setClearColor("#08131f", 1);
          if ("outputEncoding" in renderer) renderer.outputEncoding = THREE.sRGBEncoding;
          scene.add(new THREE.HemisphereLight("#bdeeff", "#17202f", 1.45));
          const key = new THREE.DirectionalLight("#7de5ff", 2.2);
          key.position.set(-3, 7, 5); scene.add(key);
          const rim = new THREE.PointLight(statusColor, 3.2, 12);
          rim.position.set(3, 2, -3); scene.add(rim);
          const cyan = new THREE.PointLight("#27b8d8", 2.1, 10);
          cyan.position.set(-4, 1, 3); scene.add(cyan);

          const assembly = new THREE.Group();
          scene.add(assembly);
          const metal = new THREE.MeshStandardMaterial({color:"#526c7c",metalness:.82,roughness:.27});
          const shell = new THREE.MeshStandardMaterial({color:"#274458",metalness:.78,roughness:.3});
          const darkMetal = new THREE.MeshStandardMaterial({color:"#172d3a",metalness:.87,roughness:.25});
          const bright = new THREE.MeshStandardMaterial({color:"#91d7e7",metalness:.72,roughness:.22});
          const accent = new THREE.MeshStandardMaterial({color:statusColor,metalness:.35,roughness:.28,emissive:statusColor,emissiveIntensity:.18});
          function axialCylinder(x, length, radiusA, radiusB, material, segments) {
            const mesh = new THREE.Mesh(new THREE.CylinderGeometry(radiusA,radiusB,length,segments || 36),material);
            mesh.rotation.z = Math.PI / 2; mesh.position.x = x; assembly.add(mesh); return mesh;
          }
          function axialRing(x,radius,tube,material) {
            const mesh = new THREE.Mesh(new THREE.TorusGeometry(radius,tube,10,48),material);
            mesh.rotation.y = Math.PI / 2; mesh.position.x = x; assembly.add(mesh);
          }
          function box(x,y,z,sx,sy,sz,material) {
            const mesh = new THREE.Mesh(new THREE.BoxGeometry(sx,sy,sz),material);
            mesh.position.set(x,y,z); assembly.add(mesh); return mesh;
          }
          box(0,-1.02,0,6.6,.18,2.1,darkMetal);
          box(0,-.89,-.77,6.25,.1,.1,metal); box(0,-.89,.77,6.25,.1,.1,metal);
          [-2.3,-1.3,1.3,2.3].forEach(x => {
            box(x,-.57,-.52,.22,.62,.22,metal); box(x,-.57,.52,.22,.62,.22,metal);
          });
          axialCylinder(0,3.8,.78,.78,shell,48);
          axialCylinder(-2.2,.72,.48,.7,metal);
          axialCylinder(2.15,.88,.71,.52,metal);
          axialCylinder(-2.72,.35,.3,.46,darkMetal);
          axialCylinder(2.77,.36,.25,.4,darkMetal);
          axialCylinder(0,5.25,.19,.19,bright,32);
          [-1.9,-1.58,-1.28,1.1,1.42,1.74,2.05].forEach((x,i) => {
            axialRing(x, i > 4 ? .58 : .79, i % 2 ? .035 : .055, i % 2 ? bright : metal);
          });
          [-.82,.82].forEach(x => {
            axialCylinder(x,.1,.82,.82,darkMetal);
            for(let i=0;i<12;i++){
              const a=i*Math.PI/6;
              const bolt=new THREE.Mesh(new THREE.SphereGeometry(.045,10,8),bright);
              bolt.position.set(x,Math.sin(a)*.82,Math.cos(a)*.82); assembly.add(bolt);
            }
          });
          [1.32,1.62,1.92].forEach((x,stage) => {
            axialCylinder(x,.1,.52,.52,darkMetal);
            for(let i=0;i<14;i++){
              const a=i*Math.PI/7+stage*.12;
              const blade=box(x,Math.sin(a)*.35,Math.cos(a)*.35,.055,.12,.42,accent);
              blade.rotation.x=a+.55;
            }
          });
          [-.7,.7].forEach(z => {
            const points=[new THREE.Vector3(-1.45,.78,z),new THREE.Vector3(-1.25,1.05,z),new THREE.Vector3(-.72,1.05,z),new THREE.Vector3(-.55,.76,z)];
            const tube=new THREE.Mesh(new THREE.TubeGeometry(new THREE.CatmullRomCurve3(points),28,.055,10,false),bright);
            assembly.add(tube);
          });
          const floor = new THREE.GridHelper(16,32,"#17445a","#10283a");
          floor.position.y=-1.13; scene.add(floor);
          assembly.rotation.x=-.08;

          let dragging=false, lastX=0, lastY=0, autoRotate=!window.matchMedia("(prefers-reduced-motion: reduce)").matches;
          canvas.addEventListener("pointerdown",e=>{dragging=true;lastX=e.clientX;lastY=e.clientY;canvas.setPointerCapture(e.pointerId);});
          canvas.addEventListener("pointerup",()=>{dragging=false;});
          canvas.addEventListener("pointercancel",()=>{dragging=false;});
          canvas.addEventListener("pointermove",e=>{
            if(!dragging)return;
            assembly.rotation.y+=(e.clientX-lastX)*.009;
            assembly.rotation.x=Math.max(-.55,Math.min(.55,assembly.rotation.x+(e.clientY-lastY)*.006));
            lastX=e.clientX;lastY=e.clientY;
          });
          canvas.addEventListener("wheel",e=>{e.preventDefault();camera.position.multiplyScalar(e.deltaY>0?1.08:.92);camera.position.clampLength(4.5,13);},{passive:false});
          document.getElementById("zoom-in").addEventListener("click",()=>camera.position.multiplyScalar(.88));
          document.getElementById("zoom-out").addEventListener("click",()=>camera.position.multiplyScalar(1.12));
          document.getElementById("pause").addEventListener("click",e=>{autoRotate=!autoRotate;e.currentTarget.textContent=autoRotate?"Ⅱ":"▶";});
          canvas.addEventListener("keydown",e=>{
            if(e.key==="ArrowLeft")assembly.rotation.y-=.12;
            if(e.key==="ArrowRight")assembly.rotation.y+=.12;
            if(e.key==="ArrowUp")assembly.rotation.x=Math.max(-.55,assembly.rotation.x-.08);
            if(e.key==="ArrowDown")assembly.rotation.x=Math.min(.55,assembly.rotation.x+.08);
          });
          function resize(){
            const rect=canvas.getBoundingClientRect();
            if(!rect.width||!rect.height)return;
            renderer.setSize(rect.width,rect.height,false);
            camera.aspect=rect.width/rect.height;camera.updateProjectionMatrix();
          }
          new ResizeObserver(resize).observe(canvas.parentElement);resize();
          function animate(){
            requestAnimationFrame(animate);
            if(autoRotate&&!dragging)assembly.rotation.y+=.0018;
            renderer.render(scene,camera);
          }
          animate();
        })();
        </script>
        """
    )
    replacements = {
        "__STATUS__": status,
        "__STATUS_COLOR__": status_color,
        "__UNIT__": f"{int(unit_id):02d}",
        "__RISK__": f"{failure_probability:.1%}",
        "__RUL__": f"{predicted_rul:.1f}",
    }
    for token, value in replacements.items():
        twin_html = twin_html.replace(token, html.escape(value))
    twin_url = "data:text/html;charset=utf-8," + quote(twin_html)
    st.iframe(twin_url, height=506, tab_index=0)


def inject_styles():
    st.markdown(
        """
        <style>
        :root { color-scheme: dark; }
        .stApp { background: radial-gradient(ellipse at 15% 0%,rgba(20,65,87,.22),transparent 34%),#08111b; color:#e4edf5; }
        [data-testid="stHeader"] { background:rgba(8,17,27,.82); }
        [data-testid="stAppViewContainer"] { background:transparent; }
        .block-container { max-width:1540px; padding:1.35rem clamp(1rem,3vw,2.8rem) 3rem; }
        h1,h2,h3 { color:#e8f1f8!important; letter-spacing:-.025em; }
        h2 { margin-top:1.3rem!important; }
        [data-testid="stCaptionContainer"] p { color:#8ea5b8; }
        [data-testid="stMetric"] { background:rgba(16,31,45,.75);border:1px solid rgba(101,157,184,.16);padding:16px 18px;border-radius:14px; }
        [data-testid="stMetricLabel"] p { color:#91aabd; }
        [data-testid="stMetricValue"] { color:#e8f5fb; }
        [data-testid="stDataFrame"],[data-testid="stTable"] { border:1px solid rgba(101,157,184,.15);border-radius:12px;overflow:hidden; }
        .command-header { display:flex;justify-content:space-between;align-items:center;gap:20px;margin:10px 0 12px;padding:20px 24px;border:1px solid rgba(91,167,194,.2);border-radius:16px;background:linear-gradient(115deg,rgba(18,39,55,.94),rgba(10,22,34,.86));box-shadow:0 14px 36px rgba(0,0,0,.18); }
        .brand-kicker { color:#59cbe5;font-size:10px;font-weight:700;letter-spacing:.22em; }
        .brand-title { margin-top:5px;color:#f0f7fb;font-size:clamp(22px,3vw,32px);font-weight:750;letter-spacing:.055em; }
        .brand-subtitle { margin-top:4px;color:#829caf;font-size:10px;letter-spacing:.2em; }
        .header-right { text-align:right;color:#7f9aae;font-size:11px;line-height:1.9; }
        .live-pill { display:inline-flex;align-items:center;gap:8px;padding:5px 10px;border:1px solid rgba(72,209,160,.25);border-radius:20px;color:#7de0b8;background:rgba(72,209,160,.06);font-size:10px;letter-spacing:.12em; }
        .live-pill i { width:7px;height:7px;background:#48d1a0;border-radius:50%;box-shadow:0 0 12px #48d1a0; }
        .top-nav { display:flex;flex-wrap:wrap;gap:9px;margin:0 0 18px; }
        .top-nav a { padding:7px 12px;border:1px solid rgba(99,155,181,.16);border-radius:20px;color:#a8bdcd;text-decoration:none;font-size:11px;transition:all .18s ease; }
        .top-nav a:hover,.top-nav a:focus-visible { border-color:#45bddb;color:#d9f6fc;background:rgba(46,156,184,.09);outline:none; }
        .kpi-card { min-height:130px;padding:17px 18px;border:1px solid rgba(92,162,190,.19);border-radius:15px;background:linear-gradient(145deg,rgba(20,39,54,.9),rgba(13,25,37,.9));box-shadow:0 12px 30px rgba(0,0,0,.12);transition:transform .18s ease,border-color .18s ease,box-shadow .18s ease; }
        .kpi-card:hover { transform:translateY(-3px);border-color:rgba(63,196,222,.48);box-shadow:0 16px 35px rgba(0,0,0,.2),0 0 22px rgba(37,161,190,.08); }
        .kpi-heading { display:flex;justify-content:space-between;align-items:center;color:#91aabd;font-size:11px;letter-spacing:.08em; }
        .kpi-icon { display:grid;place-items:center;width:30px;height:30px;border:1px solid rgba(66,193,221,.22);border-radius:9px;color:#65cee4;background:rgba(42,160,185,.09);font-size:15px; }
        .kpi-value { margin-top:14px;color:#f0f7fb;font-size:clamp(23px,2.3vw,31px);font-weight:720;letter-spacing:-.035em; }
        .kpi-note { margin-top:4px;color:#70899e;font-size:10px; }
        .section-head { display:flex;align-items:center;gap:10px;margin:26px 0 6px; }
        .section-head span { color:#54c9e2;font-size:10px;font-weight:700;letter-spacing:.16em; }
        .section-head b { color:#e2eef6;font-size:19px;font-weight:650; }
        .section-note { margin:0 0 12px;color:#8198ab;font-size:12px; }
        .panel { padding:16px 18px;border:1px solid rgba(95,154,181,.16);border-radius:14px;background:rgba(14,28,41,.68); }
        .status-chip { display:inline-block;padding:4px 9px;border:1px solid currentColor;border-radius:20px;font-size:10px;letter-spacing:.1em; }
        [data-testid="stForm"] { padding:18px;border:1px solid rgba(90,158,186,.18);border-radius:15px;background:rgba(13,26,38,.68); }
        [data-testid="stFormSubmitButton"] button,[data-testid="stButton"] button { border:1px solid rgba(71,195,221,.36);border-radius:9px;color:#e3f7fc;background:linear-gradient(100deg,#12607b,#1680a0);transition:transform .16s ease,box-shadow .16s ease; }
        [data-testid="stFormSubmitButton"] button:hover,[data-testid="stButton"] button:hover { transform:translateY(-1px);box-shadow:0 0 18px rgba(44,188,220,.2); }
        [data-testid="stTextInput"] input,[data-testid="stNumberInput"] input,[data-testid="stSelectbox"] [role="combobox"] { border-color:rgba(101,160,187,.25); }
        hr { border-color:rgba(103,155,180,.14); }
        .small-note { color:#7992a6;font-size:11px;line-height:1.6; }
        @media(max-width:720px) { .command-header { align-items:flex-start;padding:16px; }.header-right { font-size:10px; }.block-container { padding-left:.8rem;padding-right:.8rem; }.top-nav { gap:6px; }.top-nav a { font-size:10px;padding:6px 9px; } }
        @media(prefers-reduced-motion:reduce) { *,*::before,*::after { scroll-behavior:auto!important;transition:none!important;animation:none!important; } }
        </style>
        """,
        unsafe_allow_html=True,
    )


def render_kpi(label, value, note, icon, accent):
    return (
        '<div class="kpi-card" style="border-top:2px solid '
        + accent
        + '"><div class="kpi-heading"><span>'
        + html.escape(label)
        + '</span><span class="kpi-icon">'
        + html.escape(icon)
        + '</span></div><div class="kpi-value">'
        + html.escape(str(value))
        + '</div><div class="kpi-note">'
        + html.escape(note)
        + "</div></div>"
    )


def section_heading(anchor, number, title, note):
    st.markdown(
        f'<div id="{anchor}" class="section-head"><span>{number}</span><b>{title}</b></div>'
        f'<p class="section-note">{note}</p>',
        unsafe_allow_html=True,
    )


st.set_page_config(
    page_title="Asset Intelligence | Predictive Maintenance",
    page_icon="⚙️",
    layout="wide",
    initial_sidebar_state="collapsed",
)
inject_styles()

xgb_rul, xgb_clf, threshold, missing_model_paths = load_models()
if missing_model_paths:
    st.error(
        "Required model artifact file(s) not found. Checked paths:\n"
        + "\n".join(f"- {path}" for path in missing_model_paths)
    )
    st.stop()

if xgb_rul.feature_cols != xgb_clf.feature_cols:
    st.error("The loaded RUL and risk models do not share the same feature schema.")
    st.stop()

try:
    raw_df, feat_df, rul_df = load_test_data()
except FileNotFoundError as error:
    st.error(f"Could not load the demo sensor dataset: {error}")
    st.stop()

feature_cols = xgb_rul.feature_cols or get_feature_cols(feat_df)
missing_features = [col for col in feature_cols if col not in feat_df.columns]
if missing_features:
    st.error(f"Dataset is missing model features: {', '.join(missing_features)}")
    st.stop()

latest_df = get_latest_per_unit(feat_df)
X_latest = latest_df.loc[:, feature_cols]
latest_df = latest_df.copy()
latest_df["pred_rul"] = np.clip(xgb_rul.predict(X_latest), 0, None).round(1)
latest_df["failure_prob"] = xgb_clf.predict_proba(X_latest)
latest_df["alert"] = latest_df["failure_prob"] >= threshold
latest_df["health_status"] = latest_df["failure_prob"].apply(risk_color)
latest_df["risk_icon"] = latest_df["health_status"].map(
    {"HEALTHY": "●", "WARNING": "●", "CRITICAL": "●"}
)

system_online = xgb_rul is not None and xgb_clf is not None
now = datetime.now().astimezone()
st.markdown(
    f"""
    <div class="command-header">
      <div>
        <div class="brand-kicker">ASSET INTELLIGENCE</div>
        <div class="brand-title">PREDICTIVE ASSET HEALTH</div>
        <div class="brand-subtitle">INDUSTRIAL MONITORING PLATFORM · FD001 DEMO</div>
      </div>
      <div class="header-right">
        <span class="live-pill"><i></i>{"SYSTEM ONLINE" if system_online else "MODEL OFFLINE"}</span><br>
        {now.strftime("%d %b %Y · %H:%M %Z")}<br>
        SYNTHETIC DEMO DATA
      </div>
    </div>
    <nav class="top-nav" aria-label="Dashboard sections">
      <a href="#overview">OVERVIEW</a><a href="#digital-twin">DIGITAL TWIN</a>
      <a href="#fleet">FLEET HEALTH</a><a href="#prediction">PREDICTION LAB</a>
      <a href="#monitoring">MONITORING</a>
    </nav>
    """,
    unsafe_allow_html=True,
)

n_assets = int(latest_df["unit_id"].nunique())
n_high_risk = int((latest_df["health_status"] == "CRITICAL").sum())
average_rul = float(latest_df["pred_rul"].mean()) if n_assets else float("nan")

section_heading(
    "overview",
    "01 /",
    "Fleet at a glance",
    "Model-scored final observations from the generated FD001-style test fleet.",
)
kpi_columns = st.columns(4)
kpis = [
    ("ASSETS MONITORED", n_assets, "Synthetic demo units in the test set", "◈", "#39bfdc"),
    ("HIGH-RISK ASSETS", n_high_risk, "Risk probability ≥ 60% · API high-risk band", "!", "#ff7180"),
    ("AVERAGE REMAINING LIFE", f"{average_rul:.1f} cycles", "Mean predicted RUL across monitored units", "↗", "#42d5c3"),
    ("MODEL STATUS", "ONLINE" if system_online else "OFFLINE", "XGBoost RUL + risk classifier loaded", "✓", "#48d1a0"),
]
for column, item in zip(kpi_columns, kpis):
    with column:
        st.markdown(render_kpi(*item), unsafe_allow_html=True)

unit_ids = sorted(int(unit_id) for unit_id in raw_df["unit_id"].unique())
latest_by_unit = latest_df.set_index("unit_id")

section_heading(
    "digital-twin",
    "02 /",
    "Asset digital twin",
    "Interactive illustrative turbine assembly. It is not a geometric twin of the underlying demo records.",
)
twin_left, twin_right = st.columns([1.7, 1], gap="large")
with twin_right:
    selected_unit = st.selectbox(
        "Demo unit",
        unit_ids,
        format_func=lambda value: f"UNIT {value:02d} · SYNTHETIC",
        key="twin_unit",
    )
    unit_score = latest_by_unit.loc[selected_unit]
    unit_status = unit_score["health_status"]
    st.markdown(
        f'<div class="panel"><div class="small-note">CURRENT MODEL ASSESSMENT</div>'
        f'<p style="font-size:22px;font-weight:700;color:{STATUS_COLORS[unit_status]};margin:8px 0">'
        f'{unit_status}</p><div class="small-note">Failure probability</div>'
        f'<div style="font-size:20px;color:#eaf4fb">{unit_score["failure_prob"]:.1%}</div>'
        f'<div class="small-note" style="margin-top:12px">Predicted remaining life</div>'
        f'<div style="font-size:20px;color:#eaf4fb">{unit_score["pred_rul"]:.1f} cycles</div>'
        f'<div class="small-note" style="margin-top:12px">Last observed cycle · {int(unit_score["cycle"])}</div>'
        f"</div>",
        unsafe_allow_html=True,
    )
    st.caption("● HEALTHY < 30%  ·  ● WARNING 30–60%  ·  ● CRITICAL ≥ 60%")
with twin_left:
    render_digital_twin(
        selected_unit,
        float(unit_score["failure_prob"]),
        float(unit_score["pred_rul"]),
    )

section_heading(
    "fleet",
    "03 /",
    "Fleet health",
    "Risk categories use the API's probability bands; alert count uses the saved cost-based decision threshold.",
)
health_counts = (
    latest_df["health_status"]
    .value_counts()
    .reindex(["HEALTHY", "WARNING", "CRITICAL"], fill_value=0)
    .rename_axis("Health status")
    .rename("Assets")
    .reset_index()
)
health_col, rul_col = st.columns(2, gap="large")
with health_col:
    st.markdown("#### Health distribution")
    st.bar_chart(
        health_counts,
        x="Health status",
        y="Assets",
        color="Health status",
        height=250,
    )
with rul_col:
    st.markdown("#### Predicted RUL by unit")
    st.bar_chart(
        latest_df.set_index("unit_id")[["pred_rul"]],
        color="#39bfdc",
        height=250,
    )

display_df = latest_df[
    ["unit_id", "cycle", "health_status", "pred_rul", "failure_prob", "alert"]
].rename(
    columns={
        "unit_id": "Demo unit",
        "cycle": "Last cycle",
        "health_status": "Health",
        "pred_rul": "Predicted RUL (cycles)",
        "failure_prob": "Failure probability",
        "alert": "Threshold alert",
    }
)
st.dataframe(
    display_df.sort_values(
        ["Threshold alert", "Failure probability"], ascending=[False, False]
    ),
    use_container_width=True,
    height=340,
    hide_index=True,
)

unit_history = raw_df[raw_df["unit_id"] == selected_unit].sort_values("cycle").copy()
unit_features = feat_df[feat_df["unit_id"] == selected_unit].sort_values("cycle").copy()
unit_X = unit_features.loc[:, feature_cols]
unit_features["pred_rul"] = np.clip(xgb_rul.predict(unit_X), 0, None)
unit_features["failure_prob"] = xgb_clf.predict_proba(unit_X)
trend_col, sensor_col = st.columns(2, gap="large")
with trend_col:
    st.markdown(f"#### Unit {selected_unit:02d} · model trajectory")
    st.line_chart(
        unit_features.set_index("cycle")[["pred_rul"]],
        color="#39bfdc",
        height=260,
    )
    st.line_chart(
        unit_features.set_index("cycle")[["failure_prob"]],
        color="#ff7180",
        height=220,
    )
with sensor_col:
    st.markdown(f"#### Unit {selected_unit:02d} · observed sensors")
    visible_sensors = [sensor for sensor in ("s2", "s11", "s12") if sensor in unit_history]
    st.line_chart(
        unit_history.set_index("cycle")[visible_sensors],
        color=["#39bfdc", "#42d5c3", "#88a9ff"],
        height=260,
    )
    st.caption("Observed demo sensor readings; sensor units are not specified in the dataset.")

section_heading(
    "prediction",
    "04 /",
    "Prediction lab",
    "Edit a new sensor reading seeded from the selected synthetic unit's observed history. No real-time fleet feed is connected.",
)
st.markdown(
    '<p class="small-note">This interactive demo appends one entered reading to the unit history, '
    'rebuilds the same rolling, slope and cumulative features, then calls the existing XGBoost wrappers. '
    'Sensor values are shown in dataset units because physical units are not defined here.</p>',
    unsafe_allow_html=True,
)

history = raw_df[raw_df["unit_id"] == selected_unit].sort_values("cycle").copy()
seed = history.iloc[-1]
next_cycle = int(seed["cycle"]) + 1
with st.form("sensor_prediction_form"):
    reading = {
        "cycle": st.number_input(
            "Next cycle",
            min_value=1,
            value=next_cycle,
            step=1,
            key=f"cycle_{selected_unit}",
        )
    }
    st.markdown("##### Operating settings")
    setting_columns = st.columns(3)
    for column, setting in zip(setting_columns, SETTING_COLS):
        with column:
            reading[setting] = st.number_input(
                f"Operating setting {setting}",
                value=float(seed[setting]),
                format="%.4f",
                key=f"setting_{selected_unit}_{setting}",
            )
    st.markdown("##### Sensor readings · dataset units")
    sensor_columns = st.columns(3)
    for index, sensor in enumerate(SENSOR_COLS):
        with sensor_columns[index % 3]:
            reading[sensor] = st.number_input(
                sensor.upper(),
                value=float(seed[sensor]),
                format="%.4f",
                key=f"sensor_{selected_unit}_{sensor}",
            )
    submitted = st.form_submit_button("Run health assessment", use_container_width=True)

if submitted:
    if reading["cycle"] != next_cycle:
        st.error(
            f"Cycle must be exactly {next_cycle}, the next consecutive cycle after the selected unit's history."
        )
    else:
        try:
            with st.spinner("Engineering the sensor window and scoring both models..."):
                prediction_rul, prediction_risk = score_reading(
                    history, reading, xgb_rul, xgb_clf
                )
            result_status = risk_color(prediction_risk)
            result_cols = st.columns(3)
            result_cols[0].metric("Predicted remaining life", f"{prediction_rul:.1f} cycles")
            result_cols[1].metric("Failure probability", f"{prediction_risk:.1%}")
            result_cols[2].metric("Health category", result_status)
            st.progress(
                min(max(prediction_risk, 0.0), 1.0),
                text=f"Failure probability · {prediction_risk:.1%}",
            )
            if prediction_risk >= threshold:
                st.warning(
                    f"Decision threshold exceeded ({threshold:.1%}). Treat this as a demo alert, "
                    "not an operational maintenance decision."
                )
            else:
                st.success("Prediction is below the saved operating threshold for this demo.")
        except (KeyError, TypeError, ValueError) as error:
            st.error(f"Could not score this sensor reading: {error}")

section_heading(
    "monitoring",
    "05 /",
    "Model monitoring & explainability",
    "Monitor artifacts generated by the analysis notebooks. Feature importance is model-level, not a per-prediction explanation.",
)
monitor_left, monitor_right = st.columns(2, gap="large")
with monitor_left:
    st.markdown("#### Serving model status")
    model_rows = pd.DataFrame(
        [
            {"Artifact": "xgb_rul.pkl", "Status": "LOADED" if xgb_rul else "MISSING"},
            {"Artifact": "xgb_clf.pkl", "Status": "LOADED" if xgb_clf else "MISSING"},
            {
                "Artifact": "best_threshold.json",
                "Status": f"{threshold:.3f}" if os.path.isfile(os.path.join(ARTIFACTS, "best_threshold.json")) else "DEFAULT 0.500",
            },
        ]
    )
    st.dataframe(model_rows, use_container_width=True, hide_index=True)
    st.markdown("#### Classifier feature importance")
    importance = xgb_clf.feature_importance().head(10).sort_values()
    importance_df = importance.rename_axis("Feature").rename("Importance").reset_index()
    st.bar_chart(
        importance_df,
        x="Importance",
        y="Feature",
        horizontal=True,
        color="#42d5c3",
        height=330,
    )
    st.caption(
        "Uses the classifier's built-in XGBoost feature_importances_ values. "
        "These are model-level importance signals, not causal sensor effects or SHAP values."
    )

with monitor_right:
    st.markdown("#### Sensor drift")
    drift_path = os.path.join(ARTIFACTS, "last_drift_report.json")
    if os.path.isfile(drift_path):
        with open(drift_path, encoding="utf-8") as file:
            drift = json.load(file)
        drift_cols = st.columns(3)
        drift_cols[0].metric("Features checked", drift.get("n_features_checked", "—"))
        drift_cols[1].metric("Drifted", drift.get("n_drifted", "—"))
        drift_cols[2].metric("Worst KS statistic", f'{drift.get("worst_ks", 0):.3f}')
        if drift.get("n_drifted", 0) > 0:
            st.warning(
                f"Drift reported in {drift['n_drifted']} features "
                f"({drift.get('pct_drifted', 0):.1f}%). Highest KS statistic: "
                f"{drift.get('worst_feature', 'not available')}."
            )
        else:
            st.success("No significant drift recorded in the latest report.")
        st.caption("This is the saved notebook report, including simulated data-shift scenarios.")
    else:
        st.info("No drift summary is available. Run notebook 06 to create the report.")

st.divider()
st.markdown(
    '<p class="small-note">ASSET INTELLIGENCE · Synthetic FD001 demo · Predictive scores are advisory '
    'and are not safety-critical control outputs.</p>',
    unsafe_allow_html=True,
)
