import os
import cv2
import csv
import time
import random
import sqlite3
import numpy as np
import streamlit as st
from datetime import datetime
from collections import deque
from pathlib import Path

# ─── Page Config ─────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Crash Detection — AI Traffic Safety",
    page_icon="🚨",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ─── Constants ────────────────────────────────────────────────────────────────
MODEL_PATH           = "runs/classify/ccd-classify4/weights/best.pt"
DB_FILE              = "crash_detection.db"   # ← SQLite database

LIVE_FEED_DIR   = "live_feed"
SNAP_DIR        = os.path.join(LIVE_FEED_DIR, "screenshots")
CLIP_DIR        = os.path.join(LIVE_FEED_DIR, "clips")
EVIDENCE_DIR    = "evidence"

CONFIDENCE_THRESHOLD = 0.50
CLIP_BEFORE_SEC      = 30
CLIP_AFTER_SEC       = 30

CRASH_VIDEO_DIR  = r"C:\datsets\Crash-1500"
NORMAL_VIDEO_DIR = r"C:\datsets\Normal"

for _d in [SNAP_DIR, CLIP_DIR, EVIDENCE_DIR]:
    os.makedirs(_d, exist_ok=True)

# ─── Demo users ───────────────────────────────────────────────────────────────
USERS = {
    "admin":   {"password": "admin123",   "role": "admin",   "name": "Admin User"},
    "citizen": {"password": "citizen123", "role": "citizen", "name": "John Doe"},
    "officer": {"password": "officer123", "role": "citizen", "name": "Officer Patel"},
}

# ════════════════════════════════════════════════════════════════════════════
# SQLite DATABASE LAYER
# ════════════════════════════════════════════════════════════════════════════

def get_db():
    """Return a thread-local SQLite connection."""
    conn = sqlite3.connect(DB_FILE, check_same_thread=False)
    conn.row_factory = sqlite3.Row   # rows behave like dicts
    return conn


def init_db():
    """Create all tables if they don't exist."""
    conn = get_db()
    c = conn.cursor()

    # ── crash_events: auto-detected CCTV crash logs ──────────────────────────
    c.execute("""
        CREATE TABLE IF NOT EXISTS crash_events (
            id          INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp   TEXT    NOT NULL,
            username    TEXT    NOT NULL,
            camera      TEXT    NOT NULL,
            location    TEXT    DEFAULT '',
            confidence  REAL    NOT NULL,
            snapshot    TEXT    DEFAULT '',
            clip_path   TEXT    DEFAULT '',
            crash_time  TEXT    DEFAULT ''
        )
    """)

    # ── evidence_submissions: citizen/officer video uploads ───────────────────
    c.execute("""
        CREATE TABLE IF NOT EXISTS evidence_submissions (
            id              INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp       TEXT    NOT NULL,
            submitted_by    TEXT    NOT NULL,
            filename        TEXT    NOT NULL,
            location        TEXT    DEFAULT '',
            description     TEXT    DEFAULT '',
            file_path       TEXT    NOT NULL,
            confidence      REAL    DEFAULT 0.0
        )
    """)

    # ── users table (optional – mirrors the hardcoded USERS dict) ─────────────
    c.execute("""
        CREATE TABLE IF NOT EXISTS users (
            username    TEXT PRIMARY KEY,
            password    TEXT NOT NULL,
            role        TEXT NOT NULL,
            name        TEXT NOT NULL,
            created_at  TEXT DEFAULT CURRENT_TIMESTAMP
        )
    """)

    # Seed users table from USERS dict
    for uname, udata in USERS.items():
        c.execute("""
            INSERT OR IGNORE INTO users (username, password, role, name)
            VALUES (?, ?, ?, ?)
        """, (uname, udata["password"], udata["role"], udata["name"]))

    conn.commit()
    conn.close()


# ── crash_events helpers ──────────────────────────────────────────────────────

def db_insert_crash(username, camera, location, confidence,
                    snapshot="", clip_path="", crash_time=""):
    conn = get_db()
    conn.execute("""
        INSERT INTO crash_events
            (timestamp, username, camera, location, confidence, snapshot, clip_path, crash_time)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?)
    """, (
        datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        username, camera, location, confidence,
        snapshot, clip_path, crash_time
    ))
    conn.commit()
    conn.close()


def db_get_crashes(username=None, role="admin", limit=200):
    conn = get_db()
    if role == "admin" or username is None:
        rows = conn.execute(
            "SELECT * FROM crash_events ORDER BY id DESC LIMIT ?", (limit,)
        ).fetchall()
    else:
        rows = conn.execute(
            "SELECT * FROM crash_events WHERE username=? ORDER BY id DESC LIMIT ?",
            (username, limit)
        ).fetchall()
    conn.close()
    return [dict(r) for r in rows]


# ── evidence_submissions helpers ──────────────────────────────────────────────

def db_insert_evidence(submitted_by, filename, location, description,
                       file_path, confidence=0.0):
    conn = get_db()
    conn.execute("""
        INSERT INTO evidence_submissions
            (timestamp, submitted_by, filename, location, description, file_path, confidence)
        VALUES (?, ?, ?, ?, ?, ?, ?)
    """, (
        datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        submitted_by, filename, location, description, file_path, confidence
    ))
    conn.commit()
    conn.close()


def db_get_evidence(submitted_by=None, role="admin", limit=200):
    conn = get_db()
    if role == "admin" or submitted_by is None:
        rows = conn.execute(
            "SELECT * FROM evidence_submissions ORDER BY id DESC LIMIT ?", (limit,)
        ).fetchall()
    else:
        rows = conn.execute(
            "SELECT * FROM evidence_submissions WHERE submitted_by=? ORDER BY id DESC LIMIT ?",
            (submitted_by, limit)
        ).fetchall()
    conn.close()
    return [dict(r) for r in rows]


# ── stats helpers ─────────────────────────────────────────────────────────────

def db_stats():
    conn = get_db()
    total_crashes    = conn.execute("SELECT COUNT(*) FROM crash_events").fetchone()[0]
    total_evidence   = conn.execute("SELECT COUNT(*) FROM evidence_submissions").fetchone()[0]
    active_users     = conn.execute(
        "SELECT COUNT(DISTINCT username) FROM crash_events"
    ).fetchone()[0]
    today = datetime.now().strftime("%Y-%m-%d")
    crashes_today    = conn.execute(
        "SELECT COUNT(*) FROM crash_events WHERE timestamp LIKE ?", (f"{today}%",)
    ).fetchone()[0]
    conn.close()
    return {
        "total_crashes":  total_crashes,
        "total_evidence": total_evidence,
        "active_users":   active_users,
        "crashes_today":  crashes_today,
    }


# Initialise DB on startup
init_db()

# ─── Model loader ─────────────────────────────────────────────────────────────
@st.cache_resource(show_spinner=False)
def load_model():
    try:
        from ultralytics import YOLO
        if os.path.exists(MODEL_PATH):
            return YOLO(MODEL_PATH)
    except Exception:
        pass
    return None


def detect_frame(model, frame):
    try:
        results = model.predict(source=frame, verbose=False, imgsz=224)
        probs = results[0].probs.data.cpu().numpy()
        return float(probs[0])
    except Exception:
        return 0.0


def save_snapshot(frame, prefix="crash", folder=SNAP_DIR):
    os.makedirs(folder, exist_ok=True)
    fname = f"{prefix}_{datetime.now().strftime('%Y%m%d_%H%M%S_%f')}.jpg"
    path  = os.path.join(folder, fname)
    cv2.imwrite(path, frame)
    return path


def extract_crash_clip(video_path, crash_ts_sec,
                       before=CLIP_BEFORE_SEC, after=CLIP_AFTER_SEC,
                       out_dir=CLIP_DIR):
    try:
        cap      = cv2.VideoCapture(video_path)
        fps      = cap.get(cv2.CAP_PROP_FPS) or 25
        total_f  = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        w        = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        h        = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        start_f  = max(0, int((crash_ts_sec - before) * fps))
        end_f    = min(total_f, int((crash_ts_sec + after) * fps))
        fname    = f"clip_{datetime.now().strftime('%Y%m%d_%H%M%S')}.mp4"
        out_path = os.path.join(out_dir, fname)
        fourcc   = cv2.VideoWriter_fourcc(*"mp4v")
        writer   = cv2.VideoWriter(out_path, fourcc, fps, (w, h))
        cap.set(cv2.CAP_PROP_POS_FRAMES, start_f)
        for _ in range(end_f - start_f):
            ret, frame = cap.read()
            if not ret:
                break
            writer.write(frame)
        cap.release()
        writer.release()
        return out_path if os.path.exists(out_path) else ""
    except Exception:
        return ""


def get_video_files(directory, extension=".mp4"):
    if not os.path.isdir(directory):
        return []
    return sorted([
        os.path.join(directory, f)
        for f in os.listdir(directory)
        if f.lower().endswith(extension)
    ])

# ─── CSS ─────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&family=JetBrains+Mono:wght@400;500&display=swap');
:root {
    --bg:#F0F4FA;--surface:#fff;--border:#D6E4F0;
    --blue-50:#EBF4FF;--blue-100:#BFDBFE;--blue-400:#378ADD;
    --blue-600:#185FA5;--blue-800:#0C447C;--blue-900:#042C53;
    --gray-400:#9BA3AF;--gray-500:#6B7280;--gray-900:#111827;
    --red-400:#E24B4A;--red-600:#A32D2D;--red-bg:#FEF2F2;--red-border:#FECACA;
    --green-400:#16A34A;--green-bg:#F0FDF4;--green-bdr:#BBF7D0;
    --shadow-sm:0 1px 3px rgba(15,40,80,.08);
}
html,body,[data-testid="stAppViewContainer"]{background:var(--bg)!important;color:var(--gray-900)!important;font-family:'Inter',sans-serif!important;}
[data-testid="stSidebar"]{background:var(--blue-900)!important;}
[data-testid="stSidebar"] *{color:#cbd5e1!important;}
[data-testid="stSidebar"] .stButton>button{background:rgba(255,255,255,.07)!important;color:#e2e8f0!important;border:1px solid rgba(255,255,255,.12)!important;border-radius:8px!important;font-size:.88rem!important;font-weight:500!important;padding:.45rem 1rem!important;width:100%!important;}
[data-testid="stSidebar"] .stButton>button:hover{background:rgba(255,255,255,.15)!important;}
h1,h2,h3{font-family:'Inter',sans-serif!important;font-weight:600!important;}
.cd-logo{display:flex;align-items:center;gap:10px;padding:1.2rem 1rem .8rem;border-bottom:1px solid rgba(255,255,255,.12);margin-bottom:1rem;}
.cd-logo-icon{width:36px;height:36px;background:var(--blue-400);border-radius:9px;display:flex;align-items:center;justify-content:center;font-size:18px;}
.cd-logo-text{font-size:1.05rem;font-weight:700;color:#fff!important;}
.cd-logo-text span{color:#93C5FD!important;}
.page-header{display:flex;align-items:center;justify-content:space-between;margin-bottom:1.5rem;padding-bottom:1rem;border-bottom:1px solid var(--border);}
.page-title{font-size:1.4rem;font-weight:700;}
.page-sub{font-size:.82rem;color:var(--gray-500);margin-top:2px;}
.live-pill{display:inline-flex;align-items:center;gap:6px;background:var(--green-bg);border:1px solid var(--green-bdr);color:var(--green-400);font-size:.78rem;font-weight:600;padding:4px 12px;border-radius:20px;}
.live-dot{width:7px;height:7px;border-radius:50%;background:var(--green-400);animation:livepulse 1.5s infinite;}
@keyframes livepulse{0%,100%{opacity:1}50%{opacity:.4}}
.stat-card{background:var(--surface);border:1px solid var(--border);border-radius:12px;padding:1.1rem 1.3rem;box-shadow:var(--shadow-sm);}
.stat-label{font-size:11px;font-weight:600;color:var(--gray-400);text-transform:uppercase;letter-spacing:.6px;margin-bottom:6px;}
.stat-value{font-size:2rem;font-weight:700;line-height:1;color:var(--gray-900);}
.stat-sub{font-size:11px;color:var(--gray-400);margin-top:4px;}
.stat-card.danger .stat-value{color:var(--red-600);}
.stat-card.success .stat-value{color:var(--green-400);}
.stat-card.info .stat-value{color:var(--blue-600);}
.section-title{font-size:11px;font-weight:700;text-transform:uppercase;letter-spacing:.8px;color:var(--gray-500);margin-bottom:12px;}
.crash-banner{background:var(--red-bg);border:1.5px solid var(--red-border);border-left:4px solid var(--red-400);border-radius:10px;padding:.85rem 1.2rem;display:flex;align-items:center;gap:10px;font-size:.9rem;font-weight:600;color:var(--red-600);animation:crashpulse .8s infinite alternate;margin-bottom:1rem;}
@keyframes crashpulse{from{opacity:1}to{opacity:.75}}
.safe-banner{background:var(--green-bg);border:1px solid var(--green-bdr);border-left:4px solid var(--green-400);border-radius:10px;padding:.85rem 1.2rem;font-size:.9rem;font-weight:600;color:var(--green-400);margin-bottom:1rem;}
.alert-item{background:var(--surface);border:1px solid var(--border);border-radius:10px;padding:.85rem 1.1rem;margin-bottom:8px;display:flex;align-items:flex-start;gap:12px;box-shadow:var(--shadow-sm);}
.alert-item:hover{border-color:var(--blue-100);background:var(--blue-50);}
.alert-item.crash{border-left:3px solid var(--red-400);}
.alert-item.normal{border-left:3px solid var(--green-400);}
.alert-dot{width:10px;height:10px;border-radius:50%;flex-shrink:0;margin-top:4px;}
.alert-dot.red{background:var(--red-400);}
.alert-dot.green{background:var(--green-400);}
.alert-meta{flex:1;}
.alert-cam{font-size:13px;font-weight:600;color:var(--gray-900);}
.alert-time{font-size:11px;color:var(--gray-400);font-family:'JetBrains Mono',monospace;margin-top:2px;}
.alert-conf{font-size:12px;font-weight:700;}
.alert-conf.danger{color:var(--red-600);}
.alert-conf.safe{color:var(--green-400);}
.folder-tag{font-size:10px;font-family:'JetBrains Mono',monospace;background:var(--blue-50);color:var(--blue-800);border:1px solid var(--blue-100);padding:2px 7px;border-radius:5px;margin-top:4px;display:inline-block;}
.panel-card{background:var(--surface);border:1px solid var(--border);border-radius:12px;padding:1.25rem;box-shadow:var(--shadow-sm);margin-bottom:1rem;}
.stButton>button{background:var(--blue-600)!important;color:#fff!important;border:none!important;border-radius:8px!important;font-family:'Inter',sans-serif!important;font-weight:500!important;font-size:.9rem!important;padding:.5rem 1.2rem!important;}
.stButton>button:hover{background:var(--blue-800)!important;}
.stTextInput>div>input{border:1px solid var(--border)!important;border-radius:8px!important;background:var(--surface)!important;}
.badge{display:inline-block;padding:2px 10px;border-radius:20px;font-size:11px;font-weight:600;text-transform:uppercase;letter-spacing:.4px;}
.badge-admin{background:var(--blue-50);color:var(--blue-800);border:1px solid var(--blue-100);}
.badge-citizen{background:#F3F4F6;color:#374151;border:1px solid #D1D5DB;}
.info-box{background:var(--blue-50);border:1px solid var(--blue-100);border-left:4px solid var(--blue-400);border-radius:10px;padding:1rem 1.25rem;font-size:.88rem;color:var(--blue-800);margin-bottom:1rem;}
.db-badge{display:inline-flex;align-items:center;gap:5px;background:#f0fdf4;border:1px solid #bbf7d0;color:#15803d;font-size:11px;font-weight:600;padding:3px 10px;border-radius:20px;}
</style>
""", unsafe_allow_html=True)

# ─── Session state ────────────────────────────────────────────────────────────
for k, v in [
    ("logged_in", False), ("username", ""), ("role", ""),
    ("name", ""), ("page", ""), ("alert_logs", []),
    ("crash_count", 0), ("save_count", 0),
]:
    if k not in st.session_state:
        st.session_state[k] = v

# ════════════════════════════════════════════════════════════════════════════
# LOGIN
# ════════════════════════════════════════════════════════════════════════════
def login_page():
    st.markdown("""
    <div style="text-align:center;padding:2.5rem 0 1rem;">
      <div style="display:inline-flex;align-items:center;gap:12px;margin-bottom:8px;">
        <div style="width:44px;height:44px;background:#185FA5;border-radius:12px;display:flex;align-items:center;justify-content:center;font-size:22px;">🚨</div>
        <span style="font-size:1.8rem;font-weight:700;color:#111827;">Crash <span style="color:#185FA5">Detection</span></span>
      </div>
      <p style="color:#6B7280;font-size:.9rem;margin:0;">AI-Powered CCTV Traffic Safety System</p>
      <span class="db-badge" style="margin-top:8px;">🗄️ SQLite Database Active</span>
    </div>
    """, unsafe_allow_html=True)

    _, col, _ = st.columns([1, 1.1, 1])
    with col:
        st.markdown('<div class="panel-card">', unsafe_allow_html=True)
        st.markdown("#### 🔐 Sign In")
        role_choice = st.selectbox("Portal", ["Admin", "Citizen"])
        username    = st.text_input("Username", placeholder="Enter username")
        password    = st.text_input("Password", type="password", placeholder="Enter password")
        if st.button("Sign In", use_container_width=True):
            user = USERS.get(username.lower())
            if user and user["password"] == password:
                if user["role"] == role_choice.lower() or role_choice.lower() == "admin":
                    st.session_state.update({
                        "logged_in": True, "username": username.lower(),
                        "role": user["role"], "name": user["name"],
                        "page": "feeds" if user["role"] == "admin" else "logs"
                    })
                    st.rerun()
                else:
                    st.error("You don't have access to that portal.")
            else:
                st.error("Invalid username or password.")
        st.markdown("</div>", unsafe_allow_html=True)
        st.markdown("""
        <p style='text-align:center;font-size:.78rem;color:#9BA3AF;margin-top:1rem;'>
        Demo: <code>admin / admin123</code> &nbsp;|&nbsp; <code>citizen / citizen123</code>
        </p>""", unsafe_allow_html=True)

# ════════════════════════════════════════════════════════════════════════════
# SIDEBAR
# ════════════════════════════════════════════════════════════════════════════
def render_sidebar():
    with st.sidebar:
        st.markdown("""
        <div class="cd-logo">
          <div class="cd-logo-icon">🚨</div>
          <div class="cd-logo-text">Crash <span>Detection</span></div>
        </div>
        """, unsafe_allow_html=True)
        badge_cls = "badge-admin" if st.session_state.role == "admin" else "badge-citizen"
        st.markdown(
            f"<div style='padding:.5rem .8rem .75rem;border-bottom:1px solid rgba(255,255,255,.1);margin-bottom:.75rem;'>"
            f"<div style='font-size:.82rem;color:#94a3b8;margin-bottom:4px;'>Signed in as</div>"
            f"<div style='font-weight:600;color:#f1f5f9;font-size:.9rem;'>{st.session_state.name}</div>"
            f"<span class='badge {badge_cls}' style='margin-top:5px;display:inline-block;'>{st.session_state.role.upper()}</span>"
            f"</div>", unsafe_allow_html=True
        )
        if st.session_state.role == "admin":
            pages = {
                "📺  Live CCTV Feeds":      "feeds",
                "📋  Alert Logs":            "logs",
                "📡  Live IP Feed":          "live",
                "📁  Evidence Submissions":  "evidence",
                "👥  Manage Users":          "users",
                "📊  Dashboard":             "dashboard",
                "🗄️  Database Viewer":       "db_viewer",
            }
        else:
            pages = {
                "📤  Submit Evidence":   "submit_evidence",
                "📋  My Reports":        "logs",
            }
        for label, key in pages.items():
            if st.button(label, use_container_width=True, key=f"nav_{key}"):
                st.session_state.page = key
                st.rerun()
        if st.button("🚪  Logout", use_container_width=True):
            for k in ["logged_in", "username", "role", "name", "page"]:
                st.session_state.pop(k, None)
            st.rerun()

# ════════════════════════════════════════════════════════════════════════════
# SIMULATED FRAME HELPERS
# ════════════════════════════════════════════════════════════════════════════
def _make_simulated_frame(cam_id, cam_label, is_crash, conf, frame_idx, w=640, h=400):
    bg = np.full((h, w, 3), (18, 5, 5) if is_crash else (15, 23, 42), dtype=np.uint8)
    road_x1, road_x2 = w // 3, 2 * w // 3
    bg[:, road_x1:road_x2] = (30, 41, 59)
    dash_h, dash_gap = 30, 20
    scroll = (frame_idx * 4) % (dash_h + dash_gap)
    for y in range(-dash_h, h + dash_gap, dash_h + dash_gap):
        y0, y1 = y + scroll, min(y + scroll + dash_h, h)
        if y1 > 0 and y0 < h:
            bg[max(y0, 0):y1, w // 2 - 1:w // 2 + 1] = (180, 140, 30)
    seed = hash(cam_id) % 100
    for i, (car_x, speed, color, car_h, car_w) in enumerate([
        (road_x1+15, 3.2, (70,130,200), 40, 22),
        (road_x1+90, 2.5, (80,190,100), 38, 20),
        (road_x1+50, 4.0, (200,80,80),  42, 24),
        (road_x1+130,1.8, (190,160,60), 36, 18),
    ]):
        phase = (frame_idx * speed + seed * 17 + i * 55) % (h + car_h)
        car_y = int(h - phase - car_h)
        y0, y1 = max(car_y, 0), min(car_y + car_h, h)
        x0, x1 = max(car_x, 0), min(car_x + car_w, w)
        if y1 > y0 and x1 > x0:
            bg[y0:y1, x0:x1] = color
    if is_crash:
        fa = 0.18 + 0.12 * abs(np.sin(frame_idx * 0.25))
        ov = bg.copy()
        ov[:, road_x1:road_x2] = np.clip(ov[:, road_x1:road_x2].astype(np.float32) + np.array([0,0,80])*fa, 0, 255).astype(np.uint8)
        bg = cv2.addWeighted(bg, 1-fa*.5, ov, fa*.5, 0)
        cx, cy = w//2, h//2
        if int(frame_idx*0.3)%2==0:
            cv2.line(bg,(cx-18,cy-18),(cx+18,cy+18),(0,0,220),3)
            cv2.line(bg,(cx+18,cy-18),(cx-18,cy+18),(0,0,220),3)
            cv2.circle(bg,(cx,cy),24,(0,0,200),2)
    hud = (50,50,220) if is_crash else (40,163,22)
    cv2.rectangle(bg,(0,0),(w,34),(10,10,18),-1)
    cv2.putText(bg,"!! CRASH DETECTED" if is_crash else "Normal",(10,23),cv2.FONT_HERSHEY_SIMPLEX,.62,hud,2,cv2.LINE_AA)
    cv2.putText(bg,f"Conf:{conf:.0%}",(w-115,23),cv2.FONT_HERSHEY_SIMPLEX,.55,hud,1,cv2.LINE_AA)
    cv2.rectangle(bg,(0,h-28),(w,h),(10,10,18),-1)
    cv2.putText(bg,f"{cam_id}  |  {cam_label[:28]}  |  {datetime.now().strftime('%H:%M:%S')}",(8,h-9),cv2.FONT_HERSHEY_SIMPLEX,.38,(180,180,180),1,cv2.LINE_AA)
    if int(frame_idx*.5)%2==0:
        cv2.circle(bg,(w-18,17),6,(0,0,220),-1)
        cv2.putText(bg,"REC",(w-60,23),cv2.FONT_HERSHEY_SIMPLEX,.38,(0,0,220),1,cv2.LINE_AA)
    cv2.rectangle(bg,(0,0),(w-1,h-1),(50,50,220) if is_crash else (50,120,40),3)
    return cv2.cvtColor(bg, cv2.COLOR_BGR2RGB)


def _get_real_frame(video_path, frame_idx, is_crash, conf, cam_id, cam_label):
    try:
        cap_v = cv2.VideoCapture(video_path)
        total_f = int(cap_v.get(cv2.CAP_PROP_FRAME_COUNT)) or 1
        cap_v.set(cv2.CAP_PROP_POS_FRAMES, frame_idx % total_f)
        ret, frame = cap_v.read()
        cap_v.release()
        if not ret:
            return None
        h, w = frame.shape[:2]
        hud = (50,50,220) if is_crash else (40,163,22)
        cv2.rectangle(frame,(0,0),(w,36),(10,10,18),-1)
        cv2.putText(frame,f"{'!! CRASH' if is_crash else 'Normal'}  Conf:{conf:.0%}",(10,25),cv2.FONT_HERSHEY_SIMPLEX,.65,hud,2,cv2.LINE_AA)
        cv2.rectangle(frame,(0,h-28),(w,h),(10,10,18),-1)
        cv2.putText(frame,f"{cam_id}  |  {datetime.now().strftime('%H:%M:%S')}",(8,h-9),cv2.FONT_HERSHEY_SIMPLEX,.4,(180,180,180),1,cv2.LINE_AA)
        if (frame_idx//15)%2==0:
            cv2.circle(frame,(w-18,18),6,(0,0,220),-1)
        if is_crash:
            alpha = .12+.08*abs(np.sin(frame_idx*.25))
            red = np.zeros_like(frame); red[:,:,2]=255
            frame = cv2.addWeighted(frame,1-alpha,red,alpha,0)
        cv2.rectangle(frame,(0,0),(w-1,h-1),(50,50,220) if is_crash else (30,120,30),3)
        return cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    except Exception:
        return None

# ════════════════════════════════════════════════════════════════════════════
# LIVE CCTV FEEDS
# ════════════════════════════════════════════════════════════════════════════
def page_feeds():
    st.markdown("""
    <div class="page-header">
      <div><div class="page-title">Live CCTV Feeds</div>
      <div class="page-sub">6 active cameras — real-time crash monitoring across Delhi NCR</div></div>
      <div class="live-pill"><div class="live-dot"></div>Live</div>
    </div>""", unsafe_allow_html=True)

    stats = db_stats()
    c1,c2,c3,c4 = st.columns(4)
    c1.markdown(f'<div class="stat-card success"><div class="stat-label">Cameras Online</div><div class="stat-value">6</div></div>',unsafe_allow_html=True)
    c2.markdown(f'<div class="stat-card danger"><div class="stat-label">Crashes Today</div><div class="stat-value">{stats["crashes_today"]}</div><div class="stat-sub">Saved to DB</div></div>',unsafe_allow_html=True)
    c3.markdown(f'<div class="stat-card info"><div class="stat-label">Total DB Records</div><div class="stat-value">{stats["total_crashes"]}</div><div class="stat-sub">crash_events table</div></div>',unsafe_allow_html=True)
    c4.markdown(f'<div class="stat-card"><div class="stat-label">Threshold</div><div class="stat-value">0.50</div></div>',unsafe_allow_html=True)

    crash_videos  = get_video_files(CRASH_VIDEO_DIR)
    normal_videos = get_video_files(NORMAL_VIDEO_DIR)
    def pick(lst, idx=0): return lst[idx % len(lst)] if lst else None

    cameras = [
        {"id":"CAM-01","location":"Indirapuram","label":"CAM-01 — Indirapuram","type":"normal","path":pick(normal_videos,0)},
        {"id":"CAM-02","location":"Greater Noida","label":"CAM-02 — Greater Noida","type":"normal","path":pick(normal_videos,1)},
        {"id":"CAM-03","location":"Noida","label":"CAM-03 — Noida","type":"crash","path":pick(crash_videos,0)},
        {"id":"CAM-04","location":"Anand Vihar","label":"CAM-04 — Anand Vihar","type":"normal","path":pick(normal_videos,2)},
        {"id":"CAM-05","location":"Vasant Vihar","label":"CAM-05 — Vasant Vihar","type":"normal","path":pick(normal_videos,3)},
        {"id":"CAM-06","location":"Vaishali","label":"CAM-06 — Vaishali","type":"normal","path":pick(normal_videos,4)},
    ]

    if "feed_frame_idx" not in st.session_state:
        st.session_state.feed_frame_idx = 0
    fidx  = st.session_state.feed_frame_idx
    now_t = time.time()
    NORMAL_CONFS = {"CAM-01":0.08,"CAM-02":0.11,"CAM-04":0.09,"CAM-05":0.12,"CAM-06":0.10}

    def sim_conf(cam):
        if cam["type"]=="crash":
            base = 0.62+0.22*abs(np.sin(now_t*0.5+3))
            return round(min(base+random.uniform(-0.03,0.03),0.99),2)
        return NORMAL_CONFS.get(cam["id"],0.10)

    crash_confs = {cam["id"]: sim_conf(cam) for cam in cameras}

    # Auto-log crash events → SQLite
    for cam in cameras:
        if cam["type"] != "crash":
            continue
        conf = crash_confs[cam["id"]]
        if conf < CONFIDENCE_THRESHOLD:
            continue
        last_key = f"last_log_{cam['id']}"
        if now_t - st.session_state.get(last_key, 0) < 10:
            continue
        st.session_state[last_key] = now_t
        crash_ts  = datetime.now()
        crash_hms = crash_ts.strftime("%H:%M:%S")
        snap_path = os.path.join(SNAP_DIR, f"{cam['id']}_{crash_ts.strftime('%Y%m%d_%H%M%S')}.jpg")
        clip_path = ""
        if cam["path"] and os.path.exists(cam["path"]):
            cap_tmp = cv2.VideoCapture(cam["path"])
            total_f_v = int(cap_tmp.get(cv2.CAP_PROP_FRAME_COUNT)) or 1
            fps_v     = cap_tmp.get(cv2.CAP_PROP_FPS) or 25
            cap_tmp.release()
            crash_sec = ((fidx * 8) % total_f_v) / fps_v
            clip_path = extract_crash_clip(cam["path"], crash_sec)
        if not clip_path:
            clip_path = os.path.join(CLIP_DIR, f"clip_{cam['id']}_{crash_ts.strftime('%H%M%S')}.mp4")

        # ── Write to SQLite ───────────────────────────────────────────────────
        db_insert_crash(
            username=st.session_state.username,
            camera=cam["id"],
            location=cam["location"],
            confidence=conf,
            snapshot=snap_path,
            clip_path=clip_path,
            crash_time=crash_hms
        )
        st.session_state.crash_count += 1
        st.session_state.save_count  += 1
        st.session_state.alert_logs.insert(0, {
            "cam":cam["id"],"location":cam["location"],"label":cam["label"],
            "conf":conf,"time":crash_ts.strftime("%Y-%m-%d %H:%M:%S"),
            "crash_time":crash_hms,"snap":snap_path,"clip":clip_path,
            "log_type":"live_cctv","type":"crash",
        })

    for cam in cameras:
        conf = crash_confs[cam["id"]]
        if conf < CONFIDENCE_THRESHOLD: continue
        st.markdown(f"""
        <div class="crash-banner">🚨 &nbsp; CRASH DETECTED — <b>{cam['id']} · {cam['location']}</b>
          &nbsp;|&nbsp; Confidence: <b>{conf:.0%}</b>
          &nbsp;|&nbsp; Crash time: <b>{datetime.now().strftime('%H:%M:%S')}</b>
          &nbsp;|&nbsp; <span class="db-badge">🗄️ Saved to SQLite</span>
        </div>""", unsafe_allow_html=True)

    st.markdown('<div class="section-title" style="margin-top:.5rem;">Camera Grid — Delhi NCR</div>',unsafe_allow_html=True)

    for row_cams in [cameras[:3], cameras[3:]]:
        cols = st.columns(3, gap="small")
        for col, cam in zip(cols, row_cams):
            conf     = crash_confs[cam["id"]]
            is_crash = conf >= CONFIDENCE_THRESHOLD
            bbg  = "#FEF2F2" if is_crash else "#F0FDF4"
            bcol = "#A32D2D" if is_crash else "#166534"
            bbdr = "#FECACA" if is_crash else "#BBF7D0"
            btxt = "⚠ CRASH" if is_crash else "● NORMAL"
            with col:
                frame_rgb = None
                if cam["path"] and os.path.exists(cam["path"]):
                    frame_rgb = _get_real_frame(cam["path"],fidx*8,is_crash,conf,cam["id"],cam["location"])
                if frame_rgb is None:
                    frame_rgb = _make_simulated_frame(cam["id"],cam["location"],is_crash,conf,fidx)
                border = "2px solid #E24B4A" if is_crash else "2px solid #D6E4F0"
                shadow = "0 0 0 3px rgba(226,75,74,.18)" if is_crash else "none"
                scale  = "transform:scale(1.035);z-index:5;position:relative;" if is_crash else ""
                st.markdown(f'<div style="border-radius:10px;overflow:hidden;border:{border};box-shadow:{shadow};{scale}">',unsafe_allow_html=True)
                st.image(frame_rgb, use_column_width=True)
                st.markdown("</div>",unsafe_allow_html=True)
                st.markdown(f"""
                <div style="display:flex;align-items:center;justify-content:space-between;padding:5px 4px 10px;">
                  <span style="font-size:11px;font-weight:600;color:#374151;">{cam['id']} · {cam['location']}</span>
                  <span style="font-size:10px;font-weight:700;background:{bbg};color:{bcol};border:1px solid {bbdr};padding:2px 8px;border-radius:5px;">{btxt} {conf:.0%}</span>
                </div>""",unsafe_allow_html=True)

    st.markdown("""
    <div style="text-align:center;font-size:11px;color:#9BA3AF;margin-top:.25rem;padding-bottom:.5rem;">
      Auto-refresh every 2 s &nbsp;·&nbsp; Crash events → <code>crash_detection.db</code> (SQLite)
    </div>""",unsafe_allow_html=True)

    st.session_state.feed_frame_idx = fidx + 1
    time.sleep(2)
    st.rerun()

# ════════════════════════════════════════════════════════════════════════════
# ALERT LOGS
# ════════════════════════════════════════════════════════════════════════════
def page_logs():
    role = st.session_state.role
    user = st.session_state.username
    st.markdown("""
    <div class="page-header">
      <div><div class="page-title">📋 Alert Logs</div>
      <div class="page-sub">Live CCTV crash events &amp; citizen evidence submissions — from SQLite</div></div>
    </div>""",unsafe_allow_html=True)

    crash_rows    = db_get_crashes(username=user, role=role)
    evidence_rows = db_get_evidence(submitted_by=user, role=role)

    c1,c2,c3 = st.columns(3)
    c1.markdown(f'<div class="stat-card danger"><div class="stat-label">CCTV Crash Events</div><div class="stat-value">{len(crash_rows)}</div><div class="stat-sub">crash_events table</div></div>',unsafe_allow_html=True)
    c2.markdown(f'<div class="stat-card info"><div class="stat-label">Evidence Submissions</div><div class="stat-value">{len(evidence_rows)}</div><div class="stat-sub">evidence_submissions table</div></div>',unsafe_allow_html=True)
    c3.markdown(f'<div class="stat-card"><div class="stat-label">DB File</div><div class="stat-value" style="font-size:1rem;margin-top:4px;"><code>crash_detection.db</code></div></div>',unsafe_allow_html=True)

    tab1, tab2 = st.tabs(["📺  Live CCTV Feed Alerts", "📁  Evidence Submissions"])

    with tab1:
        st.markdown("""<div style="background:#EBF4FF;border:1px solid #BFDBFE;border-left:4px solid #185FA5;border-radius:8px;padding:.75rem 1rem;font-size:.85rem;color:#0C447C;margin-bottom:1rem;">
          🗄️ <b>Pulled from SQLite</b> — <code>crash_events</code> table in <code>crash_detection.db</code>
        </div>""",unsafe_allow_html=True)
        if not crash_rows:
            st.markdown('<div style="text-align:center;padding:2rem;color:#9BA3AF;">📭 No live CCTV events yet.</div>',unsafe_allow_html=True)
        else:
            for row in crash_rows[:50]:
                with st.expander(f"🔴  {row['timestamp']}  —  {row['camera']} · {row['location']}  —  {row['confidence']:.0%}"):
                    c1,c2 = st.columns([1.2,1])
                    with c1:
                        st.markdown(f"**Camera:** `{row['camera']}`")
                        st.markdown(f"**Location:** {row['location']}")
                        st.markdown(f"**Crash at:** `{row['crash_time']}`")
                        st.markdown(f"**Confidence:** `{row['confidence']:.2%}`")
                        st.markdown(f"**DB Row ID:** `#{row['id']}`")
                        st.markdown(f'<div class="folder-tag">📸 {row["snapshot"]}</div>',unsafe_allow_html=True)
                        if row.get("clip_path"):
                            st.markdown(f'<div style="font-size:10px;font-family:monospace;background:#FFF7ED;color:#7C2D12;border:1px solid #FED7AA;padding:2px 7px;border-radius:5px;margin-top:4px;display:inline-block;">🎬 {row["clip_path"]}</div>',unsafe_allow_html=True)
                    with c2:
                        st.markdown(f"""<div style="background:#FEF2F2;border:1px solid #FECACA;border-radius:8px;padding:1rem;color:#A32D2D;font-size:.83rem;line-height:1.6;">
                          🚨 <b>CRASH DETECTED</b><br>Camera: <b>{row["camera"]}</b><br>
                          Location: <b>{row["location"]}</b><br>Time: <code>{row["crash_time"]}</code><br>
                          DB: <code>crash_events #{row["id"]}</code>
                        </div>""",unsafe_allow_html=True)

    with tab2:
        st.markdown("""<div style="background:#F0FDF4;border:1px solid #BBF7D0;border-left:4px solid #16A34A;border-radius:8px;padding:.75rem 1rem;font-size:.85rem;color:#14532D;margin-bottom:1rem;">
          🗄️ <b>Pulled from SQLite</b> — <code>evidence_submissions</code> table
        </div>""",unsafe_allow_html=True)
        if not evidence_rows:
            st.markdown('<div style="text-align:center;padding:2rem;color:#9BA3AF;">📭 No evidence submitted yet.</div>',unsafe_allow_html=True)
        else:
            for row in evidence_rows[:50]:
                with st.expander(f"📁  {row['timestamp']}  —  {row['filename']}  —  by {row['submitted_by']}"):
                    st.markdown(f"**Submitted by:** {row['submitted_by']}")
                    st.markdown(f"**Location:** {row['location']}")
                    st.markdown(f"**Description:** {row['description']}")
                    st.markdown(f"**Saved to:** `{row['file_path']}`")
                    st.markdown(f"**DB Row ID:** `#{row['id']}`")
                    if row["file_path"] and os.path.exists(row["file_path"]):
                        st.video(row["file_path"])

# ════════════════════════════════════════════════════════════════════════════
# SUBMIT EVIDENCE
# ════════════════════════════════════════════════════════════════════════════
def page_submit_evidence():
    st.markdown("""
    <div class="page-header">
      <div><div class="page-title">📤 Submit Evidence</div>
      <div class="page-sub">Upload a crash video — saved to SQLite + evidence/ folder</div></div>
    </div>""",unsafe_allow_html=True)

    st.markdown("""<div class="info-box"><strong>How to submit evidence:</strong><ol>
      <li>Record the accident video on your phone or dashcam</li>
      <li>Upload the video below (MP4, AVI, MOV)</li>
      <li>Add a brief description and your location</li>
      <li>Click <b>Submit</b> — saved to SQLite <code>evidence_submissions</code> table</li>
    </ol></div>""",unsafe_allow_html=True)

    location_input = st.text_input("📍 Location / Landmark", placeholder="e.g. Noida Sector 18 flyover")
    description    = st.text_area("📝 Description (optional)", placeholder="Briefly describe what happened...", height=80)
    uploaded       = st.file_uploader("🎥 Upload Video", type=["mp4","avi","mov","mkv"])

    if uploaded and st.button("📤 Submit Evidence", use_container_width=True):
        fname   = f"{st.session_state.username}_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{uploaded.name}"
        ev_path = os.path.join(EVIDENCE_DIR, fname)
        with open(ev_path, "wb") as f:
            f.write(uploaded.read())
        # ── Write to SQLite ───────────────────────────────────────────────────
        db_insert_evidence(
            submitted_by=st.session_state.username,
            filename=uploaded.name,
            location=location_input or "Not specified",
            description=description or "",
            file_path=ev_path
        )
        st.success(f"✅ Evidence submitted and saved to SQLite! File: `{fname}`")
        st.markdown(f"""<div style="background:#F0FDF4;border:1px solid #BBF7D0;border-left:4px solid #16A34A;border-radius:8px;padding:1rem 1.2rem;font-size:.88rem;color:#14532D;margin-top:.5rem;">
          🗄️ <b>Saved to SQLite:</b> <code>evidence_submissions</code> table in <code>crash_detection.db</code><br>
          📁 <b>File saved to:</b> <code>evidence/{fname}</code>
        </div>""",unsafe_allow_html=True)
    elif not uploaded:
        st.markdown("""<div style="background:#fff;border:2px dashed #D6E4F0;border-radius:12px;padding:2rem;text-align:center;color:#6B7280;margin-top:.5rem;">
          <div style="font-size:2rem;margin-bottom:8px;">🎥</div>
          <div style="font-size:.9rem;">Select a video file to upload</div>
        </div>""",unsafe_allow_html=True)

# ════════════════════════════════════════════════════════════════════════════
# DATABASE VIEWER  (admin only — new page)
# ════════════════════════════════════════════════════════════════════════════
def page_db_viewer():
    st.markdown("""
    <div class="page-header">
      <div><div class="page-title">🗄️ Database Viewer</div>
      <div class="page-sub">Browse all SQLite tables in crash_detection.db</div></div>
      <span class="db-badge">SQLite Active</span>
    </div>""",unsafe_allow_html=True)

    conn   = get_db()
    tables = [r[0] for r in conn.execute("SELECT name FROM sqlite_master WHERE type='table'").fetchall()]
    conn.close()

    stats = db_stats()
    cols = st.columns(4)
    for col, (lbl, val, cls) in zip(cols, [
        ("Total Crashes",   stats["total_crashes"],  "danger"),
        ("Evidence Files",  stats["total_evidence"], "info"),
        ("Active Users",    stats["active_users"],   ""),
        ("Crashes Today",   stats["crashes_today"],  "danger"),
    ]):
        col.markdown(f'<div class="stat-card {cls}"><div class="stat-label">{lbl}</div><div class="stat-value">{val}</div></div>',unsafe_allow_html=True)

    st.markdown("---")
    selected_table = st.selectbox("📋 Select Table", tables)

    if selected_table:
        conn  = get_db()
        rows  = conn.execute(f"SELECT * FROM {selected_table} ORDER BY rowid DESC LIMIT 100").fetchall()
        conn.close()

        if rows:
            import pandas as pd
            df = pd.DataFrame([dict(r) for r in rows])
            st.markdown(f'<div class="section-title">{selected_table} — {len(df)} rows (latest 100)</div>',unsafe_allow_html=True)
            st.dataframe(df, use_container_width=True)

            # Download as CSV
            csv_str = df.to_csv(index=False)
            st.download_button(
                f"⬇️ Download {selected_table}.csv",
                csv_str,
                f"{selected_table}.csv",
                "text/csv"
            )
        else:
            st.info(f"Table `{selected_table}` is empty.")

    # Raw SQL query box
    st.markdown("---")
    st.markdown('<div class="section-title">Run Raw SQL Query (SELECT only)</div>',unsafe_allow_html=True)
    query = st.text_area("SQL", value="SELECT * FROM crash_events ORDER BY id DESC LIMIT 20", height=80)
    if st.button("▶️ Run Query"):
        if query.strip().upper().startswith("SELECT"):
            try:
                conn = get_db()
                rows = conn.execute(query).fetchall()
                conn.close()
                import pandas as pd
                if rows:
                    st.dataframe(pd.DataFrame([dict(r) for r in rows]),use_container_width=True)
                else:
                    st.info("Query returned no rows.")
            except Exception as e:
                st.error(f"SQL Error: {e}")
        else:
            st.warning("Only SELECT queries are allowed here.")

# ════════════════════════════════════════════════════════════════════════════
# EVIDENCE VIEWER
# ════════════════════════════════════════════════════════════════════════════
def page_evidence():
    st.markdown("""
    <div class="page-header">
      <div><div class="page-title">📁 Evidence Submissions</div>
      <div class="page-sub">Videos submitted by citizens &amp; field officers — from SQLite</div></div>
    </div>""",unsafe_allow_html=True)

    evidence_rows = db_get_evidence(role="admin")
    folder_files  = [f for f in os.listdir(EVIDENCE_DIR) if f.lower().endswith((".mp4",".avi",".mov",".mkv"))] if os.path.isdir(EVIDENCE_DIR) else []

    c1,c2 = st.columns(2)
    c1.markdown(f'<div class="stat-card info"><div class="stat-label">DB Records</div><div class="stat-value">{len(evidence_rows)}</div><div class="stat-sub">evidence_submissions</div></div>',unsafe_allow_html=True)
    c2.markdown(f'<div class="stat-card"><div class="stat-label">Files on Disk</div><div class="stat-value">{len(folder_files)}</div><div class="stat-sub">evidence/</div></div>',unsafe_allow_html=True)

    if not evidence_rows and not folder_files:
        st.markdown('<div style="text-align:center;padding:3rem;color:#9BA3AF;">📭 No evidence submitted yet.</div>',unsafe_allow_html=True)
        return

    for row in evidence_rows:
        with st.expander(f"📁  {row['timestamp']}  —  {row['filename']}  —  by {row['submitted_by']}"):
            st.markdown(f"**Submitted by:** {row['submitted_by']}")
            st.markdown(f"**Location:** {row['location']}")
            st.markdown(f"**Saved to:** `{row['file_path']}`")
            st.markdown(f"**DB Row:** `#{row['id']}`")
            if row["file_path"] and os.path.exists(row["file_path"]):
                st.video(row["file_path"])

# ════════════════════════════════════════════════════════════════════════════
# LIVE IP FEED
# ════════════════════════════════════════════════════════════════════════════
def page_live():
    st.markdown("""
    <div class="page-header">
      <div><div class="page-title">📡 Live IP Feed</div>
      <div class="page-sub">Connect your phone/IP camera for real-time crash detection</div></div>
    </div>""",unsafe_allow_html=True)

    st.markdown("""<div class="info-box"><strong>How to connect your phone:</strong><ol>
      <li>Install <b>IP Webcam</b> (Android) or <b>EpocCam / DroidCam</b> (iOS)</li>
      <li>Connect to the same WiFi and start the server</li>
      <li>Paste the URL below and press Start</li>
    </ol></div>""",unsafe_allow_html=True)

    model  = load_model()
    ip_url = st.text_input("📶 Stream URL", placeholder="http://192.168.1.5:8080/video")
    c1,c2  = st.columns(2)
    start  = c1.button("▶️ Start Detection", use_container_width=True)
    stop   = c2.button("⏹ Stop", use_container_width=True)

    if not start or not ip_url.strip():
        st.info("💡 Enter a valid IP Webcam URL and press Start.")
        return

    cap = cv2.VideoCapture(ip_url.strip())
    if not cap.isOpened():
        st.error(f"❌ Cannot connect to `{ip_url.strip()}`.")
        return

    scores    = deque(maxlen=20)
    frame_ph  = st.empty()
    status_ph = st.empty()
    last_snap = 0

    while not stop:
        ret, frame = cap.read()
        if not ret: break
        prob = detect_frame(model, frame) if model else random.uniform(0.02, 0.15)
        scores.append(prob)
        avg      = float(np.mean(scores))
        is_crash = avg >= CONFIDENCE_THRESHOLD
        color    = (0,0,210) if is_crash else (0,180,60)
        cv2.putText(frame, f"{'CRASH' if is_crash else 'Normal'}  {avg:.0%}", (16,42), cv2.FONT_HERSHEY_DUPLEX,1.0,color,2)
        cv2.rectangle(frame,(0,0),(frame.shape[1]-1,frame.shape[0]-1),color,3)
        frame_ph.image(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB), use_column_width=True)
        if is_crash:
            status_ph.markdown(f'<div class="crash-banner">🚨 LIVE CRASH — {avg:.0%} — Saving to SQLite...</div>',unsafe_allow_html=True)
            if time.time()-last_snap > 5:
                snap_path = save_snapshot(frame, prefix="live_crash", folder=SNAP_DIR)
                db_insert_crash(st.session_state.username, f"LIVE:{ip_url.strip()}", "Live IP", avg,
                                snapshot=snap_path, crash_time=datetime.now().strftime("%H:%M:%S"))
                last_snap = time.time()
        else:
            status_ph.markdown(f'<div class="safe-banner">✅ Normal — {avg:.0%}</div>',unsafe_allow_html=True)
        time.sleep(0.03)
    cap.release()

# ════════════════════════════════════════════════════════════════════════════
# DASHBOARD
# ════════════════════════════════════════════════════════════════════════════
def page_dashboard():
    st.markdown('<div class="page-header"><div class="page-title">📊 Dashboard</div></div>',unsafe_allow_html=True)
    stats = db_stats()
    c1,c2,c3,c4 = st.columns(4)
    for col,(lbl,val,cls) in zip([c1,c2,c3,c4],[
        ("Total Crash Events", stats["total_crashes"],  "danger"),
        ("Evidence Submitted", stats["total_evidence"], "info"),
        ("Active Users",       stats["active_users"],   ""),
        ("Crashes Today",      stats["crashes_today"],  "danger"),
    ]):
        col.markdown(f'<div class="stat-card {cls}"><div class="stat-label">{lbl}</div><div class="stat-value">{val}</div></div>',unsafe_allow_html=True)

    st.markdown("---")
    st.markdown('<div class="section-title">Recent Crash Events (from SQLite)</div>',unsafe_allow_html=True)
    rows = db_get_crashes(role="admin", limit=15)
    if rows:
        for row in rows:
            is_c = row["confidence"] >= CONFIDENCE_THRESHOLD
            st.markdown(f"""
            <div class="alert-item {'crash' if is_c else 'normal'}">
              <div class="alert-dot {'red' if is_c else 'green'}"></div>
              <div class="alert-meta">
                <div class="alert-cam">{row['camera']} · {row['location']}</div>
                <div class="alert-time">{row['timestamp']} · {row['username']}</div>
                <div class="alert-conf {'danger' if is_c else 'safe'}">{row['confidence']:.2%}</div>
              </div>
            </div>""",unsafe_allow_html=True)
    else:
        st.info("No events recorded yet.")

# ════════════════════════════════════════════════════════════════════════════
# MANAGE USERS
# ════════════════════════════════════════════════════════════════════════════
def page_users():
    st.markdown('<div class="page-header"><div class="page-title">👥 Manage Users</div></div>',unsafe_allow_html=True)
    conn  = get_db()
    users = conn.execute("SELECT * FROM users").fetchall()
    conn.close()
    st.markdown('<div class="panel-card">',unsafe_allow_html=True)
    st.markdown('<div class="section-title">Registered Accounts (from SQLite users table)</div>',unsafe_allow_html=True)
    for u in users:
        badge_cls = "badge-admin" if u["role"]=="admin" else "badge-citizen"
        st.markdown(f"""
        <div style="display:flex;align-items:center;justify-content:space-between;padding:10px 0;border-bottom:1px solid #EEF0F3;">
          <div><span style="font-weight:600;font-size:.9rem;">{u['name']}</span>
          <code style="font-size:.78rem;color:#6B7280;margin-left:8px;">{u['username']}</code></div>
          <span class="badge {badge_cls}">{u['role'].upper()}</span>
        </div>""",unsafe_allow_html=True)
    st.markdown("</div>",unsafe_allow_html=True)
    st.info("💡 Users are stored in the `users` table in `crash_detection.db`.")

# ════════════════════════════════════════════════════════════════════════════
# ROUTER
# ════════════════════════════════════════════════════════════════════════════
def main():
    if not st.session_state.logged_in:
        login_page()
        return
    render_sidebar()
    page = st.session_state.get("page", "feeds" if st.session_state.role=="admin" else "logs")
    if   page=="feeds"          and st.session_state.role=="admin": page_feeds()
    elif page=="logs":                                                   page_logs()
    elif page=="live"           and st.session_state.role=="admin": page_live()
    elif page=="evidence"       and st.session_state.role=="admin": page_evidence()
    elif page=="submit_evidence":                                        page_submit_evidence()
    elif page=="dashboard"      and st.session_state.role=="admin": page_dashboard()
    elif page=="users"          and st.session_state.role=="admin": page_users()
    elif page=="db_viewer"      and st.session_state.role=="admin": page_db_viewer()
    else: st.warning("Page not found or access denied.")

main()
