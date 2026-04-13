from __future__ import annotations

import inspect
import os
import re
import sys
from pathlib import Path

sys.path.insert(0, os.path.dirname(__file__))

import numpy as np
import pandas as pd
import streamlit as st
from PIL import Image

from config import DEFAULT_USER_ID
from modules.A_user_access.user_verification import (
    enrollment_status,
    enroll_face,
    verify_face,
    enroll_voice,
    verify_voice,
    enroll_password,
    verify_profile_password,
)
from modules.A_user_access.wake_word import is_wake_word
from modules.A_user_access.text_input_handler import TextInputHandler
from modules.B_voice_processing.audio_capture import AudioCapture
from modules.B_voice_processing.speech_to_text import SpeechToText
from modules.C_nlu.nlu_pipeline import understand
from modules.D_control.command_router import route_command
from modules.D_control.state_manager import get_state_manager

st.set_page_config(page_title="AutoML Workspace Assistant", page_icon="🤖", layout="wide")

_CONTAINER_KW = {"border": True} if "border" in inspect.signature(st.container).parameters else {}

# ── Premium CSS ──────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');

/* Global font */
html, body, [class*="css"] {
    font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
}

/* Header area */
header[data-testid="stHeader"] {
    background: linear-gradient(135deg, #0f0c29 0%, #1a1a3e 50%, #24243e 100%) !important;
}

/* Main background */
.stApp {
    background: linear-gradient(160deg, #0d1117 0%, #161b22 40%, #0d1117 100%);
}

/* Title styling */
h1 {
    background: linear-gradient(120deg, #58a6ff, #bc8cff, #f778ba);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    background-clip: text;
    font-weight: 700 !important;
    letter-spacing: -0.5px;
}

/* Subheaders */
h2, h3 {
    color: #c9d1d9 !important;
    font-weight: 600 !important;
}

/* Container styling */
div[data-testid="stVerticalBlock"] > div[data-testid="stVerticalBlockBorderWrapper"] {
    background: rgba(22, 27, 34, 0.85) !important;
    border: 1px solid rgba(48, 54, 61, 0.7) !important;
    border-radius: 16px !important;
    backdrop-filter: blur(12px);
    box-shadow: 0 4px 24px rgba(0,0,0,0.15);
    transition: border-color 0.2s ease;
}

div[data-testid="stVerticalBlock"] > div[data-testid="stVerticalBlockBorderWrapper"]:hover {
    border-color: rgba(88, 166, 255, 0.3) !important;
}

/* Button styling */
.stButton > button {
    background: linear-gradient(135deg, #238636, #2ea043) !important;
    color: white !important;
    border: none !important;
    border-radius: 10px !important;
    padding: 0.45rem 1.2rem !important;
    font-weight: 500 !important;
    transition: all 0.2s ease !important;
    box-shadow: 0 2px 8px rgba(35, 134, 54, 0.3);
}

.stButton > button:hover {
    transform: translateY(-1px);
    box-shadow: 0 4px 16px rgba(35, 134, 54, 0.5) !important;
}

.stButton > button[kind="primary"] {
    background: linear-gradient(135deg, #1f6feb, #388bfd) !important;
    box-shadow: 0 2px 8px rgba(31, 111, 235, 0.3);
}

.stButton > button[kind="primary"]:hover {
    box-shadow: 0 4px 16px rgba(31, 111, 235, 0.5) !important;
}

/* Tab styling */
.stTabs [data-baseweb="tab-list"] {
    gap: 4px;
    background: rgba(13, 17, 23, 0.6);
    border-radius: 12px;
    padding: 4px;
}

.stTabs [data-baseweb="tab"] {
    border-radius: 8px !important;
    color: #8b949e !important;
    font-weight: 500 !important;
    padding: 8px 16px !important;
}

.stTabs [aria-selected="true"] {
    background: rgba(56, 139, 253, 0.15) !important;
    color: #58a6ff !important;
}

/* Chat message styling */
.stChatMessage {
    border-radius: 12px !important;
    background: rgba(22, 27, 34, 0.6) !important;
    border: 1px solid rgba(48, 54, 61, 0.5) !important;
    margin-bottom: 4px !important;
}

/* Input fields */
.stTextInput > div > div > input {
    background: rgba(13, 17, 23, 0.8) !important;
    border: 1px solid rgba(48, 54, 61, 0.7) !important;
    border-radius: 10px !important;
    color: #c9d1d9 !important;
    transition: border-color 0.2s ease;
}

.stTextInput > div > div > input:focus {
    border-color: #58a6ff !important;
    box-shadow: 0 0 0 3px rgba(88, 166, 255, 0.15) !important;
}

/* Dataframe styling */
.stDataFrame {
    border-radius: 12px !important;
    overflow: hidden;
}

/* Download buttons */
.stDownloadButton > button {
    background: linear-gradient(135deg, #6e40c9, #8957e5) !important;
    box-shadow: 0 2px 8px rgba(110, 64, 201, 0.3);
}

.stDownloadButton > button:hover {
    box-shadow: 0 4px 16px rgba(110, 64, 201, 0.5) !important;
}

/* Expander styling */
.streamlit-expanderHeader {
    background: rgba(22, 27, 34, 0.6) !important;
    border-radius: 10px !important;
    font-weight: 500 !important;
    color: #8b949e !important;
}

/* Code blocks */
.stCodeBlock {
    border-radius: 12px !important;
}

/* Metric / caption text */
.stCaption, caption {
    color: #6e7681 !important;
}

/* JSON viewer */
.stJson {
    border-radius: 12px !important;
}

/* Chat input */
.stChatInput {
    border-radius: 12px !important;
}

.stChatInput > div {
    border-radius: 12px !important;
    border-color: rgba(48, 54, 61, 0.7) !important;
    background: rgba(13, 17, 23, 0.8) !important;
}

/* Status badge */
.status-badge {
    display: inline-block;
    padding: 3px 10px;
    border-radius: 20px;
    font-size: 0.8rem;
    font-weight: 600;
    letter-spacing: 0.5px;
}
.status-idle { background: rgba(110, 118, 129, 0.2); color: #8b949e; }
.status-training { background: rgba(56, 139, 253, 0.2); color: #58a6ff; }
.status-completed { background: rgba(35, 134, 54, 0.2); color: #3fb950; }
.status-paused { background: rgba(210, 153, 34, 0.2); color: #d29922; }
.status-stopped { background: rgba(218, 54, 51, 0.2); color: #f85149; }

/* Verified badge */
.verified-badge {
    display: inline-block;
    padding: 2px 8px;
    border-radius: 12px;
    font-size: 0.75rem;
    font-weight: 600;
}
.badge-yes { background: rgba(35, 134, 54, 0.2); color: #3fb950; }
.badge-no { background: rgba(218, 54, 51, 0.15); color: #f85149; }
</style>
""", unsafe_allow_html=True)

sm = get_state_manager()
handler = TextInputHandler()

if "profile_id" not in st.session_state:
    st.session_state.profile_id = ""
if "verified" not in st.session_state:
    st.session_state.verified = False
if "chat_open" not in st.session_state:
    st.session_state.chat_open = True  # Open by default — no need to click
if "chat_history" not in st.session_state:
    st.session_state.chat_history = [
        {"role": "assistant", "content": "👋 Welcome to the AutoML Assistant! Sign in or create an account to get started."}
    ]


def log(msg: str):
    sm.append_log(msg)


def strip_wake_or_bypass(text: str) -> tuple[bool, str]:
    cleaned = text.strip()
    if is_wake_word(cleaned):
        stripped = re.sub(
            r"^(hey|hi|okay|wake up|hello)\s+mello[\s,:\-]*",
            "",
            cleaned,
            flags=re.IGNORECASE,
        ).strip()
        sm.set_wake_detected(True)
        return True, stripped
    sm.set_wake_detected(False)
    return False, cleaned


def process_command(user_text: str):
    st.session_state.chat_history.append({"role": "user", "content": user_text})

    ok, cleaned = strip_wake_or_bypass(user_text)
    nlu = understand(cleaned)
    result = route_command(nlu)

    sm.set_transcript(user_text)
    sm.set_assistant_response(result["message"])
    log(f"Chat command → {nlu['intent']}")

    assistant_text = result["message"]
    if not ok:
        assistant_text = f"⚡ {assistant_text}"

    st.session_state.chat_history.append({"role": "assistant", "content": assistant_text})


def _status_class(status: str) -> str:
    return {
        "idle": "status-idle",
        "training": "status-training",
        "completed": "status-completed",
        "paused": "status-paused",
        "stopped": "status-stopped",
    }.get(status, "status-idle")


def render_dataset_panel(state: dict):
    st.markdown("#### 📚 Dataset")
    col_a, col_b = st.columns([1, 1])
    with col_a:
        ds_name = state.get("dataset") or "—"
        st.markdown(f"**Current dataset:** `{ds_name}`")
        info = state.get("dataset_info", {})
        if info:
            cols_display = {
                "Rows": info.get("rows", "—"),
                "Columns": info.get("columns", "—"),
                "Target": info.get("target_name") or "—",
            }
            st.markdown(" · ".join(f"**{k}:** {v}" for k, v in cols_display.items()))
            profile = info.get("profile", {})
            if profile:
                st.caption(f"Task: {profile.get('task_family', '—')} · Model: {profile.get('suggested_model', '—')}")
        else:
            st.caption("No dataset loaded yet. Ask: *'hey mello load iris dataset'*")
    with col_b:
        preview = state.get("dataset_preview", [])
        if preview:
            st.dataframe(pd.DataFrame(preview), use_container_width=True, height=220)
        else:
            st.caption("Preview will appear here once a dataset is loaded.")


def render_code_panel(state: dict):
    st.markdown("#### 💻 Code")
    tab_py, tab_ipynb, tab_ref = st.tabs(["🐍 Runnable .py", "📓 Notebook .ipynb", "📘 Kaggle Reference"])

    with tab_py:
        code = state.get("generated_code_py", "")
        if code:
            st.code(code, language="python")
            st.download_button(
                "⬇️ Download .py",
                data=code.encode("utf-8"),
                file_name=f"{state.get('dataset') or 'experiment'}.py",
                mime="text/x-python",
                key="dl_py_code",
            )
        else:
            st.caption("No Python code generated yet. Ask: *'hey mello load corresponding code'*")

    with tab_ipynb:
        notebook = state.get("generated_code_ipynb", "")
        if notebook:
            st.code(notebook, language="json")
            st.download_button(
                "⬇️ Download .ipynb",
                data=notebook.encode("utf-8"),
                file_name=f"{state.get('dataset') or 'experiment'}.ipynb",
                mime="application/x-ipynb+json",
                key="dl_ipynb_code",
            )
        else:
            st.caption("No notebook generated yet.")

    with tab_ref:
        ref_code = state.get("reference_code", "")
        ref_fmt = state.get("reference_code_format", "text")
        title = state.get("code_title", "")
        if title:
            st.markdown(f"**Source:** {title}")
        if ref_code:
            st.code(ref_code[:30000], language="python" if ref_fmt == "py" else "json")
        else:
            st.caption("No Kaggle reference code loaded.")


def render_outputs_panel(state: dict):
    st.markdown("#### 🖨️ Output")
    outputs = state.get("outputs", [])

    status = state.get("training_status", "idle")
    model_name = state.get("model") or "—"
    epoch_cur = state.get("epoch_current", 0)
    epoch_tot = state.get("epochs_total", 0)

    status_cls = _status_class(status)
    st.markdown(
        f'<span class="status-badge {status_cls}">{status.upper()}</span> '
        f'&nbsp; Epoch **{epoch_cur}** / **{epoch_tot}** &nbsp;·&nbsp; Model: **{model_name}**',
        unsafe_allow_html=True
    )

    if state.get("loss_history") or state.get("accuracy_history"):
        chart_df = pd.DataFrame(
            {
                "Loss": state.get("loss_history", []),
                "Accuracy": state.get("accuracy_history", []),
            }
        )
        chart_df.index = range(1, len(chart_df) + 1)
        st.line_chart(chart_df, use_container_width=True)

    if not outputs:
        code_output = state.get("code_output_text", "")
        if code_output:
            st.text(code_output)
        else:
            st.caption("No output yet. Ask me to run code or start training.")
        return

    for idx, item in enumerate(outputs):
        kind = item.get("type")
        title = item.get("title", f"Output {idx + 1}")
        st.markdown(f"**{title}**")

        if kind == "text":
            st.text(item.get("content", ""))
        elif kind == "table":
            st.dataframe(pd.DataFrame(item.get("records", [])), use_container_width=True)
        elif kind == "image":
            path = item.get("path")
            if path and Path(path).exists():
                st.image(path, use_container_width=True)
        elif kind == "file":
            path = item.get("path")
            if path and Path(path).exists():
                with open(path, "rb") as f:
                    st.download_button(
                        f"⬇️ Download {Path(path).name}",
                        data=f.read(),
                        file_name=Path(path).name,
                        mime=item.get("mime", "application/octet-stream"),
                        key=f"download_{idx}_{Path(path).name}",
                    )


def render_auth_panel():
    st.markdown("#### 🔐 Create Account or Sign In")
    st.caption("Enter a username and password to get started.")

    auth_tab_signin, auth_tab_signup, auth_tab_bio = st.tabs(["🔑 Sign In", "📝 Sign Up", "🧬 Biometrics"])

    with auth_tab_signup:
        new_user = st.text_input("Choose a username", key="chat_signup_user", placeholder="e.g. john_doe")
        new_pw = st.text_input("New password", type="password", key="chat_signup_password")
        conf_pw = st.text_input("Confirm password", type="password", key="chat_signup_confirm")
        if st.button("Create Account", key="chat_signup_btn", type="primary", use_container_width=True):
            uid = new_user.strip() if new_user.strip() else DEFAULT_USER_ID
            if new_pw != conf_pw:
                st.error("Passwords do not match.")
            elif len(new_pw) < 4:
                st.error("Password must be at least 4 characters.")
            else:
                result = enroll_password(uid, new_pw)
                if result["success"]:
                    st.session_state.profile_id = uid
                    st.success(f"✅ Account created for **{uid}**. You can sign in now.")
                    st.session_state.chat_history.append(
                        {"role": "assistant", "content": f"Account created for **{uid}**. Sign in to start using the assistant."}
                    )
                else:
                    st.error(result["message"])

    with auth_tab_signin:
        login_user = st.text_input("Username", key="chat_signin_user", placeholder="e.g. john_doe")
        password = st.text_input("Password", type="password", key="chat_signin_password")
        if st.button("Sign In", key="chat_signin_btn", type="primary", use_container_width=True):
            uid = login_user.strip() if login_user.strip() else DEFAULT_USER_ID
            result = verify_profile_password(uid, password)
            if result["verified"]:
                st.session_state.profile_id = uid
                st.session_state.verified = True
                sm.set_verified(True)
                st.session_state.chat_history.append(
                    {"role": "assistant", "content": f"✅ Welcome back, **{uid}**! You can now use voice or text commands. Try: *'hey mello load iris dataset'*"}
                )
                st.rerun()  # Immediately show the chat input
            else:
                st.error(result["message"])

    with auth_tab_bio:
        bio_user = st.text_input("Username for biometrics", key="chat_bio_user",
                                  value=st.session_state.profile_id or DEFAULT_USER_ID)
        uid = bio_user.strip() if bio_user.strip() else DEFAULT_USER_ID
        est = enrollment_status(uid)
        st.caption(f"Face enrolled: {'✅' if est['face_enrolled'] else '❌'} · Voice enrolled: {'✅' if est['voice_enrolled'] else '❌'}")

        st.markdown("**Face Enrollment / Verification**")
        face_img = st.camera_input("Capture face", key="chat_face_camera")
        c1, c2 = st.columns(2)
        if c1.button("Enroll Face", key="chat_enroll_face") and face_img is not None:
            arr = np.array(Image.open(face_img).convert("RGB"))
            result = enroll_face(uid, arr)
            st.success(result["message"]) if result["success"] else st.error(result["message"])
        if c2.button("Verify Face", key="chat_verify_face") and face_img is not None:
            arr = np.array(Image.open(face_img).convert("RGB"))
            result = verify_face(uid, arr)
            if result["verified"]:
                st.session_state.profile_id = uid
                st.session_state.verified = True
                sm.set_verified(True)
                st.session_state.chat_history.append(
                    {"role": "assistant", "content": f"✅ Face verified. Welcome, **{uid}**!"}
                )
                st.rerun()
            else:
                st.error(result["message"])

        st.markdown("**Voice Enrollment / Verification**")
        duration = st.slider("Voice duration (seconds)", 3, 8, 4, key="chat_voice_seconds")
        c3, c4 = st.columns(2)
        if c3.button("Enroll Voice", key="chat_enroll_voice"):
            try:
                audio = AudioCapture().record_fixed(duration=duration)
                result = enroll_voice(uid, audio)
                st.success(result["message"]) if result["success"] else st.error(result["message"])
            except Exception as e:
                st.error(str(e))
        if c4.button("Verify Voice", key="chat_verify_voice"):
            try:
                audio = AudioCapture().record_fixed(duration=duration)
                result = verify_voice(uid, audio)
                if result["verified"]:
                    st.session_state.profile_id = uid
                    st.session_state.verified = True
                    sm.set_verified(True)
                    st.session_state.chat_history.append(
                        {"role": "assistant", "content": f"✅ Voice verified. Welcome, **{uid}**!"}
                    )
                    st.rerun()
                else:
                    st.error(result["message"])
            except Exception as e:
                st.error(str(e))


def render_chat_panel():
    with st.container(**_CONTAINER_KW):
        # Header
        st.markdown("### 💬 Assistant")
        uid = st.session_state.profile_id or "not signed in"
        verified = st.session_state.verified
        badge_cls = "badge-yes" if verified else "badge-no"
        badge_txt = "Verified ✓" if verified else "Not signed in"
        st.markdown(
            f'👤 **{uid}** &nbsp; <span class="verified-badge {badge_cls}">{badge_txt}</span>',
            unsafe_allow_html=True
        )

        if not st.session_state.verified:
            render_auth_panel()
            return

        # Example commands expander
        with st.expander("💡 Example commands", expanded=False):
            examples = [
                "hey mello load iris dataset",
                "hey mello load corresponding code",
                "hey mello set learning rate to 0.01",
                "hey mello set layers to 4",
                "hey mello start training",
                "hey mello run code",
                "hey mello search dataset fraud detection",
            ]
            for ex in examples:
                st.code(ex, language="text")

        # Chat history
        for msg in st.session_state.chat_history:
            with st.chat_message(msg["role"]):
                st.markdown(msg["content"])

        # Voice recording + clear
        voice_col, clear_col = st.columns([2, 1])
        if voice_col.button("🎤 Record voice command (4s)", key="chat_record_btn", use_container_width=True):
            try:
                audio = AudioCapture().record_fixed(duration=4)
                tmp_path = Path("artifacts") / "last_command.wav"
                tmp_path.parent.mkdir(parents=True, exist_ok=True)
                AudioCapture().save(audio, str(tmp_path))
                transcript = SpeechToText().transcribe(str(tmp_path))
                if transcript.strip():
                    process_command(transcript)
                else:
                    st.session_state.chat_history.append({"role": "assistant", "content": "🔇 I could not transcribe the recording. Please try again."})
            except Exception as e:
                st.session_state.chat_history.append({"role": "assistant", "content": f"⚠️ Voice command failed: {e}"})

        if clear_col.button("🗑️ Clear Chat", key="clear_chat_btn", use_container_width=True):
            st.session_state.chat_history = [{"role": "assistant", "content": "Chat cleared. Ready for the next command."}]
            st.rerun()

        # Text input
        prompt = st.chat_input("Type a command (e.g. 'hey mello load iris dataset')")
        if prompt:
            process_command(prompt)
            st.rerun()


# ── Page Layout ──────────────────────────────────────────────

st.markdown(
    "<h1 style='margin-bottom:0'>🤖 AutoML Workspace Assistant</h1>"
    "<p style='color:#6e7681;margin-top:4px;font-size:0.9rem'>Workspace on the left · Assistant on the right</p>",
    unsafe_allow_html=True,
)

left, right = st.columns([2.25, 1], gap="large")

state = sm.get_state()

with left:
    with st.container(**_CONTAINER_KW):
        render_dataset_panel(state)
    with st.container(**_CONTAINER_KW):
        render_code_panel(state)
    with st.container(**_CONTAINER_KW):
        render_outputs_panel(sm.get_state())

with right:
    render_chat_panel()