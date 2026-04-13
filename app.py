"""
app.py — Full Streamlit App (Modules 1–10)
=============================================
Extends the original app_AB_demo.py with Modules C + D.

All original A + B features are preserved EXACTLY:
  ① Verification Gate        — password, face, or voice enrolment + sign-in  (Module 1)
  ② Voice or Text Command    — record audio (Module 4) or type (Module 3)
  ③ Wake Word Check          — Module 2
NEW stages added:
  ④ NLU Processing           — Intent Detection (Module 6) + Slot Filling (Module 7)
  ⑤ Command Execution        — Command Router (8) + State Manager (9) + Experiment Controller (10)
  Right column: Experiment dashboard, metrics charts, event log

Run with:  streamlit run app.py
"""

import sys, os
sys.path.insert(0, os.path.dirname(__file__))

import inspect
import re
import shutil
import urllib.request

import streamlit as st
import numpy as np
import pandas as pd
from PIL import Image

from config import DEFAULT_USER_ID

# Bordered containers need Streamlit ≥1.29; omit kwarg on older versions.
_CONTAINER_KW = (
    {"border": True}
    if "border" in inspect.signature(st.container).parameters
    else {}
)

# ── Streamlit page config ───────────────────────────────────────────────
st.set_page_config(
    page_title="AutoML Voice Assistant — Full Pipeline",
    page_icon="🤖",
    layout="wide",
)

# ── Module imports (A + B — unchanged) ──────────────────────────────────
from modules.A_user_access.user_verification import (
    verify_passcode,
    reset_state as reset_verification,
    enrollment_status,
    enroll_face,
    verify_face,
    enroll_voice,
    verify_voice,
    enroll_password,
    verify_profile_password,
    clear_biometric_enrollment,
)
from modules.A_user_access.wake_word import is_wake_word
from modules.A_user_access.text_input_handler import TextInputHandler
from modules.B_voice_processing.audio_capture import AudioCapture
from modules.B_voice_processing.speech_to_text import (
    SpeechToText,
    get_last_transcribe_error,
    probe_whisper_model,
)

# ── Module imports (C + D — NEW) ────────────────────────────────────────
from modules.C_nlu.nlu_pipeline import understand
from modules.D_control.command_router import route_command
from modules.D_control.state_manager import get_state_manager

sm = get_state_manager()

# ── Session state initialisation ────────────────────────────────────────
if "verified" not in st.session_state:
    st.session_state.verified = False
if "awake" not in st.session_state:
    st.session_state.awake = False
if "transcript" not in st.session_state:
    st.session_state.transcript = ""
if "raw_command" not in st.session_state:
    st.session_state.raw_command = ""
if "clean_command" not in st.session_state:
    st.session_state.clean_command = ""
if "nlu_result" not in st.session_state:
    st.session_state.nlu_result = None
if "exec_result" not in st.session_state:
    st.session_state.exec_result = None
if "event_log" not in st.session_state:
    st.session_state.event_log = []
if "profile_id" not in st.session_state:
    st.session_state.profile_id = DEFAULT_USER_ID

handler = TextInputHandler()


def log(msg: str):
    """Append timestamped message to both Streamlit session log and State Manager."""
    from datetime import datetime
    st.session_state.event_log.append(
        f"[{datetime.now().strftime('%H:%M:%S')}] {msg}"
    )
    sm.append_log(msg)


# ═══════════════════════════════════════════════════════════════════════
# LAYOUT
# ═══════════════════════════════════════════════════════════════════════

st.title("🤖 AutoML Voice Assistant — Full Pipeline (Modules 1–10)")
st.caption("CSI5180 Project · Sections A + B + C + D · Complete End-to-End")

col_left, col_right = st.columns([1.6, 1], gap="large")

# ── LEFT COLUMN: Pipeline Stages ────────────────────────────────────────
with col_left:

    # ══════════════════════════════════════════════════════════════════
    # STAGE 1 — User Verification  (Module 1)
    # ══════════════════════════════════════════════════════════════════
    with st.container(**_CONTAINER_KW):
        st.subheader("① User Verification  (Module 1)")

        # ── Current status indicator ──
        status_icon = "🟢 Verified" if st.session_state.verified else "🔴 Not Verified"
        st.markdown(f"**Status:** {status_icon}")

        # ── Profile ID ──
        pid_in = st.text_input(
            "Profile ID (each profile has its own enrollment data)",
            value=st.session_state.profile_id,
            key="profile_id_input",
            help="Enrollment data is stored under data/enrollment/<profile_id>/",
        )
        st.session_state.profile_id = (pid_in.strip() or DEFAULT_USER_ID)

        # ── Enrollment status badges ──
        est = enrollment_status(st.session_state.profile_id)

        badge_f, badge_v, badge_p = st.columns(3)
        with badge_f:
            if est["face_enrolled"]:
                st.markdown("📸 **Face:** :green[Enrolled ✅]")
            else:
                st.markdown("📸 **Face:** :orange[Not Enrolled]")
        with badge_v:
            if est["voice_enrolled"]:
                st.markdown("🎤 **Voice:** :green[Enrolled ✅]")
            else:
                st.markdown("🎤 **Voice:** :orange[Not Enrolled]")
        with badge_p:
            if est["password_enrolled"]:
                st.markdown("🔑 **Password:** :green[Set ✅]")
            else:
                st.markdown("🔑 **Password:** :orange[Not Set (default 1234)]")

        # ── Main authentication flow ──
        if not st.session_state.verified:
            st.divider()

            # ════════════════════════════════════════════════════════
            # EXPLICIT CHOICE: Sign Up or Sign In
            # ════════════════════════════════════════════════════════
            action = st.radio(
                "What would you like to do?",
                ["🆕 Sign Up (First-Time Enrollment)", "🔓 Sign In (Verify & Login)"],
                key="action_radio",
                horizontal=True,
            )

            # ────────────────────────────────────────────────────────
            # SIGN UP — First-Time Enrollment
            # ────────────────────────────────────────────────────────
            if "Sign Up" in action:
                st.markdown("---")
                st.markdown("### 🆕 First-Time Enrollment")
                st.caption(
                    "Enroll your password, face, or voice. "
                    "All data is stored locally on this machine only."
                )

                enroll_what = st.radio(
                    "What to enroll:",
                    ["🔑 Password", "📸 Face", "🎤 Voice"],
                    key="enroll_what_radio",
                )

                # ── PASSWORD ENROLLMENT ──
                if "Password" in enroll_what:
                    st.markdown("#### 🔑 Set Your Profile Password")
                    if est["password_enrolled"]:
                        st.warning("⚠️ A password is already set for this profile. Setting a new one will overwrite it.")
                    else:
                        st.info("🆕 No password set yet — create one below.")
                    st.caption("Your password is hashed with SHA-256 and stored in `meta.json`. Never stored in plaintext.")
                    new_pw  = st.text_input("New password (min 4 characters)", type="password", key="new_pw")
                    conf_pw = st.text_input("Confirm password", type="password", key="conf_pw")
                    if st.button("✅ Save Password", key="btn_enroll_pw", type="primary"):
                        if new_pw != conf_pw:
                            st.error("❌ Passwords do not match.")
                        else:
                            er = enroll_password(st.session_state.profile_id, new_pw)
                            if er["success"]:
                                log(f"🔑 Password set — profile={st.session_state.profile_id}")
                                st.success(er["message"])
                                st.rerun()
                            else:
                                st.error(er["message"])

                # ── FACE ENROLLMENT ──
                elif "Face" in enroll_what:
                    st.markdown("#### 📸 Face Enrollment")
                    if est["face_enrolled"]:
                        st.warning(
                            "⚠️ Face is **already enrolled** for this profile. "
                            "Re-enrolling will **overwrite** your existing template."
                        )

                    with st.expander("ℹ️ How face enrollment works"):
                        st.markdown(
                            "1. **Capture** — Webcam takes a photo.  \n"
                            "2. **Detect & Align** — MTCNN neural network finds and aligns the face.  \n"
                            "3. **Encode** — FaceNet (InceptionResnetV1) creates a 512-dim identity embedding.  \n"
                            "4. **Store** — Embedding saved as `face_encoding.npy`.  \n"
                            "5. **Future logins** — New photo → new embedding → cosine similarity ≥ threshold."
                        )

                    face_img = st.camera_input(
                        "📷 Capture your face (clear, frontal, good lighting)",
                        key="face_cam_enroll",
                    )

                    if face_img is not None:
                        if st.button("✅ Enroll My Face", key="btn_do_enroll_face", type="primary"):
                            with st.spinner("Detecting face and saving template…"):
                                arr = np.array(Image.open(face_img).convert("RGB"))
                                er = enroll_face(st.session_state.profile_id, arr)
                            if er["success"]:
                                log(f"📸 Face enrolled — profile={st.session_state.profile_id}")
                                st.success(er["message"])
                                st.balloons()
                                st.rerun()
                            else:
                                st.error(er["message"])
                    else:
                        st.caption("👆 Take a photo above to begin enrollment.")

                # ── VOICE ENROLLMENT ──
                else:
                    st.markdown("#### 🎤 Voice Enrollment")
                    if est["voice_enrolled"]:
                        st.warning(
                            "⚠️ Voice is **already enrolled** for this profile. "
                            "Re-enrolling will **overwrite** your existing voiceprint."
                        )

                    with st.expander("ℹ️ How voice enrollment works — READ THIS"):
                        st.markdown(
                            "**What gets compared:** Speaker *identity* (pitch, timbre, resonance) — "
                            "**NOT** the words you say. You can speak any sentence.  \n\n"
                            "1. **Record** — Microphone captures your voice at 16 kHz.  \n"
                            "2. **Embed** — Resemblyzer creates a 256-dim speaker embedding.  \n"
                            "3. **Store** — Saved as `voice_embedding.npy`."
                        )

                    from config import VOICE_COSINE_THRESHOLD
                    st.caption(f"Current voice threshold: **{VOICE_COSINE_THRESHOLD}** (set in `config.py`)")
                    vdur = st.slider("Recording duration (seconds)", 3, 12, 7, key="voice_dur_enroll")
                    st.info('🎙️ Speak naturally — say anything. Longer clips give better results.')

                    if st.button("🎤 Record & Enroll My Voice", key="btn_do_enroll_voice", type="primary"):
                        try:
                            with st.spinner(f"🔴 Recording for {vdur}s — please speak now…"):
                                cap = AudioCapture()
                                audio = cap.record_fixed(duration=vdur)
                            peak = float(np.max(np.abs(audio)))
                            if peak < 0.01:
                                st.warning("⚠️ Recording is very quiet. Check your microphone.")
                            from modules.A_user_access.voice_biometrics import LAST_RECORDING_PATH, save_wav
                            save_wav(np.asarray(audio, dtype=np.float32).flatten(), LAST_RECORDING_PATH)
                            if os.path.isfile(LAST_RECORDING_PATH):
                                st.markdown("🔊 **Listen to your recording:**")
                                st.audio(LAST_RECORDING_PATH, format="audio/wav")
                            with st.spinner("Creating speaker embedding…"):
                                er = enroll_voice(st.session_state.profile_id, audio)
                            if er["success"]:
                                log(f"🎙️ Voice enrolled — profile={st.session_state.profile_id}")
                                st.success(er["message"])
                                st.balloons()
                                st.rerun()
                            else:
                                st.error(er["message"])
                        except Exception as e:
                            st.error(f"Microphone error: {e}")

            # ────────────────────────────────────────────────────────
            # SIGN IN — Verify & Login
            # ────────────────────────────────────────────────────────
            else:
                st.markdown("---")
                st.markdown("### 🔓 Sign In — Verify Your Identity")

                signin_method = st.radio(
                    "Verification method:",
                    ["🔑 Password", "📸 Face Recognition", "🎤 Voice Recognition"],
                    key="signin_method_radio",
                )

                # ── PASSWORD SIGN IN ──
                if "Password" in signin_method:
                    st.markdown("#### 🔑 Password Sign In")
                    if est["password_enrolled"]:
                        st.success("✅ Profile password set. Enter it below.")
                    else:
                        st.warning(
                            "⚠️ No profile password set yet — using **default passcode (1234)**. "
                            "Go to Sign Up → Password to set your own."
                        )
                    passcode = st.text_input("Password", type="password", key="passcode_input")
                    if st.button("🔓 Verify Password & Sign In", key="btn_verify_pw", type="primary"):
                        result = verify_profile_password(st.session_state.profile_id, passcode)
                        if result["verified"]:
                            st.session_state.verified = True
                            sm.set_verified(True)
                            log(f"✅ Verified (password) profile={st.session_state.profile_id}")
                            st.success(result["message"])
                            st.rerun()
                        else:
                            st.error(result["message"])

                # ── FACE SIGN IN ──
                elif "Face" in signin_method:
                    st.markdown("#### 📸 Face Recognition Sign In")
                    if not est["face_enrolled"]:
                        st.error(
                            "❌ **Face is not enrolled** for this profile!  \n"
                            "Please go to **Sign Up** first to enroll your face."
                        )
                    else:
                        st.success("✅ Face template found. Capture a new photo to verify.")
                        face_img = st.camera_input("📷 Verify your face", key="face_cam_verify")
                        if face_img is not None:
                            if st.button("🔓 Verify Face & Sign In", key="btn_do_verify_face", type="primary"):
                                with st.spinner("Comparing against stored template…"):
                                    arr = np.array(Image.open(face_img).convert("RGB"))
                                    result = verify_face(st.session_state.profile_id, arr)
                                if result["verified"]:
                                    st.session_state.verified = True
                                    sm.set_verified(True)
                                    log(f"✅ Verified (face) profile={st.session_state.profile_id}")
                                    st.success(result["message"])
                                    st.rerun()
                                else:
                                    st.error(result["message"])
                        else:
                            st.caption("👆 Take a photo above to verify your identity.")

                # ── VOICE SIGN IN ──
                elif "Voice" in signin_method:
                    st.markdown("#### 🎤 Voice Recognition Sign In")
                    if not est["voice_enrolled"]:
                        st.error(
                            "❌ **Voice is not enrolled** for this profile!  \n"
                            "Please go to **Sign Up** first to enroll your voice."
                        )
                    else:
                        from config import VOICE_COSINE_THRESHOLD
                        st.success("✅ Voiceprint found. Record any short phrase to verify.")
                        st.info(
                            "💡 **Note:** This matches your voice *characteristics* (who you are), "
                            "not the specific words you say. Say anything in your normal voice."
                        )
                        st.caption(
                            f"Threshold: **{VOICE_COSINE_THRESHOLD}** — "
                            f"same speaker typically 0.75–0.95, different speakers 0.40–0.75."
                        )
                        vdur = st.slider("Recording duration (seconds)", 3, 12, 7, key="voice_dur_verify")

                        if st.button("🎤 Record & Verify Voice", key="btn_do_verify_voice", type="primary"):
                            try:
                                with st.spinner(f"🔴 Recording for {vdur}s — please speak now…"):
                                    cap = AudioCapture()
                                    audio = cap.record_fixed(duration=vdur)
                                peak = float(np.max(np.abs(audio)))
                                if peak < 0.01:
                                    st.warning("⚠️ Very quiet recording. Check mic and speak louder.")

                                from modules.A_user_access.voice_biometrics import LAST_RECORDING_PATH, save_wav
                                save_wav(np.asarray(audio, dtype=np.float32).flatten(), LAST_RECORDING_PATH)
                                if os.path.isfile(LAST_RECORDING_PATH):
                                    st.markdown("🔊 **Listen to your recording:**")
                                    st.audio(LAST_RECORDING_PATH, format="audio/wav")

                                with st.spinner("Comparing against stored voiceprint…"):
                                    result = verify_voice(st.session_state.profile_id, audio)

                                m = re.search(r"similarity=([\d.]+)", result.get("message", ""))
                                if m:
                                    score = float(m.group(1))
                                    delta = score - VOICE_COSINE_THRESHOLD
                                    c1, c2 = st.columns(2)
                                    c1.metric("🎤 Similarity Score", f"{score:.3f}",
                                              delta=f"{delta:+.3f} vs threshold",
                                              delta_color="normal")
                                    c2.metric("🎯 Threshold", f"{VOICE_COSINE_THRESHOLD}")

                                if result["verified"]:
                                    st.session_state.verified = True
                                    sm.set_verified(True)
                                    log(f"✅ Verified (voice) profile={st.session_state.profile_id}")
                                    st.success(result["message"])
                                    st.rerun()
                                else:
                                    st.error(result["message"])
                            except Exception as e:
                                st.error(f"Microphone error: {e}")


            # ── Enrollment data management ────────────────────────
            with st.expander("🔧 Enrollment Data & Privacy"):
                st.markdown(
                    "Biometrics are stored **only on this machine** under "
                    "`data/enrollment/<profile_id>/`:\n\n"
                    "| File | Contents |\n"
                    "|------|----------|\n"
                    "| `face_encoding.npy` | 512-dim FaceNet identity embedding |\n"
                    "| `voice_embedding.npy` | 256-dim Resemblyzer speaker embedding |\n"
                    "| `meta.json` | Enrollment flags and metadata |\n\n"
                    "Passwords are compared to a **SHA-256 hash** — never stored in plaintext."
                )
                c1, c2 = st.columns(2)
                if c1.button("🗑️ Clear Face Data", key="clr_face"):
                    clear_biometric_enrollment(st.session_state.profile_id, "face")
                    log("🗑️ Face enrollment cleared")
                    st.rerun()
                if c2.button("🗑️ Clear Voice Data", key="clr_voice"):
                    clear_biometric_enrollment(st.session_state.profile_id, "voice")
                    log("🗑️ Voice enrollment cleared")
                    st.rerun()
                if st.button("🗑️ Clear ALL Biometric Data for This Profile", key="clr_bio_all"):
                    clear_biometric_enrollment(st.session_state.profile_id, "all")
                    log("🗑️ All biometric data cleared")
                    st.rerun()

        else:
            # ── Already verified — show log-out button ──
            if st.button("🔒 Lock / Log Out", key="btn_logout"):
                st.session_state.verified = False
                st.session_state.awake = False
                sm.set_verified(False)
                sm.set_wake_detected(False)
                reset_verification()
                log("🔒 Session locked.")
                st.rerun()

    st.divider()

    # ══════════════════════════════════════════════════════════════════
    # STAGE 2 — Input (Voice or Text)  (Modules 3 & 4)
    # ══════════════════════════════════════════════════════════════════
    with st.container(**_CONTAINER_KW):
        st.subheader("② Input — Voice or Text  (Modules 3 & 4)")

        if not st.session_state.verified:
            st.warning("⚠️ Verify your identity first.")
        else:
            input_mode = st.radio(
                "Input mode",
                ["📝 Type a command", "🎤 Record voice"],
                horizontal=True,
                key="input_mode",
            )

            raw_text = ""

            if input_mode == "📝 Type a command":
                typed = st.text_input(
                    "Type your command:",
                    placeholder='e.g.  "hey mello set learning rate to 0.01"',
                    key="typed_cmd",
                )
                if st.button("Submit Text Command", key="btn_text"):
                    processed = handler.process(typed)
                    if processed["valid"]:
                        raw_text = processed["text"]
                        log(f"📝 Text input received: '{raw_text}'")
                    else:
                        st.warning(processed["message"])

            else:  # Voice recording
                duration = st.slider("Recording duration (seconds)", 2, 10, 5)
                if st.button("🎤 Start Recording", key="btn_record"):
                    audio = None
                    with st.spinner(f"Recording for {duration}s…"):
                        try:
                            capture = AudioCapture()
                            audio = capture.record_fixed(duration=duration)
                            capture.save(audio, "/tmp/demo_clip.wav")
                            log(f"🎤 Audio captured ({duration}s).")
                        except Exception as e:
                            st.error(f"Microphone error: {e}")
                            audio = None

                    if audio is not None:
                        with st.spinner("Transcribing with Whisper…"):
                            try:
                                stt = SpeechToText()
                                transcript = stt.transcribe("/tmp/demo_clip.wav")
                                st.session_state.transcript = transcript
                                raw_text = transcript
                                log(f"🗣️ Transcribed: '{transcript}'")
                                if not transcript.strip():
                                    hint = get_last_transcribe_error() or "No transcript returned."
                                    st.warning(hint)
                                    log(f"⚠️ Transcription empty: {hint}")
                                else:
                                    st.success(f"**Heard:** {transcript}")
                            except Exception as e:
                                st.error(f"Transcription error: {e}")

            # ── Push raw_text into session if we got something ──────
            if raw_text:
                st.session_state.raw_command = raw_text
                sm.set_transcript(raw_text)

    st.divider()

    # ══════════════════════════════════════════════════════════════════
    # STAGE 3 — Wake Word Detection  (Module 2)
    # ══════════════════════════════════════════════════════════════════
    with st.container(**_CONTAINER_KW):
        st.subheader("③ Wake Word Detection  (Module 2)")

        cmd = st.session_state.raw_command
        if cmd:
            woke = is_wake_word(cmd)
            st.session_state.awake = woke
            sm.set_wake_detected(woke)
            if woke:
                st.success(f"✅ Wake word detected in: *\"{cmd}\"*")
                log("🔔 Wake word detected.")

                # Strip the wake word from the command before passing downstream
                clean_cmd = re.sub(
                    r"(hey|hi|okay|wake up|hello)\s+mello\s*",
                    "",
                    cmd,
                ).strip()
                st.session_state.clean_command = clean_cmd
                st.markdown("**Command (wake word stripped):**")
                st.code(clean_cmd, language="text")
            else:
                st.warning(
                    f"⚠️ No wake word found in: *\"{cmd}\"*  "
                    "— Try starting with **'hey mello'**"
                )
        else:
            st.info("Waiting for input…")

    st.divider()

    # ══════════════════════════════════════════════════════════════════
    # STAGE 4 — NLU Processing  (Modules 6 & 7) — NEW
    # ══════════════════════════════════════════════════════════════════
    with st.container(**_CONTAINER_KW):
        st.subheader("④ NLU Processing  (Modules 6 & 7)")

        if st.session_state.awake and st.session_state.get("clean_command"):
            clean = st.session_state.clean_command
            nlu_result = understand(clean)
            st.session_state.nlu_result = nlu_result

            c1, c2 = st.columns(2)
            with c1:
                intent_color = "green" if nlu_result["intent"] != "unknown_intent" else "red"
                st.markdown(f"**Detected Intent:** :{intent_color}[`{nlu_result['intent']}`]")
                st.markdown(f"**Category:** `{nlu_result['intent_category']}`")
            with c2:
                if nlu_result["slots"]:
                    st.markdown("**Extracted Slots:**")
                    for k, v in nlu_result["slots"].items():
                        st.markdown(f"  • `{k}` = `{v}` *({type(v).__name__})*")
                else:
                    st.markdown("**Slots:** *(none required)*")

            if nlu_result["missing_slots"]:
                st.warning(f"⚠️ Missing required parameter(s): **{', '.join(nlu_result['missing_slots'])}**")
            if nlu_result["invalid_slots"]:
                for slot_name, err in nlu_result["invalid_slots"].items():
                    st.error(f"❌ Invalid `{slot_name}`: {err}")

            with st.expander("📋 Full NLU output (JSON)"):
                st.json(nlu_result)
            log(f"🧠 NLU → intent={nlu_result['intent']}, slots={nlu_result['slots']}")
        else:
            st.info("Awaiting a valid wake-word command…")

    st.divider()

    # ══════════════════════════════════════════════════════════════════
    # STAGE 5 — Command Execution  (Modules 8, 9, 10) — NEW
    # ══════════════════════════════════════════════════════════════════
    with st.container(**_CONTAINER_KW):
        st.subheader("⑤ Command Execution  (Modules 8, 9, 10)")

        nlu = st.session_state.nlu_result
        if nlu is not None and st.session_state.awake:
            exec_result = route_command(nlu)
            st.session_state.exec_result = exec_result
            sm.set_assistant_response(exec_result["message"])

            if exec_result["success"]:
                st.success(f"✅ **{exec_result['intent']}** — {exec_result['message']}")
            else:
                st.error(f"❌ **{exec_result['intent']}** — {exec_result['message']}")

            st.markdown(f"**Route:** `{exec_result['category']}` · **Intent:** `{exec_result['intent']}`")
        else:
            st.info("Awaiting NLU output…")


# ── RIGHT COLUMN: System State + Dashboard ──────────────────────────────
with col_right:

    # ═══════════════════════════════════════════════════════════
    # Experiment State Dashboard  (Module 9)
    # ═══════════════════════════════════════════════════════════
    st.subheader("🖥️ Experiment Dashboard  (Module 9)")

    state = sm.get_state()

    # Status badge
    status_map = {
        "idle": "⚪ Idle",
        "training": "🟢 Training",
        "paused": "🟡 Paused",
        "stopped": "🔴 Stopped",
        "completed": "✅ Completed",
    }
    st.markdown(f"**Training:** {status_map.get(state['training_status'], state['training_status'])}")

    # Experiment config table
    kv_data = {
        "Dataset":       state["dataset"] or "—",
        "Model":         state["model"] or "—",
        "Learning Rate": state["learning_rate"],
        "Batch Size":    state["batch_size"],
        "Epochs":        f"{state['epoch_current']} / {state['epochs_total']}",
    }
    for label, val in kv_data.items():
        st.markdown(f"**{label}:** `{val}`")

    # ═══════════════════════════════════════════════════════════
    # Metrics Charts
    # ═══════════════════════════════════════════════════════════
    if state["loss_history"] or state["accuracy_history"]:
        st.divider()
        st.subheader("📈 Training Metrics")

        if state["loss_history"] and state["accuracy_history"]:
            chart_data = pd.DataFrame({
                "Loss": state["loss_history"],
                "Accuracy": state["accuracy_history"],
            })
            chart_data.index = range(1, len(chart_data) + 1)
            chart_data.index.name = "Epoch"

            tab_loss, tab_acc, tab_both = st.tabs(["📉 Loss", "📈 Accuracy", "📊 Both"])
            with tab_loss:
                st.line_chart(chart_data["Loss"], use_container_width=True)
            with tab_acc:
                st.line_chart(chart_data["Accuracy"], use_container_width=True)
            with tab_both:
                st.line_chart(chart_data, use_container_width=True)

            mc1, mc2 = st.columns(2)
            mc1.metric("📉 Latest Loss", f"{state['loss_history'][-1]:.4f}")
            mc2.metric("📈 Latest Accuracy", f"{state['accuracy_history'][-1]:.4f}")

    # ═══════════════════════════════════════════════════════════
    # Pipeline State
    # ═══════════════════════════════════════════════════════════
    st.divider()
    st.subheader("🔗 Pipeline State")

    pipe_data = {
        "verified":    st.session_state.verified,
        "awake":       st.session_state.awake,
        "transcript":  st.session_state.transcript or st.session_state.raw_command or "—",
        "raw_command": st.session_state.raw_command or "—",
    }
    for k, v in pipe_data.items():
        icon = "✅" if v is True else ("❌" if v is False else "📄")
        st.markdown(f"**`{k}`** {icon}  \n`{v}`")

    # ═══════════════════════════════════════════════════════════
    # Event Log
    # ═══════════════════════════════════════════════════════════
    st.divider()
    st.subheader("📋 Event Log")
    if st.session_state.event_log:
        for entry in reversed(st.session_state.event_log[-15:]):
            st.text(entry)
    else:
        st.caption("No events yet.")

    c1, c2 = st.columns(2)
    if c1.button("Clear Log", key="btn_clear_log"):
        st.session_state.event_log = []
        sm.clear_event_log()
        st.rerun()
    if c2.button("Reset Experiment", key="btn_reset_exp"):
        sm.reset_experiment()
        log("🔄 Experiment state reset.")
        st.rerun()

    # ═══════════════════════════════════════════════════════════
    # STT Health Check (from original app)
    # ═══════════════════════════════════════════════════════════
    st.divider()
    st.subheader("🩺 STT Health Check")
    st.caption("Records ~1s from mic, verifies HTTPS to Google, tries Whisper, checks ffmpeg.")
    if st.button("Run STT health checks", key="btn_stt_health"):
        with st.spinner("Running checks…"):
            out: dict = {}
            try:
                capture = AudioCapture()
                audio = np.asarray(capture.record_fixed(duration=1), dtype=np.float32).flatten()
                peak = float(np.max(np.abs(audio)))
                out["microphone"] = {"ok": peak > 1e-5, "detail": f"peak={peak:.4f}"}
            except Exception as e:
                out["microphone"] = {"ok": False, "detail": repr(e)}
            whisper_ok, whisper_msg = probe_whisper_model()
            out["whisper"] = {"ok": whisper_ok, "detail": whisper_msg}
            ffmpeg = shutil.which("ffmpeg")
            out["ffmpeg"] = {"ok": bool(ffmpeg), "detail": ffmpeg or "Not found"}
        st.json(out)

    # ═══════════════════════════════════════════════════════════
    # Quick Test (no mic)
    # ═══════════════════════════════════════════════════════════
    st.divider()
    st.subheader("🧪 Quick Test (no mic)")
    st.caption("Test the full C+D pipeline without a microphone.")
    test_text = st.text_area(
        "Test text",
        value="hey mello load the titanic dataset",
        height=70,
        key="quick_test_text",
    )
    if st.button("▶ Run Quick Test", key="btn_quick_test"):
        p = handler.process(test_text)
        if p["valid"]:
            woke = is_wake_word(p["text"])
            if woke:
                clean = re.sub(r"(hey|hi|okay|wake up|hello)\s+mello\s*", "", p["text"]).strip()
                nlu = understand(clean)
                result = route_command(nlu)
                sm.set_assistant_response(result["message"])

                st.markdown("---")
                st.markdown(f"**Wake word:** ✅ Detected")
                st.markdown(f"**Clean command:** `{clean}`")
                st.markdown(f"**Intent:** `{nlu['intent']}` ({nlu['intent_category']})")
                if nlu["slots"]:
                    st.markdown(f"**Slots:** `{nlu['slots']}`")
                if nlu["missing_slots"]:
                    st.warning(f"⚠️ Missing: {nlu['missing_slots']}")
                if nlu["invalid_slots"]:
                    for sn, se in nlu["invalid_slots"].items():
                        st.error(f"❌ {sn}: {se}")

                if result["success"]:
                    st.success(f"✅ {result['message']}")
                else:
                    st.error(f"❌ {result['message']}")

                log(f"🧪 Quick test: '{clean}' → {nlu['intent']}")
            else:
                st.warning("⚠️ No wake word detected. Try starting with 'hey mello'.")
        else:
            st.warning(p["message"])
