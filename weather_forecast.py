from dotenv import load_dotenv
load_dotenv()

import os
import io
import tempfile
import requests
import streamlit as st
import google.generativeai as genai
from gtts import gTTS
import re
import json
from contextlib import suppress

# Voice deps
import speech_recognition as sr
from streamlit_mic_recorder import mic_recorder

# -----------------------------
# PAGE CONFIG
# -----------------------------
st.set_page_config(page_title="Gemini Weather Assistant", page_icon="🌤️", layout="centered")

# Small CSS polish
st.markdown("""
<style>
.block-container {padding-top: 2rem; padding-bottom: 3rem;}
h1 {margin-bottom:.25rem}
.subtitle {color:#64748b;margin-bottom:1.25rem}
.card {border:1px solid #e9ecef;border-radius:12px;padding:1.1rem;background:#fff;box-shadow:0 4px 18px rgba(0,0,0,.04)}
.divider {height:1px;background:#eef2f7;margin:.75rem 0 1rem}
.small {color:#64748b;font-size:.9rem}
</style>
""", unsafe_allow_html=True)

# -----------------------------
# MODEL CONFIG
# -----------------------------
genai.configure(api_key=os.getenv("GENAI_API_KEY"))
model = genai.GenerativeModel("models/gemini-1.5-flash")

# -----------------------------
# HEADER
# -----------------------------
st.markdown("## 🌤️ Gemini Weather Assistant")
st.markdown('<div class="subtitle">Ask about the weather by typing or speaking. You’ll also hear your forecast.</div>',
            unsafe_allow_html=True)

# -----------------------------
# UTILS
# -----------------------------
def transcribe_wav_bytes(wav_bytes: bytes) -> str | None:
    """Transcribe WAV bytes using SpeechRecognition (no PyAudio)."""
    try:
        r = sr.Recognizer()
        with sr.AudioFile(io.BytesIO(wav_bytes)) as source:
            data = r.record(source)
        return r.recognize_google(data)  # uses Google Web Speech API
    except Exception as e:
        st.warning(f"Transcription failed: {e}")
        return None

def extract_city_keywords(q: str) -> tuple[str, str]:
    prompt = f"""
    Extract the city and intent from this weather-related question: '{q}'.
    Respond ONLY in this format:
    City: <city>
    Keywords: <comma-separated keywords like storm, hurricane, rain, wind, heat, etc.>
    """
    resp = model.generate_content(prompt)
    lines = [ln.strip() for ln in (resp.text or "").splitlines() if ln.strip()]
    city = lines[0].replace("City:", "").strip() if len(lines) > 0 else ""
    keywords = lines[1].replace("Keywords:", "").strip() if len(lines) > 1 else ""
    return city, keywords

def fetch_weather(city: str) -> tuple[str, float]:
    key = os.getenv("WEATHER_API_KEY")
    url = f"http://api.openweathermap.org/data/2.5/weather?q={city}&appid={key}&units=metric"
    data = requests.get(url, timeout=15).json()
    if data.get("cod") != 200:
        raise ValueError(data.get("message", "Failed to fetch weather"))
    desc = data["weather"][0]["description"]
    temp = float(data["main"]["temp"])
    return desc, temp

def summarize_forecast(city: str, desc: str, temp: float, keywords: str) -> str:
    p = f"""
    The current weather in {city} is {desc} with a temperature of {temp:.1f}°C.
    The user is interested in: {keywords}.
    Please write a warm, friendly 2‑sentence weather forecast.
    """
    return (model.generate_content(p).text or "").strip()

def speak(text: str):
    tts = gTTS(text=text, lang="en")
    with tempfile.NamedTemporaryFile(delete=False, suffix=".mp3") as fp:
        tts.save(fp.name)
        st.audio(fp.name, format="audio/mp3")

# -----------------------------
# INPUT: VOICE + TEXT
# -----------------------------
st.markdown("### 🎙️ Voice input")
st.caption("Click **Start recording**, speak your question, then click **Stop**. We’ll transcribe it automatically.")

audio = mic_recorder(
    start_prompt="Start recording",
    stop_prompt="Stop",
    just_once=True,
    format="wav",            # <- important: WAV for SpeechRecognition
    use_container_width=True,
    key="mic_raw",
)

recognized = None
if audio and "bytes" in audio:
    recognized = transcribe_wav_bytes(audio["bytes"])
    if recognized:
        st.success(f"Recognized: {recognized}")

st.markdown("### ✍️ Or type your question")
default_text = st.session_state.get("question", recognized or "")
user_question = st.text_input(
    "Your question:",
    value=default_text,
    placeholder="e.g., What’s the weather in Dallas today?",
    label_visibility="collapsed",
    key="question",
)

col_run, col_reset = st.columns([1, 1])
with col_run:
    run = st.button("Get Forecast", type="primary", use_container_width=True)
with col_reset:
    if st.button("Reset", use_container_width=True):
        st.session_state.clear()
        st.experimental_rerun()

# -----------------------------
# PIPELINE
# -----------------------------
if run:
    if not user_question.strip():
        st.warning("Please provide a question by voice or text first.")
        st.stop()

    debug = {}
    with st.spinner("Thinking…"):
        try:
            city, keywords, raw_gemini = extract_city_keywords(user_question)
            debug["gemini_extract_raw"] = raw_gemini
            debug["city"] = city
            debug["keywords"] = keywords

            if not city:
                st.error("I couldn't extract a city from your question. Try rephrasing (e.g., 'What's the weather in Dallas today?').")
                st.stop()

            desc, temp, weather_json = fetch_weather(city)
            debug["openweather_json"] = weather_json

            forecast = summarize_forecast(city, desc, temp, keywords)
            debug["forecast_text"] = forecast

            # -------- Present result nicely --------
            st.markdown('<div class="card">', unsafe_allow_html=True)
            st.markdown("### 🌍 Location")
            st.write(f"**{city}**")

            st.markdown('<div class="divider"></div>', unsafe_allow_html=True)
            st.markdown("### 🌡️ Current conditions")
            st.write(f"**{desc.capitalize()}**, **{temp:.1f}°C**")

            if keywords:
                st.markdown('<div class="divider"></div>', unsafe_allow_html=True)
                st.markdown("### 🔑 Focus")
                st.write(keywords)

            st.markdown('<div class="divider"></div>', unsafe_allow_html=True)
            st.markdown("### 🗣️ Forecast")
            st.write(forecast)

            st.markdown('<div class="divider"></div>', unsafe_allow_html=True)
            st.markdown("#### 🔊 Listen")
            speak(forecast)

            st.markdown('</div>', unsafe_allow_html=True)

        except Exception as e:
            st.error("Something went wrong while generating your forecast.")
            st.exception(e)  # shows full traceback
        finally:
            with st.expander("Debug"):
                st.json(debug)
                st.write("Question:", user_question)
                st.write("ENV has GENAI_API_KEY:", bool(os.getenv("GENAI_API_KEY")))
                st.write("ENV has WEATHER_API_KEY:", bool(os.getenv("WEATHER_API_KEY")))