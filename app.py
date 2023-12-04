import streamlit as st
import tempfile
from utils.audio import TextToSpeech
from gtts import gTTS
from googletrans import Translator
from pydub import AudioSegment

audio_file = None

def init_session_state():
    if 'hinditext' not in st.session_state:
        st.session_state.hinditext = ''
    if 'englishtext' not in st.session_state:
        st.session_state.englishtext = ''
    if 'gttsText' not in st.session_state:
        st.session_state.gttsText = ''
    if 'audio_file' not in st.session_state:
        st.session_state.audio_file = None

# Function to convert text to speech using gtts
def text_to_speech_gtts(text, lang='en', speed=1.0):
    try:
        tts = gTTS(text=text, lang=lang, slow=False)
        with tempfile.NamedTemporaryFile(delete=False, suffix=".mp3") as temp_file:
            temp_filename = temp_file.name
            tts.save(temp_filename)
        return temp_filename
    except Exception as e:
        st.error(f"Error during text-to-speech conversion: {e}")
        return None

# Function to translate English text to Hindi using googletrans
def translate_to_hindi(text):
    try:
        translator = Translator()
        translation = translator.translate(text, src='en', dest='hi')
        return translation.text
    except Exception as e:
        st.error(f"Translation error: {e}")
        return None

# Function to adjust pitch and speed of audio using pydub
def adjust_audio(audio_file, speed=1.0, pitch=1.0):
    if audio_file:
        try:
            audio = AudioSegment.from_mp3(audio_file)
        except:
            try:
                audio = AudioSegment.from_wav(audio_file)
            except:
                audio = AudioSegment.from_mp4(audio_file)

        # Adjust pitch and speed
        new_audio = audio._spawn(audio.raw_data, overrides={
            "frame_rate": int(audio.frame_rate * speed),
            "pitch": pitch
        })

        # Export modified audio
        with tempfile.NamedTemporaryFile(delete=False, suffix=".mp3") as temp_file:
            temp_filename = temp_file.name
            new_audio.export(temp_filename, format="mp3")

        return temp_filename
    else:
        st.warning("Please convert text to speech first.")
        return None

# Streamlit UI
def main():
    st.set_page_config(page_title="Text to Speech Converter", page_icon="🔊", layout="centered", initial_sidebar_state="collapsed")

    st.title("Text to Speech Converter 🎙️")

    engine = st.radio("Select Text to Speech Engine 🛠️", ("🤖 Robot voice", "🗣️ Realistic voice"), horizontal=True, help="Select the engine to use for text-to-speech conversion.")

    if engine == "🗣️ Realistic voice":
        col1, col2 = st.columns(2)

        with col1:
            englishtext = st.text_area("Enter text in English 🇬🇧", st.session_state.englishtext, height=100)
            st.session_state.englishtext = englishtext

        with col2:
            hinditext = st.text_area("Enter text in Hindi 🇮🇳", st.session_state.hinditext, height=100)
            st.session_state.hinditext = hinditext

        _, col4, _ = st.columns(3)
        with col4:
            if st.button("Translate to Hindi 🔄"):
                if st.session_state.englishtext:
                    translated_text = translate_to_hindi(st.session_state.englishtext)
                    if translated_text:
                        st.session_state.hinditext = translated_text
                        st.success("Translation successful! 🌐")
                    else:
                        st.warning("Translation failed. 😟")
                else:
                    st.warning("Please enter text in English to translate to Hindi. 🇬🇧➡️🇮🇳")

    elif engine == "🤖 Robot voice":
        gttsText = st.text_area("Enter text in Hindi or English 🇮🇳🇬🇧", height=100)
        st.session_state.gttsText = gttsText

    _, col3, _ = st.columns(3)
    with col3:
        if st.button("Convert to speech 🔊"):
            if engine == "🗣️ Realistic voice":
                tts = TextToSpeech()
                if st.session_state.hinditext:
                    audio_file = tts.text_to_audio(text=st.session_state.hinditext, play_chunks=False, play_combined=True)
                    st.session_state.audio_file = audio_file
                else:
                    st.warning("Please enter text in Hindi text area to convert to speech. 🇮🇳")
            elif engine == "🤖 Robot voice":
                if st.session_state.gttsText:
                    audio_file = text_to_speech_gtts(st.session_state.gttsText)
                    st.session_state.audio_file = audio_file
                else:
                    st.warning("Please enter text in Hindi or English to convert to speech. 🇮🇳🇬🇧")

    if st.session_state.audio_file:
        st.audio(st.session_state.audio_file, format='audio/mp3')

    # Audio options using pydub pitch and speed
    st.subheader("Audio options 🎧")
    col5, col6 = st.columns(2)
    with col5:
        pitch = st.slider("Pitch", min_value=0.5, max_value=2.0, value=1.0, step=0.1, help="Adjust the pitch of the audio. 🎵")
    with col6:
        speed = st.slider("Speed", min_value=0.5, max_value=2.0, value=1.0, step=0.1, help="Adjust the speed of the audio. ⏩")

    if st.button("Apply 🔄"):
        new_audio = adjust_audio(st.session_state.audio_file, speed=speed, pitch=pitch)
        if new_audio:
            st.audio(new_audio, format='audio/mp3')

if __name__ == '__main__':
    init_session_state()
    main()
