from transformers import VitsModel, AutoTokenizer
import torch
import numpy as np
from scipy.io import wavfile
import tempfile


class TextToSpeech:
    def __init__(self, model_name="facebook/mms-tts-hin"):
        self.model = VitsModel.from_pretrained(model_name)
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.original_text = ""
        self.chunks = []

    def preprocess_text(self, text):
        # Split the text and clean it
        text = text.split("\n")
        text = [i.strip() for i in text if i.strip() != ""]
        text = " ".join(text)
        return text

    def split_text_into_chunks(self, text, chunk_size=256):
        # Split text into chunks ensuring that chunks end at word boundaries
        words = text.split()
        chunks = []
        current_chunk = ""
        for word in words:
            if len(current_chunk) + len(word) + 1 <= chunk_size:  # +1 for space
                current_chunk += word + " "
            else:
                chunks.append(current_chunk.strip())
                current_chunk = word + " "
        if current_chunk.strip():
            chunks.append(current_chunk.strip())
        return chunks

    def generate_audio(self, text):
        inputs = self.tokenizer(text, return_tensors="pt", padding=True, truncation=True, max_length=512)
        inputs["input_ids"] = inputs["input_ids"].long()

        with torch.no_grad():
            output = self.model(**inputs).waveform

        return output.numpy()

    def process_and_combine_text(self, text, play_chunks=False):
        self.original_text = text
        cleaned_text = self.preprocess_text(text)
        self.chunks = self.split_text_into_chunks(cleaned_text)

        # Generate and play audio for each chunk
        combined_audio = []
        max_length = 0  # Track the maximum length of chunks
        for i, chunk in enumerate(self.chunks):
            print(f"Processing chunk {i+1}/{len(self.chunks)}")
            audio_chunk = self.generate_audio(chunk)
            
            # Normalize and play the chunk
            audio_chunk_normalized = audio_chunk / np.max(np.abs(audio_chunk))
            combined_audio.append(audio_chunk_normalized)
            print(f'audio_chunk {i} shape: {audio_chunk_normalized.shape}')
            max_length = max(max_length, len(audio_chunk_normalized))


        combined_audio = np.concatenate(combined_audio, axis=1)
        return combined_audio
        


    def text_to_audio(self, text, play_chunks=False,play_combined=False):
        print("Processing text...")
        combined_audio = self.process_and_combine_text(text, play_chunks)

        if combined_audio.size == 0:
            print("No audio data to display.")
            return



        # Save the combined audio to a mp3 file using tmpfile
        with tempfile.NamedTemporaryFile(delete=False, suffix=".mp3") as temp_file:
            temp_filename = temp_file.name
            combined_audio_normalized = combined_audio / np.max(np.abs(combined_audio))
            combined_audio_normalized = combined_audio_normalized * 32767
            combined_audio_normalized = combined_audio_normalized.astype(np.int16)
            wavfile.write(temp_filename, self.model.config.sampling_rate, combined_audio_normalized.T)
            print(f"Audio saved to {temp_filename}")







# Example usage
if __name__ == "__main__":
    tts = TextToSpeech()


    text_to_speak = """
    हम तेरे बिन अब रह नहीं सकते तेरे बिन क्या वजूद मेरा हम तेरे बिन अब रह नहीं सकते तेरे बिन क्या वजूद मेरा तुझसे जुदा गर हो जाएंगे तो ख़ुद से ही हो जाएंगे जुदा क्यूंकि तुम ही हो अब तुम ही हो ज़िन्दगी अब तुम ही हो चैन भी, मेरा दर्द भी मेरी आशिक़ी अब तुम ही हो तेरा मेरा रिश्ता है कैसा एक पल दूर गवारा नहीं तेरे लिए हर रोज़ हैं जीते तुझ को दिया मेरा वक़्त सभी कोई लम्हा मेरा ना हो तेरे बिना हर सांस पे नाम तेरा.. """
    tts.text_to_audio(text=text_to_speak,filename='output.wav',play_chunks=False,play_combined=True)


