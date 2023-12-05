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
        #specify torch to use cpu
        self.model.to('cpu')


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
        with tempfile.NamedTemporaryFile(delete=False, suffix="wav") as temp_file:
            temp_filename = temp_file.name
            combined_audio_normalized = combined_audio / np.max(np.abs(combined_audio))
            combined_audio_normalized = combined_audio_normalized * 32767
            combined_audio_normalized = combined_audio_normalized.astype(np.int16)
            wavfile.write(temp_filename, self.model.config.sampling_rate, combined_audio_normalized.T)
            print(f"Audio saved to {temp_filename}")

        return temp_filename


# Example usage
if __name__ == "__main__":
    tts = TextToSpeech()


    text_to_speak = """
    हम तेरे बिन अब रह नहीं सकते तेरे बिन क्या वजूद मेरा हम तेरे बिन अब रह नहीं सकते तेरे बिन क्या वजूद मेरा तुझसे जुदा गर हो जाएंगे तो ख़ुद से ही हो जाएंगे जुदा क्यूंकि तुम ही हो अब तुम ही हो ज़िन्दगी अब तुम ही हो चैन भी, मेरा दर्द भी मेरी आशिक़ी अब तुम ही हो तेरा मेरा रिश्ता है कैसा एक पल दूर गवारा नहीं तेरे लिए हर रोज़ हैं जीते तुझ को दिया मेरा वक़्त सभी कोई लम्हा मेरा ना हो तेरे बिना हर सांस पे नाम तेरा.. """
    tts.text_to_audio(text=text_to_speak,filename='output.wav',play_chunks=False,play_combined=True)





# Some weights of VitsModel were not initialized from the model checkpoint at facebook/mms-tts-hin and are newly initialized: ['flow.flows.3.wavenet.res_skip_layers.2.parametrizations.weight.original1', 'flow.flows.1.wavenet.in_layers.0.parametrizations.weight.original1', 'flow.flows.1.wavenet.in_layers.2.parametrizations.weight.original0', 'flow.flows.3.wavenet.in_layers.3.parametrizations.weight.original0', 'posterior_encoder.wavenet.in_layers.4.parametrizations.weight.original1', 'flow.flows.0.wavenet.in_layers.1.parametrizations.weight.original1', 'flow.flows.1.wavenet.in_layers.1.parametrizations.weight.original1', 'posterior_encoder.wavenet.in_layers.2.parametrizations.weight.original1', 'posterior_encoder.wavenet.in_layers.10.parametrizations.weight.original1', 'posterior_encoder.wavenet.res_skip_layers.1.parametrizations.weight.original0', 'flow.flows.1.wavenet.in_layers.2.parametrizations.weight.original1', 'flow.flows.0.wavenet.in_layers.3.parametrizations.weight.original0', 'flow.flows.2.wavenet.res_skip_layers.1.parametrizations.weight.original1', 'posterior_encoder.wavenet.in_layers.9.parametrizations.weight.original1', 'flow.flows.1.wavenet.res_skip_layers.1.parametrizations.weight.original0', 'flow.flows.2.wavenet.in_layers.3.parametrizations.weight.original0', 'flow.flows.2.wavenet.res_skip_layers.2.parametrizations.weight.original0', 'posterior_encoder.wavenet.res_skip_layers.5.parametrizations.weight.original0', 'flow.flows.0.wavenet.res_skip_layers.1.parametrizations.weight.original0', 'flow.flows.0.wavenet.in_layers.2.parametrizations.weight.original1', 'flow.flows.3.wavenet.in_layers.1.parametrizations.weight.original1', 'flow.flows.1.wavenet.res_skip_layers.0.parametrizations.weight.original0', 'posterior_encoder.wavenet.in_layers.5.parametrizations.weight.original1', 'flow.flows.1.wavenet.res_skip_layers.3.parametrizations.weight.original1', 'posterior_encoder.wavenet.res_skip_layers.12.parametrizations.weight.original0', 'flow.flows.0.wavenet.res_skip_layers.3.parametrizations.weight.original1', 'posterior_encoder.wavenet.res_skip_layers.7.parametrizations.weight.original1', 'flow.flows.0.wavenet.res_skip_layers.1.parametrizations.weight.original1', 'flow.flows.1.wavenet.in_layers.3.parametrizations.weight.original1', 'posterior_encoder.wavenet.in_layers.7.parametrizations.weight.original1', 'flow.flows.3.wavenet.res_skip_layers.0.parametrizations.weight.original1', 'posterior_encoder.wavenet.in_layers.3.parametrizations.weight.original1', 'flow.flows.2.wavenet.res_skip_layers.0.parametrizations.weight.original0', 'flow.flows.1.wavenet.in_layers.1.parametrizations.weight.original0', 'posterior_encoder.wavenet.in_layers.1.parametrizations.weight.original1', 'posterior_encoder.wavenet.in_layers.10.parametrizations.weight.original0', 'flow.flows.0.wavenet.in_layers.1.parametrizations.weight.original0', 'flow.flows.3.wavenet.in_layers.2.parametrizations.weight.original0', 'flow.flows.2.wavenet.res_skip_layers.3.parametrizations.weight.original0', 'posterior_encoder.wavenet.in_layers.8.parametrizations.weight.original1', 'posterior_encoder.wavenet.in_layers.6.parametrizations.weight.original0', 'posterior_encoder.wavenet.res_skip_layers.2.parametrizations.weight.original1', 'flow.flows.1.wavenet.res_skip_layers.2.parametrizations.weight.original0', 'flow.flows.2.wavenet.in_layers.1.parametrizations.weight.original0', 'posterior_encoder.wavenet.res_skip_layers.15.parametrizations.weight.original0', 'flow.flows.2.wavenet.in_layers.2.parametrizations.weight.original1', 'flow.flows.0.wavenet.in_layers.2.parametrizations.weight.original0', 'posterior_encoder.wavenet.res_skip_layers.13.parametrizations.weight.original1', 'flow.flows.3.wavenet.in_layers.0.parametrizations.weight.original0', 'posterior_encoder.wavenet.res_skip_layers.0.parametrizations.weight.original1', 'flow.flows.0.wavenet.res_skip_layers.0.parametrizations.weight.original0', 'posterior_encoder.wavenet.res_skip_layers.0.parametrizations.weight.original0', 'posterior_encoder.wavenet.res_skip_layers.13.parametrizations.weight.original0', 'flow.flows.2.wavenet.in_layers.1.parametrizations.weight.original1', 'posterior_encoder.wavenet.in_layers.15.parametrizations.weight.original0', 'posterior_encoder.wavenet.res_skip_layers.9.parametrizations.weight.original1', 'flow.flows.3.wavenet.res_skip_layers.3.parametrizations.weight.original1', 'posterior_encoder.wavenet.res_skip_layers.4.parametrizations.weight.original1', 'flow.flows.0.wavenet.res_skip_layers.0.parametrizations.weight.original1', 'posterior_encoder.wavenet.res_skip_layers.1.parametrizations.weight.original1', 'posterior_encoder.wavenet.res_skip_layers.2.parametrizations.weight.original0', 'flow.flows.2.wavenet.in_layers.2.parametrizations.weight.original0', 'posterior_encoder.wavenet.res_skip_layers.11.parametrizations.weight.original1', 'posterior_encoder.wavenet.in_layers.5.parametrizations.weight.original0', 'posterior_encoder.wavenet.res_skip_layers.5.parametrizations.weight.original1', 'posterior_encoder.wavenet.in_layers.6.parametrizations.weight.original1', 'posterior_encoder.wavenet.res_skip_layers.10.parametrizations.weight.original0', 'posterior_encoder.wavenet.res_skip_layers.11.parametrizations.weight.original0', 'flow.flows.0.wavenet.res_skip_layers.2.parametrizations.weight.original1', 'posterior_encoder.wavenet.in_layers.4.parametrizations.weight.original0', 'posterior_encoder.wavenet.in_layers.2.parametrizations.weight.original0', 'flow.flows.2.wavenet.in_layers.0.parametrizations.weight.original1', 'flow.flows.3.wavenet.res_skip_layers.1.parametrizations.weight.original0', 'flow.flows.2.wavenet.res_skip_layers.1.parametrizations.weight.original0', 'posterior_encoder.wavenet.res_skip_layers.8.parametrizations.weight.original1', 'posterior_encoder.wavenet.in_layers.12.parametrizations.weight.original1', 'flow.flows.1.wavenet.res_skip_layers.2.parametrizations.weight.original1', 'flow.flows.3.wavenet.in_layers.1.parametrizations.weight.original0', 'flow.flows.1.wavenet.in_layers.3.parametrizations.weight.original0', 'posterior_encoder.wavenet.in_layers.11.parametrizations.weight.original1', 'posterior_encoder.wavenet.in_layers.13.parametrizations.weight.original1', 'flow.flows.3.wavenet.res_skip_layers.3.parametrizations.weight.original0', 'flow.flows.3.wavenet.res_skip_layers.1.parametrizations.weight.original1', 'posterior_encoder.wavenet.res_skip_layers.8.parametrizations.weight.original0', 'posterior_encoder.wavenet.in_layers.15.parametrizations.weight.original1', 'flow.flows.3.wavenet.in_layers.0.parametrizations.weight.original1', 'flow.flows.1.wavenet.in_layers.0.parametrizations.weight.original0', 'posterior_encoder.wavenet.in_layers.9.parametrizations.weight.original0', 'posterior_encoder.wavenet.res_skip_layers.12.parametrizations.weight.original1', 'flow.flows.2.wavenet.res_skip_layers.0.parametrizations.weight.original1', 'flow.flows.0.wavenet.in_layers.3.parametrizations.weight.original1', 'flow.flows.3.wavenet.in_layers.2.parametrizations.weight.original1', 'flow.flows.3.wavenet.res_skip_layers.0.parametrizations.weight.original0', 'posterior_encoder.wavenet.res_skip_layers.9.parametrizations.weight.original0', 'flow.flows.0.wavenet.in_layers.0.parametrizations.weight.original1', 'flow.flows.0.wavenet.in_layers.0.parametrizations.weight.original0', 'posterior_encoder.wavenet.in_layers.12.parametrizations.weight.original0', 'flow.flows.0.wavenet.res_skip_layers.2.parametrizations.weight.original0', 'posterior_encoder.wavenet.in_layers.7.parametrizations.weight.original0', 'posterior_encoder.wavenet.res_skip_layers.6.parametrizations.weight.original1', 'flow.flows.2.wavenet.in_layers.0.parametrizations.weight.original0', 'flow.flows.0.wavenet.res_skip_layers.3.parametrizations.weight.original0', 'flow.flows.2.wavenet.in_layers.3.parametrizations.weight.original1', 'posterior_encoder.wavenet.res_skip_layers.14.parametrizations.weight.original1', 'posterior_encoder.wavenet.in_layers.14.parametrizations.weight.original0', 'flow.flows.2.wavenet.res_skip_layers.2.parametrizations.weight.original1', 'posterior_encoder.wavenet.in_layers.0.parametrizations.weight.original0', 'flow.flows.3.wavenet.in_layers.3.parametrizations.weight.original1', 'posterior_encoder.wavenet.in_layers.3.parametrizations.weight.original0', 'flow.flows.1.wavenet.res_skip_layers.1.parametrizations.weight.original1', 'posterior_encoder.wavenet.in_layers.13.parametrizations.weight.original0', 'flow.flows.2.wavenet.res_skip_layers.3.parametrizations.weight.original1', 'posterior_encoder.wavenet.res_skip_layers.3.parametrizations.weight.original0', 'posterior_encoder.wavenet.res_skip_layers.3.parametrizations.weight.original1', 'flow.flows.1.wavenet.res_skip_layers.3.parametrizations.weight.original0', 'posterior_encoder.wavenet.res_skip_layers.6.parametrizations.weight.original0', 'posterior_encoder.wavenet.in_layers.14.parametrizations.weight.original1', 'posterior_encoder.wavenet.res_skip_layers.15.parametrizations.weight.original1', 'posterior_encoder.wavenet.in_layers.8.parametrizations.weight.original0', 'posterior_encoder.wavenet.res_skip_layers.10.parametrizations.weight.original1', 'posterior_encoder.wavenet.res_skip_layers.7.parametrizations.weight.original0', 'flow.flows.1.wavenet.res_skip_layers.0.parametrizations.weight.original1', 'posterior_encoder.wavenet.in_layers.0.parametrizations.weight.original1', 'posterior_encoder.wavenet.res_skip_layers.14.parametrizations.weight.original0', 'flow.flows.3.wavenet.res_skip_layers.2.parametrizations.weight.original0', 'posterior_encoder.wavenet.in_layers.11.parametrizations.weight.original0', 'posterior_encoder.wavenet.res_skip_layers.4.parametrizations.weight.original0', 'posterior_encoder.wavenet.in_layers.1.parametrizations.weight.original0']

# You should probably TRAIN this model on a down-stream task to be able to use it for predictions and inference.

# Some weights of the model checkpoint at facebook/mms-tts-hin were not used when initializing VitsModel: ['posterior_encoder.wavenet.in_layers.8.weight_v', 'posterior_encoder.wavenet.res_skip_layers.7.weight_g', 'posterior_encoder.wavenet.res_skip_layers.6.weight_g', 'posterior_encoder.wavenet.res_skip_layers.1.weight_g', 'flow.flows.0.wavenet.res_skip_layers.1.weight_g', 'flow.flows.3.wavenet.res_skip_layers.1.weight_v', 'flow.flows.1.wavenet.res_skip_layers.0.weight_g', 'posterior_encoder.wavenet.in_layers.10.weight_g', 'posterior_encoder.wavenet.in_layers.2.weight_v', 'flow.flows.1.wavenet.res_skip_layers.1.weight_v', 'posterior_encoder.wavenet.in_layers.2.weight_g', 'posterior_encoder.wavenet.res_skip_layers.10.weight_g', 'posterior_encoder.wavenet.res_skip_layers.8.weight_g', 'flow.flows.0.wavenet.in_layers.3.weight_v', 'posterior_encoder.wavenet.res_skip_layers.14.weight_g', 'flow.flows.3.wavenet.res_skip_layers.0.weight_v', 'flow.flows.2.wavenet.res_skip_layers.2.weight_v', 'flow.flows.3.wavenet.in_layers.2.weight_g', 'flow.flows.3.wavenet.in_layers.1.weight_v', 'posterior_encoder.wavenet.in_layers.13.weight_g', 'posterior_encoder.wavenet.in_layers.9.weight_g', 'flow.flows.2.wavenet.in_layers.3.weight_v', 'posterior_encoder.wavenet.in_layers.4.weight_g', 'posterior_encoder.wavenet.res_skip_layers.5.weight_v', 'flow.flows.3.wavenet.in_layers.2.weight_v', 'flow.flows.2.wavenet.in_layers.3.weight_g', 'flow.flows.0.wavenet.in_layers.0.weight_v', 'posterior_encoder.wavenet.in_layers.12.weight_g', 'flow.flows.0.wavenet.in_layers.2.weight_v', 'flow.flows.2.wavenet.in_layers.1.weight_g', 'flow.flows.1.wavenet.in_layers.3.weight_g', 'flow.flows.0.wavenet.in_layers.0.weight_g', 'posterior_encoder.wavenet.in_layers.5.weight_g', 'posterior_encoder.wavenet.in_layers.11.weight_g', 'flow.flows.0.wavenet.res_skip_layers.2.weight_v', 'posterior_encoder.wavenet.in_layers.9.weight_v', 'posterior_encoder.wavenet.res_skip_layers.14.weight_v', 'flow.flows.2.wavenet.res_skip_layers.1.weight_g', 'flow.flows.3.wavenet.in_layers.1.weight_g', 'flow.flows.2.wavenet.res_skip_layers.0.weight_g', 'posterior_encoder.wavenet.res_skip_layers.8.weight_v', 'posterior_encoder.wavenet.in_layers.11.weight_v', 'flow.flows.3.wavenet.res_skip_layers.3.weight_v', 'flow.flows.1.wavenet.in_layers.1.weight_v', 'flow.flows.1.wavenet.in_layers.3.weight_v', 'posterior_encoder.wavenet.in_layers.15.weight_v', 'flow.flows.1.wavenet.res_skip_layers.3.weight_v', 'posterior_encoder.wavenet.res_skip_layers.6.weight_v', 'posterior_encoder.wavenet.in_layers.4.weight_v', 'flow.flows.0.wavenet.res_skip_layers.1.weight_v', 'posterior_encoder.wavenet.in_layers.15.weight_g', 'flow.flows.1.wavenet.res_skip_layers.2.weight_g', 'posterior_encoder.wavenet.in_layers.12.weight_v', 'posterior_encoder.wavenet.res_skip_layers.2.weight_v', 'flow.flows.2.wavenet.res_skip_layers.2.weight_g', 'flow.flows.2.wavenet.in_layers.0.weight_g', 'posterior_encoder.wavenet.in_layers.7.weight_g', 'posterior_encoder.wavenet.res_skip_layers.4.weight_g', 'posterior_encoder.wavenet.res_skip_layers.15.weight_g', 'posterior_encoder.wavenet.res_skip_layers.3.weight_g', 'flow.flows.2.wavenet.res_skip_layers.3.weight_g', 'flow.flows.1.wavenet.res_skip_layers.2.weight_v', 'flow.flows.0.wavenet.in_layers.1.weight_v', 'flow.flows.0.wavenet.res_skip_layers.3.weight_g', 'flow.flows.3.wavenet.res_skip_layers.2.weight_v', 'posterior_encoder.wavenet.res_skip_layers.0.weight_g', 'posterior_encoder.wavenet.res_skip_layers.15.weight_v', 'posterior_encoder.wavenet.in_layers.7.weight_v', 'flow.flows.3.wavenet.in_layers.0.weight_v', 'posterior_encoder.wavenet.in_layers.6.weight_g', 'flow.flows.3.wavenet.res_skip_layers.1.weight_g', 'posterior_encoder.wavenet.res_skip_layers.1.weight_v', 'flow.flows.1.wavenet.in_layers.2.weight_v', 'posterior_encoder.wavenet.res_skip_layers.12.weight_v', 'flow.flows.0.wavenet.res_skip_layers.2.weight_g', 'posterior_encoder.wavenet.in_layers.6.weight_v', 'flow.flows.1.wavenet.res_skip_layers.1.weight_g', 'posterior_encoder.wavenet.res_skip_layers.11.weight_g', 'posterior_encoder.wavenet.in_layers.1.weight_v', 'flow.flows.1.wavenet.res_skip_layers.3.weight_g', 'posterior_encoder.wavenet.res_skip_layers.9.weight_g', 'flow.flows.2.wavenet.in_layers.2.weight_g', 'flow.flows.0.wavenet.res_skip_layers.0.weight_g', 'flow.flows.3.wavenet.res_skip_layers.2.weight_g', 'flow.flows.3.wavenet.in_layers.3.weight_g', 'posterior_encoder.wavenet.res_skip_layers.12.weight_g', 'flow.flows.3.wavenet.in_layers.3.weight_v', 'flow.flows.2.wavenet.res_skip_layers.1.weight_v', 'posterior_encoder.wavenet.in_layers.10.weight_v', 'posterior_encoder.wavenet.res_skip_layers.10.weight_v', 'posterior_encoder.wavenet.res_skip_layers.5.weight_g', 'flow.flows.1.wavenet.in_layers.0.weight_v', 'flow.flows.2.wavenet.in_layers.0.weight_v', 'posterior_encoder.wavenet.in_layers.0.weight_g', 'flow.flows.0.wavenet.in_layers.2.weight_g', 'flow.flows.3.wavenet.res_skip_layers.0.weight_g', 'posterior_encoder.wavenet.in_layers.14.weight_v', 'flow.flows.2.wavenet.in_layers.1.weight_v', 'posterior_encoder.wavenet.in_layers.8.weight_g', 'flow.flows.1.wavenet.in_layers.0.weight_g', 'flow.flows.3.wavenet.res_skip_layers.3.weight_g', 'posterior_encoder.wavenet.in_layers.3.weight_v', 'posterior_encoder.wavenet.in_layers.13.weight_v', 'flow.flows.1.wavenet.res_skip_layers.0.weight_v', 'flow.flows.1.wavenet.in_layers.1.weight_g', 'flow.flows.0.wavenet.res_skip_layers.3.weight_v', 'posterior_encoder.wavenet.in_layers.3.weight_g', 'flow.flows.2.wavenet.res_skip_layers.3.weight_v', 'flow.flows.3.wavenet.in_layers.0.weight_g', 'flow.flows.1.wavenet.in_layers.2.weight_g', 'flow.flows.2.wavenet.res_skip_layers.0.weight_v', 'posterior_encoder.wavenet.in_layers.1.weight_g', 'posterior_encoder.wavenet.in_layers.14.weight_g', 'posterior_encoder.wavenet.res_skip_layers.7.weight_v', 'flow.flows.0.wavenet.in_layers.1.weight_g', 'posterior_encoder.wavenet.res_skip_layers.0.weight_v', 'posterior_encoder.wavenet.res_skip_layers.4.weight_v', 'flow.flows.2.wavenet.in_layers.2.weight_v', 'posterior_encoder.wavenet.in_layers.0.weight_v', 'flow.flows.0.wavenet.res_skip_layers.0.weight_v', 'posterior_encoder.wavenet.res_skip_layers.13.weight_g', 'flow.flows.0.wavenet.in_layers.3.weight_g', 'posterior_encoder.wavenet.res_skip_layers.9.weight_v', 'posterior_encoder.wavenet.res_skip_layers.3.weight_v', 'posterior_encoder.wavenet.res_skip_layers.13.weight_v', 'posterior_encoder.wavenet.res_skip_layers.11.weight_v', 'posterior_encoder.wavenet.in_layers.5.weight_v', 'posterior_encoder.wavenet.res_skip_layers.2.weight_g']

# - This IS expected if you are initializing VitsModel from the checkpoint of a model trained on another task or with another architecture (e.g. initializing a BertForSequenceClassification model from a BertForPreTraining model).

# - This IS NOT expected if you are initializing VitsModel from the checkpoint of a model that you expect to be exactly identical (initializing a BertForSequenceClassification model from a BertForSequenceClassification model).


#to solve this issue we can use the following code

# from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
# model = AutoModelForSeq2SeqLM.from_pretrained("facebook/m2m100_418M")
