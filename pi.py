import os
import librosa
import time
from transformers import WhisperProcessor, AutoConfig
from optimum.onnxruntime import ORTModelForSpeechSeq2Seq

def run(model_name, audio):
    #define odel path
    model_path = os.path.join("/home/carol/mp/quantize", model_name)
    # Load the model and processor
    processor = WhisperProcessor.from_pretrained(model_name)
    model_config = AutoConfig.from_pretrained(model_name)
    sessions = ORTModelForSpeechSeq2Seq.load_model(
                os.path.join(model_path, 'encoder_model.onnx'),
                os.path.join(model_path, 'decoder_model.onnx'),
                os.path.join(model_path, 'decoder_with_past_model.onnx'))
    model = ORTModelForSpeechSeq2Seq(sessions[0], sessions[1], model_config, model_path, sessions[2])

    # Load the audio file
    audio_data, sample_rate = librosa.load(audio, sr=16000, mono=True)

    # Preprocess the audio
    input_features = processor(audio_data, sampling_rate=sample_rate, return_tensors="pt").input_features
    forced_decoder_ids = processor.get_decoder_prompt_ids(language="english",task="translate")
    
    # Measure the time taken for inference
    start_time = time.time()
    predicted_ids = model.generate(input_features,forced_decoder_ids=forced_decoder_ids)[0]
    # Generate transcription
    transcription = processor.decode(predicted_ids, skip_special_tokens=True)
    inference_time = time.time() - start_time

     # model size
    total_size = 0
    for dirpath, dirnames, filenames in os.walk(model_path):
        for f in filenames:
            fp = os.path.join(dirpath, f)
            total_size += os.path.getsize(fp)
    size = total_size / (1024 * 1024)  # Convert to MB

    print()
    print("Model name = ",model_name)
    print()
    print(transcription)
    print()
    print("Inference Time = ",inference_time)
    print("Model size = ",size) 



run("q-tiny-arm","test2.wav")