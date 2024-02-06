import pyaudio
import wave
from openai import OpenAI
import subprocess

# 오디오 설정
FORMAT = pyaudio.paInt16
CHANNELS = 1
RATE = 44100
CHUNK = 1024
RECORD_SECONDS = 10  
WAVE_OUTPUT_FILENAME = "output1.wav"
audio = pyaudio.PyAudio()

stream = audio.open(format=FORMAT, channels=CHANNELS,
                    rate=RATE, input=True,
                    frames_per_buffer=CHUNK)
print("녹음 시작")

frames = []
for i in range(0, int(RATE / CHUNK * RECORD_SECONDS)):
    data = stream.read(CHUNK)
    frames.append(data)

print("녹음 종료")
stream.stop_stream()
stream.close()
audio.terminate()

# WAV 파일로 저장
wf = wave.open(WAVE_OUTPUT_FILENAME, 'wb')
wf.setnchannels(CHANNELS)
wf.setsampwidth(audio.get_sample_size(FORMAT))
wf.setframerate(RATE)
wf.writeframes(b''.join(frames))
wf.close()

client = OpenAI(
    api_key='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'
)
audio_file = open("/Users/nam/Desktop/gpt_api/output1.wav", "rb")
transcript = client.audio.transcriptions.create(
    model="whisper-1", 
    file=audio_file
)
text = transcript.text
print(text)
# 분류모델의 결과에 따라 프롬프트 앞에 나 어린이 인데 + 라는 말 추가?

# GPT API의 응답 길이 제한
max_tokens = 100  
completion = client.chat.completions.create(
    model="gpt-3.5-turbo",
    messages=[
        {"role": "system", "content": text},
        {"role": "user", "content": text}
    ],
    max_tokens=max_tokens  # 응답 길이 제한 설정
)
task = completion.choices[0].message.content.replace('\n', '')

response = client.audio.speech.create(
    model="tts-1",
    voice="alloy",
    input=task,
)

# 저장한 mp3 파일을 d
response.stream_to_file("testing1.wav")
subprocess.call(["afplay", "testing1.wav"]) 
