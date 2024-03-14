## My Own RasberryPI AI speaker


### Server
- Listen request
### Client
- Raspberry Pi board
- request tasks (post recorded wave file) to server
### model
- making appropriate answer text
- speaker Embedding model to get personal information (sex, age)
- give prompt to GPT model for additional specific information

--------------------------------------------------------------------------------------------

we use OpenAI Whisper model API and Google Cloud API for STT / TTS
we finetuned pretrained Whisper model for classification about personal information (sex, age)
Audio dataset: '어린이 음성 데이터셋' and '한국어 음성 데이터셋' from AIHUB

server.py: utilize server
virtual_model.py: executed when server recalls. first recall stt, and track sex, age from the audio file. (kinda main.py)
client.py: Recorded.wav file(=audio input), save it to recorded_audio file, and execute this file (with appropriate path designated)
stt.py: speech-to-text
tts.py: text-to-speech
ask_api.py: requires extra info and message from audio inputs
