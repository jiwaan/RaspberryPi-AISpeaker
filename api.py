from openai import OpenAI
import openai
client = OpenAI(
    api_key = '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'
)

audio_file= open("/Users/nam/Desktop/gpt_api/일반남여_일반통합06_F_1524374974_25_수도권_실내_06990.wav", "rb")
transcript = client.audio.transcriptions.create(
model="whisper-1", 
file=audio_file
)
text=transcript.text
# from openai import OpenAI
# client = OpenAI(
#     api_key = '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'
# )

# audio_file= open("/path/to/file/audio.mp3", "rb")
# transcript = client.audio.transcriptions.create(
#   model="whisper-1", 
#   file=audio_file
# )
# 여기부터 생성임

completion = client.chat.completions.create(
model="gpt-3.5-turbo",
messages=[
    {"role": "system", "content": "You are a poetic assistant, skilled in explaining complex programming concepts with creative flair."},
    {"role": "user", "content": "한국말로 짧은 말 해봐"}
]
)
task=completion.choices[0].message.content.replace('\n', '')

response = client.audio.speech.create(
    model="tts-1",
    voice="alloy",
    input=task,
)

response.stream_to_file("rererererreallllllloutput.mp3")

# import os
# from openai import OpenAI
# client = OpenAI()
# OpenAI.api_key = os.getenv('OPENAI_API_KEY')

# completion = client.completions.create(
#   model="gpt-3.5-turbo-instruct",
#   prompt="Say this is a test",
#   max_tokens=7,
#   temperature=0
# )

# print(completion.choices[0].text)
# import openai

# OPENAI_API_KEY = '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'
# openai.api_key = OPENAI_API_KEY
# model = "gpt-3.5-turbo"
# query = "나 어린이인데, 좀 쉽게 "+"텍스트를 이미지로 그려주는 모델에 대해 알려줘."

# messages = [{
#     "role": "system",
#     "content": "You are a helpful assistant."
# }, {
#     "role": "user",
#     "content": query
# }]
# response = openai.ChatCompletion.create(model=model, messages=messages)
# answer = response['choices'][0]['message']['content']
# print(answer)