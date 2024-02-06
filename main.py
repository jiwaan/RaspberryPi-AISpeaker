import pyaudio
import wave
from openai import OpenAI
# import subproces
from inference import Inferencer
# import inference.Inferencer

def main():
    print('hello_worlds')
    client = OpenAI(
        api_key='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'
    )
    audio_file = open('11001005.wav', "rb")
    transcript = client.audio.transcriptions.create(
        model="whisper-1", 
        file=audio_file
    )
    text = transcript.text

    print(text)
    inferencer = Inferencer()
    output = inferencer.inference('11001005.wav')

    print(output)
    if output==0:
        text= '나 어린아이인데, 아이한테 얘기하듯이 답변 해줘, 내 질문은'+text
    else:
        text= '내 질문은'+text

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

    # 저장한 wav 파일을 저장하고 바로 말하는 버전
    response.stream_to_file("testing1.wav")
    # subprocess.call(["afplay", "testing1.wav"])

main()