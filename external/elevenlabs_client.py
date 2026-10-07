from elevenlabs.client import ElevenLabs
from elevenlabs.play import play

import os
from dotenv import load_dotenv

load_dotenv()


class ElevenLabsClient:
    def __init__(self, api_key: str, model: str, voice: str):
        self.api_key = api_key 
        self.model = model
        self.voice = voice
        self.output = "mp3"

        self.client = ElevenLabs(api_key=self.api_key)

    def tts(self, text: str):

        audio = self.client.text_to_speech.convert(
            text=text,
            voice_id=(self.voice),
            model_id=(self.model),
            output_format="mp3_44100_128"
        )

        play(audio)
    

    def stt(self, text):
        text = self.client.speech_to_text.convert(
            file=""
        )

        

        


    