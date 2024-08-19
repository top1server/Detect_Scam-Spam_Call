import speech_recognition as sr
from queue import Queue

class Convert:
    def __init__(self, queue, rate=44100, width=2, channels=1, language="vi-VN"):
        self.queue = queue
        self.rate = rate
        self.width = width
        self.channels = channels
        self.language = language
        self.recognizer = sr.Recognizer()
        
    def convert(self):
        print("Converting audio stream...")
        accumulated_audio = b''
        while True:
            data = self.queue.get()
            if data is None:
                break
            
            accumulated_audio += data
            
            if len(accumulated_audio) > self.rate * self.width * 5:
                try:
                    audio_source = sr.AudioData(accumulated_audio, self.rate, self.width)
                    text = self.recognizer.recognize_google(audio_source, language=self.language)
                    print("Recognized text:", text)
                except sr.UnknownValueError:
                    print("Could not understand audio")
                except sr.RequestError as e:
                    print("Could not request results; {0}".format(e))
                
                accumulated_audio = b''
        