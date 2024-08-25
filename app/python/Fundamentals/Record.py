import pyaudio as pa
from queue import Queue
import wave
from threading import Thread
import sys
sys.dont_write_bytecode = True
class Record:
    def __init__(self, format=pa.paInt16, channels=1, rate=44100, chunk=4096, record_seconds=None):
        self.format = format
        self.channels = channels
        self.rate = rate
        self.chunk = chunk
        self.record_seconds = record_seconds
        self.audio = pa.PyAudio()
        self.queue = Queue()
        self.frames = []
        self.is_recording = False
        
    def startRecord(self):
        self.frames = []
        self.is_recording = True
        self.stream = self.audio.open(format=self.format, channels=self.channels,
                                      rate=self.rate, input=True,
                                      frames_per_buffer=self.chunk)
        self.record_thread = Thread(target=self.record)
        self.record_thread.start()

    def record(self):
        print("Start recording...")
        while self.is_recording:
            data = self.stream.read(self.chunk)
            self.queue.put(data)
            self.frames.append(data)
        
        print("Stop recording")
        self.stream.stop_stream()
        self.stream.close()

    def stopRecord(self, filename):
        self.is_recording = False
        self.record_thread.join()
        self.audio.terminate()

        with wave.open(filename, 'wb') as wf:
            wf.setnchannels(self.channels)
            wf.setsampwidth(self.audio.get_sample_size(self.format))
            wf.setframerate(self.rate)
            wf.writeframes(b''.join(self.frames))