import pyaudio as pa
from queue import Queue

class Record:
    def __init__(self, format=pa.paInt16, channels=1, rate=44100, chunk=4096, record_seconds=None):
        self.format = format
        self.channels = channels
        self.rate = rate
        self.chunk = chunk
        self.record_seconds = record_seconds
        self.audio = pa.PyAudio()
        self.queue = Queue()

    def startRecord(self):
        stream = self.audio.open(format=self.format, channels=self.channels,
                                 rate=self.rate, input=True,
                                 frames_per_buffer=self.chunk)
        print("Start recording...")
        while True:
            data = stream.read(self.chunk)
            self.queue.put(data)
            if self.record_seconds and (stream.get_time() >= self.record_seconds):
                break
        
        print("Stop recording\n")
        stream.stop_stream()
        stream.close()
        self.queue.put(None)

    def terminate(self):
        self.audio.terminate()