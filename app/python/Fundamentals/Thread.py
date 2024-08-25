import threading
import sys
import os
sys.path.append(os.path.abspath('/Detect_scam-spam_call'))
import threading
from app.python.Fundamentals.Record import Record
from app.python.Fundamentals.Convert import Convert

class Thread:
    def __init__(self, record, convert):
        self.record = record
        self.convert = convert
        self.text = ""
        self.thread = None
        self.stop_event = threading.Event()

    def start(self):
        self.stop_event.clear()
        self.thread = threading.Thread(target=self.run)
        self.thread.start()

    def stop(self):
        self.stop_event.set()
        if self.thread is not None:
            self.thread.join()

    def run(self):
        print("Recording and converting...")
        self.record.startRecord()

        while not self.stop_event.is_set():
            text = self.convert.convert()
            if text:
                self.text += text + " "
            print("Accumulated text:", self.text)

        self.record.stopRecord("recording.wav")
        print("Stopped recording and converting.")
