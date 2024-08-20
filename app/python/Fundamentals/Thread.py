import sys
import os
sys.dont_write_bytecode = True
sys.path.append(os.path.abspath('/Detect_scam-spam_call'))
from app.python.Fundamentals.Record import Record
from app.python.Fundamentals.Convert import Convert
import threading

class Thread:
    @staticmethod
    def main():
        record_seconds = None
        
        recorder = Record(record_seconds=record_seconds)
        converter = Convert(queue=recorder.queue, rate=recorder.rate, width=recorder.audio.get_sample_size(recorder.format))
        record_thread = threading.Thread(target=recorder.startRecord)
        stt_thread = threading.Thread(target=converter.convert)

        record_thread.start()
        stt_thread.start()
        record_thread.join()
        stt_thread.join()
        recorder.terminate()
    