import sys
import os
sys.dont_write_bytecode = True
sys.path.append(os.path.abspath('/Detect_scam-spam_call'))
from app.python.Fundamentals.Thread import Thread

if __name__ == "__main__":
    Thread.main()