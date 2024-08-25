from kivy.clock import Clock
class Time:
    def __init__(self, label):
        self.label = label
        self.start_time = 0
        self.time_event = None

    def start(self):
        self.start_time = 0
        self.time_event = Clock.schedule_interval(self.update_time, 1)

    def stop(self):
        if self.time_event:
            Clock.unschedule(self.time_event)
        self.label.text = '00:00:00'

    def update_time(self, dt):
        self.start_time += 1
        hours, remainder = divmod(self.start_time, 3600)
        minutes, seconds = divmod(remainder, 60)
        self.label.text = f'{hours:02}:{minutes:02}:{seconds:02}'