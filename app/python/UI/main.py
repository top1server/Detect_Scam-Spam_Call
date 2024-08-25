from kivy.app import App
from kivy.uix.button import Button
from kivy.uix.floatlayout import FloatLayout
from kivy.uix.label import Label
from kivy.clock import Clock
from kivy.graphics import Color, Rectangle
import sys
import os
import threading
import json
sys.path.append(os.path.abspath('/Detect_scam-spam_call'))
from app.python.Fundamentals.Record import Record
from app.python.Fundamentals.Convert import Convert
from app.python.Fundamentals.Thread import Thread
from app.python.Models.LogisticRegressionClassifier import LogisticRegressionClassifier
from app.python.UI.Time import Time

sys.dont_write_bytecode = True

class MyApp(App):
    def build(self):
        self.layout = FloatLayout()
        with self.layout.canvas.before:
            Color(1, 1, 1, 1)
            self.rect = Rectangle(size=self.layout.size, pos=self.layout.pos)
        self.layout.bind(size=self._update_rect, pos=self._update_rect)

        self.button = Button(
            text='Bắt đầu ghi âm',
            size_hint=(None, None),
            size=(200, 50),
            pos_hint={'center_x': 0.5, 'center_y': 0.4}
        )
        self.button.bind(on_press=self.on_button_press)
        self.layout.add_widget(self.button)

        self.time_label = Label(
            text='00:00:00',
            size_hint=(None, None),
            pos_hint={'center_x': 0.5, 'center_y': 0.5},
            font_size='30sp',
            color=(0, 0, 0, 1)
        )
        self.layout.add_widget(self.time_label)

        self.status_label = Label(
            text='',
            size_hint=(None, None),
            pos_hint={'center_x': 0.5, 'center_y': 0.3},
            font_size='24sp',
            color=(0, 0, 0, 1)
        )
        self.layout.add_widget(self.status_label)

        self.dots_count = 0
        self.time_display = Time(self.time_label)
        self.record = Record()
        self.convert = Convert(queue=self.record.queue, rate=self.record.rate, width=self.record.audio.get_sample_size(self.record.format))
        self.classifier = LogisticRegressionClassifier(data_path='D:\\Detect_Scam-Spam_Call\\app\\python\\Models\\dataset.json')
        self.classifier.train(*self.classifier.load_data())
        self.thread = Thread(self.record, self.convert)

        self.blink_interval = 0.5
        self.blink_event = None
        self.current_index = 0
        self.text_to_display = "Phát hiện lừa đảo!!!"

        return self.layout

    def _update_rect(self, instance, value):
        self.rect.pos = instance.pos
        self.rect.size = instance.size

    def on_button_press(self, instance):
        if instance.text == "Bắt đầu ghi âm":
            instance.text = "Dừng ghi âm"
            self.time_display.start()
            self.status_event = Clock.schedule_interval(self.update_status, 0.5)
            self.thread.start()
            self.status_label.text = "Đang ghi âm"
        else:
            instance.text = "Bắt đầu ghi âm"
            self.time_display.stop()
            Clock.unschedule(self.status_event)
            self.thread.stop()
            self.status_label.text = ""
            if self.blink_event:
                Clock.unschedule(self.blink_event)

    def update_status(self, dt):
        self.dots_count = (self.dots_count + 1) % 4
        if self.dots_count == 0:
            self.status_label.text = "Đang ghi âm"
        else:
            self.status_label.text = "Đang ghi âm" + "." * self.dots_count
        
        text = self.thread.text.strip()
        if text:
            predicted_label = self.classifier.predict_label(text)
            if predicted_label == 1:
                if not self.blink_event:
                    self.blink_event = Clock.schedule_interval(self.update_text, self.blink_interval)
            else:
                self.status_label.text = "Đang ghi âm"
                if self.blink_event:
                    Clock.unschedule(self.blink_event)
                    self.blink_event = None
                    self.current_index = 0
                    self.status_label.text = "Đang ghi âm"

    def update_text(self, dt):
        if self.current_index < len(self.text_to_display):
            self.status_label.text = self.text_to_display[:self.current_index + 1]
            self.current_index += 1
        else:
            self.current_index = 0
            self.status_label.text = ""

if __name__ == "__main__":
    MyApp().run()
