import os
from typing import List, Dict, Optional
import sys

import numpy as np
from PyQt5.QtWidgets import QApplication, QWidget, QVBoxLayout, QLabel, QHBoxLayout, QPushButton, QStatusBar, QSlider
from PyQt5.QtCore import Qt
from pyqtgraph import PlotWidget, InfiniteLine, mkColor
import simpleaudio as sa
from scipy.io.wavfile import write

from dict_parser import parse_dict
from src.utils.config import get_settings, set_settings
from sound_utils import record_sample, create_forced_alignment
from src.utils.data import Phoneme
from src.utils.tsv import TSVEntry


class LabeledWidget(QWidget):
    def __init__(self, label: str, widget: QWidget):
        super().__init__()
        layout = QHBoxLayout()
        self.labelWidget = QLabel('{}: '.format(label))
        self.widget = widget
        layout.addWidget(self.labelWidget)
        layout.addWidget(self.widget)
        self.setLayout(layout)


class ButtonBar(QWidget):
    def __init__(self, labels: List[str]):
        super().__init__()
        self.widgetDict: Dict[str, QPushButton] = {}
        layout = QHBoxLayout()
        for label in labels:
            self.widgetDict[label] = QPushButton(text=label)
            layout.addWidget(self.widgetDict[label])
        self.setLayout(layout)

    def __getitem__(self, item: str) -> Optional[QPushButton]:
        if item in self.widgetDict:
            return self.widgetDict[item]
        return None

    def __contains__(self, item: str) -> bool:
        return item in self.widgetDict

    def update_label(self, old_label: str, new_label: str):
        if old_label in self.widgetDict:
            widget = self.widgetDict[old_label]
            widget.setText(new_label)
            self.widgetDict.pop(old_label)
            self.widgetDict[new_label] = widget


class Display(QWidget):
    def __init__(self):
        super().__init__()
        self.settings = get_settings()
        self.entries = parse_dict(self.settings['word_dict'])[self.settings['start_index']:]
        self.entry_index = 0

        layout = QVBoxLayout()

        self.word_widget = LabeledWidget('Word', QLabel('Sample Word'))
        self.word_phonemes = LabeledWidget('Phonemes', QLabel('Sample Phonemes'))

        layout.addWidget(self.word_widget)
        layout.addWidget(self.word_phonemes)

        self.plot = PlotWidget(title='Recording')

        layout.addWidget(self.plot)

        self.buttonBar = ButtonBar([
            'Skip',
            'Record',
            'Play',
            'Save'
        ])
        self.buttonBar['Record'].clicked.connect(self.record_event)
        self.buttonBar['Skip'].clicked.connect(self.skip_event)
        self.buttonBar['Play'].clicked.connect(lambda x: self.play_sample())
        self.buttonBar['Save'].clicked.connect(self.save_event)
        self.record_mode = True
        self.recording_data: Optional[List[float]] = None
        self.play_obj: Optional[sa.PlayObject] = None
        self.alignment: Optional[List[Phoneme]] = None
        layout.addWidget(self.buttonBar)

        self.start_line: Optional[InfiniteLine] = None
        self.stop_line: Optional[InfiniteLine] = None

        self.setLayout(layout)
        self.setup_item()

    def disable_recording(self, disabled: bool = True):
        self.buttonBar['Save'].setDisabled(disabled)
        self.buttonBar['Play'].setDisabled(disabled)

    def setup_item(self):
        item = self.entries[self.entry_index]
        self.word_widget.widget.setText(item.term)
        self.word_phonemes.widget.setText(item.phoneme_display())
        self.buttonBar.update_label('Clear', 'Record')
        self.record_mode = True
        self.disable_recording()

    def update_recording_data(self, update_lines: bool = True):
        self.disable_recording(self.record_mode)
        if self.recording_data:
            start, stop = 0, len(self.recording_data)
            self.plot.clear()
            self.plot.plot(list(range(len(self.recording_data))), self.recording_data)
            if update_lines:
                self.update_trim(start, stop)

    def update_trim(self, pstart: float, pstop: float):
        self.start_line = self.plot.addLine(pstart, movable=True, pen=mkColor(52, 232, 235), label='Start',
                                            labelOpts={
                                                'position': 0.75
                                            })
        self.stop_line = self.plot.addLine(pstop, movable=True, pen=mkColor(52, 232, 235),
                                           label='Stop', labelOpts={
                'position': 0.75
            })
        self.start_line.sigPositionChangeFinished.connect(self.slider_released_event)
        self.stop_line.sigPositionChangeFinished.connect(self.slider_released_event)
        self.play_sample()

    def update_alignment(self):
        self.alignment = create_forced_alignment(
            np.array(self.recording_data[round(self.start_line.value()):round(self.stop_line.value())]),
            self.entries[self.entry_index].term,
            self.settings['recording_rate'])
        if self.alignment:
            for phone in self.alignment:
                self.plot.addLine(phone.start_index + self.start_line.value())
                self.plot.addLine(phone.stop_index + self.start_line.value())

    def save_event(self, event):
        self.settings['start_index'] += 1
        set_settings(self.settings)
        full_audio = self.recording_data[round(self.start_line.value()):round(self.stop_line.value())]
        if not self.settings['recording_dir'].exists():
            os.makedirs(self.settings['recording_dir'], exist_ok=True)
        for pi, phone in enumerate(self.alignment):
            phone_dir = self.settings['recording_dir'] / phone.name
            if not phone_dir.exists():
                os.makedirs(phone_dir, exist_ok=True)
            phoneme_audio = np.expand_dims(np.array(full_audio[phone.start_index:phone.stop_index]), axis=1)
            phoneme_filename = phone_dir / '{}_{}.wav'.format(self.settings['start_index'] - 1, pi)
            write(phoneme_filename, self.settings['recording_rate'], phoneme_audio)
        self.skip_event()

    def skip_event(self, event=None):
        self.entry_index += 1
        self.setup_item()

    def record_event(self, event):
        if self.record_mode:
            self.buttonBar.update_label('Record', 'Recording...')
            self.buttonBar['Recording...'].repaint()
            self.recording_data = record_sample(self.settings['recording_seconds'], self.settings['recording_rate'])\
                .flatten().tolist()
            self.buttonBar.update_label('Recording...', 'Clear')
            self.buttonBar['Clear'].repaint()
        else:
            self.buttonBar.update_label('Clear', 'Record')
            self.recording_data = None

        self.record_mode = not self.record_mode
        self.update_recording_data()

        if not self.record_mode:
            self.update_alignment()

    def slider_released_event(self):
        self.update_recording_data(False)
        self.update_alignment()
        start, stop = self.start_line.value() if self.start_line else 0, \
            self.stop_line.value() if self.stop_line else len(self.recording_data)
        self.update_trim(start, stop)

    def play_sample(self):
        audio = np.array(self.recording_data[round(self.start_line.value()):round(self.stop_line.value())])
        audio *= 32767 / np.max(np.abs(audio))
        audio = audio.astype(np.int16)
        self.play_obj = sa.play_buffer(audio, 1, 2, self.settings['recording_rate'])


if __name__ == '__main__':
    app = QApplication(sys.argv)
    display = Display()
    display.show()
    app.exec_()
