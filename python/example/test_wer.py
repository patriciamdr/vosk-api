#!/usr/bin/env python3
import argparse
import glob

from jiwer import wer
from vosk import Model, KaldiRecognizer, SetLogLevel
import os
import wave

SetLogLevel(0)


def speech_to_text(args):
    if not os.path.exists(os.path.join('models', args.model)):
        print(
            "Please download the model from https://alphacephei.com/vosk/models and unpack to 'models' folder.")
        exit(1)

    for filepath in glob.iglob(os.path.join(os.getcwd(), args.data, '*.wav')):
        print(filepath)

        wf = wave.open(args.data, "rb")
        if wf.getnchannels() != 1 or wf.getsampwidth() != 2 or wf.getcomptype() != "NONE":
            print("Audio file must be WAV format mono PCM.")
            exit(1)

        model = Model(args.model)
        rec = KaldiRecognizer(model, wf.getframerate())

        while True:
            data = wf.readframes(4000)
            if len(data) == 0:
                break
            if rec.AcceptWaveform(data):
                print(rec.Result())
            else:
                print(rec.PartialResult())

        print(rec.FinalResult())

        hypothesis_path = os.path.join(args.hypothesis, filepath.split('.')[0] + '.txt')
        with open(hypothesis_path, 'w') as hypothesis:
            hypothesis.write(rec.FinalResult())


def calculate_wer(args):
    ground_truth = []
    hypothesis = []

    for filepath in glob.iglob(os.path.join(os.getcwd(), args.ground_truth, "*.txt")):
        with open(filepath) as file:
            ground_truth.append(file.read())

    for filepath in glob.iglob(os.path.join(os.getcwd(), args.hypothesis, "*.txt")):
        with open(filepath) as file:
            hypothesis.append(file.read())

    error = wer(ground_truth, hypothesis)
    print(error)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model',
                        default='model-en-us',
                        type=str)
    parser.add_argument('--audio_data',
                        default='audio_data',
                        help='Directory with audio files',
                        type=str)
    parser.add_argument('--ground_truth',
                        default='ground_truth',
                        help='Directory with ground truth',
                        type=str)
    parser.add_argument('--hypothesis',
                        default='hypothesis',
                        help='Directory where to save recognized text output',
                        type=str)
    args = parser.parse_args()

    speech_to_text(args)
    calculate_wer(args)
