#!/usr/bin/python
import pygame
import alsaaudio
import numpy as np
import sys
import scipy.io.wavfile as wav
import time


FORMAT = alsaaudio.PCM_FORMAT_S16_LE
CHANNELS = 1
RATE = 44100
#INPUT_BLOCK_TIME = 0.05
#INPUT_FRAMES_PER_BLOCK = int(RATE * INPUT_BLOCK_TIME)
INPUT_FRAMES_PER_BLOCK = 1024
prefix = sys.argv[1]
card_name = sys.argv[2]


class Recorder(object):
    def __init__(self):
        self.stream = self.open_mic_stream()
        self.done = False

    def stop(self):
        self.stream.close()

    def open_mic_stream(self):
        stream = alsaaudio.PCM(alsaaudio.PCM_CAPTURE, card=card_name)
        #stream = alsaaudio.PCM(alsaaudio.PCM_CAPTURE)
        stream.setchannels(CHANNELS)
        stream.setrate(RATE)
        stream.setformat(FORMAT)
        stream.setperiodsize(INPUT_FRAMES_PER_BLOCK)
        return stream

    def record_sample(self, name):
        finished = False
        print(name)
        samples = []
        while not finished:
            for event in pygame.event.get():
                if event.type == pygame.KEYDOWN:
                    finished = True
                    if event.key == pygame.K_ESCAPE:
                        self.done = True
                    break
            l, data = self.stream.read()
            samples.append(data)
        frames = np.fromstring(''.join(samples), dtype='int16')
        wav.write(name, RATE, frames)

    def record_samples(self, prefix):
        print("Press any key to begin...")
        done = False
        while not done:
            for event in pygame.event.get():
                if event.type == pygame.KEYDOWN:
                    done = True
                    break
        print("Beginning recording")
        i = 1
        while not self.done:
            name = "{}{}.wav".format(prefix, i)
            self.record_sample(name)
            print("Sample {}".format(i))
            i += 1


if __name__ == "__main__":
    pygame.init()
    screen = pygame.display.set_mode((640, 480))
    pygame.display.set_caption('Basic Pygame program')
    background = pygame.Surface(screen.get_size())
    background = background.convert()
    background.fill((0, 0, 0))
    screen.blit(background, (0, 0))
    R = Recorder()
    R.record_samples(prefix)
    pygame.quit()
