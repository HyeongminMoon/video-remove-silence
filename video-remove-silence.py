#!/usr/bin/env python3

import argparse
import collections
import math
import os
import re
import subprocess
import sys
import tempfile
import wave
import glob

from pydub import AudioSegment


parser = argparse.ArgumentParser()
parser.add_argument('--path', type=str, default='files', help='path to video')
parser.add_argument('--threshold-level', type=float, default=-35, help='threshold level in dB')
parser.add_argument('--threshold-duration', type=float, default=0.2, help='threshold duration in seconds')
parser.add_argument('--constant', type=float, default=0, help='duration constant transform value')
parser.add_argument('--sublinear', type=float, default=0, help='duration sublinear transform factor')
parser.add_argument('--linear', type=float, default=0.1, help='duration linear transform factor')
parser.add_argument('--save-silence', type=str, help='filename for saving silence')
parser.add_argument('--recalculate-time-in-description', type=str, help='path to text file')
args = parser.parse_args()

import json
from subprocess import check_output


def find_silences(filename):
    global args
    blend_duration = 0.005
    with wave.open(filename) as wav:
        size = wav.getnframes()
        channels = wav.getnchannels()
        sample_width = wav.getsampwidth()
        frame_rate = wav.getframerate()
        max_value = 1 << (8 * sample_width - 1)
        half_blend_frames = int(blend_duration * frame_rate / 2)
        blend_frames = half_blend_frames * 2
        assert size > blend_frames > 0
        square_threshold = max_value ** 2 * 10 ** (args.threshold_level / 10)
        blend_squares = collections.deque()
        blend = 0

        def get_values():
            frames_read = 0
            while frames_read < size:
                frames = wav.readframes(min(0x1000, size - frames_read))
                frames_count = len(frames) // sample_width // channels
                for frame_index in range(frames_count):
                    yield frames[frame_index*channels*sample_width:(frame_index+1)*channels*sample_width]
                frames_read += frames_count

        def get_is_silence(blend):
            results = 0
            frames = get_values()
            for index in range(half_blend_frames):
                frame = next(frames)
                square = 0
                for channel in range(channels):
                    value = int.from_bytes(frame[sample_width*channel:sample_width*channel+sample_width], 'little', signed=True)
                    square += value*value
                blend_squares.append(square)
                blend += square
            for index in range(size-half_blend_frames):
                frame = next(frames)
                square = 0
                for channel in range(channels):
                    value = int.from_bytes(frame[sample_width*channel:sample_width*channel+sample_width], 'little', signed=True)
                    square += value*value
                blend_squares.append(square)
                blend += square
                if index < half_blend_frames:
                    yield blend < square_threshold * channels * (half_blend_frames + index + 1)
                else:
                    result = blend < square_threshold * channels * (blend_frames + 1)
                    if result:
                        results += 1
                    yield result
                    blend -= blend_squares.popleft()
            for index in range(half_blend_frames):
                blend -= blend_squares.popleft()
                yield blend < square_threshold * channels * (blend_frames - index)

        is_silence = get_is_silence(blend)

        def to_regions(iterable):
            iterator = enumerate(iterable)
            while True:
                try:
                    index, value = next(iterator)
                except StopIteration:
                    return
                if value:
                    start = index
                    while True:
                        try:
                            index, value = next(iterator)
                            if not value:
                                yield start, index
                                break
                        except StopIteration:
                            yield start, index+1
                            return

        threshold_frames = int(args.threshold_duration * frame_rate)
        silence_regions = ( (start, end) for start, end in to_regions(is_silence) if end-start >= blend_duration )
        silence_regions = ( (start + (half_blend_frames if start > 0 else 0), end - (half_blend_frames if end < size else 0)) for start, end in silence_regions )
        silence_regions = [ (start, end) for start, end in silence_regions if end-start >= threshold_frames ]
        including_end = len(silence_regions) == 0 or silence_regions[-1][1] == size
        silence_regions = [ (start/frame_rate, end/frame_rate) for start, end in silence_regions ]
        # print(args.save_silence)
        if args.save_silence:
            with wave.open(args.save_silence, 'wb') as out_wav:
                out_wav.setnchannels(channels)
                out_wav.setsampwidth(sample_width)
                out_wav.setframerate(frame_rate)
                for start, end in silence_regions:
                    wav.setpos(start)
                    frames = wav.readframes(end-start)
                    out_wav.writeframes(frames)

    return silence_regions, including_end

def transform_duration(duration):
    global args
    return args.constant + args.sublinear * math.log(duration + 1) + args.linear * duration

def format_offset(offset):
    return '{}:{}:{}'.format(int(offset) // 3600, int(offset) % 3600 // 60, offset % 60)

def closest_frames(duration, frame_rate):
    return int((duration + 1 / frame_rate / 2) // (1 / frame_rate))

def compress_audio(wav, start_frame, end_frame, result_frames):
    # print(start_frame, end_frame, result_frames)
    if result_frames == 0:
        return b''
    elif result_frames == end_frame - start_frame:
        # print("same")
        wav.setpos(start_frame)
        return wav.readframes(result_frames)
    else:
        channels = wav.getnchannels()
        sample_width = wav.getsampwidth()
        frame_width = sample_width*channels
        if result_frames*2 <= end_frame - start_frame:
            left_length = result_frames
            right_length = result_frames
        else:
            left_length = (end_frame - start_frame + 1) // 2
            right_length = end_frame - start_frame - left_length
        crossfade_length = right_length + left_length - result_frames
        crossfade_start = (result_frames - crossfade_length) // 2
        wav.setpos(start_frame)
        left_frames = wav.readframes(left_length)
        wav.setpos(end_frame - right_length)
        right_frames = wav.readframes(right_length)
        result = bytearray(b'\x00'*result_frames*frame_width)
        result[:(left_length-crossfade_length)*frame_width] = left_frames[:-crossfade_length*frame_width]
        result[-(right_length-crossfade_length)*frame_width:] = right_frames[crossfade_length*frame_width:]
        for i in range(crossfade_length):
            r = i / (crossfade_length - 1)
            l = 1 - r
            for channel in range(channels):
                signal_left = int.from_bytes(left_frames[(left_length-crossfade_length+i)*frame_width+channel*sample_width:(left_length-crossfade_length+i)*frame_width+(channel+1)*sample_width], 'little', signed=True)
                signal_right = int.from_bytes(right_frames[i*frame_width+channel*sample_width:i*frame_width+(channel+1)*sample_width], 'little', signed=True)
                result[(left_length-crossfade_length+i)*frame_width+channel*sample_width:(left_length-crossfade_length+i)*frame_width+(channel+1)*sample_width] = int(signal_left*l + signal_right*r).to_bytes(sample_width, 'little', signed=True)
        return result


audio_file = tempfile.NamedTemporaryFile(delete=False)
audio_file.close()

print('Extracting audio...')
if os.path.isdir(args.path):
    paths = glob.glob(args.path +'/*')
else:
    paths = [args.path]
cnt = 0
for path in paths:
    cnt+=1
    print(cnt, path.split('/')[-1])
    silences, including_end = find_silences(path)

    total_duration = sum(( end-start for start, end in silences ))
    # print(silences)
    
    if len(silences) == 0:
        print('Everything is fine')
        sys.exit(0)
    
    # print('Found {} gaps, {:.1f} seconds total'.format(len(silences), total_duration))
    regions = []
    if silences[0][0] > 0:
        regions.append((0, silences[0][0], False))
    for silence, next_silence in zip(silences[:-1], silences[1:]):
        regions.append((silence[0], silence[1], True))
        regions.append((silence[1], next_silence[0], False))
    if including_end:
        regions.append((silences[-1][0], None, True))
    else:
        regions.append((silences[-1][0], silences[-1][1], True))
        regions.append((silences[-1][1], None, False))
    
    frame_rate = 44100
    # audio_track = tempfile.NamedTemporaryFile(delete=False)
    # audio_track.close()
    
    wav = wave.open(path)
    save_dir = os.path.join(os.getcwd(), 'result/')
    if not os.path.isdir(save_dir):
        os.mkdir(save_dir)
    
    dst = save_dir + path.split('files')[-1]
    out_wav = wave.open(dst , 'wb')
    # out_wav = wave.open(audio_track.name, 'wb')
    size = wav.getnframes()
    channels = wav.getnchannels()
    sample_width = wav.getsampwidth()
    audio_frame_rate = wav.getframerate()
    out_wav.setnchannels(channels)
    out_wav.setsampwidth(sample_width)
    out_wav.setframerate(audio_frame_rate)

    
    _, start, start_sil = regions[0]
    end, _, end_sil = regions[-1]
    
    if not start_sil:
        start = 0.0
    if not end_sil or end == 0:
        end = None
    
    audio_start_frame = min(int(start * audio_frame_rate), size)
    audio_end_frame = size if end is None else min(int(end * audio_frame_rate), size)
    audio_result_frames = audio_end_frame-audio_start_frame

    out_wav.writeframes(compress_audio(wav, audio_start_frame, audio_end_frame, audio_result_frames))

    wav.close()
    out_wav.close()
    
    # encoder.stdin.close()
    seg = AudioSegment.silent(duration=300)
    song = AudioSegment.from_wav(dst)
    final_song = seg + song + seg
    
    final_song.export(dst, format="wav")

    # print()
#     exit()
