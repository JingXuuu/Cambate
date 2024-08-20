# modified from PDVC/visualization/visualization.py

import matplotlib.pyplot as plt
import time
import numpy as np
import cv2
import json
from PIL import Image, ImageFont, ImageDraw
import argparse
import os
from tqdm import tqdm
import textwrap
import concurrent.futures

from moviepy.editor import VideoFileClip


# utils
def get_frame_caption(frame_time, captions):
    frame_captions = []
    for caption in captions:
        start_time, end_time = caption['timestamp']
        if start_time <= frame_time <= end_time:
            frame_captions.append(caption)
    return frame_captions

def paint_text(im, text, font, pos, color):
    img_PIL = Image.fromarray(cv2.cvtColor(im, cv2.COLOR_BGR2RGB))
    draw = ImageDraw.Draw(img_PIL)
    wrapped_text = textwrap.fill(text, width=68)
    draw.text(pos, wrapped_text, font=font, fill=color)
    img = cv2.cvtColor(np.asarray(img_PIL), cv2.COLOR_RGB2BGR)
    return img

def process_img(img, cur_time, title, frame_captions, n_caption=3):
    # add caption to the image (frame)
    scale = 1.0
    basic_text_height = 50
    text_height = int(basic_text_height * scale)
    font_size = int(text_height * 0.8)

    h, w, c = img.shape
    last_time = cur_time
    cur_time = time.time()
    img_fps = 1. / (cur_time - last_time + 1e-8)
    bg_img = np.zeros_like(img)
    cv2.rectangle(bg_img, (0, 0), (len(title) * text_height // 2, text_height), (120, 120, 120), -1, 1, 0)
    cv2.rectangle(bg_img, (0, h - text_height * (n_caption + 2)), (w, h), (120, 120, 120), -1, 1, 0)
    mask = bg_img / 255.
    alpha = 0.5
    img = img * (mask == 0) + alpha * img * (mask > 0) + (1 - alpha) * mask
    img = img.astype('uint8')

    font = ImageFont.truetype("visualization/Arial.ttf", font_size)
    
    img = paint_text(img, title, font, (10, 0), color=(255, 255, 255))
    for i, caption_info in enumerate(frame_captions[:n_caption]):
        caption, timestamp = caption_info['sentence'], caption_info['timestamp']
        caption_text = '{:2.1f}s-{:2.1f}s: {}'.format(timestamp[0], timestamp[1], caption)
        pt_text = (10, h - text_height * (n_caption + 2) + i * text_height)
        img = paint_text(img, caption_text, font, pt_text, color=(255, 255, 255))

    return img, cur_time, img_fps

def generate_caption(captions_path):
    # This is a placeholder function. Implement your caption generation logic here.
    # The output should be a list of dictionaries with 'timestamp' and 'sentence' keys.
    # Example format: [{'timestamp': [0, 2], 'sentence': 'A person is playing football.'}, ...]

    captions = []
    with open(captions_path, 'r') as f:
        captions = json.load(f)

    return captions


# add captions
def vid_show(vid_path, captions, save_mp4, save_mp4_path):
    start_time = time.time()
    cur_time = time.time()
    video = cv2.VideoCapture(vid_path)
    fps = video.get(cv2.CAP_PROP_FPS)
    frame_count = video.get(cv2.CAP_PROP_FRAME_COUNT)
    duration = frame_count / fps
    print('fps: {}, duration: {}, frames: {}'.format(fps, duration, frame_count))
    img_fps = fps
    n = 0
    if save_mp4:
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        video_writer = cv2.VideoWriter(save_mp4_path, fourcc, fps, (1280, 720))

    captions = sorted(captions, key=lambda p: p['timestamp'])

    for frame_id in tqdm(range(int(frame_count))):
        ret, frame = video.read()
        if n >= int(fps / img_fps) or save_mp4:
            n = 0
        else:
            n += 1
            continue
        if not ret:
            break
        frame = cv2.resize(frame, (1280, 720))
        frame_time = frame_id / fps
        frame_captions = get_frame_caption(frame_time, captions)
        n_caption = min(3, len(frame_captions))

        title = '{:.1f}s/{:.1f}s'.format(frame_time, duration)
        frame, cur_time, img_fps = process_img(frame, cur_time, title, frame_captions, n_caption=n_caption)
        if not save_mp4:
            plt.axis('off')
            plt.imshow(frame[:, :, ::-1])
            plt.show()
        if save_mp4:
            video_writer.write(frame)

    if save_mp4:
        video_writer.release()
        print('output video saved at {}, process time: {} s'.format(save_mp4_path, cur_time - start_time))




# add captions with parallel processing
def vid_show_segment(vid_path, captions, save_mp4, save_mp4_path, start_frame, end_frame, total_frame):

    video = cv2.VideoCapture(vid_path)
    fps = video.get(cv2.CAP_PROP_FPS)
    video.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
    if save_mp4:
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        video_writer = cv2.VideoWriter(save_mp4_path, fourcc, fps, (1280, 720))

    captions = sorted(captions, key=lambda p: p['timestamp'])

    for frame_id in tqdm(range(start_frame, end_frame)):
        ret, frame = video.read()
        if not ret:
            break
        frame = cv2.resize(frame, (1280, 720)) # adjust the size of the frame(resolution)
        frame_time = frame_id / fps
        frame_captions = get_frame_caption(frame_time, captions)
        n_caption = min(3, len(frame_captions))

        title = '{:.1f}s/{:.1f}s'.format(frame_time, total_frame / fps)
        frame, cur_time, img_fps = process_img(frame, time.time(), title, frame_captions, n_caption=n_caption)
        if save_mp4:
            video_writer.write(frame)

    if save_mp4:
        video_writer.release()

def process_video_parallel(input_video_path, captions, output_video_path, num_threads=4):
    temp_output_video_path = f"temp{output_video_path}"
    video = cv2.VideoCapture(input_video_path)
    frame_count = int(video.get(cv2.CAP_PROP_FRAME_COUNT))
    segment_length = frame_count // num_threads

    futures = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=num_threads) as executor:
        for i in range(num_threads):
            start_frame = i * segment_length
            end_frame = (i + 1) * segment_length if i != num_threads - 1 else frame_count
            segment_output_path = f"{temp_output_video_path}_part{i}.mp4"
            futures.append(executor.submit(vid_show_segment, input_video_path, captions, True, segment_output_path, start_frame, end_frame, frame_count))

    # Wait for all threads to complete
    concurrent.futures.wait(futures)

    # Combine the segments
    combine_segments(f"{temp_output_video_path}", num_threads)

    # extract audio from original video and add to the output video
    # Load the input video
    input_video = VideoFileClip(input_video_path)
    input_audio = input_video.audio

    # Add the audio to the output video
    output_video = VideoFileClip(f"{temp_output_video_path}")
    output_video_with_audio = output_video.set_audio(input_audio)

    # Write the output video with audio
    output_video_with_audio.write_videofile(output_video_path, codec='libx264', audio_codec='aac')

    # Remove the temporary video
    os.remove(f"{temp_output_video_path}")

def combine_segments(output_video_path, num_parts):
    video_writer = None
    for i in range(num_parts):
        segment_path = f"{output_video_path}_part{i}.mp4"
        segment_video = cv2.VideoCapture(segment_path)
        while True:
            ret, frame = segment_video.read()
            if not ret:
                break
            if video_writer is None:
                fourcc = cv2.VideoWriter_fourcc(*"mp4v")
                fps = segment_video.get(cv2.CAP_PROP_FPS)
                frame_size = (frame.shape[1], frame.shape[0])
                video_writer = cv2.VideoWriter(output_video_path, fourcc, fps, frame_size)
            video_writer.write(frame)
        segment_video.release()
        os.remove(segment_path)
    if video_writer:
        video_writer.release()

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_video_path', type=str, required=True, help='Path to the input video file.')
    parser.add_argument('--output_video_path', type=str, default='output.mp4', help='Path to save the output video file.')
    opt = parser.parse_args()

    input_video_path = opt.input_video_path
    output_video_path = opt.output_video_path
    captions_path = 'results.json'

    
    print('Generating captions for video:', input_video_path)
    captions = generate_caption(captions_path)

    print('Processing video and adding captions...')
    # vid_show(input_video_path, captions, save_mp4=True, save_mp4_path=output_video_path)
    process_video_parallel(input_video_path, captions, output_video_path, num_threads=8)