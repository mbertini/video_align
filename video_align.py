# Description: This script aligns two videos by finding the best alignment point in the first video
#              using the peak signal-to-noise ratio (PSNR) and structural similarity index (SSIM) metrics.
#              The aligned videos are then composited side-by-side into a single video.
# Usage: python video_align.py video_a_path video_b_path [--output_dir OUTPUT_DIR] [--output_filename OUTPUT_FILENAME]
#        [--metrics METRICS [METRICS ...]] [--verbose]
# Example: python video_align.py long_high_quality.mp4 short_compressed.mp4 --output_dir . --output_filename aligned_composite_ --metrics psnr ssim --verbose

import argparse
import subprocess
import json
import os
import numpy as np
from PIL import Image
import tempfile
from skimage.metrics import structural_similarity, peak_signal_noise_ratio
import av


def get_video_info(video_path):
    cmd = [
        'ffprobe',
        '-v', 'quiet',
        '-print_format', 'json',
        '-show_format',
        '-show_streams',
        video_path
    ]
    result = subprocess.run(cmd, capture_output=True, text=True)
    try:
        info = json.loads(result.stdout)
        return info
    except json.JSONDecodeError:
        print(f"Error decoding JSON for {video_path}")
        return None


def get_frame_count(info):
    if 'streams' in info:
        for stream in info['streams']:
            if stream.get('codec_type') == 'video':
                if 'nb_frames' in stream:
                    return int(stream['nb_frames'])
                elif 'duration' in stream and 'avg_frame_rate' in stream:
                    duration = float(stream['duration'])
                    fps = eval(stream['avg_frame_rate'])
                    return int(duration * fps)
    print("Could not determine frame count")
    return None


def get_video_duration(info):
    if 'format' in info and 'duration' in info['format']:
        return float(info['format']['duration'])
    elif 'streams' in info:
        for stream in info['streams']:
            if 'duration' in stream:
                return float(stream['duration'])
    print("Could not determine video duration")
    return None


def get_video_fps(info):
    if 'streams' in info:
        for stream in info['streams']:
            if stream.get('codec_type') == 'video' and 'avg_frame_rate' in stream:
                return eval(stream['avg_frame_rate'])
    print("Could not determine video fps")
    return None


def check_video_files(video_a_path, video_b_path):
    for path in [video_a_path, video_b_path]:
        if not os.path.exists(path):
            print(f"Error: The file {path} does not exist.")
            return False

    for name, path in [("Video A", video_a_path), ("Video B", video_b_path)]:
        info = get_video_info(path)
        if info is None:
            print(f"Error: Could not get info for {name} at {path}")
            return False

        duration = get_video_duration(info)
        fps = get_video_fps(info)
        frame_count = get_frame_count(info)

        if duration is None or fps is None or frame_count is None:
            print(f"Error: Could not get duration, FPS, or frame count for {name} at {path}")
            return False

        print(f"{name}:")
        print(f"  Path: {path}")
        print(f"  Duration: {duration:.2f} seconds")
        print(f"  FPS: {fps}")
        print(f"  Frame count: {frame_count}")
        print()

    return True


def extract_frame(video_path, time, output_path):
    cmd = [
        'ffmpeg',
        '-i', video_path,
        '-ss', str(time),
        '-vframes', '1',
        '-q:v', '2',
        output_path
    ]
    subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)


def compute_psnr(img1, img2):
    return peak_signal_noise_ratio(img1, img2)
    mse = np.mean((img1 - img2) ** 2)
    if mse == 0:
        return float('inf')
    max_pixel = 255.0
    return 20 * np.log10(max_pixel / np.sqrt(mse))


def compute_ssim(img1, img2):
    return structural_similarity(img1, img2, channel_axis=2, data_range=255)


def compute_similarities(img1, img2):
    return {
        'psnr': compute_psnr(img1, img2),
        'ssim': compute_ssim(img1, img2),
    }


def align_videos(video_a_path, video_b_path):
    if not check_video_files(video_a_path, video_b_path):
        return None

    info_a = get_video_info(video_a_path)
    info_b = get_video_info(video_b_path)

    duration_a = get_video_duration(info_a)
    duration_b = get_video_duration(info_b)

    fps_a = get_video_fps(info_a)
    fps_b = get_video_fps(info_b)

    if duration_a < duration_b:
        video_a_path, video_b_path = video_b_path, video_a_path
        duration_a, duration_b = duration_b, duration_a
        fps_a, fps_b = fps_b, fps_a
        info_a, info_b = info_b, info_a

    frame_count_b = int(duration_b * fps_b)
    sample_frames = [0, frame_count_b // 2, frame_count_b - 1]

    with tempfile.TemporaryDirectory() as temp_dir:
        print(f"Using temporary directory: {temp_dir}")

        # Extract and store frames from video B
        frames_b = {}
        for i, frame_b in enumerate(sample_frames):
            time_b = frame_b / fps_b
            frame_b_path = os.path.join(temp_dir, f'frame_b_{i}.png')
            extract_frame(video_b_path, time_b, frame_b_path)
            frames_b[i] = np.array(Image.open(frame_b_path).convert('RGB'))
            os.remove(frame_b_path)  # Remove the temporary file

        # First stage: Find 10 best candidates for each metric using only the first frame
        candidates = {'psnr': [], 'ssim': [], 'ncc': []}
        total_frames = int(duration_a * fps_a) - frame_count_b + 1

        with av.open(video_a_path) as container:
            stream = container.streams.video[0]
            # stream.codec_context.skip_frame = 'NONKEY'
            frame_idx = 0

            for frame in container.decode(video=0):
                if frame_idx >= total_frames:
                    break
                if frame_idx % 100 == 0:
                    print(f"Processing frame {frame_idx}/{total_frames} in Video A (First stage)")
                frame_idx += 1

                img_a = frame.to_ndarray(format='rgb24')

                similarities = compute_similarities(img_a, frames_b[0])

                for metric, value in similarities.items():
                    candidates[metric].append((frame_idx, value))

        # Sort candidates for each metric
        for metric in candidates:
            if metric == 'psnr':
                candidates[metric].sort(key=lambda x: x[1])  # Lower is better for PSNR
            else:
                candidates[metric].sort(key=lambda x: x[1], reverse=True)  # Higher is better for SSIM
            candidates[metric] = candidates[metric][:10]  # Keep top 10 candidates

        # Second stage: Evaluate top candidates for each metric
        best_alignments = {'psnr': None, 'ssim': None}
        best_similarity_sums = {'psnr': float('inf'), 'ssim': float('-inf')}

        for metric in ['psnr', 'ssim']:
            print(f"\nProcessing top candidates for {metric.upper()}")
            for idx, (start_frame, _) in enumerate(candidates[metric]):
                print(f"Processing candidate {idx + 1}/10: frame {start_frame} in Video A (Second stage)")
                similarity_sum = 0
                for i, frame_b in enumerate(sample_frames):
                    time_a = (start_frame + frame_b) / fps_a

                    frame_a_path = os.path.join(temp_dir, f'frame_a_{i}.png')
                    extract_frame(video_a_path, time_a, frame_a_path)  # we need random access so use extract_frame

                    img_a = np.array(Image.open(frame_a_path).convert('RGB'))

                    if metric == 'psnr':
                        similarity = compute_psnr(img_a, frames_b[i])
                    elif metric == 'ssim':
                        similarity = compute_ssim(img_a, frames_b[i])

                    similarity_sum += similarity

                    os.remove(frame_a_path)

                if (metric == 'psnr' and similarity_sum < best_similarity_sums[metric]) or \
                        (metric != 'psnr' and similarity_sum > best_similarity_sums[metric]):
                    best_similarity_sums[metric] = similarity_sum
                    best_alignments[metric] = start_frame / fps_a
                    print(f"New best alignment using {metric.upper()} (score: {best_similarity_sums[metric]}): Video B starts at {best_alignments[metric]:.2f} seconds (frame: {start_frame}) in Video A")

    return best_alignments


def create_composite_video(video_a_path, video_b_path, alignment_time, output_path):
    info_b = get_video_info(video_b_path)
    duration_b = float(info_b['format']['duration'])

    # Set a high bitrate value (e.g., 60M for 60 Mbps)
    high_bitrate = '60M'

    cmd = [
        'ffmpeg',
        '-i', video_a_path,
        '-i', video_b_path,
        '-filter_complex',
        f'[0:v]trim=start={alignment_time}:duration={duration_b},setpts=PTS-STARTPTS[a];'
        f'[1:v]setpts=PTS-STARTPTS[b];'
        f'[a][b]hstack=inputs=2[out]',
        '-map', '[out]',
        '-t', str(duration_b),
        '-b:v', high_bitrate,
        output_path
    ]
    subprocess.run(cmd, check=True)


if __name__ == '__main__':
    # parse command line arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('video_a_path', nargs='?', help='Path to the first video file', default='./long_high_quality.mp4')
    parser.add_argument('video_b_path', nargs='?', help='Path to the second video file', default='./short_compressed.mp4')
    parser.add_argument('--output_dir', help='Path to the output composite video file', default='.')
    parser.add_argument('--output_filename', help='Base name of the output composite video file',
                        default='aligned_composite_')
    parser.add_argument('--metrics', nargs='+', default=['psnr', 'ssim'], help='Metrics to use for video alignment')
    parser.add_argument('--verbose', action='store_true', help='Print verbose output')
    args = parser.parse_args()

    video_a_path = args.video_a_path
    video_b_path = args.video_b_path
    print(f"Video A: {video_a_path}")
    print(f"Video B: {video_b_path}")
    print("Aligning videos using metrics: {}".format(', '.join(args.metrics)))
    print("Output directory: {}".format(args.output_dir))
    print("Output base filename: {}".format(args.output_filename))

    alignment_times = align_videos(video_a_path, video_b_path)

    for metric, alignment_time in alignment_times.items():
        if alignment_time is not None:
            print(f"Best alignment using {metric.upper()}: Video B starts at {alignment_time:.2f} seconds in Video A")
            output_path = args.output_dir + os.sep + args.output_filename + metric + '.mp4'
            if os.path.exists(output_path):
                os.remove(output_path)
            create_composite_video(video_a_path, video_b_path, alignment_time, output_path)
            print(f"Composite video created: {output_path}")
        else:
            print(f"Video alignment failed using {metric.upper()}")
