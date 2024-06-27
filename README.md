Description: This script aligns two videos by finding the best alignment point in the first video
              using the peak signal-to-noise ratio (PSNR) and structural similarity index (SSIM) metrics.
              The aligned videos are then composited side-by-side into a single video.
Usage: python video_align.py video_a_path video_b_path [--output_dir OUTPUT_DIR] [--output_filename OUTPUT_FILENAME]
        [--metrics METRICS [METRICS ...]] [--verbose]
Example: python video_align.py long_high_quality.mp4 short_compressed.mp4 --output_dir . --output_filename aligned_composite_ --metrics psnr ssim --verbose
