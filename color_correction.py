"""
Color Correction - Gray World Algorithm

The "Gray World" assumption: on average, a well-lit natural scene should
average out to gray (equal R, G, B). If the average color is skewed
(e.g. too blue, too warm), that skew is treated as a color cast and
corrected by scaling each channel so the three channel averages match.

Usage:
    python color_correction.py
Edit INPUT_IMAGE below to point at the image you want to correct.
"""

import cv2
import numpy as np

INPUT_IMAGE = "input.png"
OUTPUT_IMAGE = "output_color_corrected.png"


def gray_world_correction(image):
    """
    Apply Gray World white balance correction.

    Steps:
    1. Split the image into B, G, R channels.
    2. Compute the average intensity of each channel.
    3. Compute the overall average across all channels (the "gray" target).
    4. Scale each channel so its average matches the overall gray average.
    """
    image = image.astype(np.float32)

    b, g, r = cv2.split(image)

    b_avg = np.mean(b)
    g_avg = np.mean(g)
    r_avg = np.mean(r)

    gray_avg = (b_avg + g_avg + r_avg) / 3.0

    # Scale factor per channel: push each channel's average toward gray_avg
    b_gain = gray_avg / b_avg
    g_gain = gray_avg / g_avg
    r_gain = gray_avg / r_avg

    b = b * b_gain
    g = g * g_gain
    r = r * r_gain

    corrected = cv2.merge([b, g, r])

    # Clip back to valid 0-255 range and convert back to uint8
    corrected = np.clip(corrected, 0, 255).astype(np.uint8)

    return corrected


def main():
    image = cv2.imread(INPUT_IMAGE, cv2.IMREAD_COLOR)
    if image is None:
        raise FileNotFoundError(f"Could not read {INPUT_IMAGE}")

    corrected = gray_world_correction(image)

    cv2.imwrite(OUTPUT_IMAGE, corrected)
    print(f"Saved color-corrected image: {OUTPUT_IMAGE}")


if __name__ == "__main__":
    main()
