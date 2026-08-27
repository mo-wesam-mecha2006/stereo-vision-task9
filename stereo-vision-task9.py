"""
Task 9.1 - Stereo Vision: Disparity Map, Depth Estimation, 3D Point Cloud

Dataset format (Middlebury-style):
    data/<scene_name>/im0.png      -> left image
    data/<scene_name>/im1.png      -> right image
    data/<scene_name>/calib.txt    -> calibration info, e.g.:
        cam0=[fx 0 cx0; 0 fy cy0; 0 0 1]
        cam1=[fx 0 cx1; 0 fy cy1; 0 0 1]
        doffs=...      (difference between cx0 and cx1)
        baseline=...   (mm)
        width=...
        height=...
        ndisp=...      (suggested max disparity)

How to run:
    python stereo_vision.py
This processes every scene folder found under DATA_DIR automatically.
"""

import os
import re

import cv2
import numpy as np

# ============================================================
# CONFIG
# ============================================================

DATA_DIR = "data"          # each subfolder here is one scene (im0.png, im1.png, calib.txt)
OUTPUT_DIR = "output"       # where disparity maps + point clouds get saved

BLOCK_SIZE = 15             # must be odd

# Pixel to report depth for, as a fraction of image width/height (so it works
# for any resolution). (0.5, 0.5) = center of the image.
CHECK_PIXEL_FRACTION = (0.5, 0.5)


# ============================================================
# STEP 1: Parse calib.txt (Middlebury format)
# ============================================================

def parse_calib(path):
    """
    Reads a Middlebury-style calib.txt and pulls out the numbers we need:
    fx, fy, cx (of the left camera), cy, doffs, baseline, width, height, ndisp.
    """
    with open(path) as f:
        text = f.read()

    cam0 = re.search(r"cam0=\[([^\]]+)\]", text).group(1)
    nums = [float(x) for x in re.split(r"[\s;]+", cam0.strip()) if x]
    fx, cx0, fy, cy0 = nums[0], nums[2], nums[4], nums[5]

    doffs = float(re.search(r"doffs=([\d.]+)", text).group(1))
    baseline = float(re.search(r"baseline=([\d.]+)", text).group(1))
    width = int(re.search(r"width=(\d+)", text).group(1))
    height = int(re.search(r"height=(\d+)", text).group(1))
    ndisp = int(re.search(r"ndisp=(\d+)", text).group(1))

    return {
        "fx": fx, "fy": fy, "cx": cx0, "cy": cy0,
        "doffs": doffs, "baseline": baseline,
        "width": width, "height": height, "ndisp": ndisp,
    }


def num_disparities_from_ndisp(ndisp):
    """StereoBM needs numDisparities divisible by 16 -> round ndisp up."""
    return int(np.ceil(ndisp / 16.0) * 16)


# ============================================================
# STEP 2: Load images
# ============================================================

def load_images(scene_dir):
    left_path = os.path.join(scene_dir, "im0.png")
    right_path = os.path.join(scene_dir, "im1.png")

    left_gray = cv2.imread(left_path, cv2.IMREAD_GRAYSCALE)
    right_gray = cv2.imread(right_path, cv2.IMREAD_GRAYSCALE)
    left_color = cv2.imread(left_path, cv2.IMREAD_COLOR)

    if left_gray is None or right_gray is None:
        raise FileNotFoundError(f"Could not read images in {scene_dir}")

    return left_gray, right_gray, left_color


# ============================================================
# STEP 3: Compute disparity map using StereoBM
# ============================================================

def compute_disparity(left_gray, right_gray, num_disparities):
    stereo = cv2.StereoBM_create(numDisparities=num_disparities, blockSize=BLOCK_SIZE)
    raw_disparity = stereo.compute(left_gray, right_gray)

    # StereoBM outputs fixed-point disparity scaled by 16 -> convert back to real values
    disparity = raw_disparity.astype(np.float32) / 16.0
    return disparity


def save_disparity_images(disparity, gray_out, color_out):
    # normalize to 0-255 so it can be saved/viewed as a normal image
    disp_norm = cv2.normalize(disparity, None, 0, 255, cv2.NORM_MINMAX)
    disp_uint8 = disp_norm.astype(np.uint8)

    cv2.imwrite(gray_out, disp_uint8)

    disp_color = cv2.applyColorMap(disp_uint8, cv2.COLORMAP_JET)
    cv2.imwrite(color_out, disp_color)


# ============================================================
# STEP 4: Depth estimation at a specific pixel
# ============================================================
# Middlebury-style depth formula: Z = (fx * baseline) / (disparity + doffs)
# doffs accounts for the two cameras having different principal points (cx).

def get_depth_at_pixel(disparity, x, y, calib):
    d = disparity[y, x]
    if d <= 0:
        return None  # no valid match at this pixel

    Z = (calib["fx"] * calib["baseline"]) / (d + calib["doffs"])
    return Z  # in mm, since baseline was given in mm


# ============================================================
# STEP 5 (BONUS): Build 3D point cloud and save as .ply
# ============================================================

def build_pointcloud(disparity, left_color, calib):
    h, w = disparity.shape
    fx, fy, cx, cy = calib["fx"], calib["fy"], calib["cx"], calib["cy"]
    baseline, doffs = calib["baseline"], calib["doffs"]

    ys, xs = np.where(disparity > 0)
    d = disparity[ys, xs]

    Z = (fx * baseline) / (d + doffs)
    X = (xs - cx) * Z / fx
    Y = (ys - cy) * Z / fy

    points = np.stack([X, Y, Z], axis=1).astype(np.float32)

    bgr = left_color[ys, xs]
    colors = bgr[:, ::-1].astype(np.uint8)  # BGR -> RGB

    return points, colors


def save_ply(points, colors, path):
    with open(path, "w") as f:
        f.write("ply\n")
        f.write("format ascii 1.0\n")
        f.write(f"element vertex {len(points)}\n")
        f.write("property float x\nproperty float y\nproperty float z\n")
        f.write("property uchar red\nproperty uchar green\nproperty uchar blue\n")
        f.write("end_header\n")
        for (x, y, z), (r, g, b) in zip(points, colors):
            f.write(f"{x} {y} {z} {r} {g} {b}\n")


# ============================================================
# MAIN - process every scene folder under DATA_DIR
# ============================================================

def process_scene(scene_name, scene_dir):
    print(f"\n=== Scene: {scene_name} ===")

    calib_path = os.path.join(scene_dir, "calib.txt")
    calib = parse_calib(calib_path)
    num_disp = num_disparities_from_ndisp(calib["ndisp"])
    print(f"fx={calib['fx']}, baseline={calib['baseline']}mm, doffs={calib['doffs']}, "
          f"ndisp={calib['ndisp']} -> numDisparities={num_disp}")

    left_gray, right_gray, left_color = load_images(scene_dir)

    disparity = compute_disparity(left_gray, right_gray, num_disp)

    gray_out = os.path.join(OUTPUT_DIR, f"{scene_name}_disparity_gray.png")
    color_out = os.path.join(OUTPUT_DIR, f"{scene_name}_disparity_color.png")
    save_disparity_images(disparity, gray_out, color_out)
    print(f"Saved disparity maps: {gray_out}, {color_out}")

    h, w = disparity.shape
    px = int(w * CHECK_PIXEL_FRACTION[0])
    py = int(h * CHECK_PIXEL_FRACTION[1])
    depth = get_depth_at_pixel(disparity, px, py, calib)
    if depth is None:
        print(f"Pixel ({px}, {py}): no valid disparity")
    else:
        print(f"Estimated distance at pixel ({px}, {py}): {depth:.1f} mm ({depth/1000:.3f} m)")

    points, colors = build_pointcloud(disparity, left_color, calib)
    ply_out = os.path.join(OUTPUT_DIR, f"{scene_name}_pointcloud.ply")
    save_ply(points, colors, ply_out)
    print(f"Saved point cloud ({len(points)} points): {ply_out}")


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    scene_names = sorted(
        d for d in os.listdir(DATA_DIR)
        if os.path.isdir(os.path.join(DATA_DIR, d))
    )

    for scene_name in scene_names:
        process_scene(scene_name, os.path.join(DATA_DIR, scene_name))


if __name__ == "__main__":
    main()