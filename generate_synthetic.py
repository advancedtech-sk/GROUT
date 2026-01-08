import os
import random
import numpy as np
import cv2
from scipy.spatial import Voronoi
from tqdm import tqdm


# ==================== HELPER FUNCTIONS ====================

def get_base_theme():
    """Returns a base color (BGR) and variance."""
    themes = [
        ((220, 220, 220), 15),  # White/Grey
        ((180, 160, 140), 25),  # Beige/Stone
        ((100, 100, 110), 15),  # Slate
        ((90, 60, 50), 20),     # Terracotta
    ]
    return random.choice(themes)


def get_jittered_color(base_color, variance):
    """Returns a color close to base."""
    if variance == 0:
        return base_color  # No jitter for monochrome mode
    b, g, r = base_color
    noise = np.random.randint(-variance, variance, 3)
    return (
        int(np.clip(b + noise[0], 0, 255)),
        int(np.clip(g + noise[1], 0, 255)),
        int(np.clip(r + noise[2], 0, 255))
    )


def draw_wobbly_line(img, pt1, pt2, color, thickness, segments=4):
    """Draws a jagged, imperfect line adapted for very thin grout."""
    pt1 = np.array(pt1)
    pt2 = np.array(pt2)
    vec = pt2 - pt1
    dist = np.linalg.norm(vec)

    if dist < 10:  # Too short to wobble
        cv2.line(img, tuple(pt1), tuple(pt2), color, thickness)
        return

    points = [pt1]
    for i in range(1, segments):
        alpha = i / segments
        p_base = pt1 + vec * alpha
        # Perpendicular jitter
        perp = np.array([-vec[1], vec[0]])
        perp = perp / (np.linalg.norm(perp) + 1e-6)

        # Reduced jitter for thinner lines (0.5 to 1.0 px)
        jitter = perp * random.uniform(-0.8, 0.8)
        points.append((p_base + jitter).astype(int))
    points.append(pt2)

    for i in range(len(points) - 1):
        # Very subtle thickness variation (e.g. 1px vs 2px)
        t_jitter = thickness
        if thickness > 1 and random.random() > 0.7:
            t_jitter = max(1, thickness + random.randint(-1, 0))

        cv2.line(img, tuple(points[i]), tuple(points[i + 1]), color, t_jitter)


def draw_line_robust(img, pt1, pt2, color, thickness, is_wobbly=True):
    """
    Decides whether to draw a 'Nasty' wobbly line or a 'Modern' perfect line.
    """
    if is_wobbly:
        draw_wobbly_line(img, pt1, pt2, color, thickness)
    else:
        # PERFECT STRAIGHT LINE (Anti-aliased)
        cv2.line(img, tuple(pt1), tuple(pt2), color, thickness, cv2.LINE_AA)


def add_dirt_overlay(img):
    """Adds global dirt to disguise clean edges."""
    h, w = img.shape[:2]
    overlay = np.zeros((h // 2, w // 2), dtype=np.uint8)
    cv2.randn(overlay, 128, 40)
    overlay = cv2.resize(overlay, (w, h), interpolation=cv2.INTER_CUBIC)
    dirt_mask = overlay < 110
    img = img.astype(np.int16)
    img[dirt_mask] -= 25
    return np.clip(img, 0, 255).astype(np.uint8)


# ==================== GEOMETRY GENERATORS ====================

def generate_running_bond(img_size, row_height=30):
    """
    Generates points in parallel rows with offset vertical joints (Brick/Running Bond).
    Creates the "straight parallel lines divided by short lines" pattern.
    """
    points = []

    # Rotate the whole system randomly
    angle = random.uniform(0, 180)
    rad = np.deg2rad(angle)
    c, s = np.cos(rad), np.sin(rad)
    rot_mat = np.array(((c, -s), (s, c)))

    # Generate on a larger canvas to avoid empty corners after rotation
    oversize = int(img_size * 1.5)
    start, end = -oversize // 2, oversize // 2

    for y in range(start, end, row_height):
        # Randomize row starting offset (The "Brick" effect)
        row_offset = random.randint(0, row_height)

        # Decide tile width for this row
        tile_width = random.randint(20, 45)

        for x in range(start, end, tile_width):
            # Base point
            px = x + row_offset
            py = y

            # Add slight jitter so lines aren't laser-perfect
            # Jitter Y less than X to preserve the "Parallel Lines" look
            jx = px + random.randint(-2, 2)
            jy = py + random.randint(-1, 1)

            # Rotate
            pt = np.dot(rot_mat, np.array([jx, jy])) + [img_size / 2, img_size / 2]

            if -50 <= pt[0] <= img_size + 50 and -50 <= pt[1] <= img_size + 50:
                points.append([int(pt[0]), int(pt[1])])

    return np.array(points, dtype=np.int32)


def generate_vermiculatum(img_size):
    """
    Generates concentric circles with RANDOM centers and varied tile sizes.
    Can produce fine detail (8-15px) or coarse tiles (20-40px).
    """
    points = []

    # 1. Randomize Center (Can be anywhere in image)
    center_x = random.randint(50, img_size - 50)
    center_y = random.randint(50, img_size - 50)
    center = np.array([center_x, center_y])

    # 2. Decide Size Scale for this image
    # "Fine" = tiny tiles (8-15px), "Coarse" = big tiles (20-40px)
    is_fine_detail = random.random() < 0.5

    # Start from center and move out
    current_radius = random.randint(5, 20)

    # Go way past corners to ensure coverage if center is off-screen
    max_radius = int(img_size * 1.5)

    while current_radius < max_radius:
        # Determine Row Height (Radial thickness of the tile)
        if is_fine_detail:
            row_height = random.randint(8, 16)
        else:
            row_height = random.randint(18, 35)

        # Circumference at this radius
        circumference = 2 * np.pi * current_radius

        # Determine Tile Width (Arc length)
        tile_width = max(5, int(row_height * random.uniform(0.8, 1.2)))

        num_tiles = int(circumference / tile_width)
        if num_tiles == 0:
            num_tiles = 1

        # Angle step
        theta_step = 2 * np.pi / num_tiles

        # Offset angle (Stagger joints)
        angle_offset = random.uniform(0, np.pi / 4)

        for i in range(num_tiles):
            theta = i * theta_step + angle_offset

            # Polar to Cartesian
            x = center[0] + current_radius * np.cos(theta)
            y = center[1] + current_radius * np.sin(theta)

            # Add slight noise (Hand-laid imperfection)
            jx = x + random.uniform(-1.5, 1.5)
            jy = y + random.uniform(-1.5, 1.5)

            # Only keep points relevant to the image + buffer
            if -50 <= jx <= img_size + 50 and -50 <= jy <= img_size + 50:
                points.append([int(jx), int(jy)])

        # Move to next ring
        current_radius += row_height

    return np.array(points, dtype=np.int32)


# ==================== MOSAIC GENERATOR ====================

def generate_mosaic(index, output_dir, img_size=512):
    """
    V13 Mosaic Generator - Safe masks with thicker training targets.

    Geometry modes:
    - 40% Running Bond (Parallel lines with offset joints)
    - 30% Opus Vermiculatum (Concentric circles, random center)
    - 30% Voronoi (Random, high density support)

    Features:
    - 15% chance of monochrome tiles
    - Adaptive grout color (contrast against tiles)
    - SEPARATE THICKNESS: Image (realistic 1-4px) vs Mask (min 2px for training stability)
    """
    img_dir = os.path.join(output_dir, "images")
    mask_dir = os.path.join(output_dir, "masks")
    os.makedirs(img_dir, exist_ok=True)
    os.makedirs(mask_dir, exist_ok=True)

    img = np.zeros((img_size, img_size, 3), dtype=np.uint8)
    mask = np.zeros((img_size, img_size), dtype=np.uint8)

    # 1. Select Geometry Mode
    mode_roll = random.random()

    if mode_roll < 0.40:
        # Running Bond
        rh = random.randint(10, 40)
        points = generate_running_bond(img_size, row_height=rh)
        is_wobbly = False

    elif mode_roll < 0.70:
        # Vermiculatum
        points = generate_vermiculatum(img_size)
        is_wobbly = True

    else:
        # Voronoi
        num_seeds = random.randint(400, 1500)
        points = np.random.randint(-50, img_size + 50, (num_seeds, 2))
        is_wobbly = True

    # 2. Generate Voronoi
    try:
        vor = Voronoi(points)
    except Exception:
        return

    # 3. Theme
    base_color, variance = get_base_theme()
    img[:] = base_color

    # Monochrome check (15%)
    is_monochrome = random.random() < 0.15
    if is_monochrome:
        variance = 0

    # Zero grout mode (10%) - no visible grout but mask still has lines
    # Forces model to learn tile boundaries from color differences alone
    is_zero_grout = not is_monochrome and random.random() < 0.10

    # 4. Fill Tiles
    if vor.regions:
        for region in vor.regions:
            if not -1 in region and len(region) > 0:
                pts = np.array([vor.vertices[i] for i in region], np.int32)
                tile_color = get_jittered_color(base_color, variance)
                cv2.fillPoly(img, [pts], tile_color)

    # 5. Draw Grout
    if vor.ridge_vertices:
        for vpair in vor.ridge_vertices:
            if vpair[0] >= 0 and vpair[1] >= 0:
                pt1 = vor.vertices[vpair[0]].astype(int)
                pt2 = vor.vertices[vpair[1]].astype(int)

                if -50 <= pt1[0] <= img_size + 50 or -50 <= pt2[0] <= img_size + 50:
                    if is_zero_grout:
                        # ZERO GROUT MODE: No visible grout, but mask has 2px lines
                        cv2.line(mask, tuple(pt1), tuple(pt2), 255, 2)
                        # Don't draw anything on image - tiles touch directly
                    else:
                        # --- NORMAL THICKNESS LOGIC ---
                        # Image thickness: Realistic (can be 1px)
                        if len(points) > 1000:  # High density
                            img_thickness = random.randint(1, 2)
                        else:
                            img_thickness = random.randint(1, 4)

                        # Mask thickness: TRAINING STABILITY (2-3px range)
                        # Dilate the mask slightly compared to the image, but cap at 3px
                        mask_thickness = min(3, max(2, img_thickness + 1))

                        # Contrast check
                        mean_val = np.mean(base_color)
                        if mean_val > 128:
                            grout_c = (random.randint(20, 60),) * 3
                        else:
                            grout_c = (random.randint(180, 220),) * 3

                        # DRAW MASK (Always thicker for training stability)
                        cv2.line(mask, tuple(pt1), tuple(pt2), 255, mask_thickness)

                        # DRAW IMAGE (Realistic thin lines)
                        draw_line_robust(img, pt1, pt2, grout_c, img_thickness, is_wobbly=is_wobbly)

    # 6. Realism
    noise = np.random.normal(0, 5, (img_size, img_size, 3)).astype(np.int16)
    img = np.clip(img.astype(np.int16) + noise, 0, 255).astype(np.uint8)

    if is_wobbly:
        img = add_dirt_overlay(img)
        img = cv2.GaussianBlur(img, (3, 3), 0)
    else:
        img = cv2.GaussianBlur(img, (1, 1), 0)

    # Save
    cv2.imwrite(f"{img_dir}/syn_{index:05d}.jpg", img)
    cv2.imwrite(f"{mask_dir}/syn_{index:05d}.png", mask)


# ==================== MAIN ====================

if __name__ == "__main__":
    print("Generating 1,500 V13 Images (Safe Masks)...")
    for i in tqdm(range(1500)):
        generate_mosaic(i, "data_local")
