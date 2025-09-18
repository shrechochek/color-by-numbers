#!/usr/bin/env python3
import os
import math
import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

# ------------------ CONFIG: настройте параметры здесь ------------------
INPUT_FILENAME = "test.jpg"     # имя вашего файла (положите его в ту же папку)
OUTPUT_SHEET = "color_by_numbers_sheet.png"
OUTPUT_PALETTE = "palette.png"
OUTPUT_PREVIEW = "colorized_preview.png"

NUM_COLORS = 20
BLUR_KERNEL = 5
KMEANS_ATTEMPTS = 4
SMOOTHING_ITER = 2
MIN_REGION_AREA = 400
OUTLINE_THICKNESS = 2
FONT_PATH = None
FONT_SIZE = 20
IMAGE_MAX_DIM = 1600
DOWNSCALE_FOR_COMPUTE = True
# -----------------------------------------------------------------------

def load_image(path):
    img = cv2.imread(path, cv2.IMREAD_COLOR)
    if img is None:
        raise FileNotFoundError(f"Не могу открыть файл: {path}")
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

def resize_if_needed(img, max_dim):
    h, w = img.shape[:2]
    scale = 1.0
    if max(h, w) > max_dim:
        scale = max_dim / max(h, w)
        new_size = (int(w*scale), int(h*scale))
        img = cv2.resize(img, new_size, interpolation=cv2.INTER_AREA)
    return img, scale

def quantize_colors(img, k=8, blur_kernel=0, attempts=4):
    work = img.copy()
    if blur_kernel and blur_kernel >= 3:
        if blur_kernel % 2 == 0: blur_kernel += 1
        work = cv2.GaussianBlur(work, (blur_kernel, blur_kernel), 0)
    data = work.reshape((-1,3)).astype(np.float32)
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 40, 0.1)
    ret, labels, centers = cv2.kmeans(data, k, None, criteria, attempts, cv2.KMEANS_PP_CENTERS)
    centers = np.clip(centers.astype(np.uint8), 0, 255)
    labels = labels.flatten()
    quant = centers[labels].reshape(img.shape)
    return quant, labels.reshape((img.shape[0], img.shape[1])), centers

def remove_small_regions_by_reassignment(label_img, color_img, min_area):
    h,w = label_img.shape
    out = label_img.copy()
    k = int(out.max()) + 1
    for c in range(k):
        mask = (out == c).astype(np.uint8)
        if mask.sum() == 0:
            continue
        num, labs, stats, centroids = cv2.connectedComponentsWithStats(mask, connectivity=8)
        for comp in range(1, num):
            area = stats[comp, cv2.CC_STAT_AREA]
            if area < min_area:
                comp_mask = (labs == comp).astype(np.uint8)
                dil = cv2.dilate(comp_mask, np.ones((3,3), np.uint8), iterations=1)
                border = dil & (~comp_mask)
                neighbor_labels = out[border.astype(bool)]
                if neighbor_labels.size == 0:
                    continue
                vals, counts = np.unique(neighbor_labels, return_counts=True)
                best = vals[np.argmax(counts)]
                out[comp_mask.astype(bool)] = best
    return out

def get_text_size(draw, text, font):
    """
    Универсальная функция: пытается несколько способов получить размер текста,
    чтобы избежать зависимости от конкретной версии Pillow.
    Возвращает (width, height).
    """
    # 1) textbbox (Pillow >= 8.0)
    try:
        bbox = draw.textbbox((0,0), text, font=font)
        return bbox[2]-bbox[0], bbox[3]-bbox[1]
    except Exception:
        pass
    # 2) textsize (may exist)
    try:
        return draw.textsize(text, font=font)
    except Exception:
        pass
    # 3) font.getsize
    try:
        return font.getsize(text)
    except Exception:
        pass
    # 4) font.getbbox
    try:
        bbox = font.getbbox(text)
        return bbox[2]-bbox[0], bbox[3]-bbox[1]
    except Exception:
        pass
    # fallback (approx)
    try:
        return (len(text) * getattr(font, "size", 10), getattr(font, "size", 10))
    except Exception:
        return (10 * len(text), 10)

def create_contour_sheet(label_img, palette, outline_thickness=2, add_numbers=True, font=None):
    h,w = label_img.shape
    sheet = Image.new("RGB", (w,h), (255,255,255))
    draw = ImageDraw.Draw(sheet)
    k = int(label_img.max()) + 1
    region_centroids = {}
    for lbl in range(k):
        mask = (label_img == lbl).astype(np.uint8) * 255
        if mask.sum() == 0:
            continue
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            continue
        for cnt in contours:
            pts = [(int(p[0][0]), int(p[0][1])) for p in cnt]
            if len(pts) > 2:
                draw.line(pts + [pts[0]], fill=(0,0,0), width=outline_thickness)
        M = cv2.moments(mask)
        if M["m00"] != 0:
            cx = int(M["m10"]/M["m00"])
            cy = int(M["m01"]/M["m00"])
            region_centroids[lbl] = (cx, cy)
    if add_numbers:
        for lbl, (cx,cy) in region_centroids.items():
            text = str(lbl+1)
            tw, th = get_text_size(draw, text, font)
            draw.text((cx - tw/2, cy - th/2), text, fill=(0,0,0), font=font)
    return sheet

def create_palette_image(centers, order_indices=None, per_row=5, box_size=80, margin=10, font=None):
    if order_indices is None:
        order = list(range(len(centers)))
    else:
        order = order_indices
    n = len(order)
    cols = per_row
    rows = math.ceil(n / cols)
    width = cols * box_size + (cols+1) * margin
    height = rows * box_size + (rows+1) * margin + 40
    pal = Image.new("RGB", (width, height), (255,255,255))
    draw = ImageDraw.Draw(pal)
    for idx, oi in enumerate(order):
        r = idx // cols
        c = idx % cols
        x = margin + c * (box_size + margin)
        y = margin + r * (box_size + margin)
        color = tuple(int(v) for v in centers[oi])
        draw.rectangle([x,y, x+box_size, y+box_size], fill=color, outline=(0,0,0))
        text = str(oi+1)
        tw, th = get_text_size(draw, text, font)
        tx = x + (box_size - tw)/2
        ty = y + box_size + 4
        draw.text((tx,ty), text, fill=(0,0,0), font=font)
    return pal

def main():
    if not os.path.exists(INPUT_FILENAME):
        print(f"Файл {INPUT_FILENAME} не найден в текущей папке ({os.getcwd()}). Поместите изображение рядом со скриптом.")
        return
    img = load_image(INPUT_FILENAME)
    orig_h, orig_w = img.shape[:2]

    work_img, scale = resize_if_needed(img, IMAGE_MAX_DIM) if DOWNSCALE_FOR_COMPUTE else (img.copy(), 1.0)
    print(f"Исходный размер: {orig_w}x{orig_h}. Использую для вычислений: {work_img.shape[1]}x{work_img.shape[0]} (масштаб {scale:.3f})")

    quant_img, labels2d, centers = quantize_colors(work_img, k=NUM_COLORS, blur_kernel=BLUR_KERNEL, attempts=KMEANS_ATTEMPTS)

    for i in range(SMOOTHING_ITER):
        labels2d = cv2.medianBlur(labels2d.astype(np.uint8), 3)

    labels2d = remove_small_regions_by_reassignment(labels2d, quant_img, MIN_REGION_AREA)

    if DOWNSCALE_FOR_COMPUTE and scale != 1.0:
        labels_full = cv2.resize(labels2d.astype(np.int32), (orig_w, orig_h), interpolation=cv2.INTER_NEAREST)
    else:
        labels_full = labels2d

    h,w = labels_full.shape
    preview = np.zeros((h,w,3), dtype=np.uint8)
    for i, c in enumerate(centers):
        preview[labels_full == i] = c
    preview_pil = Image.fromarray(preview)
    preview_pil.save(OUTPUT_PREVIEW)
    print(f"Сохранён предварительный colorized preview: {OUTPUT_PREVIEW}")

    try:
        if FONT_PATH:
            font = ImageFont.truetype(FONT_PATH, FONT_SIZE)
        else:
            font = ImageFont.load_default()
    except Exception:
        font = ImageFont.load_default()

    sheet = create_contour_sheet(labels_full, centers, outline_thickness=OUTLINE_THICKNESS, add_numbers=True, font=font)
    sheet.save(OUTPUT_SHEET)
    print(f"Сохранён лист раскраски: {OUTPUT_SHEET}")

    counts = [(i, int((labels_full==i).sum())) for i in range(len(centers))]
    counts.sort(key=lambda x: -x[1])
    order = [i for i,_ in counts if _>0] + [i for i,_ in counts if _==0]
    palette = create_palette_image(centers, order_indices=order, per_row=5, box_size=80, margin=10, font=font)
    palette.save(OUTPUT_PALETTE)
    print(f"Сохранена палитра: {OUTPUT_PALETTE}")
    print("Готово. Откройте файлы и проверьте результат.")

if __name__ == "__main__":
    main()
