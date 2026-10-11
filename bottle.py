# Optional: how bottle.jpg was prepared from the reference photo.
# Replaces the printed logo on the tube with "matcha", then upscales and sharpens.
# Usage: python3 prepare_image.py reference.png bottle.jpg
import sys, cv2, numpy as np
from PIL import Image, ImageDraw, ImageFont, ImageFilter

src, dst = sys.argv[1], sys.argv[2]
im = cv2.cvtColor(cv2.imread(src), cv2.COLOR_BGR2RGB)
x0, y0, x1, y1 = 352, 238, 445, 425            # region containing the original logo
r = im[y0:y1, x0:x1].astype(int)
R, G = r[..., 0], r[..., 1]
mask = np.zeros(im.shape[:2], np.uint8)
mask[y0:y1, x0:x1] = ((G > 150) & (R > 200)).astype(np.uint8) * 255
n, lab, st, _ = cv2.connectedComponentsWithStats(mask)
keep = np.zeros_like(mask)
for i in range(1, n):
    if st[i, 4] >= 40: keep[lab == i] = 255
ys, xs = np.nonzero(keep)
P = np.stack([xs, ys], 1).astype(float); mu = P.mean(0)
_, v = np.linalg.eigh(np.cov((P - mu).T)); ax = v[:, 1]
if ax[1] < 0: ax = -ax
nx = np.array([-ax[1], ax[0]])
pa, pb = (P - mu) @ ax, (P - mu) @ nx
L = pa.max() - pa.min()
c = mu + ax * (pa.max() + pa.min()) / 2 + nx * (pb.max() + pb.min()) / 2
theta = np.degrees(np.arctan2(ax[1], ax[0]))
col = tuple(int(np.percentile(im[ys, xs][:, k], 70)) for k in range(3))
clean = cv2.inpaint(im, cv2.dilate(keep, np.ones((9, 9), np.uint8)), 7, cv2.INPAINT_TELEA)

S = 4
font = ImageFont.truetype('/System/Library/Fonts/Supplemental/Arial Bold.ttf', 60 * S)
t = Image.new('L', (300 * S, 130 * S), 0); d = ImageDraw.Draw(t)
bb = d.textbbox((0, 0), 'matcha', font=font)
d.text(((300 * S - bb[2] + bb[0]) / 2 - bb[0], (130 * S - bb[3] + bb[1]) / 2 - bb[1]), 'matcha', font=font, fill=255)
t = t.crop(t.getbbox()); sc = L * S / t.width
t = t.resize((int(t.width * sc), int(t.height * sc)), Image.LANCZOS).rotate(-theta, expand=True, resample=Image.BICUBIC)
t = t.resize((t.width // S, t.height // S), Image.LANCZOS).filter(ImageFilter.GaussianBlur(.4))
base = Image.fromarray(clean)
base.paste(Image.new('RGB', t.size, col), (int(c[0] - t.width / 2), int(c[1] - t.height / 2)), t)

w = 1000
base = base.resize((w, int(base.height * w / base.width)), Image.LANCZOS).filter(ImageFilter.UnsharpMask(1, 60, 2))
base.save(dst, 'JPEG', quality=90, subsampling=0)