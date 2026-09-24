"""Renders the inspection report as scanned pages.

Deterministic: a fixed RNG seed, a bundled font where one is found, and no
timestamps in the output. Re-running it produces byte-identical files, which is
what lets the fixture manifest pin a sha-256 that stays true.

Why render rather than photograph something: the pages have to be
nonconfidential (PS-K) and their ground truth has to be exactly known, because
the whole point is grading an OCR result against it. A real scan of a real
report gives neither.

The degradation is the part that matters. A clean render of text is not a test
of OCR -- it is a test of nothing, because the model reads it perfectly and the
pipeline's handling of uncertainty is never exercised. So each page carries
rotation, gaussian noise, a bleed-through ghost and a slight blur, and page 3
carries a region degraded past legibility on purpose.

    python fixtures/agent-system/v1/sources/scan/make-scans.py
"""

import hashlib
import random
from pathlib import Path

from PIL import Image, ImageDraw, ImageFilter, ImageFont

HERE = Path(__file__).resolve().parent
SEED = 26117  # The problem statement number. Arbitrary, and written down.

WIDTH, HEIGHT = 1700, 2200  # ~200 dpi, A4-ish
MARGIN = 150


def font(size, bold=False):
    candidates = [
        r"C:\Windows\Fonts\georgiab.ttf" if bold else r"C:\Windows\Fonts\georgia.ttf",
        r"C:\Windows\Fonts\timesbd.ttf" if bold else r"C:\Windows\Fonts\times.ttf",
        "/usr/share/fonts/truetype/dejavu/DejaVuSerif.ttf",
    ]
    for path in candidates:
        if Path(path).exists():
            return ImageFont.truetype(path, size)
    return ImageFont.load_default(size)


PAGES = [
    (
        "page-01-readings.png",
        [
            ("h1", "ULTRASONIC THICKNESS INSPECTION"),
            ("h2", "Vessel PV-2201  |  Crude Distillation Unit"),
            ("rule", ""),
            ("p", "Inspection date:  2026-08-12"),
            ("p", "Inspector:  P. Shetty"),
            ("p", "Vessel:  horizontal separator, hydrocarbon service"),
            ("p", "Commissioned:  2009"),
            ("p", "Nominal wall thickness:  12.0 mm"),
            ("gap", ""),
            ("h2", "READINGS - LOWER SHELL COURSE"),
            ("gap", ""),
            ("row", "Point          Thickness (mm)"),
            ("rule", ""),
            ("row", "A                        9.4"),
            ("row", "B                        8.9"),
            ("row", "C                        8.2"),
            ("row", "D                        9.1"),
            ("rule", ""),
            ("gap", ""),
            ("p", "Four ultrasonic readings were taken on the lower shell"),
            ("p", "course. The governing (lowest) measurement is 8.2 mm."),
            ("gap", ""),
            ("p", "The governing measurement is below the minimum allowable"),
            ("p", "thickness for hydrocarbon service."),
            ("gap", ""),
            ("p", "Previous inspection 2024-03-18: governing reading 9.6 mm"),
            ("p", "at point C, above the minimum allowable at that time."),
        ],
    ),
    (
        "page-02-condition.png",
        [
            ("h1", "CONDITION OF THE SHELL COURSE"),
            ("h2", "Vessel PV-2201  |  page 2 of 3"),
            ("rule", ""),
            ("p", "The lower shell course shows general thinning consistent"),
            ("p", "with the service history of the vessel."),
            ("gap", ""),
            ("p", "EXTERNAL PITTING is present over roughly one third of"),
            ("p", "the course, deepest adjacent to point C where the"),
            ("p", "governing reading was taken. No through-wall"),
            ("p", "penetration was observed."),
            ("gap", ""),
            ("p", "INTERNAL INSPECTION WAS NOT PERFORMED on this visit."),
            ("p", "The vessel was not opened, so internal surface condition"),
            ("p", "is inferred from the ultrasonic readings alone and from"),
            ("p", "the previous internal inspection dated 2024-03-18."),
            ("gap", ""),
            ("h2", "INSULATION AND COATING"),
            ("p", "Insulation was removed at four points and reinstated."),
            ("p", "The vapour barrier was intact at three of four points."),
            ("p", "At point C the barrier had failed, consistent with the"),
            ("p", "pitting observed there, and is the most likely route for"),
            ("p", "corrosion under insulation."),
            ("gap", ""),
            ("p", "Coating over the inspected area is degraded and should"),
            ("p", "be renewed at the next opportunity."),
        ],
    ),
    (
        "page-03-signature-illegible.png",
        [
            ("h1", "NOTES FOR THE ASSESSOR"),
            ("h2", "Vessel PV-2201  |  page 3 of 3"),
            ("rule", ""),
            ("p", "This report records measurements only. It does not"),
            ("p", "constitute a fitness-for-service assessment, and no"),
            ("p", "determination of remaining life has been made here."),
            ("gap", ""),
            ("p", "Assessment against the applicable SOP revision is the"),
            ("p", "responsibility of Technical Services."),
            ("gap", ""),
            ("gap", ""),
            ("h2", "INSPECTOR SIGNATURE"),
            ("smudge", "P. Shetty   NDT Level II   Cert 4471-B"),
            ("gap", ""),
            ("p", "Countersigned:  ____________________"),
        ],
    ),
]


def render(name, blocks, rng):
    image = Image.new("L", (WIDTH, HEIGHT), 250)
    draw = ImageDraw.Draw(image)
    f_h1, f_h2, f_p, f_row = font(58, True), font(40, True), font(36), font(36)

    y = MARGIN
    smudge_box = None
    for kind, text in blocks:
        if kind == "gap":
            y += 34
        elif kind == "rule":
            draw.line([(MARGIN, y + 10), (WIDTH - MARGIN, y + 10)], fill=90, width=3)
            y += 40
        elif kind == "h1":
            draw.text((MARGIN, y), text, font=f_h1, fill=20)
            y += 84
        elif kind == "h2":
            draw.text((MARGIN, y), text, font=f_h2, fill=35)
            y += 62
        elif kind == "row":
            draw.text((MARGIN + 40, y), text, font=f_row, fill=30)
            y += 54
        elif kind == "smudge":
            # Drawn, then destroyed. The text is really there and really cannot
            # be read, which is what "unreadable" has to mean for the fail-03
            # case to be testing anything.
            draw.text((MARGIN, y), text, font=f_p, fill=40)
            smudge_box = (MARGIN - 20, y - 15, WIDTH - MARGIN, y + 60)
            y += 64
        else:
            draw.text((MARGIN, y), text, font=f_p, fill=30)
            y += 52

    if smudge_box:
        region = image.crop(smudge_box).filter(ImageFilter.GaussianBlur(7))
        image.paste(region, smudge_box[:2])
        over = ImageDraw.Draw(image)
        for _ in range(240):
            x = rng.randint(smudge_box[0], smudge_box[2])
            yy = rng.randint(smudge_box[1], smudge_box[3])
            over.ellipse(
                [x, yy, x + rng.randint(4, 16), yy + rng.randint(4, 16)],
                fill=rng.randint(120, 215),
            )

    # Bleed-through from the reverse side, which is what makes a real scan of a
    # double-sided page harder than a render of one.
    ghost = image.transpose(Image.FLIP_LEFT_RIGHT).point(lambda v: min(255, v + 34))
    image = Image.blend(image, ghost, 0.06)

    pixels = image.load()
    for _ in range(int(WIDTH * HEIGHT * 0.02)):
        x, yy = rng.randrange(WIDTH), rng.randrange(HEIGHT)
        pixels[x, yy] = max(0, min(255, pixels[x, yy] + rng.randint(-40, 40)))

    image = image.rotate(rng.uniform(-0.8, 0.8), resample=Image.BICUBIC, fillcolor=250)
    image = image.filter(ImageFilter.GaussianBlur(0.45))

    out = HERE / name
    image.convert("L").save(out, "PNG", optimize=True)
    return out


def main():
    rng = random.Random(SEED)
    written = []
    for name, blocks in PAGES:
        path = render(name, blocks, rng)
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        written.append((path.name, path.stat().st_size, digest))
        print(f"{path.name}  {path.stat().st_size:>8} bytes  sha256 {digest}")
    return written


if __name__ == "__main__":
    main()
