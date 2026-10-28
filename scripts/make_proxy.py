"""Draw a correctly scaled stand-in of the seated young woman on the middle seat of the near bench.

Geometry measured on work/plates/plate_1536.png (1536x864): seat front edge (847,427)->(987,657), seat back edge
(967,443)->(1153,643), ground ~120-130 px below the seat (seat height 0.45 m => ~270 px/m at the middle seat).
The stand-in is refined into a person by make_start.py (inpainting at partial strength), which keeps its
position and size.
"""
import sys

from PIL import Image, ImageDraw, ImageFilter

plate = Image.open(sys.argv[1]).convert("RGB")
out_img, out_mask = sys.argv[2], sys.argv[3]
im = plate.copy()
d = ImageDraw.Draw(im)
RED, RED2, BLACK, SKIN, WHITE, HAIR = (165, 28, 32), (125, 20, 24), (25, 25, 28), (214, 170, 140), (236, 234, 228), (40, 30, 26)

# lower legs (black jeans) from knees at the seat front edge down to the ground, feet forward (left)
d.polygon([(905, 548), (930, 552), (902, 652), (880, 650)], fill=BLACK)
d.polygon([(925, 560), (948, 566), (918, 662), (897, 658)], fill=BLACK)
d.ellipse([862, 642, 906, 662], fill=WHITE)          # trainers
d.ellipse([880, 652, 924, 672], fill=WHITE)
# thighs + seat of coat: from hips (back of seat) forward to knees
d.polygon([(1035, 520), (1050, 560), (950, 575), (900, 552), (912, 528)], fill=RED)
# torso, leaning slightly forward over the phone
d.polygon([(960, 392), (1012, 384), (1045, 455), (1046, 540), (965, 548), (942, 470)], fill=RED)
# arms down to the hands on her lap
d.polygon([(948, 405), (968, 400), (962, 470), (938, 492), (925, 482)], fill=RED2)
d.polygon([(1000, 395), (1022, 400), (1000, 470), (962, 498), (955, 482)], fill=RED2)
d.ellipse([925, 478, 950, 500], fill=SKIN)           # hands holding phone
d.rectangle([930, 470, 946, 486], fill=(20, 20, 24))   # phone
d.rectangle([955, 505, 1000, 538], fill=BLACK)        # bag on lap
d.line([(1000, 390), (975, 505)], fill=BLACK, width=3)  # bag strap
# head bowed forward: dark hair at back, face below, white beanie on top
d.ellipse([952, 330, 1000, 386], fill=SKIN)
d.ellipse([968, 338, 1008, 392], fill=HAIR)
d.chord([948, 316, 1006, 368], 180, 360, fill=WHITE)
d.rectangle([949, 338, 1005, 350], fill=WHITE)
d.ellipse([990, 378, 1008, 398], fill=HAIR)          # low ponytail
# soft shadow on the ground in front (sun from behind the bench, shadows fall left-down)
sh = Image.new("L", im.size, 0)
ImageDraw.Draw(sh).polygon([(870, 650), (930, 665), (820, 760), (760, 740)], fill=90)
sh = sh.filter(ImageFilter.GaussianBlur(10))
im = Image.composite(Image.new("RGB", im.size, (40, 40, 44)), im, sh)
im.save(out_img)

mask = Image.new("L", im.size, 0)
ImageDraw.Draw(mask).polygon([(940, 300), (1030, 300), (1065, 450), (1068, 575), (960, 600), (935, 690),
                              (850, 690), (845, 620), (880, 540), (915, 400)], fill=255)
mask = mask.filter(ImageFilter.MaxFilter(21))
mask.save(out_mask)
print("ok")
