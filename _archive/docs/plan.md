Key idea:
Treat this as a detection + counting + anomaly problem.
• Use YOLOv8n to detect & count:
• pina
• worker (workers)
• optionally object (things you already see: tools, big chunks of trash, etc.)
• On top of that, use a simple anomaly layer to flag “anything weird that isn’t one of the above” in defined ROIs.

Below I’ll go in order: 1. How to reduce redundant frames and what to keep. 2. How to label piñas / humans / foreign objects (and what not to do). 3. A concrete step-by-step plan (data → labeling → training → active learning → deployment).

⸻

1. Redundant frames: what to actually label

Your sample frames are extremely similar (steady cameras, slow changes). You don’t need thousands of near-identical images for YOLO; hundreds of diverse frames per viewpoint is usually enough for v1.

1.1. High-level target

From the ~12–13k frames you listed, I’d aim for something like:
• Initial labeling set: ~600–1000 frames total.
• ~60% with piñas.
• ~20% “empty / almost empty” bays (like the DIFUSOR images).
• ~20% with humans and misc stuff.

You can keep the rest untouched for later active learning.

1.2. Stronger de-duplication strategy

You already cluster with cosine distance on embeddings and pick the sharpest per cluster. Likely you’re just being too conservative with thresholds.

A practical pipeline per video: 1. Temporal thinning first
• Sample, say, 1 frame every 1–2 seconds as a base.
• You can adaptively shorten this when motion is high. 2. Motion-based filter
• Compute frame-to-frame SSIM or mean absolute difference.
• Only keep a frame if it differs by at least a small amount from the last kept one.
• Example rule:

# pseudocode

if ssim(curr, last_kept) < 0.98:
keep

    •	For your super-steady piles, this alone will nuke most duplicates.

    3.	Embedding clustering for diversity
    •	Take the remaining frames and embed with a pre-trained model (ResNet, CLIP, whatever you already used).
    •	Cluster with k-medoids or k-means for a target K per camera (e.g. 100–200).
    •	Keep only the cluster medoids → these are maximally diverse.
    4.	Per-camera quotas
    •	Enforce something like 100–150 frames per camera/angle in the first labeling round.
    •	Make sure you keep:
    •	Different fill levels (empty, mid, overflowing).
    •	Different lighting (day, night, glare).
    •	Frames with workers present in different positions.

That will give you a much smaller and richer set than just “sharpest per cluster”.

⸻

2. Labeling strategy (piñas, humans, and non-targets)

2.1. Classes

Given your requirements, I’d start with 3 classes: 1. pina 2. worker (just call it person in labels if you like) 3. object (only things you actually see for now: tools, large debris, logs, tires if they really appear, etc.)

Later, you can split object into sub-classes if needed (e.g., tire, log), but initially a single “foreign” class is simpler and matches your “alert if anything else is in there” requirement.

2.2. Bounding boxes vs segmentation

For counting and alerts, bounding boxes are enough.
• Use YOLOv8n detection (not segmentation) to keep labeling cheap.
• Segmentation would be overkill unless you need exact area estimates or pixel masks.

2.3. How to label piñas (very important)

Your heaps are dense and highly occluded. To make YOLO actually learn:

Do:
• Draw a bounding box per visible piña where you can reasonably distinguish it.
• Include partially visible piñas if at least ~30% of the object is visible and you’d want it counted by a human auditor.
• Make boxes tight but not insanely tight—some background is fine.
• Be consistent:
• If a piña is on the edge and only a tiny sliver is visible and you wouldn’t count it manually, decide that this is “uncountable” and don’t label those anywhere.

Do NOT:
• Do partial labeling on full frames (e.g., label only 20 piñas out of 200 in the same image) without an ignore mechanism.
YOLO treats unlabeled positives as background, which will actively teach the model that many piñas are “background” and hurt performance badly.

If you want to limit labeling effort while still having dense piles, you have two good options:

Option A – Crop patches and fully label per patch 1. For labeling, generate image tiles (e.g., 640×640 or 800×800) cut from the full frame. 2. In each tile, label every piña you can see following your visibility rule. 3. Ignore the rest of the frame; you don’t train on the uncropped full frame at all (so no unlabeled piñas exist in your training images).

This keeps each labeling task manageable.

Option B – “Ignore regions” (more advanced)
If you really want to annotate only part of the heap on the full frame:
• Define ignore polygons in Label Studio over the area you’re not labeling.
• Modify the YOLOv8 training pipeline to skip loss for anchors inside ignore regions.

This is doable but requires customizing the dataloader & loss. For a first project, I’d strongly recommend Option A (tiling).

2.4. Labeling humans
• Any worker with PPE: label as worker (or simply person).
• Include them in all positions: near pit edge, inside pit (if that ever happens), side walkways.
• Don’t worry too much about perfectly distinguishing PPE vs non-PPE for now unless you have a safety requirement around that.

2.5. Labeling foreign objects
• Only label what you actually care about and what appears in your data:
• Large debris piles, tools left in the pit, tires, big rocks/logs, etc.
• For the small shredded waste on the floor (like the bits in your DIFUSOR images):
• Decide a rule: e.g., only label foreign objects bigger than X pixels or bigger than a piña. Tiny fragments might not matter.
• Label them with a bounding box and class object.

Later, the alert logic can say: “If object_count > 0 in ROI while conveyor is supposed to be clear → raise alert.”

⸻

3. Project plan (end-to-end)

I’ll lay this out as concrete steps you can actually execute.

Step 0 – Define the outputs & ROIs

Per camera, define:
• Regions of interest (polygons) where:
• Piñas are expected (hopper, conveyor, etc.).
• Foreign objects are dangerous / unwanted (pit area, moving machinery).
• For counting, you likely care about piñas within a specific zone (e.g., in hopper, not scattered outside).

All later counting and alerts will be restricted to these ROIs to reduce noise.

⸻

Step 1 – Frame selection & splitting 1. Run the frame-thinning pipeline over each clip:
• Sample 1 frame per second.
• Use SSIM / frame difference to keep only frames that change.
• Cluster remaining frames per camera and select K medoids (e.g., 100–150). 2. From those candidate frames:
• Manually skim and mark:
• Hopper with different fill levels.
• Empty/clean pit.
• Pit with trash and humans.
• Any weird lighting or unusual states. 3. Split into datasets:
• Train: ~70%
• Val: ~15%
• Test: ~15%
• Make sure each camera and each condition is represented in all splits.

⸻

Step 2 – Prepare Label Studio project 1. Create one project with three object labels:
• pina
• worker
• object 2. Decide on workflow:
• Either upload full frames or pre-generated tiles.
• I’d recommend tiles for the dense hopper views:
• You can precompute tiles offline, upload them, and label fully. 3. Write a short annotation guide for your labelers:
• Example rules for piñas: visibility threshold, how tight boxes should be, when not to label.
• Example rules for foreign objects: minimum size & what counts.
• Some visual examples from your frames.

⸻

Step 3 – First labeling round

Aim for something like:
• 400–800 images/tiles with piñas.
• 100–200 images with workers.
• 100–200 images with empty/clean pits and/or foreign objects.

Make sure:
• No training image has unlabeled piñas in regions you care about (unless you went with ignore regions).
• Each class appears in enough images (min ~150–200 images containing each class is a good start).

⸻

Step 4 – Train YOLOv8n

Using Ultralytics (Python API or CLI): 1. Build a YOLO dataset structure:

# data.yaml

path: /path/to/dataset
train: images/train
val: images/val

names:
0: pina
1: worker
2: object

    2.	Train:

yolo detect train \
 model=yolov8n.pt \
 data=data.yaml \
 imgsz=640 \
 epochs=100 \
 batch=16 \
 workers=8 \
 optimizer=sgd

    3.	Recommended augmentations:
    •	Standard YOLOv8: flips, scale, HSV.
    •	Mosaic and random cropping are very useful here: they help the model see partial piles and different densities.
    •	Avoid crazy rotations since your CCTV orientation is fixed.

⸻

Step 5 – Evaluate detection + counting

From the trained model: 1. Measure detection metrics on the test set:
• mAP50, mAP50-95 for each class.
• Precision / recall per class. 2. For counting, compute:

    •	pina_count_pred = number of pina detections above threshold within ROI.
    •	pina_count_true = ground-truth piña count within ROI.

Compute MAE / RMSE:

MAE = mean(abs(pina_count_pred - pina_count_true))
RMSE = sqrt(mean((pina_count_pred - pina_count_true)\*\*2))

Do the same for worker and object counts where relevant.

Tune detection confidence threshold to get a good trade-off between over-counting and under-counting (e.g., F1 or minimal count error).

⸻

Step 6 – Active learning loop

Once v1 is trained: 1. Run the model on all remaining unlabeled frames (including the ones you didn’t pick initially). 2. For each frame:
• Collect:
• Max detection confidence.
• Average detection confidence.
• Number of detections.
• Or use entropy from softmax if you add a small head. 3. Select frames for the next labeling batch:
• High uncertainty (low confidence or inconsistent predictions).
• New unusual scenarios (visually different embeddings or where the model frequently outputs object). 4. Label those and retrain/fine-tune the model.

This gives you maximum value per labeled frame and helps with rare edge cases.

⸻

Step 7 – Non-target / anomaly handling

You want: 1. Count humans and foreign objects → YOLO already gives this. 2. Alert when “any other weird object” appears even if you haven’t explicitly trained that class.

You can do a very practical two-layer approach per frame:

Layer 1 – YOLO detections
• If object_count > 0 in critical ROI:
• Raise an alert (with bbox snapshots).
• If worker inside restricted area ROI:
• Raise a safety alert.

Layer 2 – Simple anomaly detector on background

For each camera and ROI: 1. Collect a bunch of nominal frames (normal conditions). 2. Extract embeddings (e.g., with a pre-trained CNN) from:
• The whole ROI, or
• A grid of patches in the ROI. 3. Fit a simple model:
• kNN on embeddings,
• or IsolationForest,
• or just mean + covariance with Mahalanobis distance. 4. At runtime:
• For each frame/patch in ROI, compute embedding and its distance to normal.
• If distance > threshold and YOLO doesn’t explain it with a known class (pina, worker), trigger an “unknown object” alert.

This covers the “tires, rocks, logs, etc” that you don’t yet have in your training data.

⸻

Step 8 – Deployment logic

When you wire everything up:

For each incoming frame per camera: 1. Run YOLOv8n. 2. Filter detections to your ROI. 3. Compute:
• count_pina
• count_person
• count_foreign 4. Apply rules:
• If count_foreign > 0 → foreign object alert.
• If count_person > 0 in unsafe ROI → safety alert.
• Optionally log count_pina over time to monitor throughput / inventory. 5. Run anomaly detector:
• If anomaly score high and there’s no YOLO detection explaining it, raise “unknown object” alert.
