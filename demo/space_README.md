---
title: WIUT Traffic Event Detection
emoji: 🚦
colorFrom: gray
colorTo: blue
sdk: gradio
app_file: app.py
pinned: false
license: agpl-3.0
short_description: Traffic events + causal risk curve on your road-camera clip
---

Live demo of our WIUT Hackathon 2026 CV-track submission. Upload a road-camera
clip (up to 120 s, 200 MB): it runs the submitted pipeline on CPU and returns
the detected events, an annotated playback, one clip per event, the risk curve
and the prediction in the harness's JSON format.

The event rules are drawn for one intersection (the competition's camera);
on other scenes the app says so and shows detections and the risk curve only.
Code: <REPO_URL>. Licence: AGPL-3.0 (Ultralytics YOLO11 is AGPL-3.0).
