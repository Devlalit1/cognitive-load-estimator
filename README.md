# 🧠 Cognitive Load Estimator

Real-time cognitive load estimation from facial behaviour using MediaPipe FaceMesh, OpenCV, and scikit-learn — presented in a live Streamlit dashboard.

---

## Project Structure

```
cognitive-load-estimator/
├── src/
│   ├── capture.py          # Webcam feed handler (30 FPS, BGR→RGB)
│   ├── face_mesh.py        # MediaPipe FaceMesh – 468 landmark extraction
│   ├── features.py         # EAR, blink rate, PERCLOS, head pose, AU4, gaze
│   ├── window_buffer.py    # 30-s sliding window aggregation → feature vector
│   ├── model.py            # Random Forest / SVM training & prediction wrapper
│   └── dashboard.py        # Streamlit real-time UI
├── data/
│   ├── sessions/           # Feature CSVs + task result CSVs per session
│   └── labels/             # NASA-TLX scores (ground truth)
├── experiments/
│   ├── n_back_task.py      # 3-back working memory task (High load)
│   ├── stroop_task.py      # Stroop color-word test (High load validation)
│   └── nasa_tlx.py         # NASA-TLX survey (Streamlit)
├── notebooks/
│   └── analysis.ipynb      # EDA, model training, plots
├── report/                 # IEEE-format final report (LaTeX / PDF)
├── requirements.txt
└── README.md
```

---

## Quick Start

### 1. Set up environment

```bash
python -m venv cogload-env
# Windows
cogload-env\Scripts\activate
# macOS/Linux
source cogload-env/bin/activate

pip install -r requirements.txt
```

### 2. Verify webcam + landmark tracking

```bash
cd src
python face_mesh.py          # import/lifecycle check
```

MediaPipe 1.x uses the Tasks API and requires the Face Landmarker model asset.
Place `face_landmarker.task` in the project `models/` directory (the detector
looks there automatically), or set an alternate path before starting:

```powershell
$env:FACE_LANDMARKER_MODEL = "C:\path\to\face_landmarker.task"
```

The detector also supports the legacy `mp.solutions.face_mesh` API when using
an older MediaPipe release.

### 3. Run data-collection session

```bash
# Terminal 1 – run dashboard (records features automatically)
streamlit run src/dashboard.py

# Terminal 2 – run cognitive task (after dashboard is running)
python experiments/n_back_task.py <session_id>

# Terminal 3 – after each task, open TLX survey
streamlit run experiments/nasa_tlx.py
```

### 4. Train the model (after collecting ≥10 participants)

Open `notebooks/analysis.ipynb` and follow the cells.

### 5. Run the full real-time dashboard

```bash
streamlit run src/dashboard.py
```

The dashboard records one aggregated 30-second feature row at a time. Use
**Record this session** to capture estimates. The **Session brief** tab shows
the load distribution, dominant estimate, high-load share, signal averages,
face-detection coverage, and a short interpretation. Download the brief as
Markdown and the recorded estimates as CSV, or save the CSV under
`data/sessions/`. Add a matching NASA-TLX label file under `data/labels/`
before training:

```bash
python src/model.py --train
```

### Deployment note

The current dashboard opens the webcam with OpenCV on the machine running
Streamlit. It works when run locally, but a cloud deployment cannot access
the webcam attached to your browser's computer. Remote webcam use requires
browser-side camera capture and sending frames to the app.

Streamlit Community Cloud also needs the Linux `libgl1` system package for
OpenCV; this is declared in the root `packages.txt`.

---

## Features Extracted

| Feature | Description |
|---|---|
| `ear_mean_avg` | Mean Eye Aspect Ratio over 30-s window |
| `ear_mean_std` | EAR variability |
| `blink_rate` | Blinks per minute |
| `perclos` | % frames with eyes ≥80% closed |
| `pitch/yaw/roll` | Head rotation angles (mean & std) |
| `head_stability` | Inverse of angular variance |
| `au4_mean/std` | Brow-furrow proxy (AU4) |
| `gaze_x/y_std` | Fixation stability |

---

## Cognitive Load Labels

| Label | NASA-TLX Range | Task |
|---|---|---|
| Low    | 0–33  | Relaxing video |
| Medium | 34–66 | Article + MCQ  |
| High   | 67–100 | N-back (n=3) / Stroop |

---

## References

- Sweller, J. (1988). *Cognitive load during problem solving*. Cognitive Science.
- Solovey et al. (2015). *Classifying driver workload*. CHI.
- Fridman et al. (2018). *MIT Advanced Vehicle Technology Study*. arXiv.
- Appel et al. (2019). *Predicting cognitive load from eye tracking*. ACM CHI.
- Zandi et al. (2023). *Cognitive load estimation in the wild*. IEEE TAFFC.
