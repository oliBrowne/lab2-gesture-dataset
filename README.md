# Lab 2 gesture dataset — raw data (work in progress)

Phone accelerometer recordings of three air-drawn gestures plus `idle`, for a Magic-Wand-style classifier.

## Gestures

| Label | Motion |
|---|---|
| `wing` | draw a capital **W**, left to right |
| `ring` | draw one **clockwise circle** |
| `slope` | diagonal stroke **up-right, then straight down** |
| `idle` | no gesture: hold still, set the phone down, hold it while talking, walk a few steps |

## Layout

```
data/raw/<gesture>/<gesture>_p<person>_s<session>_<take>.csv
data/capture_log.csv   one row per file: person, hand, phone, OS, date, time, session, reps, notes
```

The **label is the folder name** (always the same as the first part of the file name). `p1` = person, `s2` = that person's session 2, `01` = take 1.

Every CSV has the header `t,ax,ay,az`: `t` = seconds since the recording started, `ax ay az` = total acceleration **including gravity**, in m/s², in the phone's own axes. Sampled at ~100 Hz.

## How to add your recordings

1. Install **Sensor Logger** (iOS or Android). Settings: **Accelerometer** and **Gravity** on, everything else off, sampling rate **100 Hz**, export format **CSV**.
2. Hold the phone **flat in your hand, screen facing up**.
3. Each recording: press record and hold still ~2 s, do the gesture **10 times** with a **2–3 s completely still pause** between reps (mix slow, normal and fast), then stop. For `idle`, record ~40 s.
4. Per session, record **2 takes of each of the 4 classes** (8 recordings). Try to do 2–3 sessions at different times/places.
5. Export each recording and **rename it immediately** to `<gesture>_p<your number>_s<session>_<take>`, e.g. `ring_p2_s1_01.zip`. Ask which person number is yours (p1 is taken).
6. Send the exported .zip files to p1, who converts them into the CSV format above, or commit them yourself in that format, and add a row per file to `data/capture_log.csv`.
