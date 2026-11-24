# Beatwise

Classifies ECG heartbeats into four groups (normal, supraventricular, ventricular, fusion) and runs entirely in the browser. Live: https://aditya2375.github.io/beatwise/

Data: MIT-BIH Arrhythmia Database (PhysioNet, ODC-By 1.0). Split by patient (de Chazal): train on DS1, test on DS2.

## Results on the 22 unseen DS2 patients (49,684 beats)

| Class | Found | Precision |
|---|---|---|
| Normal | 96.2% | 96.3% |
| Supraventricular | 10.7% | 16.5% |
| Ventricular | 93.0% | 80.9% |
| Fusion | 34.8% | 22.8% |

Overall accuracy 92.3%. A gradient-boosted baseline on timing plus coarse shape gets 93.0%, higher overall, but lower F1 on ventricular and fusion beats. Supraventricular detection is weak, mostly because record 232 holds 1,382 of the 1,837 such beats.

Honest note: a first convolutional design scored 86.2% on DS2 and was dropped. Further changes used patient-wise cross-validation on DS1 only, but DS2 was seen once before the redesign.

## Method

Each beat is compared with the median of the same patient's previous 30 beats (correlation, error, width, peak shift), plus nine timing features. A five-seed MLP ensemble (15-128-64-4) classifies it. The JavaScript port is checked against Python in `test/parity.test.js`.

## Run

Static files, no build. `python3 -m http.server` in this folder, or `node --test test/`. Training scripts are in `train/` and need the MIT-BIH records.

Not a medical device. Research demo only.

MIT licence.
