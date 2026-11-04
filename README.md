# NBA Shot Make Probability

Interactive Streamlit dashboard for a logistic-regression model that predicts whether an NBA shot goes in, trained on 425,719 player-tracking shots.

**Author:** Terrance Dessaure

## What's inside
| Tab | What it shows |
|---|---|
| Overview | FG% by shot type and by distance, with sidebar filters |
| Shot Chart | Half-court heat map of FG% in 2×2 ft zones |
| Defense | Effect of defender distance (split 2PT/3PT), contesters, closing speed |
| Model | Validation AUC / accuracy / log loss vs. baseline, coefficients, calibration |
| Shot Simulator | Set up any shot with sliders and get its make probability and expected points |

## Results (20% held-out validation set)
| Metric | Model | Baseline |
|---|---|---|
| ROC AUC | 0.649 | 0.500 |
| Accuracy | 62.3% | 54.2% |
| Log loss | 0.642 | 0.690 |

## Run it locally
```bash
pip install -r requirements.txt
python model.py          # optional self-check, prints metrics
streamlit run app.py     # opens http://localhost:8501
```
`training.csv.gz` must sit in the same folder as `app.py`.

## Files
- `model.py` – feature engineering, training, evaluation (no Streamlit; testable on its own)
- `app.py` – dashboard UI
- `requirements.txt` – dependencies for local runs and Streamlit Community Cloud

## Features
Distance, closest-defender distance, defender closing speed (parsed from the 4-frame `closestdefapproach` array), number of contesters, shot clock, dribbles, shooter speed, catch-and-shoot and rushed-shot flags, and one-hot shot type / game state.