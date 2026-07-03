# Anime Score Predictor

A personal recommender that predicts **the score you'd give an anime**, not the crowd average. It's trained on your own MyAnimeList (MAL) rating history using XGBoost, and can rank your Plan to Watch (PTW) list by predicted score so you know what to watch next.

This isn't a "people who liked X also liked Y" system. It's a regression model trained on one person's taste, using anime metadata (genre, type, source, popularity, etc.) as features.

---

## What it does

- `predict("title")` — search for any anime and get your predicted score for it, with a breakdown of the anime's metadata and (if you've already watched it) how far off the model was.
- `rank_ptw()` — pulls everything in your Plan to Watch list, predicts a score for each, and ranks the whole list so you can prioritize what to watch next.
- Full error-metrics report (MAE, RMSE, R², Spearman correlation, MAPE) so you know how much to trust the numbers.

---

## How it works — pipeline

1. **Load your MAL export** (XML) and keep only entries marked `Completed` with a score > 0. These are the labeled examples — everything else in your list can't be used for training since there's no target score.
2. **Load the anime metadata dataset** (Kaggle CSV) and clean it up (numeric fields have things like `#` and `,` in them, some rows have parser-breaking quote nesting).
3. **Merge** your watch history with the metadata on MAL anime ID. Anything you've watched that isn't in the dataset (usually very new releases) gets dropped and reported.
4. **Feature engineering** — turn genres/type/source into multi-hot/one-hot vectors, impute missing numeric fields, drop genres you've barely watched (see below).
5. **Train/validation split** (80/20).
6. **Hyperparameter search** with Optuna — 20 trials, 5-fold CV inside the 80% training split, optimizing validation RMSE.
7. **Final training** on the training split with early stopping (a 15% holdout from the training data decides when to stop adding trees).
8. **Evaluation** on the held-out validation set.
9. **Inference** — `predict()` for one-off lookups, `rank_ptw()` for ranking your whole backlog.

---

## Data sources

- **Your watch history**: exported and unzipped from MAL as XML. Export your own list here: **https://myanimelist.net/panel.php?go=export**
- **Anime metadata**: [MyAnimeList 2025 dataset on Kaggle](https://www.kaggle.com/datasets/syahrulapriansyah2/myanimelist-2025) — provides genres, type, source material, episode count, popularity/members/favorites counts, MAL score, and release year for the anime universe your watch history gets matched against.

Anime released after the Kaggle dataset's snapshot date won't be in it, so very recent titles can't be scored — the notebook reports how many of your entries fall into this gap at merge time.

---

## Features used

All features come from the merged (your history) × (Kaggle metadata) table. Every anime is represented as a single feature vector:

**Genres — multi-hot** (one column per genre, `1` if the anime has that genre)
Built from MAL's `Genres`, `Demographic`, and `Themes` columns combined into one tag set (see engineering notes below). Full candidate list before filtering:
`Action, Adventure, Avant Garde, Award Winning, Cars, Comedy, Demons, Drama, Ecchi, Fantasy, Game, Gourmet, Harem, Hentai, Historical, Horror, Josei, Martial Arts, Mecha, Military, Music, Mystery, Parody, Police, Psychological, Romance, Samurai, School, Sci-Fi, Seinen, Shoujo, Shounen, Slice of Life, Space, Sports, Super Power, Supernatural, Suspense, Vampire, Work Life` (39 total before sparse-genre dropping — see below).

**Type — one-hot** (5 categories)
`Movie, ONA, OVA, Special, TV`. MAL's `TV Special` type is folded into `Special` since it's a rare distinction with too few examples to model separately.

**Source material — one-hot** (11 categories)
`4-koma manga, Game, Light novel, Manga, Mixed media, Music, Novel, Original, Other, Visual novel, Web manga`

**Numeric features** (6 total)
- `num_episodes` — episode count
- `mal_score` — the anime's *general* MAL community score (used as a feature, not the target — your personal score is the target)
- `popularity_rank` — MAL popularity ranking
- `favorites_count` — number of users who favorited it
- `members_count` — number of users who have it on their list
- `start_year` — release year

Target variable: `my_score` — your personal 1–10 rating from the MAL export.

---

## Feature engineering details

- **Genre merging**: the Kaggle dataset splits genre-like tags across three separate columns (`Genres`, `Demographic`, `Themes`). These are unioned into a single tag set per anime before multi-hot encoding, so e.g. `Shounen` (a demographic tag) and `Isekai` (a theme tag) end up in the same feature space as `Action` (a genre tag).
- **Genre aliasing**: the dataset lists `Erotica` as distinct from `Hentai`; these are folded into a single `Hentai` tag to avoid splitting an already-small category further.
- **Sparse genre dropping**: any genre you've watched fewer than `GENRE_MIN_COUNT` (default 3) titles of gets dropped from the feature set entirely. With a personal dataset of a few hundred to a few thousand rows, one-hot columns for genres you've barely touched are pure noise the model can latch onto and overfit — this is a direct anti-overfitting measure, and the notebook prints exactly which genres got cut and why.
- **Missing value imputation**: `num_episodes` → training-set median, `mal_score` → training-set mean, `start_year` → training-set median. These stats are computed once on the merged training data and reused at inference time for consistency.
- **Type folding**: `TV Special` → `Special`, same rationale as genre sparsity — too few examples to justify a separate category.
- **Title search**: `search_anime()` is punctuation-insensitive and does substring + word-overlap (Jaccard/coverage blend) matching, so `"Steins Gate"` finds *Steins;Gate* and Japanese titles work interchangeably with English ones.

---

## Model & training

- **Algorithm**: XGBoost regressor (`reg:squarederror`), predicting a continuous score later clipped to `[1, 10]`.
- **Hyperparameter search**: Optuna, 20 trials, TPE sampler, minimizing mean validation RMSE across 5-fold CV run inside the 80% training split. Search space:
  - `max_depth`: 2–6
  - `learning_rate`: 0.01–0.2 (log scale)
  - `subsample`: 0.5–0.9
  - `colsample_bytree`: 0.5–0.9
  - `reg_alpha` (L1): 0.0–5.0
  - `reg_lambda` (L2): 1.0–10.0
  - `min_child_weight`: 1–15
  - `n_estimators` fixed at a ceiling of 500, with early stopping (20 rounds) deciding the real number used per fold.
- **Final fit**: the winning hyperparameters are retrained on a 85/15 split of the training data, using the 15% as an early-stopping holdout (separate from the validation set used for reporting).
- **Why RMSE as the tuning objective but a full metric suite for reporting**: RMSE penalizes big misses harder, which is what you want during hyperparameter search (avoid wildly wrong predictions). But RMSE alone doesn't tell you if the model gets your *relative* preferences right — hence the fuller report below.

---

## Evaluation metrics

The notebook reports training vs. validation for all of these, so you can see the train/val gap (a proxy for overfitting) as well as absolute performance:

| Metric | What it tells you |
|---|---|
| **MAE** | Average absolute error in points, on the 1–10 scale. The most directly interpretable number — "predictions are off by ±X points on average." |
| **RMSE** | Same units as MAE but penalizes large misses more heavily. A model with a few very wrong predictions will show a bigger MAE↔RMSE gap. |
| **R²** | Fraction of the variance in your actual scores the model explains. |
| **Spearman rank correlation** | How well the model gets the *ordering* of anime right, independent of exact score. This is the metric that matters most for `rank_ptw()` — you care more about "will I like A more than B" than "is my score exactly 8.3." |
| **MAPE (%)** | Average error as a percentage of the true score — gives a scale-independent read on error size. |

Alongside these, the notebook prints an actual-vs-average-predicted breakdown (grouped by your true score) and the top 15 features by importance, so you can sanity-check what the model is actually keying off of.

---

## Usage

```python
predict("Fullmetal Alchemist Brotherhood")
predict("One Punch Man")
predict("Steins Gate")               # punctuation optional
predict("Shingeki no Kyojin")        # JP title for Attack on Titan
predict("Kimetsu no Yaiba")          # JP title for Demon Slayer
predict("Shigatsu wa Kimi no Uso")   # Your Lie in April

ptw_rankings = rank_ptw()            # ranks your whole Plan to Watch list
```

`predict()` shows the top `top_n` (default 3) title matches sorted by MAL popularity, each with type/source/year/episode/genre info, the predicted score with a bar visualization, and — if you've already watched it — your actual score and how far off the model was.

`rank_ptw()` returns a DataFrame and prints a ranked table of your whole backlog, plus a list of any PTW titles too new to be in the metadata dataset.

---

## Setup

**Requirements**: `pandas`, `numpy`, `xgboost`, `optuna`, `scikit-learn`, `scipy` (installed via the first notebook cell).

**Config** (top of notebook, edit to match your files):
```python
ANIMELIST_XML     = "animelistv1.xml"   # your exported MAL XML — see export link below
ANIME_CSV         = "mal_anime.csv"     # Kaggle dataset CSV
N_CV_FOLDS        = 5
OPTUNA_TRIALS     = 20
EARLY_STOP_ROUNDS = 20
MAX_ESTIMATORS    = 500
```

**Getting your own data**:
1. Export your MAL list as XML: **https://myanimelist.net/panel.php?go=export**
2. Drop both files in the notebook's working directory and update the config cell with their filenames.

---

## Limitations

- **Personal model** — trained on one person's ratings, so it won't generalize to anyone else's taste. Rerunning it on someone else's export just trains a different personal model.
- **Cold start** — needs a reasonably sized completed-and-scored history to learn anything meaningful; a very small watch history will overfit badly (watch for a large train/val gap in the metrics report).
- **Coverage gap** — anime released after the Kaggle dataset's snapshot can't be scored at all, since they're simply not in the metadata table. Very recent/ongoing seasonal titles are the most likely to hit this.
- **Metadata-only features** — the model has no access to plot, characters, animation style, or anything qualitative; it's reasoning purely from genre/type/source/popularity/episode-count patterns in your history. Two anime that "feel" completely different to you but share metadata will get similar predictions.
