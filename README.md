
## LoL Draft Probability Estimator
"/reports" contains the project proposal and progress reports. lol_draft_project_docs goes more in-depth on why certain things were done along with dated progress (there is more passion here)

## Project Notes (as of 4/6/2026)
Cool stuff unique to my project worth discussing:

### 1. **Caching**
Caching is implemented in the Riot API collection pipeline (`build_draft_dataset.py`). The main motivation was that as the project evolved, I kept adding and testing new player-based features. Without caching, that would have meant repeatedly querying the same player profiles and treating them as if they were new, which would waste both time and API calls.

The cache stores each player’s ranked solo-queue profile keyed by PUUID, including:
- wins
- losses
- games played
- win rate
- tier
- rank
- LP

A few implementation details were important:
- **`save_player_cache()`** writes to a temporary `.tmp` file before replacing the original cache file. This prevents corruption if the script stops midway through a run.
- **`maybe_cache_entry()`** inserts player information into the cache as soon as it becomes available.
- **`get_player_ranked_profile()`** is the final lookup step: before making an API request, it first checks whether the player’s PUUID is already in the cache. If so, the cached profile is returned immediately.
- The cache is written back to disk every **200 matches** and again at the end of the run.

The practical benefit is that repeated dataset builds can reuse previously fetched player data instead of re-querying Riot’s API. This made later feature-engineering iterations much faster.

---

### 2. **Canonical Teams**
Riot’s raw match data is naturally stored in terms of **blue side** and **red side**, but side itself is not the signal I wanted the model to learn. In fact, because red side often corresponds to slightly higher average MMR, leaving matches in blue/red form risks preserving side-specific information that is not central to the actual composition.

To avoid that, teams are converted into a **canonical `team_a` vs `team_b` format**, with both sides ordered consistently. The goal is to make the representation symmetric and prevent the same composition from being split into two different training cases solely because it appeared once on blue and once on red.

For example, suppose the composition:
- Aatrox, Udyr, Syndra, Jinx, Rakan

appears on blue side in one game and on red side in another. If both are stored as raw blue/red outcomes, then the model effectively sees:
- “blue side wins” once
- “red side wins” once

instead of learning that the **same composition won twice**.

Because the dataset is already limited by Riot API rate constraints, this matters a lot. Canonicalization prevents side information from artificially fragmenting the dataset and makes the effective learning problem more consistent.

---

### 3. **Standardization**
Numeric features in this project live on very different scales. For example:
- win-rate differences are usually tiny decimal values (e.g. `0.505` vs `0.506`)
- games played can range from very small values up to the hundreds

If these are fed directly into the model without scaling, larger-magnitude features can dominate optimization simply because of their units rather than because they are more informative.

To address this, numeric features were standardized using the usual transformation:

\[
x' = \frac{x - \mu_{\text{train}}}{\sigma_{\text{train}}}
\]

where the mean and standard deviation were computed on the **training split only**.

This was done carefully to avoid leakage:
- fitted preprocessing statistics were computed from the **training split**
- the **validation split** was used only for model selection
- the **test split** was reserved for final evaluation

So this is not a fancy trick, but it is an important correctness point: all learned preprocessing steps were fit on training data only.

---

### 4. **Draft Alone Was Weak**
At the start of the project, I tried a very naive setup: use only the 10 drafted champions and the game result. Intuitively, this seemed like it might work, because draft should contain some predictive signal.

In practice, it did not. A draft-only model produced probabilities clustered near `0.5` for most games and behaved close to random.

In hindsight, this makes sense. Solo queue is extremely noisy:
- player execution varies a lot
- coordination is inconsistent
- in-game decisions matter heavily
- many matches are decided by factors not visible from draft alone

This motivated adding:
- player win-rate features
- games-played / confidence features
- engineered draft abstractions such as subclasses and scaling types

These additions gave the model enough structure to move away from near-random behavior and produce meaningful probabilities.

It is also worth noting that many of these abstractions were **hardcoded from domain knowledge**. For example, subclass labels such as bruiser, tank, enchanter, or mage were assigned manually rather than learned automatically. That is imperfect, but it still provided useful signal and made the draft representation more learnable.

---

### 5. **Late Fusion vs. Numeric Token**
In the final model, numeric features bypass the transformer and are concatenated with the final draft representation after the encoder. Initially this was mostly a simplicity choice, but it later became an explicit architecture comparison.

Two variants were tested:

1. **Late fusion:** numeric features appended after the transformer  
2. **Early fusion:** numeric features projected into a token and inserted into the transformer sequence

The early-fusion version did **not** show a meaningful benefit under matched settings. In fact, it was slightly weaker, although the difference was small.

Because the more complex variant did not improve results, the final model kept the simpler **late-fusion** design:
- easier to explain
- easier to interpret
- same or slightly better performance

The numeric-token experiment was still saved separately in its own Git branch.

---

### 6. **Why Calibration Mattered**
The main goal of the project was not just to predict which team would win, but to estimate a **usable pre-game win probability**.

That distinction matters:
- **Accuracy** only tells us how often the model’s binary decision is correct.
- **Log loss** measures overall probability quality.
- **Calibration** checks whether predicted probabilities actually match realized frequencies.

For this use case, calibration is especially important. If the model says a team has a **70%** chance of winning, then games like that should really be won about **70%** of the time. That is what makes the prediction interpretable as a probability rather than just a ranking score.

The model was trained with **binary cross-entropy with logits**, which optimizes log loss, not calibration directly. Calibration was therefore evaluated separately using:
- calibration plots
- Expected Calibration Error (ECE)
- Brier score

This ended up being one of the most important project takeaways. Several models, including logistic regression and XGBoost, achieved similar log-loss values, but differed more clearly in calibration quality. Since the intended use case is a pre-game probability estimator, calibration became the main lens through which final performance was interpreted.

