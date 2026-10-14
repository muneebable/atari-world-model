# Atari world model + inverse dynamics

A small, end-to-end version of the recipe behind action-conditioned world models, trained on Atari Breakout:

1. **Collect** gameplay as `(frame, action)` pairs.
2. **Inverse dynamics model (IDM):** infer the action a player took from the frames around it, so unlabelled video can be turned into training data (the VPT idea).
3. **Tokenizer:** a VQ-VAE that turns each 64×64 frame into 8×8 discrete tokens.
4. **World model:** a causal transformer over frame tokens interleaved with action tokens. Given a few real frames and a sequence of actions, it imagines the future frame by frame.
5. **Evaluate** with numbers rather than only nice GIFs (see below).

```
gym env ──► (frame, action) dataset ──┬──► IDM: frames → action ──► relabel data (actions_idm)
                                      └──► VQ-VAE tokens ──► transformer world model ◄── actions (true or IDM)
                                                                    │
                     eval: PSNR vs copy-last-frame baseline, IDM-on-imagined-video consistency, counterfactual GIFs
```

## Results

_To fill in after training on a Colab/Kaggle GPU (`runs/*_eval.json`)._

| | true action labels | IDM pseudo-labels |
|---|---|---|
| IDM accuracy on held-out episodes | — | — |
| Tokenizer reconstruction PSNR (upper bound) | — | — |
| World model PSNR, mean over 12 imagined steps | — | — |
| Copy-last-frame baseline PSNR | — | — |
| Action consistency (IDM reads imagined video) | — | — |

## How the evaluation works

- **PSNR against two references.** Copying the last real frame is the baseline a world model has to beat. Decoding the ground-truth tokens is the ceiling a token-based model can reach. Scores are reported per imagined step, so error accumulation over the horizon is visible.
- **Action consistency.** The IDM reads the imagined video and we check whether it recovers the actions the world model was conditioned on, compared with the IDM on real video. A model that ignores its actions can still look plausible; this metric catches that.
- **Counterfactuals.** From the same real start, each column holds one action down for the whole rollout (`*_counterfactual.gif`). LEFT and RIGHT should move the paddle in opposite directions.
- **Pseudo-label experiment.** The world model is trained twice: once on true actions and once on IDM-inferred actions. The gap shows how much a world model loses when its labels come from an IDM instead of the game, which is the question that matters when the training data is unlabelled video.

## Design notes

- **Sticky actions off** (`repeat_action_probability=0`). With the ALE default of 0.25, the logged action is sometimes not the one executed, which would put label noise into both models.
- **Max-pool downscaling.** Area-averaging 210×160 → 64×64 turns the ball into one dim pixel that a reconstruction loss ignores. Max-pooling keeps it bright.
- **Random policy with held actions** (1–8 steps), so the data has sustained paddle movement rather than jitter.
- **Ambiguous actions are expected.** In Breakout, FIRE only matters when no ball is in play, so FIRE and NOOP usually produce identical frames, and LEFT at the left wall equals NOOP. The IDM confusion matrix (`runs/idm_eval.json`) shows this; it is a property of the game, not an IDM bug.
- **Cosine-similarity VQ codebook** (unit-norm codes and latents) to keep codebook usage high without extra tricks.

## Run

On Colab or Kaggle with a GPU, open `colab.ipynb`. Locally:

```bash
pip install -e ".[dev]"
python -m wm.collect --episodes 500            # ~115k frames
python -m wm.idm train && python -m wm.idm label
python -m wm.tokenizer train && python -m wm.tokenizer encode
python -m wm.dynamics train                                            # true labels
python -m wm.dynamics train --actions actions_idm --name dynamics_idm  # IDM labels
python -m wm.evaluate --dynamics dynamics
python -m wm.evaluate --dynamics dynamics_idm
```

Tests (CPU, about 20 seconds; the pipeline test runs every stage on a few real episodes):

```bash
pytest -q
```
