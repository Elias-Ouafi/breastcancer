# Running the demo

> **Research Use Only — Not for diagnostic use.**

The DCE-MRI lesion-localisation demo runs from a fresh clone, with no dataset download:
the checkpoint and three curated cases are versioned. A French version of this
walkthrough is in [DEMO.md](../DEMO.md).

```bash
pip install -r requirements.txt
python run_demo.py
```

## Running it

**The day before**, confirm the machine is ready without occupying a port:

```bash
python run_demo.py --check
```

**On the day:**

1. `python run_demo.py`
2. Open <http://127.0.0.1:5000>
3. Click **Cas 1** — no file picker to fumble with. (Drag-and-drop still works for
   your own exam.)
4. Walk through the result, in this order:
   - the verdict and confidence, with the compute time (~70 ms warm, ~0.6 s on the first call)
   - the annotated slice, accent box on the enhancing region
   - **drag the slider** — the lesion appears, peaks and fades. This is the moment
     that convinces.
   - **Vue MIP** — the maximum-intensity projection, how a radiologist reads it
   - unfold *Détail technique* if you get questions
5. Follow with **Comment ça marche** (link at the bottom) if they want the pipeline.

## What to say yourself

Three sentences, before anyone has to ask:

> The slice was picked in advance by a human. The model segments a lesion well
> **once shown the right slice** — 88 % sensitivity. It cannot find that slice on its
> own yet: a dedicated classifier took that from 0 % to 43 %, and that work is
> ongoing.

> The 0.53 Dice is not comparable to the ~0.80 in the literature: our training masks
> are bounding boxes, not expert contours.

> Nothing is clinically validated. This is research on a 186-patient sample.

The app already states all of this in its "Limites connues" panel. Saying it before
they read it buys credibility rather than costing it.

## If it goes wrong

| Symptom | Fix |
|---------|-----|
| `run_demo.py` refuses to start | It names the missing prerequisite; follow the `->` line it prints |
| Port already in use | `python run_demo.py --port 5001` |
| Engine badge reads `mock` | The launcher was bypassed — start again with `run_demo.py` |
| No annotated slice shown | The upload is not a preprocessed `.npz`; use a `data/curated_data/demo_cases/` file |

**Safety net:** screenshot a successful result the day before. If the live run fails,
you carry on without a gap.
