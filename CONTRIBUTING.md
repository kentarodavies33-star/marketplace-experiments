# Working together

## Sharing the project
1. One of us creates a GitHub repository and pushes this folder to it.
2. Add the other person under **Settings → Collaborators**.
3. The other person runs `git clone <repo-url>` and opens the folder in VS Code.

## Getting credit for shared work
Each commit records who made it. For work done together (for example in a Live Share
session), add a co-author line at the end of the commit message so GitHub credits both of us:

```
Add block-length sweep

Co-authored-by: KD <kd-email@example.com>
```

The email has to be one linked to that person's GitHub account.

## Working at the same time
Use the **Live Share** extension: the host clicks *Live Share* in the status bar and sends
the link. Commit and push when the session ends.

## Before pushing
- `pytest -q` passes.
- If you changed the simulation, rerun `python run_experiments.py` so `results/` matches the code.

## Ideas for next steps (split these up)
- **Dasher response:** let Dasher supply react to earnings (more orders → more Dashers log on).
- **Spatial spillover:** let neighbouring regions share Dashers, so switchbacks leak at region borders.
- **Smarter estimators:** a regression with region and hour fixed effects, or CUPAC (an ML prediction as the covariate).
- **Power analysis:** how many days of switchback do we need to detect a 2% lift?
- **Burn-in:** drop the first period of each block and measure the bias/variance trade-off.
