"""Make the README pictures from the exported model data.

Run from the repo root (after export_data.py has created docs/data/):
    python make_plots.py              # writes to images/
    python make_plots.py --out figs   # or somewhere else

Needs: numpy, matplotlib.
"""
import argparse
import collections
import json
import math
import os

import matplotlib
matplotlib.use("Agg")                       # draw to files, no window needed
import matplotlib.pyplot as plt
import numpy
from matplotlib.colors import LinearSegmentedColormap, PowerNorm

DATA = "docs/data"
BG, INK, MUTED, ACCENT = "#0b0b0b", "#f3efe7", "#a39e94", "#FF7A30"
BLUE = "#6fa8ff"

plt.rcParams.update({
    "figure.facecolor": BG, "axes.facecolor": BG, "savefig.facecolor": BG,
    "text.color": INK, "axes.labelcolor": INK, "axes.edgecolor": "#3a3a3a",
    "xtick.color": MUTED, "ytick.color": MUTED, "axes.titlecolor": INK,
    "font.size": 10, "axes.titlesize": 12, "axes.titleweight": "bold",
    "axes.spines.top": False, "axes.spines.right": False,
    "grid.color": "#262626", "legend.frameon": False,
})
HEAT = LinearSegmentedColormap.from_list("heat", ["#0b0b0b", "#5a2a10", ACCENT, "#fff3e6"])


def load(name):
    with open(os.path.join(DATA, name)) as f:
        return json.load(f)


def softmax_rows(scores, temperature=1.0):
    z = numpy.asarray(scores, dtype=float) / temperature
    z = z - z.max(axis=1, keepdims=True)
    e = numpy.exp(z)
    return e / e.sum(axis=1, keepdims=True)


def entropy_rows(probs):
    return -(probs * numpy.log(numpy.clip(probs, 1e-12, None))).sum(axis=1)


def save(fig, out, name):
    path = os.path.join(out, name)
    fig.savefig(path, dpi=160, bbox_inches="tight")
    plt.close(fig)
    print(f"  {path}")


def best_possible_loss(tokens):
    """Lowest loss any one-word model can reach on this text: the data's own
    uncertainty, H(next word | current word), estimated from counts."""
    pair_counts = collections.Counter(zip(tokens[:-1], tokens[1:]))
    cur_counts = collections.Counter(tokens[:-1])
    total = sum(pair_counts.values())
    return -sum(c * math.log(c / cur_counts[a]) for (a, _), c in pair_counts.items()) / total


# ---------------------------------------------------------------- figures
def fig_training(out, meta, losses, snaps, tokens, vocab):
    w2i = {w: i for i, w in enumerate(vocab)}
    cur_ids = [w2i[t] for t in tokens[:-1]]
    floor = best_possible_loss(tokens)
    guess = meta["loss_random_guess"]
    steps = numpy.arange(1, len(losses) + 1)

    fig, (a, b) = plt.subplots(1, 2, figsize=(13, 4.6))
    a.plot(steps, losses, color=ACCENT, lw=2)
    a.axhline(guess, color=MUTED, ls="--", lw=1)
    a.text(1.2, guess + 0.06, f"random guessing  ln(112) = {guess:.2f}", color=MUTED, fontsize=9)
    a.axhline(floor, color=BLUE, ls=":", lw=1.3)
    a.text(1.2, floor + 0.07, f"best possible for this text  ≈ {floor:.2f}", color=BLUE, fontsize=9)
    for s in snaps:
        if s["step"] in (100, 1000, 20000):
            a.scatter([s["step"]], [s["loss"]], color=INK, s=28, zorder=5)
            a.annotate(f"step {s['step']:,}\nloss {s['loss']:.2f}", (s["step"], s["loss"]),
                       textcoords="offset points", xytext=(8, 8), fontsize=8.5, color=INK)
    a.set_xscale("log")
    a.set_xlabel("training step (log scale)")
    a.set_ylabel("loss")
    a.set_title("Loss falls fast, then crawls toward a floor")
    a.set_ylim(floor - 0.2, guess + 0.35)
    a.grid(True, alpha=.6)

    sx = [s["step"] for s in snaps]
    ent = [float(entropy_rows(softmax_rows(numpy.array(s["weights"])))[cur_ids].mean()) for s in snaps]
    sx_plot = [max(x, 1) for x in sx]
    b.plot(sx_plot, ent, color=ACCENT, lw=2, marker="o", ms=5)
    b.axhline(math.log(len(vocab)), color=MUTED, ls="--", lw=1)
    b.text(1.2, math.log(len(vocab)) + 0.07, "completely unsure (all 112 words equally likely)", color=MUTED, fontsize=9)
    b.axhline(floor, color=BLUE, ls=":", lw=1.3)
    b.text(1.2, floor + 0.1, "as unsure as the text itself is", color=BLUE, fontsize=9)
    b.set_xscale("log")
    b.set_xlabel("training step (log scale)")
    b.set_ylabel("average uncertainty of its guesses (entropy, nats)")
    b.set_title("It goes from “no idea” to “as sure as the text allows”")
    b.set_ylim(floor - 0.2, math.log(len(vocab)) + 0.35)
    b.grid(True, alpha=.6)
    fig.tight_layout()
    save(fig, out, "training.png")


def fig_weight_maps(out, snaps, tokens, vocab, k=28):
    freq = collections.Counter(tokens)
    order = sorted(range(len(vocab)), key=lambda i: -freq[vocab[i]])[:k]   # the k most common words
    labels = [vocab[i] for i in order]
    show = [s for s in snaps if s["step"] in (0, 100, 1000, 20000)]
    fig, axes = plt.subplots(1, 4, figsize=(18, 5.6))
    for ax, s in zip(axes, show):
        P = softmax_rows(numpy.array(s["weights"]))[numpy.ix_(order, order)]
        im = ax.imshow(P, cmap=HEAT, norm=PowerNorm(0.45, vmin=0, vmax=1), interpolation="nearest")
        ax.set_title(f"step {s['step']:,}   (loss {s['loss']:.2f})", fontsize=11)
        ax.set_xticks(range(k)); ax.set_xticklabels(labels, rotation=90, fontsize=7)
        ax.set_yticks(range(k)); ax.set_yticklabels(labels, fontsize=7)
        ax.set_xlabel("next word")
        if ax is axes[0]:
            ax.set_ylabel("current word")
        for sp in ax.spines.values():
            sp.set_visible(False)
    cb = fig.colorbar(im, ax=axes, fraction=0.010, pad=0.01)
    cb.set_label("probability of the next word", color=INK)
    fig.suptitle(f"The model is a table of odds. Here are the {k} most common words, noise at first, structure after training",
                 y=0.99, fontsize=13, fontweight="bold")
    save(fig, out, "weight_maps.png")


def fig_predictions(out, snaps, vocab):
    w2i = {w: i for i, w in enumerate(vocab)}
    first = next(s for s in snaps if s["step"] == 0)
    last = snaps[-1]
    P0 = softmax_rows(numpy.array(first["weights"]))
    P1 = softmax_rows(numpy.array(last["weights"]))
    words = [w for w in ("the", "is", "octopus", "sun") if w in w2i]
    fig, axes = plt.subplots(1, len(words), figsize=(15, 4.2))
    for ax, w in zip(axes, words):
        i = w2i[w]
        ranked = numpy.argsort(P1[i])[::-1]
        top = [j for j in ranked[:6] if P1[i][j] > 0.005] or list(ranked[:2])
        y = numpy.arange(len(top))
        ax.barh(y + .2, P0[i][top] * 100, height=.38, color="#4a4a4a", label="before training")
        ax.barh(y - .2, P1[i][top] * 100, height=.38, color=ACCENT, label="after 20,000 steps")
        ax.set_yticks(y); ax.set_yticklabels([vocab[j] for j in top])
        ax.invert_yaxis()
        ax.set_xlabel("chance of coming next (%)")
        ax.set_title(f"after “{w}”")
        for yy, j in zip(y, top):
            ax.text(P1[i][j] * 100 + 1, yy - .2, f"{P1[i][j] * 100:.0f}%", va="center", fontsize=8.5, color=INK)
        ax.set_xlim(0, max(P1[i][top].max() * 100 * 1.25, 12))
    axes[0].legend(loc="lower right", fontsize=8.5)
    fig.suptitle("What it believes comes next, before and after training", y=1.03, fontsize=13, fontweight="bold")
    fig.tight_layout()
    save(fig, out, "predictions.png")


def fig_temperature(out, snaps, vocab):
    w2i = {w: i for i, w in enumerate(vocab)}
    W = numpy.array(snaps[-1]["weights"])
    i = w2i["the"]
    top = numpy.argsort(softmax_rows(W)[i])[::-1][:8]
    temps = [0.3, 0.7, 1.0, 1.5]
    fig, axes = plt.subplots(1, 4, figsize=(15, 4), sharey=True)
    for ax, t in zip(axes, temps):
        p = softmax_rows(W, t)[i]
        ax.barh(numpy.arange(len(top)), p[top] * 100, color=ACCENT if t == 1.0 else "#8a8a8a")
        ax.set_yticks(numpy.arange(len(top))); ax.set_yticklabels([vocab[j] for j in top])
        ax.invert_yaxis()
        ax.set_title(f"temperature {t}")
        ax.set_xlabel("chance (%)")
        for k, j in enumerate(top):
            ax.text(p[j] * 100 + .8, k, f"{p[j] * 100:.0f}%", va="center", fontsize=8.5)
        ax.set_xlim(0, 62)
    fig.suptitle("The creativity dial: same model, same word (“the”), different temperature", y=1.03, fontsize=13, fontweight="bold")
    fig.tight_layout()
    save(fig, out, "temperature.png")


def fig_model_vs_text(out, snaps, tokens, vocab, k=28):
    """Does the trained table match the habits actually in the text?"""
    freq = collections.Counter(tokens)
    w2i = {w: i for i, w in enumerate(vocab)}
    order = sorted(range(len(vocab)), key=lambda i: -freq[vocab[i]])[:k]
    labels = [vocab[i] for i in order]
    counts = numpy.zeros((len(vocab), len(vocab)))
    for a, b in zip(tokens[:-1], tokens[1:]):
        counts[w2i[a], w2i[b]] += 1
    text_p = counts / counts.sum(axis=1, keepdims=True).clip(min=1)
    model_p = softmax_rows(numpy.array(snaps[-1]["weights"]))
    rows = [i for i in range(len(vocab)) if counts[i].sum() > 0]
    tv = 0.5 * numpy.abs(text_p[rows] - model_p[rows]).sum(axis=1)             # total variation per word
    weight = counts[rows].sum(axis=1)
    avg_gap = float((tv * weight).sum() / weight.sum())

    fig, axes = plt.subplots(1, 2, figsize=(13, 6.2))
    for ax, M, title in ((axes[0], text_p, "What the text actually does"),
                         (axes[1], model_p, "What the trained model predicts")):
        im = ax.imshow(M[numpy.ix_(order, order)], cmap=HEAT, norm=PowerNorm(0.45, vmin=0, vmax=1), interpolation="nearest")
        ax.set_title(title)
        ax.set_xticks(range(k)); ax.set_xticklabels(labels, rotation=90, fontsize=7.5)
        ax.set_yticks(range(k)); ax.set_yticklabels(labels, fontsize=7.5)
        ax.set_xlabel("next word")
        for sp in ax.spines.values():
            sp.set_visible(False)
    axes[0].set_ylabel("current word")
    cb = fig.colorbar(im, ax=axes, fraction=0.025, pad=0.02)
    cb.set_label("probability", color=INK)
    fig.suptitle(f"The model learned the text’s habits: on average its odds are within {avg_gap * 100:.1f} percentage points of the real counts",
                 y=0.99, fontsize=12.5, fontweight="bold")
    save(fig, out, "model_vs_text.png")
    print(f"  (average gap between model and text: {avg_gap * 100:.2f} percentage points)")


def main():
    parser = argparse.ArgumentParser(description="Make the README pictures.")
    parser.add_argument("--out", default="images")
    args = parser.parse_args()
    os.makedirs(args.out, exist_ok=True)

    meta, vocab, tokens = load("meta.json"), load("vocab.json"), load("tokens.json")
    losses, snaps = load("loss_curve.json"), load("snapshots.json")
    print(f"writing pictures to {args.out}/")
    fig_training(args.out, meta, losses, snaps, tokens, vocab)
    fig_weight_maps(args.out, snaps, tokens, vocab)
    fig_predictions(args.out, snaps, vocab)
    fig_temperature(args.out, snaps, vocab)
    fig_model_vs_text(args.out, snaps, tokens, vocab)
    print("done.")


if __name__ == "__main__":
    main()
