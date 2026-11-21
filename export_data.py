"""Export everything the website needs as small JSON files.

Run from the repo root:
    python export_data.py                 # writes to docs/data/
    python export_data.py --out some/dir  # or somewhere else

It reuses train() from lm.py without changing it. Training is deterministic
(fixed seed), so training for k steps gives exactly the weights you have at step k.
"""
import argparse
import json
import math
import os

import numpy

from lm import (LEARNING_RATE, SEED, STEPS, load_text, make_pairs, softmax,
                text_to_ids, tokenization, train)

TEXT_FILES = ["data/ocean.txt", "data/space.txt"]
SNAPSHOT_STEPS = [0, 10, 50, 100, 250, 500, 1000, 2000, 5000, 10000, 20000]
TEMPERATURES = [0.3, 0.7, 1.0, 1.5]
SAMPLE_SEEDS = [0, 1, 2, 3, 4]
SENTENCE_LENGTH = 8      # words after the start word, same as generate_text()
DECIMALS = 4             # rounding keeps the files small


def rounded(array):
    """NumPy array -> plain nested lists of rounded floats (JSON friendly)."""
    return numpy.round(array, DECIMALS).tolist()


def loss_of(W, current, target):
    """Average loss of weights W on the training pairs."""
    probs = softmax(W[current])
    return float(-numpy.mean(numpy.log(probs[numpy.arange(len(current)), target])))


def sample_sentence(W, word_to_id, id_to_word, start, length, seed, temperature):
    """Same dice-rolling loop as generate_text(), but it returns the sentence
    and has a temperature: scores are divided by it before softmax."""
    rng = numpy.random.default_rng(seed)
    probs = softmax(W / temperature)
    V = len(id_to_word)
    current_w = word_to_id[start]
    words = [start]
    for _ in range(length):
        current_w = int(rng.choice(V, p=probs[current_w]))
        words.append(id_to_word[current_w])
    return " ".join(words)


def top_next(probs, id_to_word, k=8):
    """For every word: its k most likely next words with probabilities."""
    result = {}
    for i in range(len(id_to_word)):
        best = numpy.argsort(probs[i])[::-1][:k]
        result[id_to_word[i]] = [[id_to_word[int(j)], round(float(probs[i][j]), DECIMALS)]
                                 for j in best]
    return result


def write_json(folder, name, data):
    path = os.path.join(folder, name)
    with open(path, "w") as f:
        json.dump(data, f, separators=(",", ":"))
    print(f"  {name:<20} {os.path.getsize(path) / 1024:8.1f} KB")


def main():
    parser = argparse.ArgumentParser(description="Export model data for the website.")
    parser.add_argument("--out", default="docs/data", help="output folder")
    args = parser.parse_args()
    os.makedirs(args.out, exist_ok=True)

    # ---- data, same steps as lm.py ----
    text = load_text(TEXT_FILES)
    tokens, vocab, word_to_id, id_to_word = tokenization(text)
    current, target = make_pairs(text_to_ids(tokens, word_to_id))
    V = len(vocab)
    start_word = "the" if "the" in word_to_id else vocab[0]

    # ---- the full training run (loss at every step) ----
    print(f"training {STEPS} steps for the full loss curve ...")
    losses, W_final = train(current, target, vocab)

    # ---- snapshots: retrain for k steps, same seed = same weights at step k ----
    snapshots = []
    for step in SNAPSHOT_STEPS:
        print(f"snapshot at step {step} ...")
        _, W = train(current, target, vocab, steps=step)
        snapshots.append({
            "step": step,
            "loss": round(loss_of(W, current, target), DECIMALS),
            "weights": rounded(W),
            "sentences": [sample_sentence(W, word_to_id, id_to_word, start_word,
                                          SENTENCE_LENGTH, s, 1.0) for s in range(3)],
        })

    # ---- sanity check: the last snapshot must equal the full run ----
    last = snapshots[-1]
    assert last["step"] == STEPS, "last snapshot step must equal STEPS"
    assert abs(last["loss"] - loss_of(W_final, current, target)) < 1e-3, "snapshot != full run"

    # ---- samples at several temperatures ----
    samples = {str(t): {str(s): sample_sentence(W_final, word_to_id, id_to_word, start_word,
                                                SENTENCE_LENGTH, s, t)
                        for s in SAMPLE_SEEDS} for t in TEMPERATURES}

    probs_final = softmax(W_final)

    print(f"writing files to {args.out}/")
    write_json(args.out, "meta.json", {
        "tokens": len(tokens), "vocab_size": V, "pairs": len(current),
        "weights": V * V, "seed": SEED, "learning_rate": LEARNING_RATE, "steps": STEPS,
        "loss_start": round(losses[0], DECIMALS), "loss_end": round(losses[-1], DECIMALS),
        "loss_random_guess": round(math.log(V), DECIMALS),
        "start_word": start_word, "snapshot_steps": SNAPSHOT_STEPS,
        "temperatures": TEMPERATURES, "decimals": DECIMALS,
        "text_files": TEXT_FILES,
    })
    write_json(args.out, "vocab.json", vocab)
    write_json(args.out, "tokens.json", tokens)
    write_json(args.out, "loss_curve.json", [round(l, DECIMALS) for l in losses])
    write_json(args.out, "weights_final.json", rounded(W_final))
    write_json(args.out, "top_next.json", top_next(probs_final, id_to_word))
    write_json(args.out, "samples.json", samples)
    write_json(args.out, "snapshots.json", snapshots)
    print("done.")


if __name__ == "__main__":
    main()
