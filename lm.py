import numpy

# Store data from all texts in one variable
def load_text(paths):
    parts = []
    for path in paths:
        with open(path, "r") as f:
            parts.append(f.read())
    return " ".join(parts)

# Tokenization
def tokenization(text):
        tokens = text.split()                   # 324 tokens
        vocab = sorted(set(tokens))     # 112 unique tokens
        word_to_id = {w: i for i, w in enumerate(vocab)}
        id_to_word = {i: w for i, w in enumerate(vocab)}
        return tokens, vocab, word_to_id, id_to_word

SEED = 0
LEARNING_RATE = 1.0
STEPS = 20000

# Transform the normal text into ids because the model works with numbers
def text_to_ids(tokens, word_to_id):
        id_text = []
        for i in range(len(tokens)):
                id_text.append(word_to_id[tokens[i]])
        return id_text

def make_pairs(id_text):
        # Find the current word
        current = id_text[:-1]
        # Find the target word that is after the current word in the id_text
        target = id_text[1:]
        return current, target

# turn a list of raw scores into probabilities between 0 and 1
def softmax(scores):
    e = numpy.exp(scores - scores.max(axis=1, keepdims=True))
    total = e.sum(axis=1, keepdims=True)
    return e / total

def train(current, target, vocab, seed=SEED, learning_rate=LEARNING_RATE, steps=STEPS):
        # now we will create a grid holding the position of our current word
        # by using the lenght of current (323) as the rows and vocab (112) as col
        # this will map where our current word is located at on the grid
        N = len(current)
        V = len(vocab)
        # Now we assign random weights that get adjusted after each run
        W = numpy.random.default_rng(seed).normal(scale=0.01, size=(V, V))
        losses = []
        for step in range(steps):
                # get current scores from W[current] row and turn them into
                # probabilities for the next word after the current one
                probs = softmax(W[current])
                # calculate loss to see how wrong we are about the prediction
                loss = -numpy.mean(numpy.log(probs[numpy.arange(N), target]))

                # now here instead of using the one-hot method which i learned that
                # is rather slow we use this method that searches through the weights
                # and instead of multiplying everything by 0 except the predicted word
                # we copy the probability matrix (2 dimensional array) and subtract 1
                # from where the target is supposed to be. In that way we check if the
                # result was -0.88 (0.12 - 1 = -0.88) a big negative number
                # but if the result is 0.12 (0.12 - 0 = 0.12) nothing changes on wrong words
                d = probs.copy()
                # this acts as the error signal
                d[numpy.arange(N), target] -= 1
                # when we divide by N that means that we get the average the error
                d = d / N

                # the slope helps us understant where we are on the 'bowl' -
                # whether we are climbing the slope and its going up or its going down
                slope = numpy.zeros_like(W)
                # this line here helps us avoid repetition inside our slope
                # for example if we have the word 'the' in index 0 and 5 instead of
                # adding it two times, add.at combines all repetitive words
                numpy.add.at(slope, current, d)
                # the slope calculates in which direction the weight (W) should move (left/right)
                W = W - learning_rate * slope
                losses.append(loss)
        return losses, W

def generate_text(word_to_id, id_to_word, W, start="the", length=8, seeds=(0, 1, 2)):
        if start not in word_to_id:
                raise ValueError(f"start word '{start}' is not in the vocabulary")
        V = len(id_to_word)
        all_probs = softmax(W)
        for i in seeds:
                rng = numpy.random.default_rng(i)
                sentence = start + " "
                current_w = word_to_id[start]
                for _ in range(length):
                        next_id = rng.choice(V, p=all_probs[current_w])
                        next_word = id_to_word[next_id]
                        current_w = next_id
                        sentence += next_word + " "
                print(sentence)

if __name__ == "__main__":
       text = load_text(["data/ocean.txt", "data/space.txt"])
       tokens, vocab, word_to_id, id_to_word = tokenization(text)
       id_text = text_to_ids(tokens, word_to_id)
       current, target = make_pairs(id_text)
       losses, W = train(current, target, vocab)
       print(f"tokens: {len(tokens)} | vocab: {len(vocab)} | pairs: {len(current)}")
       print(f"loss at start: {losses[0]:.3f} | loss at end: {losses[-1]:.3f}")
       generate_text(word_to_id, id_to_word, W)