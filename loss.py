import numpy as np 

class CategoricalCrossentropyLogits():
    def __init__(self):
        pass

    def loss(self, logits, labels):
        logits = logits - np.max(logits, axis=-1, keepdims=True)
        logits_at_labels = logits[np.arange(len(logits)), labels]
        crossentropy = -logits_at_labels + np.log(np.sum(np.exp(logits), axis=-1))
        return crossentropy

    def grad(self, logits, labels):
        labels_dummy = np.zeros_like(logits)
        labels_dummy[np.arange(len(logits)), labels] = 1
        logits = logits - np.max(logits, axis=-1, keepdims=True)
        softmax = np.exp(logits) / np.exp(logits).sum(axis=-1,keepdims=True)
        return (-labels_dummy + softmax) / logits.shape[0]