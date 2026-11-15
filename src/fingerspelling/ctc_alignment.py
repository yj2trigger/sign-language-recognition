"""Existing alignment extracted unchanged from src/fingerspelling_partial_compose.py."""
def align_tokens(reference, prediction):
    """Minimum Levenshtein path. Ties prefer diagonal, deletion, insertion.

    Indices point into the original arrays. Insertion ref_gap is the number
    of consumed reference tokens, allowing role analysis without inventing a
    ground-truth syllable role for an inserted prediction.
    """
    costs = [[0] * (len(prediction) + 1) for _ in range(len(reference) + 1)]
    for i in range(len(reference) + 1):
        costs[i][0] = i
    for j in range(len(prediction) + 1):
        costs[0][j] = j
    for i, expected in enumerate(reference, 1):
        for j, actual in enumerate(prediction, 1):
            costs[i][j] = min(costs[i - 1][j - 1] + (expected != actual),
                              costs[i - 1][j] + 1, costs[i][j - 1] + 1)
    path = []
    i, j = len(reference), len(prediction)
    while i or j:
        if i and j and costs[i][j] == costs[i - 1][j - 1] + (reference[i - 1] != prediction[j - 1]):
            path.append(dict(op="M" if reference[i - 1] == prediction[j - 1] else "S",
                             ref_index=i - 1, pred_index=j - 1, ref_gap=i - 1,
                             expected=reference[i - 1], predicted=prediction[j - 1]))
            i, j = i - 1, j - 1
        elif i and costs[i][j] == costs[i - 1][j] + 1:
            path.append(dict(op="D", ref_index=i - 1, pred_index=None, ref_gap=i - 1,
                             expected=reference[i - 1], predicted=None))
            i -= 1
        else:
            path.append(dict(op="I", ref_index=None, pred_index=j - 1, ref_gap=i,
                             expected=None, predicted=prediction[j - 1]))
            j -= 1
    return list(reversed(path))
