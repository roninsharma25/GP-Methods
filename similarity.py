import numpy as np
import pandas as pd

# Ranges is the an array with the range of each input column (ex: [1, 1, 1, 1])
def detectSimilarPoints(data, ranges, thres):
    data = pd.read_csv(data).to_numpy(dtype="float")

    for row1 in data:
        lineNumber = 1
        for row2 in data:
            lineNumber += 1
            if (not np.array_equal(row1, row2)):
                differenceScore = differenceScore(row1, row2, ranges)
                if (differenceScore < thres ** 2): # squared to reduce computational complexity
                    print(differenceScore)
                    print(lineNumber)
                    print('---')

def differenceScore(row1, row2, ranges):
    score = 0
    for i in range(len(row1)):
        score += ((row1[i] - row2[i]) / ranges[i]) ** 2
    return score
