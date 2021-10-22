import numpy as np
import pandas as pd

def differenceScore(row1, row2, ranges):
    score = 0
    for i in range(len(row1)):
        score += ((row1[i] - row2[i]) / ranges[i]) ** 2
    return score

# Ranges is the an array with the range of each input column (ex: [1, 1, 1, 1])
def detectSimilarPoints(prev, new, ranges, thres):
    newData = []
    problemData = []

    for row1 in prev:
        lineNumber = 1
        for row2 in new:
            lineNumber += 1
            if (row2 not in problemData):
                score = differenceScore(row1, row2, ranges)
                if (score < thres ** 2): # squared to reduce computational complexity
                    # print(score)
                    # print(lineNumber)
                    # print('---')
                    problemData.append(row2)
                else:
                    newData.append(row2)

    # for row1 in combined:
    #     lineNumber = 1
    #     for row2 in combined:
    #         lineNumber += 1
    #         if (not np.array_equal(row1, row2) and row2 not in newData and row2 not in problemData):
    #             differenceScore = differenceScore(row1, row2, ranges)
    #             if (differenceScore < thres ** 2): # squared to reduce computational complexity
    #                 print(differenceScore)
    #                 print(lineNumber)
    #                 print('---')
    #                 problemData.append(row2)
    #             else:
    #                 newData.append(row2)

    return np.array(newData, dtype='float')
