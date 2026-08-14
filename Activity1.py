students = ["Alice", "Bob", "Charlie", "David"]
scores = [85, 90, None, 70]

scores[2] = (scores[0] + scores[1] + scores[3])/3

def calculateAverage(scores):
    return sum(scores) / len(scores)

average = calculateAverage(scores)

for student in range(len(students)):
    if scores[student] > average:
        print(f"Student {students[student]} has a score of {scores[student]} which is greater than the average score of {average}")
    else:
        print(f"Student {students[student]} has a score of {scores[student]} which is not greater than the average score of {average}")
