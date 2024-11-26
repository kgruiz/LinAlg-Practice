import os
import re


def ListFilesInDir(directory):

    try:

        files = [
            f
            for f in os.listdir(directory)
            if f.endswith(".py") and f != os.path.basename(__file__)
        ]
        return files

    except FileNotFoundError:

        print(f"The directory {directory} does not exist.")
        return []


filesInDir = ListFilesInDir(".")

print("Files in directory:")
for fileName in filesInDir:

    print(f"  {fileName}")

print(f"{'-'*100}")

uniqueMatches = set()

for filePath in filesInDir:

    with open(filePath, "r") as file:

        content = file.read()

    pattern = r"def ([A-Z])(.+)\("
    matches = re.findall(pattern, content)

    uniqueMatches.update(matches)

uniqueMatches = sorted(uniqueMatches, key=lambda x: f"{x[0].lower()}{x[1]}")

overallResults = {}

for filePath in filesInDir:

    with open(filePath, "r") as file:

        content = file.read()

    results = {}

    for found in uniqueMatches:

        foundStr = rf"{found[0].lower()}{found[1]}"

        firstChar = found[0].upper()

        content, count = re.subn(
            rf"{re.escape(foundStr)}", rf"{firstChar}{found[1]}", content
        )

        results[foundStr] = count

    with open(filePath, "w") as file:
        file.write(content)

    for term, count in results.items():

        if term not in overallResults:

            overallResults[term] = []

        overallResults[term].append((filePath, count))

print(f"{'-'*100}")

for term, occurrences in overallResults.items():

    total = sum(count for _, count in occurrences)

    print(f"'{term}' found {total} times in:")

    for filePath, count in occurrences:

        if count > 0:

            print(f"  {filePath}: {count} times")

    print(f"{'-'*100}")
