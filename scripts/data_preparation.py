import pandas as pd

def load_file(path, label):
    with open(path, "r", encoding="utf-8") as f:
        lines = [line.strip() for line in f if len(line.strip()) > 10]
    return pd.DataFrame({"text": lines, "label": label})

formal = load_file("../data/formal.txt", "formal")
informal = load_file("../data/informal.txt", "informal")

df = pd.concat([formal, informal], ignore_index = True)
df.to_csv("../data/labeled_texts.csv", index = False)

print("Data prepared!")
