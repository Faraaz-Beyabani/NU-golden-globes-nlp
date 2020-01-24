import json

def main():
    preprocess()
    pass

def preprocess(filepath = "./data/gg2020.json"):
    with open(filepath, encoding="utf8") as f:
        for line in f:
            tweet = json.loads(str(line))
            # print(tweet["text"]) does exactly what you think it does, don't actually run this, it crashes the shell

if __name__=="__main__":
    main()