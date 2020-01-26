import re
import json
import nltk

ignore = ['Golden', 'Globes', 'Globe', 'Award', 'Best']

host_count = {}
award_template = {"Presenters": None, "Nominees": [], "Winner": None}

def main():
    nltk.download('averaged_perceptron_tagger')
    process_tweets()

def process_tweets(filepath = "./data/gg2020.json"):
    global host_count

    with open(filepath, encoding="utf8") as f:
        for line in f:
            tweet = json.loads(str(line))
            get_host(tweet)
        
    hosts = [k for k, _ in sorted(host_count.items(), key=lambda item: item[1], reverse=True)]
    hosts = hosts[:2]
    print(hosts)

def get_host(tweet):
    global host_count
    tknzr = nltk.tokenize.casual.TweetTokenizer(strip_handles=True)
    text = tweet['text']
    if ' host' in text:
        tags = nltk.pos_tag(tknzr.tokenize(text))
        for i in range(len(tags)-1):
            first_tag = tags[i][1] == "NNP"
            first_word = tags[i][0]
            sec_tag = tags[i+1][1] == "NNP"
            sec_word = tags[i+1][0]
            if first_tag and sec_tag and first_word not in ignore and sec_word not in ignore:
                maybe_name = f"{first_word} {sec_word}"
                if maybe_name in host_count:
                    host_count[maybe_name] += 1
                else:
                    host_count[maybe_name] = 0
        

if __name__=="__main__":
    main()