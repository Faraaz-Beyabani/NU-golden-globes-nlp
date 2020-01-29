import json
import nltk
import re
from nltk.corpus import state_union
from nltk.tokenize import PunktSentenceTokenizer

ignore = ['Golden', 'Globes', 'Globe']

OFFICIAL_AWARDS_1819 = ['best motion picture - drama', 'best motion picture - musical or comedy', 'best performance by an actress in a motion picture - drama', 'best performance by an actor in a motion picture - drama', 'best performance by an actress in a motion picture - musical or comedy', 'best performance by an actor in a motion picture - musical or comedy', 'best performance by an actress in a supporting role in any motion picture', 'best performance by an actor in a supporting role in any motion picture', 'best director - motion picture', 'best screenplay - motion picture', 'best motion picture - animated', 'best motion picture - foreign language', 'best original score - motion picture', 'best original song - motion picture', 'best television series - drama', 'best television series - musical or comedy', 'best television limited series or motion picture made for television', 'best performance by an actress in a limited series or a motion picture made for television', 'best performance by an actor in a limited series or a motion picture made for television', 'best performance by an actress in a television series - drama', 'best performance by an actor in a television series - drama', 'best performance by an actress in a television series - musical or comedy', 'best performance by an actor in a television series - musical or comedy', 'best performance by an actress in a supporting role in a series, limited series or motion picture made for television', 'best performance by an actor in a supporting role in a series, limited series or motion picture made for television', 'cecil b. demille award']




award_names = []

award_template = {"Presenters": [], "Nominees": [], "Winner": None}

def main():
    nltk.download('averaged_perceptron_tagger')

    process_tweets()

def process_tweets(filepath = "./data/gg2020.json"):
    tweets = []
    with open(filepath, encoding="utf8") as f:
        for line in f:
            tweets.append(json.loads(str(line)))
    #get_host(tweets)
    get_awards(tweets)
        
    
    # if(host_plural < host_sing):
    #     print(hosts[:1])
    # else:

def get_host(tweets):
    host_count = {}
    host_plural = 0
    host_tweets = 0
    tknzr = nltk.tokenize.casual.TweetTokenizer(strip_handles=True)
    for tweet in tweets:
        text = tweet['text']
        if ' next ' in text:
            continue

        if ' host' in text:
            host_tweets += 1
            if ' and ' in text:
                host_plural += 1

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
    hosts = [k for k, _ in sorted(host_count.items(), key=lambda item: item[1], reverse=True)][:10]
    print(hosts)

def get_awards(tweets):
    dic = {}
    for tweet in tweets:
        text = tweet['text']
        match = re.search(r'\(?(best).*((drama)|(film)|(musical)|(motion picture)|(television)|(comedy))\)?', text.lower())
        if match:
            result = ''.join(c for c in match.group() if c.isalpha() or c == ' ')
            result = result.replace('  ',' ')
            if result in dic:
                dic[result][1] += 1
            else:
                dic[result] = [match.group(), 0]
    for key in dic:
        if dic[key][1] > 10:
            print(dic[key][0]+'\n')

        

if __name__=="__main__":
    main()