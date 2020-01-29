import json
import nltk
import re
from nltk.corpus import state_union
from nltk.tokenize import PunktSentenceTokenizer

ignore = ['golden', 'globes', 'globe']

OFFICIAL_AWARDS_1819 = ['best motion picture - drama', 'best motion picture - musical or comedy', 'best performance by an actress in a motion picture - drama', 'best performance by an actor in a motion picture - drama', 'best performance by an actress in a motion picture - musical or comedy', 'best performance by an actor in a motion picture - musical or comedy', 'best performance by an actress in a supporting role in any motion picture', 'best performance by an actor in a supporting role in any motion picture', 'best director - motion picture', 'best screenplay - motion picture', 'best motion picture - animated', 'best motion picture - foreign language', 'best original score - motion picture', 'best original song - motion picture', 'best television series - drama', 'best television series - musical or comedy', 'best television limited series or motion picture made for television', 'best performance by an actress in a limited series or a motion picture made for television', 'best performance by an actor in a limited series or a motion picture made for television', 'best performance by an actress in a television series - drama', 'best performance by an actor in a television series - drama', 'best performance by an actress in a television series - musical or comedy', 'best performance by an actor in a television series - musical or comedy', 'best performance by an actress in a supporting role in a series, limited series or motion picture made for television', 'best performance by an actor in a supporting role in a series, limited series or motion picture made for television', 'cecil b. demille award']

award_template = {"Presenters": [], "Nominees": [], "Winner": None}

def main():
    nltk.download('averaged_perceptron_tagger')

    process_tweets()

def process_tweets(filepath = "./data/gg2020.json"):
    tweets = []
    with open(filepath, encoding="utf8") as f:
        for line in f:
            tweets.append(json.loads(str(line)))

    # get_host(tweets)
    # get_awards(tweets)
    get_presenters(tweets)

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
                        host_count[maybe_name] = 1
    hosts = [k for k, _ in sorted(host_count.items(), key=lambda item: item[1], reverse=True)][:10]
    if host_plural/host_tweets < 0.5:
        print(hosts[:1])
    else:
        print(hosts[:2])

def get_awards(tweets):
    dic = {}
    for tweet in tweets:
        text = tweet['text']
        match = re.search(r'\(?((cecil)|(best)).*((award)|(drama)|(film)|(musical)|(motion picture)|(television)|(comedy))\)?', text.lower())
        if match:
            result = ''.join(c for c in match.group() if c.isalpha() or c == ' ')
            result = result.replace('  ',' ')
            if result in dic:
                dic[result][1] += 1
            else:
                dic[result] = [match.group(), 0]
    for key in dic:
        if dic[key][1] > 10:
            # print(dic[key][0]+'\n')
            pass

def get_presenters(tweets):
    present_count = {k:{"presenters": {}, "present_plural": 0, "present_tweets": 0} for k in OFFICIAL_AWARDS_1819}
    found_award = None

    tknzr = nltk.tokenize.casual.TweetTokenizer(strip_handles=True)

    for tweet in tweets:
        text = tweet['text']
        match = re.search(r'\(?((cecil)|(best)).*((award)|(drama)|(film)|(musical)|(motion picture)|(television)|(comedy))\)?', text.lower())

        if not match:
            continue
        
        for award in OFFICIAL_AWARDS_1819:
            closest_award = None
            best_score = float('-inf')

        # FIX THIS CODE PLS

        if ' present' in text:
            present_count[found_award]["present_tweets"] += 1
            if ' and ' in text:
                present_count[found_award]["present_plural"] += 1

            tags = nltk.pos_tag(tknzr.tokenize(text))
            for i in range(len(tags)-1):
                first_tag = tags[i][1] == "NNP"
                first_word = tags[i][0]
                sec_tag = tags[i+1][1] == "NNP"
                sec_word = tags[i+1][0]
                if first_tag and sec_tag and first_word.lower() not in ignore and sec_word.lower() not in ignore:
                    maybe_name = f"{first_word} {sec_word}"
                    if maybe_name in present_count[found_award]:
                        present_count[found_award]["presenters"][maybe_name] += 1
                    else:
                        present_count[found_award]["presenters"][maybe_name] = 1

    # for k in present_count.keys():
    #     present_count[k]["presenters"] = sorted(present_count[k]["presenters"].items(), key=lambda item: item[1], reverse=True)[:10]

    print(present_count["best actor drama"]["presenters"])

        

if __name__=="__main__":
    main()