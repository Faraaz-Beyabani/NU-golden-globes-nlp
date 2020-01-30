import json
import nltk
import re
from nltk.corpus import state_union
from nltk.tokenize import PunktSentenceTokenizer
from nltk import distance
from collections import defaultdict

class GoldenGlobesParser:

    tweets = []
    hosts = []
    awards = []
    presenters = {}
    nominees = {}
    winners = {}

    award_template = {"Presenters": [], "Nominees": [], "Winner": None}
    ignore = ['golden', 'globes', 'globe']
        
    # THIS LIST IS NOT RETURNED. IT IS USED TO FIND PRESENTERS, NOMINEES, AND WINNERS FOR ALL THE AWARDS IN ACCORDANCE WITH THE PROJECT GUIDELINES
    # WE ARE USING THIS LIST TO AVOID CASCADING ERROR
    OFFICIAL_AWARDS_1819 = ['best motion picture - drama', 'best motion picture - musical or comedy', 'best performance by an actress in a motion picture - drama', 'best performance by an actor in a motion picture - drama', 'best performance by an actress in a motion picture - musical or comedy', 'best performance by an actor in a motion picture - musical or comedy', 'best performance by an actress in a supporting role in any motion picture', 'best performance by an actor in a supporting role in any motion picture', 'best director - motion picture', 'best screenplay - motion picture', 'best motion picture - animated', 'best motion picture - foreign language', 'best original score - motion picture', 'best original song - motion picture', 'best television series - drama', 'best television series - musical or comedy', 'best television limited series or motion picture made for television', 'best performance by an actress in a limited series or a motion picture made for television', 'best performance by an actor in a limited series or a motion picture made for television', 'best performance by an actress in a television series - drama', 'best performance by an actor in a television series - drama', 'best performance by an actress in a television series - musical or comedy', 'best performance by an actor in a television series - musical or comedy', 'best performance by an actress in a supporting role in a series, limited series or motion picture made for television', 'best performance by an actor in a supporting role in a series, limited series or motion picture made for television', 'cecil b. demille award']
    tweetized_awards = []
    categorized_winners = defaultdict(dict)

    def __init__(self, year = 2020):
        self.year = year

    def process_tweets(self):
        nltk.download('averaged_perceptron_tagger')

        with open(f"./data/gg{self.year}.json", encoding="utf8") as f:
            try:
                self.tweets = json.load(f)
            except:
                print(f"./data/gg{self.year}.json is not in JSON format!")
                self.tweets = []
                for line in f:
                    self.tweets.append(json.loads(str(line)))

        # self.get_host(self.tweets)
        # self.get_awards(self.tweets)

        # self.process_awards()
        # self.categorize_awards()
        # self.get_presenters(self.tweets)
        # self.get_winners(self.tweets)

    def process_awards(self):
        ignore = ['award', 'motion', 'performance', 'picture', 'limited', 'original', 'series', 'series,']#, " -", ' by', " an", ' in a', ' in',  ' a', ' or', ' made', ' for']
        special = ['best', 'song']
        replace = {'television': 'tv'}
        for award in self.OFFICIAL_AWARDS_1819:
            curr = award
            temp = curr.split()

            for i, word in enumerate(temp):
                if word in temp[:i]:
                    temp[i] = ""

                if word not in special:
                    if len(word) <= 4 or word in ignore:
                        temp[i] = ""

                for k,v in replace.items():
                    if word == k:
                        temp[i] = v

            curr = ' '.join([x for x in temp if x])

            self.tweetized_awards.append(curr)
        # print(self.tweetized_awards)
      
    def categorize_awards(self):
        tknzr = nltk.tokenize.casual.TweetTokenizer(strip_handles=True)

        for award in self.tweetized_awards:
            self.categorized_winners[award] = defaultdict(int)
            for tweet in self.tweets:
                if self.match_award(tweet, award):
                    if ('actor' in tweet['text'] and 'actor' not in award) or ('actress' in tweet['text'] and 'actress' not in award) or ('tv' in tweet['text'] and 'tv' not in award):
                        continue

                    tags = nltk.pos_tag(tknzr.tokenize(tweet["text"]))
                    for i in range(len(tags)-1):
                        first_tag = tags[i][1] == "NNP"
                        first_word = tags[i][0]
                        sec_tag = tags[i+1][1] == "NNP"
                        sec_word = tags[i+1][0]
                        if first_tag and sec_tag and first_word.lower() not in self.ignore and sec_word.lower() not in self.ignore:
                            maybe_name = f"{first_word} {sec_word}"
                            self.categorized_winners[award][maybe_name] += 1
        
            obsolete = []
            for name in self.categorized_winners[award].keys():
                for other_name in self.categorized_winners[award].keys():
                    if name == other_name:
                        continue
                
                diff = nltk.distance.edit_distance(name, other_name, transpositions=True)
                if diff < 7:
                    self.categorized_winners[award][name] += self.categorized_winners[award][other_name]
                    if other_name not in obsolete:
                        obsolete.append(other_name)

            for obs in obsolete:
                del self.categorized_winners[award][obs]


        for k,v in self.categorized_winners.items():
            best_n = float('-inf')
            best_c = None
            for c, n in v.items():
                if n > best_n:
                    best_n = n
                    best_c = c
            print(f"{best_c} won {k} with {best_n} occurrences\n")

    def match_award(self, tweet, award):
        for word in award.split():
            if word not in tweet["text"]:
                return False

        return True
        

    def get_host(self, tweets):
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
                    if first_tag and sec_tag and first_word.lower() not in self.ignore and sec_word.lower() not in self.ignore:
                        maybe_name = f"{first_word} {sec_word}"
                        if maybe_name in host_count.keys():
                            host_count[maybe_name] += 1
                        else:
                            host_count[maybe_name] = 0
        hosts = [k for k, _ in sorted(host_count.items(), key=lambda item: item[1], reverse=True)][:10]

        if host_plural/host_tweets < 0.5:
            self.hosts = hosts[:1]
        else:
            self.hosts = hosts[:2]
    
    def get_awards(self, tweets):
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
                self.awards.append(dic[key][0])

        print(self.awards)
        print(len(self.awards))
    
    def get_presenters(self, tweets): 
        present_count = {"presenters": {}, "present_plural": 0, "present_tweets": 0}
        tknzr = nltk.tokenize.casual.TweetTokenizer(strip_handles=True)

        for tweet in tweets:
            should_ignore = None

            text = tweet['text']
            if " present" not in text:
                continue
            
            # for word in [" next ", " last ", " host", " nominat"]:
            #     if word in text:
            #         should_ignore = True

            # if should_ignore:
            #     continue
            
            print(text)

        

        

if __name__=="__main__":
    dog = GoldenGlobesParser()
    dog.process_tweets()





    # for tweet in tweets:
    #         text = tweet['text']

    #         best_score = float("-inf")
    #         best_award = ""
    #         best_index = 0

    #         match = re.search(r'\(?((cecil)|(best)).*((award)|(drama)|(film)|(musical)|(motion picture)|(television)|(comedy))\)?', text.lower())

    #         if not match:
    #             continue
            
    #         for i, award in enumerate(self.tweetized_awards):
    #             score = 0
    #             award_key = ''.join(c for c in award if c.isalpha() or c==' ').split(' ')

    #             for w in award_key:
    #                 if w in match.group():
    #                     score += 1

    #             if score > best_score:
    #                 best_score = score
    #                 best_award = award
    #                 best_index = i

    #         if ' present' in text:
    #             present_count[self.OFFICIAL_AWARDS_1819[best_index]]["present_tweets"] += 1
    #             if ' and ' in text:
    #                 present_count[self.OFFICIAL_AWARDS_1819[best_index]]["present_plural"] += 1

    #             tags = nltk.pos_tag(tknzr.tokenize(text))
    #             for i in range(len(tags)-1):
    #                 first_tag = tags[i][1] == "NNP"
    #                 first_word = tags[i][0]
    #                 sec_tag = tags[i+1][1] == "NNP"
    #                 sec_word = tags[i+1][0]
    #                 if first_tag and sec_tag and first_word.lower() not in self.ignore and sec_word.lower() not in self.ignore:
    #                     maybe_name = f"{first_word} {sec_word}"
    #                     if maybe_name in present_count[self.OFFICIAL_AWARDS_1819[best_index]]["presenters"].keys():
    #                         present_count[self.OFFICIAL_AWARDS_1819[best_index]]["presenters"][maybe_name] += 1
    #                     else:
    #                         present_count[self.OFFICIAL_AWARDS_1819[best_index]]["presenters"][maybe_name] = 1

    #     # for k in present_count.keys():
    #     #     present_count[k]["presenters"] = sorted(present_count[k]["presenters"].items(), key=lambda item: item[1], reverse=True)[:10]

    #     print(present_count)