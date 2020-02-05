import json
import nltk
import re

from nltk.tokenize import sent_tokenize, word_tokenize
from collections import defaultdict

class GoldenGlobesParser:

    tweets = []
    hosts = []
    awards = []
    presenters = {}
    nominees = {}
    winners = {}

    ignore = ['golden', 'globes', 'globe', 'goldenglobes', 'goldenglobe', 'gg2020']
    tweetized_awards = {}
    award_words = []
        
    # THIS LIST IS NOT RETURNED. IT IS USED TO FIND PRESENTERS, NOMINEES, AND WINNERS FOR ALL THE AWARDS IN ACCORDANCE WITH THE PROJECT GUIDELINES
    # WE ARE USING THIS LIST TO AVOID CASCADING ERROR
    OFFICIAL_AWARDS_1819 = ['best motion picture - drama', 'best motion picture - musical or comedy', 'best performance by an actress in a motion picture - drama', 'best performance by an actor in a motion picture - drama', 'best performance by an actress in a motion picture - musical or comedy', 'best performance by an actor in a motion picture - musical or comedy', 'best performance by an actress in a supporting role in any motion picture', 'best performance by an actor in a supporting role in any motion picture', 'best director - motion picture', 'best screenplay - motion picture', 'best motion picture - animated', 'best motion picture - foreign language', 'best original score - motion picture', 'best original song - motion picture', 'best television series - drama', 'best television series - musical or comedy', 'best television limited series or motion picture made for television', 'best performance by an actress in a limited series or a motion picture made for television', 'best performance by an actor in a limited series or a motion picture made for television', 'best performance by an actress in a television series - drama', 'best performance by an actor in a television series - drama', 'best performance by an actress in a television series - musical or comedy', 'best performance by an actor in a television series - musical or comedy', 'best performance by an actress in a supporting role in a series, limited series or motion picture made for television', 'best performance by an actor in a supporting role in a series, limited series or motion picture made for television', 'cecil b. demille award']
    OFFICIAL_AWARDS_1315 = ['cecil b. demille award', 'best motion picture - drama', 'best performance by an actress in a motion picture - drama', 'best performance by an actor in a motion picture - drama', 'best motion picture - comedy or musical', 'best performance by an actress in a motion picture - comedy or musical', 'best performance by an actor in a motion picture - comedy or musical', 'best animated feature film', 'best foreign language film', 'best performance by an actress in a supporting role in a motion picture', 'best performance by an actor in a supporting role in a motion picture', 'best director - motion picture', 'best screenplay - motion picture', 'best original score - motion picture', 'best original song - motion picture', 'best television series - drama', 'best performance by an actress in a television series - drama', 'best performance by an actor in a television series - drama', 'best television series - comedy or musical', 'best performance by an actress in a television series - comedy or musical', 'best performance by an actor in a television series - comedy or musical', 'best mini-series or motion picture made for television', 'best performance by an actress in a mini-series or motion picture made for television', 'best performance by an actor in a mini-series or motion picture made for television', 'best performance by an actress in a supporting role in a series, mini-series or motion picture made for television', 'best performance by an actor in a supporting role in a series, mini-series or motion picture made for television']
    official_awards = []
    
    award_template = {"Presenters": [], "Nominees": [], "Winner": None}
    
    categorized_winners = defaultdict(dict)
    categorized_noms = defaultdict(dict)

    def __init__(self, year = None):
        self.year = year or 2020
        if self.year == 2013 or self.year == 2015:
            self.official_awards = self.OFFICIAL_AWARDS_1315
        else:
            self.official_awards = self.OFFICIAL_AWARDS_1315

    def parse_json(self, file):
        tweets = []
        try:
            tweets = json.load(file)
        except:
            print(f"./data/gg{self.year}.json is not in JSON format!")
        finally:
            return tweets

    def process_tweets(self):
        nltk.download('punkt')
        nltk.download('averaged_perceptron_tagger')
        nltk.download('maxent_ne_chunker')
        nltk.download('words')

        punc = ['.', ',', '-', '\'', '"']

        with open(f"./data/gg{self.year}.json", encoding="utf8") as f:    
            self.tweets = self.parse_json(f)

        if len(self.tweets) == 0:
            with open(f"./data/gg{self.year}.json", encoding="utf8") as f:
                for line in f:
                    self.tweets.append(json.loads(str(line)))
            

        self.extract_host(self.tweets)
        self.get_awards(self.tweets)

        self.process_awards()
        self.extract_winners()
        self.extract_noms()
        
        # self.get_presenters(self.tweets)

    def process_awards(self):
        ignore = ['award', 'motion', 'performance', 'picture', 'original', 'series,', 'limited']
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

            self.tweetized_awards[award] = curr

        self.award_words = set(['tv', 'movie', 'wins', 'won', 'film', 'feature'])
        for award in self.official_awards:
            for word in award.split():
                if len(word) > 3:
                    self.award_words.add(word)
        
    def extract_host(self, tweets):
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
                    first_word = tags[i][0].lower()
                    sec_tag = tags[i+1][1] == "NNP"
                    sec_word = tags[i+1][0].lower()
                    if first_tag and sec_tag and first_word not in self.ignore and sec_word not in self.ignore:
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

        print(self.hosts)
    
    def get_awards(self, tweets):
        dic = {}
        for tweet in tweets:
            text = tweet['text']
            match = re.search(r'for best .* for', text.lower())
            if match:
                result = ''.join(c for c in match.group() if c.isalpha() or c == ' ')
                result = result.replace('  ',' ')
                if result in dic:
                    dic[result][1] += 1
                else:
                    dic[result] = [result, 0]
        for key in dic:
            if dic[key][1] > 2:
                self.awards.append(dic[key][0])

        for i, a in enumerate(self.awards):
            self.awards[i] = ' '.join(a.split()[1:-1])

        print(self.awards)

    def extract_winners(self):
        aw_map = {}
        person_key = ['actor', 'actress', 'director', 'screenplay', 'award']

        for original, award in self.tweetized_awards.items():
            aw_map[award] = defaultdict(int)
            for tweet in self.tweets:
                if not self.match_award(tweet, award):
                    continue

                text = tweet['text'].replace('elevision', 'v')

                sent = nltk.sent_tokenize(text)
                word = [nltk.word_tokenize(s) for s in sent]
                tags = [nltk.pos_tag(w) for w in word]
                for tag in tags:
                    for chunk in nltk.ne_chunk(tag):
                        if type(chunk) == nltk.tree.Tree:
                            good = ['PERSON'] if 'actor' in award or 'actress' in award or 'director' in award or 'screenplay' in award or 'cecil' in award else ['GPE']
                            if chunk.label() in good:
                                maybe_name = ' '.join([c[0] for c in chunk])
                                check = maybe_name.lower()
                                for i in self.ignore:
                                    if i in check:
                                        maybe_name = ''
                                for aw in self.award_words:
                                    if aw in check:
                                        maybe_name = ''
                                for pk in person_key:
                                    if pk in award and len(maybe_name.split()) > 3:
                                        maybe_name = ''

                                if maybe_name:
                                    aw_map[award][maybe_name] += 1

            for a,b in aw_map[award].items():
                for c,d in aw_map[award].items():
                    if a == c:
                        continue
                    if c in a:
                        for e in a.split():
                            if e in aw_map[award].keys() and aw_map[award][e] > 3:
                                aw_map[award][a] += d
                                aw_map[award][c] = 0


            # best_5 = {e:f for e, f in sorted(aw_map[award].items(), key=lambda item: item[1], reverse=True)[:5]}
            self.winners[original] = sorted(aw_map[award].items(), key=lambda item: item[1], reverse=True)[0][0]
            print(f"The winner of {original} is {self.winners[original]}")


    def match_award(self, tweet, award):
        spec = ['actor', 'actress', 'tv']
        text = tweet['text'].lower()
        for word in award.split():
            if word not in text:
                return False

        for s in spec:
            if s in text and s not in award:
                return False

        return True

    def extract_noms(self):
        aw_map = {}
        person_key = ['actor', 'actress', 'director', 'screenplay', 'award']

        for original, award in self.tweetized_awards.items():
            aw_map[award] = defaultdict(int)
            for tweet in self.tweets:
                if not self.match_award(tweet, award) and not self.match_award(tweet, self.winners[original]):
                    continue

                text = tweet['text'].replace('elevision', 'v')

                if not ("nominat" in text or "should" in text or "instead" in text):
                    continue

                sent = nltk.sent_tokenize(text)
                word = [nltk.word_tokenize(s) for s in sent]
                tags = [nltk.pos_tag(w) for w in word]
                for tag in tags:
                    for chunk in nltk.ne_chunk(tag):
                        if type(chunk) == nltk.tree.Tree:
                            good = ['PERSON'] if 'actor' in award or 'actress' in award or 'director' in award or 'screenplay' in award or 'cecil' in award else ['GPE']
                            if chunk.label() in good:
                                maybe_name = ' '.join([c[0] for c in chunk])
                                check = maybe_name.lower()
                                for i in self.ignore:
                                    if i in check:
                                        maybe_name = ''
                                for aw in self.award_words:
                                    if aw in check:
                                        maybe_name = ''
                                for pk in person_key:
                                    if pk in award and len(maybe_name.split()) > 3:
                                        maybe_name = ''

                                if maybe_name:
                                    aw_map[award][maybe_name] += 1

            for a,_ in aw_map[award].items():
                for c,d in aw_map[award].items():
                    if a == c:
                        continue
                    if c in a:
                        for e in a.split():
                            if e in aw_map[award].keys() and aw_map[award][e] > 3:
                                aw_map[award][a] += d
                                aw_map[award][c] = 0


            best_5 = {e:f for e, f in sorted(aw_map[award].items(), key=lambda item: item[1], reverse=True)[:5]}
            print(f"The nominees for {original} were {best_5}")
    
    # def get_presenters(self, tweets): 
    #     present_count = {"presenters": {}, "present_plural": 0, "present_tweets": 0}
    #     tknzr = nltk.tokenize.casual.TweetTokenizer(strip_handles=True)

    #     for tweet in tweets:
    #         should_ignore = None

    #         text = tweet['text']
    #         if " present" not in text:
    #             continue

if __name__=="__main__":
    dog = GoldenGlobesParser(2015)
    dog.process_tweets()