import json
import nltk
import re
import time
from nltk.tokenize import sent_tokenize, word_tokenize
from collections import defaultdict

class GoldenGlobesParser:

    tweets = []

    hosts = []
    awards = []
    winners = {}
    presenters = defaultdict(str)
    nominees = defaultdict(str)

    award_words = set(['tv', 'movie', 'wins', 'won', 'film', 'feature'])
    ignore = ['golden', 'globes', 'globe', 'goldenglobes', 'goldenglobe', 'gg2020']
    tweetized_awards = {}
        
    # THIS LIST IS NOT RETURNED. IT IS USED TO FIND PRESENTERS, NOMINEES, AND WINNERS FOR ALL THE AWARDS IN ACCORDANCE WITH THE PROJECT GUIDELINES
    # WE ARE USING THIS LIST TO AVOID CASCADING ERROR
    OFFICIAL_AWARDS_1819 = ['best motion picture - drama', 'best motion picture - musical or comedy', 'best performance by an actress in a motion picture - drama', 'best performance by an actor in a motion picture - drama', 'best performance by an actress in a motion picture - musical or comedy', 'best performance by an actor in a motion picture - musical or comedy', 'best performance by an actress in a supporting role in any motion picture', 'best performance by an actor in a supporting role in any motion picture', 'best director - motion picture', 'best screenplay - motion picture', 'best motion picture - animated', 'best motion picture - foreign language', 'best original score - motion picture', 'best original song - motion picture', 'best television series - drama', 'best television series - musical or comedy', 'best television limited series or motion picture made for television', 'best performance by an actress in a limited series or a motion picture made for television', 'best performance by an actor in a limited series or a motion picture made for television', 'best performance by an actress in a television series - drama', 'best performance by an actor in a television series - drama', 'best performance by an actress in a television series - musical or comedy', 'best performance by an actor in a television series - musical or comedy', 'best performance by an actress in a supporting role in a series, limited series or motion picture made for television', 'best performance by an actor in a supporting role in a series, limited series or motion picture made for television', 'cecil b. demille award']
    OFFICIAL_AWARDS_1315 = ['cecil b. demille award', 'best motion picture - drama', 'best performance by an actress in a motion picture - drama', 'best performance by an actor in a motion picture - drama', 'best motion picture - comedy or musical', 'best performance by an actress in a motion picture - comedy or musical', 'best performance by an actor in a motion picture - comedy or musical', 'best animated feature film', 'best foreign language film', 'best performance by an actress in a supporting role in a motion picture', 'best performance by an actor in a supporting role in a motion picture', 'best director - motion picture', 'best screenplay - motion picture', 'best original score - motion picture', 'best original song - motion picture', 'best television series - drama', 'best performance by an actress in a television series - drama', 'best performance by an actor in a television series - drama', 'best television series - comedy or musical', 'best performance by an actress in a television series - comedy or musical', 'best performance by an actor in a television series - comedy or musical', 'best mini-series or motion picture made for television', 'best performance by an actress in a mini-series or motion picture made for television', 'best performance by an actor in a mini-series or motion picture made for television', 'best performance by an actress in a supporting role in a series, mini-series or motion picture made for television', 'best performance by an actor in a supporting role in a series, mini-series or motion picture made for television']
    official_awards = []

    def __init__(self, year = 2020):
        self.year = year
        if year == 2013 or year == 2015 or year == '2013' or year == '2015':
            self.official_awards = self.OFFICIAL_AWARDS_1315
        else:
            self.official_awards = self.OFFICIAL_AWARDS_1819

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

        with open(f"./data/gg{self.year}.json", encoding="utf8") as f:    
            self.tweets = self.parse_json(f)

        if len(self.tweets) == 0:
            with open(f"./data/gg{self.year}.json", encoding="utf8") as f:
                for line in f:
                    self.tweets.append(json.loads(str(line)))

        for tweet in self.tweets:
            tweet['text'] = tweet['text'].replace('&amp;', 'and').replace('RT ','')

        for tweet in self.tweets:
            text = tweet['text']
            if '#' in text:
                hashtag = re.findall(r'\B#\w\w*', text)
                if hashtag:
                    for h in hashtag:
                        phrase = h[1:]
                        indices = []
                        for i in range(len(phrase) - 1):
                            if phrase[i].islower() and phrase[i+1].isupper():
                                indices.append(i+1)
                        clone = list(phrase)
                        for i in indices[::-1]:
                            clone.insert(i, ' ')
                        tweet['text'] = tweet['text'].replace(h, ''.join(clone))

        self.process_awards()

        # self.extract_host()
        # self.extract_awards()
        self.extract_winners()
        self.extract_prenom()

    def process_awards(self):
        ignore = ['award', 'motion', 'performance', 'picture', 'original', 'series,']
        special = ['best', 'song']
        replace = {'television': 'tv', 'musical': '', 'comedy': 'musical/comedy', 'limited': '', 'series':'limited/series', 'mini-series':'mini'}
        for original in self.official_awards:
            award = original
            split_original = award.split()

            for i, w in enumerate(split_original):
                word = w.lower()
                if word in split_original[:i]:
                    split_original[i] = ""

                if word not in special:
                    if len(word) <= 4 or word in ignore:
                        split_original[i] = ""

                for k,v in replace.items():
                    if word == k:
                        split_original[i] = v

            award = ' '.join([x for x in split_original if x])

            self.tweetized_awards[original] = award

        for award in self.official_awards:
            for word in award.split():
                if len(word) > 3:
                    self.award_words.add(word)
                    
    def match_phrase(self, text, phrase):
        spec = ['actor', 'actress', 'tv', 'supporting']
        text_lower = text.lower()
        for word in phrase.split():
            if '/' in word:
                subwords = word.split('/')
                if not any(sw in text_lower for sw in subwords):
                    return False
            else:
                if word not in text_lower:
                    return False

        for s in spec:
            if s in text_lower and s not in phrase:
                return False

        return True
        
    def extract_host(self):
        host_count = {}
        host_plural = 0
        host_tweets = 0
        for tweet in self.tweets:
            text = tweet['text']
            if ' next ' in text:
                continue

            if ' host' in text:
                host_tweets += 1
                if ' and ' in text:
                    host_plural += 1

                tags = nltk.pos_tag(nltk.tokenize.casual.TweetTokenizer(strip_handles=True).tokenize(text))
                for i in range(len(tags)-1):
                    first_tag = tags[i][1] == "NNP"
                    first_word = tags[i][0].lower()
                    sec_tag = tags[i+1][1] == "NNP"
                    sec_word = tags[i+1][0].lower()
                    if first_tag and sec_tag and first_word not in self.ignore and sec_word not in self.ignore:
                        maybe_name = f"{first_word.capitalize()} {sec_word.capitalize()}"
                        if maybe_name in host_count.keys():
                            host_count[maybe_name] += 1
                        else:
                            host_count[maybe_name] = 0
        hosts = [k for k, _ in sorted(host_count.items(), key=lambda item: item[1], reverse=True)][:10]

        if host_plural/host_tweets < 0.5:
            self.hosts = hosts[:1]
        else:
            self.hosts = hosts[:2]

        return self.hosts
    
    def extract_awards(self):
        dic = {}
        for tweet in self.tweets:
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
            if dic[key][1] > 3:
                self.awards.append(dic[key][0])

        for i, a in enumerate(self.awards):
            self.awards[i] = ' '.join(a.split()[1:-1])

        return self.awards

    def extract_winners(self):
        person_key = ['actor', 'actress', 'director', 'cecil']

        aw_map = {}
        for award in self.tweetized_awards.values():
            aw_map[award] = defaultdict(int)
        
        for i, tweet in enumerate(self.tweets):

            if i % 10 == 0:
                continue

            tagged = None
            regexed = []
            text = tweet['text'].replace('elevision', 'v')

            for original, award in self.tweetized_awards.items():

                if not self.match_phrase(text, award):
                    continue

                if any(p in award for p in person_key):
                    if not tagged:
                        tagged = nltk.pos_tag(nltk.tokenize.casual.TweetTokenizer(strip_handles=True).tokenize(text))

                    name = []
                    names = []
                    for i in range(len(tagged)-1):
                        if tagged[i][1] == "NNP":
                            name.append(tagged[i][0].lower())
                        else:
                            if not name:
                                continue
                            if any(mn.lower() in self.award_words for mn in name):
                                continue
                            if all(mn.lower() not in self.ignore for mn in name):
                                maybe_name = ' '.join([mn for mn in name])
                                aw_map[award][maybe_name] += 1
                                names.append(maybe_name)
                            name = []
                else:
                    if len(regexed) < 1:
                        r1 = re.search(r'".*?"', text.lower())
                        r2 = re.search(r"'.*?'", text.lower())
                        regexed = [r1, r2]
                    
                    for r in regexed:
                        if r:
                            maybe_movie = ''.join(c for c in r.group() if c.isalnum() or c == ' ')

                            if any(i in maybe_movie for i in self.ignore) or any(aw in maybe_movie for aw in self.award_words) or len(maybe_movie) > 30:
                                continue

                            aw_map[award][' '.join([w for w in maybe_movie.split()])] += 1

        for original, award in self.tweetized_awards.items():
            # print(original)
            if not aw_map[award]:
                # print('\n')
                continue
            self.winners[original] = sorted(aw_map[award].items(), key=lambda item: item[1], reverse=True)[0][0]           
            # print(f"The winner was {self.winners[original]}")
        print(self.winners)
        return self.winners

    def extract_prenom(self):
        p_map = defaultdict(dict)
        # n_map = {}
        for original, award in self.tweetized_awards.items():
            p_map[original] = defaultdict(int)
            # n_map[award] = defaultdict(int)

        for tweet in self.tweets:
            text = tweet['text']
            text_l = text.lower()
            present_keywords = [' present', ' handing', ' giving']
            chunked = None
            regexed = []

            if any(p_k in text_l for p_k in present_keywords):
                for a, w in self.winners.items():
                    t_a = self.tweetized_awards[a]
                    if self.match_phrase(text_l, w.replace(' ', '/').lower()) or self.match_phrase(text_l, t_a):

                        p_t = nltk.pos_tag(nltk.tokenize.casual.TweetTokenizer(strip_handles=True).tokenize(text))
                        name = []
                        for i in range(len(p_t)-1):
                            if p_t[i][1] == "NNP":
                                name.append(p_t[i][0].lower())
                            else:
                                if not name:
                                    continue
                                if any(mn in w.lower() for mn in name) or any(mn in a.lower() for mn in name):
                                    continue
                                if all(mn not in self.ignore for mn in name):
                                    maybe_name = ' '.join([mn.capitalize() for mn in name])
                                    p_map[a][maybe_name] += 1
                                name = []

        for a, w in self.winners.items():
            ps = {k:j for k, j in sorted(p_map[a].items(), key=lambda item: item[1], reverse=True)[:10]}
            # sorted_presenters = sorted(p_map[a].items(), key=lambda item: item[1], reverse=True)
            # if sorted_presenters:
            #     self.presenters[a] = [ps.strip() for ps in sorted_presenters[0][0].split('and')]
            # else:
            #     self.presenters[a] = ['']
            self.nominees[a] = ['', '', '', '', '']
            print(f"{a}: {ps}\n")

            

if __name__=="__main__":
    start_time = time.time()
    dog = GoldenGlobesParser(2020)
    dog.process_tweets()
    print(f"{time.time() - start_time} seconds")
