import sys
import json
import re
import time

import nltk
from imdb import IMDb
from nltk.tokenize import sent_tokenize, word_tokenize
from collections import defaultdict

class GoldenGlobesParser:

    award_words = set(['tv', 'movie', 'wins', 'won', 'film', 'feature', 'award'])
    ignore = ['golden', 'globes', 'globe', 'goldenglobes', 'goldenglobe', 'gg2020']
        
    # THIS LIST IS NOT RETURNED. IT IS USED TO FIND PRESENTERS, NOMINEES, AND WINNERS FOR ALL THE AWARDS IN ACCORDANCE WITH THE PROJECT GUIDELINES
    # WE ARE USING THIS LIST TO AVOID CASCADING ERROR
    OFFICIAL_AWARDS_1819 = ['best motion picture - drama', 'best motion picture - musical or comedy', 'best performance by an actress in a motion picture - drama', 'best performance by an actor in a motion picture - drama', 'best performance by an actress in a motion picture - musical or comedy', 'best performance by an actor in a motion picture - musical or comedy', 'best performance by an actress in a supporting role in any motion picture', 'best performance by an actor in a supporting role in any motion picture', 'best director - motion picture', 'best screenplay - motion picture', 'best motion picture - animated', 'best motion picture - foreign language', 'best original score - motion picture', 'best original song - motion picture', 'best television series - drama', 'best television series - musical or comedy', 'best television limited series or motion picture made for television', 'best performance by an actress in a limited series or a motion picture made for television', 'best performance by an actor in a limited series or a motion picture made for television', 'best performance by an actress in a television series - drama', 'best performance by an actor in a television series - drama', 'best performance by an actress in a television series - musical or comedy', 'best performance by an actor in a television series - musical or comedy', 'best performance by an actress in a supporting role in a series, limited series or motion picture made for television', 'best performance by an actor in a supporting role in a series, limited series or motion picture made for television', 'cecil b. demille award']
    OFFICIAL_AWARDS_1315 = ['cecil b. demille award', 'best motion picture - drama', 'best performance by an actress in a motion picture - drama', 'best performance by an actor in a motion picture - drama', 'best motion picture - comedy or musical', 'best performance by an actress in a motion picture - comedy or musical', 'best performance by an actor in a motion picture - comedy or musical', 'best animated feature film', 'best foreign language film', 'best performance by an actress in a supporting role in a motion picture', 'best performance by an actor in a supporting role in a motion picture', 'best director - motion picture', 'best screenplay - motion picture', 'best original score - motion picture', 'best original song - motion picture', 'best television series - drama', 'best performance by an actress in a television series - drama', 'best performance by an actor in a television series - drama', 'best television series - comedy or musical', 'best performance by an actress in a television series - comedy or musical', 'best performance by an actor in a television series - comedy or musical', 'best mini-series or motion picture made for television', 'best performance by an actress in a mini-series or motion picture made for television', 'best performance by an actor in a mini-series or motion picture made for television', 'best performance by an actress in a supporting role in a series, mini-series or motion picture made for television', 'best performance by an actor in a supporting role in a series, mini-series or motion picture made for television']

    def __init__(self, year = 2020):
        self.year = int(year)
        if year == 2013 or year == 2015:
            self.official_awards = self.OFFICIAL_AWARDS_1315
        else:
            self.official_awards = self.OFFICIAL_AWARDS_1819

        self.tweets = []
        self.tweetized_awards = {}

        self.hosts = []
        self.awards = []
        self.winners = {}
        self.presenters = defaultdict(str)
        self.nominees = defaultdict(str)

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

        with open(f"./gg{self.year}.json", encoding="utf8") as f:    
            self.tweets = self.parse_json(f)

        if len(self.tweets) == 0:
            with open(f"./gg{self.year}.json", encoding="utf8") as f:
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

        print(f'\n{self.year} Hosts: {self.hosts}\n')

        return self.hosts
    
    def extract_awards(self):
        dic = {}
        for tweet in self.tweets:
            text = tweet['text']
            match = re.search(r'Best.*?goes', text)
            if match:
                result = ''.join(c for c in match.group() if c.isalpha() or c == ' ')
                result = result.replace('  ',' ').lower().replace('tv','television')
                if result in dic:
                    dic[result][1] += 1
                else:
                    dic[result] = [result, 1]
        for key in dic:
            
            if dic[key][1] > 6 * len(self.tweets) / 170000:
                self.awards.append(dic[key][0])
                

        print(f'Awards:')
        for i, a in enumerate(self.awards):
            self.awards[i] = ' '.join(a.split()[:-1])
            print(self.awards[i])
        print('\n')

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
                        regexed = [r1]
                    
                    for r in regexed:
                        if r:
                            maybe_movie = ''.join(c for c in r.group() if c.isalnum() or c == ' ')

                            if any(i in maybe_movie for i in self.ignore) or any(aw in maybe_movie for aw in self.award_words) or len(maybe_movie) > 30:
                                continue

                            aw_map[award][' '.join([w for w in maybe_movie.split()])] += 1

        for original, award in self.tweetized_awards.items():
            if not aw_map[award]:
                continue
            self.winners[original] = sorted(aw_map[award].items(), key=lambda item: item[1], reverse=True)[0][0]           
        return self.winners

    def extract_prenom(self):
        p_map = {}
        n_map = {}
        for original, award in self.tweetized_awards.items():
            p_map[original] = defaultdict(int)
            n_map[original] = defaultdict(int)

        person_key = ['actor', 'actress', 'director', 'cecil']
        present_keywords = [' present', ' handing', ' giving']
        nominee_keywords = [' nominat', ' beat', 'empty-handed', ' instead of']

        for tweet in self.tweets:
            text = tweet['text']
            text_l = text.lower()

            tagged = None
            regexed = []

            for award, winner in self.winners.items():
                award_key = self.tweetized_awards[award]
                winner_key = '/'.join([w for w in winner.split() if len(w) > 2])

                if self.match_phrase(text_l, winner_key.lower()) or self.match_phrase(text_l, award_key):
                    if any(p_k in text_l for p_k in present_keywords):
                        if not tagged:
                            tagged = nltk.pos_tag(nltk.tokenize.casual.TweetTokenizer(strip_handles=True).tokenize(text))
                        name = []
                        for i in range(len(tagged)-1):
                            if tagged[i][1] == "NNP":
                                name.append(tagged[i][0].lower())
                            else:
                                if not name:
                                    continue
                                if any(mn in winner.lower() for mn in name) or any(mn in award.lower() for mn in name):
                                    continue
                                if all(mn not in self.ignore for mn in name):
                                    maybe_name = ' '.join([mn for mn in name])
                                    p_map[award][maybe_name] += 1
                                name = []
                    if any(n_k in text_l for n_k in nominee_keywords):
                        if any(p in award_key for p in person_key):
                            if not tagged:
                                tagged = nltk.pos_tag(nltk.tokenize.casual.TweetTokenizer(strip_handles=True).tokenize(text))
                            name = []
                            for i in range(len(tagged)-1):
                                if tagged[i][1] == "NNP":
                                    name.append(tagged[i][0].lower())
                                else:
                                    if not name or len(name) <= 1:
                                        name = []
                                        continue
                                    if all(len(mn) <= 2 for mn in name):
                                        name = []
                                        continue
                                    if any(mn in winner.lower() for mn in name) or any(mn in award.lower() for mn in name):
                                        name = []
                                        continue
                                    if all(mn not in self.ignore for mn in name):
                                        maybe_name = ' '.join([mn for mn in name])
                                        n_map[award][maybe_name] += 1
                                    name = []
                        else:
                            if len(regexed) < 1:
                                r1 = re.findall(r'".*"', text_l)
                                regexed = [r1]
                        
                            for i, match in enumerate(regexed):
                                for r in match:
                                    split = r
                                    if i == 0:
                                        split = r.split('"')
                                    else:
                                        split = r.split("'")
                                    for s in split:
                                        maybe_movie = s.strip()
                                        if s and len(maybe_movie) > 3:
                                            if any(i in maybe_movie for i in self.ignore) or any(aw in maybe_movie for aw in self.award_words) or len(maybe_movie) > 30 or any(mn in winner.lower() for mn in maybe_movie.split()):
                                                continue

                                            n_map[award][' '.join([w for w in maybe_movie.split()])] += 1

        ia = IMDb()

        for award, winner in self.winners.items():
            print(f'{award}')
            print(f'Winner: {" ".join([w.capitalize() for w in winner.split()])}')

            ns = sorted(n_map[award].items(), key=lambda item: item[1], reverse=True)[:10]
            self.nominees[award] = set()

            if 'cecil' in award:
                continue

            if any(p in award_key for p in person_key):
                for nominee in ns:
                    if len(self.nominees[award]) == 4:
                        break

                    q = ia.search_person(nominee[0])

                    if q:
                        self.nominees[award].add(q[0]['name'])

            else:
                for nominee in ns:
                    if len(self.nominees[award]) == 4:
                        break

                    q = ia.search_movie(nominee[0])

                    if q:
                        self.nominees[award].add(q[0]['name'])

            print(f'Nominees: {self.nominees[award] or {}}')
            
            #///////////////////////////////////////////

            sorted_presenters = sorted(p_map[award].items(), key=lambda item: item[1], reverse=True)
            self.presenters[award] = set()

            for presenter in sorted_presenters:
                if len(self.presenters[award]) == 2:
                    break

                q = ia.search_person(presenter[0])

                if q:
                    self.presenters[award].add(q[0]['name'])

            print(f'Presenters: {self.presenters[award] or {}}')
            print('\n')

    def get_presenters(self):
        return self.presenters

    def get_nominees(self):
        return self.nominees

    def red_carpet(self):
        categories = ['best dressed', 'worst dressed']
        rc_map = {c:defaultdict(int) for c in categories}
        for tweet in self.tweets:
            text = tweet['text']
            text_l = text.lower()
            tagged = None

            for categ in categories:
                if categ in text_l:
                    if not tagged:
                        tagged = nltk.pos_tag(nltk.tokenize.casual.TweetTokenizer(strip_handles=True).tokenize(text))
                    name = []
                    for i in range(len(tagged)-1):
                        if tagged[i][1] == "NNP":
                            name.append(tagged[i][0].lower())
                        else:
                            if not name or len(name) <= 1:
                                name = []
                                continue
                            if all(len(mn) <= 2 for mn in name):
                                name = []
                                continue
                            if all(mn not in self.ignore for mn in name):
                                maybe_name = ' '.join([mn for mn in name])
                                rc_map[categ][maybe_name] += 1
                            name = []
            
        ia = IMDb()
        print('\n')
        for categ in categories:
            rc = sorted(rc_map[categ].items(), key=lambda item: item[1], reverse=True)[:10]
            for person, _ in rc:
                q = ia.search_person(person)
                if q:
                    print(f'{categ}:')
                    print(f'{q[0]["name"]}\n')
                    break    

if __name__=="__main__":
    start_time = time.time()
    parser = None

    if len(sys.argv) > 1:
        parser = GoldenGlobesParser(sys.argv[1])
    else:
        parser = GoldenGlobesParser(2020)
    
    parser.process_tweets()
    parser.process_awards()

    parser.extract_host()
    parser.extract_awards()
    parser.extract_winners()
    parser.extract_prenom()

    parser.red_carpet()
    
    print(f"{time.time() - start_time} seconds")
