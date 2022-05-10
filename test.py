import nltk
from nltk import Tree
import spacy
import random
from nltk.stem.wordnet import WordNetLemmatizer

class test:
    def __init__(self):
        self.active_sentences = [
            'I am happy.',
            'I am sad.',
            'I am angry.',
            'I hit Emily.',
            'I hate Emily.',
            'I like to sail.',
            'I like to swim.',
            'The quick brown fox jumps over the lazy dog.',
            'The orange fish eats a cow.',
            'The red cow eats a fish.',
            'The lazy dog eats a red cow.',
            'On Wednesday, George eats cake.',
            'On Tuesday, Henry changed his oil.',
            'Suzy drives Mike to school.',
            'Henry drove his son to school.',
            'In the winter, I put on my snow tires.',
            'In the summer, I put on my summer tires.',
            'I just got my starbucks coffee.',
            'Henry is a really cool dude.',
            'George is a really cool dude.',
            'I like to hang out with George and Henry, my friends.',
            'Programming is hard.',
            'Programming is fun, however.',
            'Programming with other people is always a blast.',
            'Programming is hard, but fun.',
            'Programming with a beer in hand is great.',
            'Eating chips on the couch is a great passtime.',
            'Watching TV is a great way to relax.',
            'Going to the gym is a great way to get fit.',
            'George just started going to the gym.',
            'I\'m so glad George is starting his fitness journey.',
        ]
        self.passive_sentences = [
            'I was killed',
            'George was dropped as a baby',
            'I was kicked out of the house',
            'George is hated by his mother',
            'George is also hated by his father',
            'George was traumatized by his parents',
            'Henry is loved by his mother',
            'Henry\'s loved by his mother',
            'Grant was dropped off at school',
            'Henry was kicked out of the house',
            'Hippy the hotdog was eaten',
            'Hippy the hotdog was eaten by Henry',
            'Sammy the starfish was blown away',
            'The cow was taken by the aliens',
            'The seeds were planted by the farmer',
            'The car was fixed by the mechanic',
            'The car was blown up',
            'The lock was picked',
            'The door was kicked down',
            'Jeremy was slapped',
            'The cashews were eaten by George',
            'The chair was sat in',
            'The drink was drunk',
            'The program was coded',
            'The phone was held',
            'cats were thrown by Henry',
            'drawings were drawn',
            'graffiti was made',
            'the sun was burned',
            'the hair was groomed',
            'the computer was hacked',
        ]
        # {Word : POS}
        self.pos_dict = {}  # This assumes that there is only one instance of each word
        # {Word : Parent}
        self.parent_dict = {}  # also assumes the same
        self.phrases = []
        self.tree = None

    def test_ap(self):
        combined_list = self.active_sentences + self.passive_sentences
        random.shuffle(combined_list)
        en_nlp = spacy.load('en_core_web_sm')
        active_correct = 0
        passive_correct = 0

        for sentence in combined_list:
            sen = en_nlp(sentence)
            isActive = self.isActive(sen)
            if sentence in self.active_sentences and isActive:
                active_correct += 1
                print(sentence)
                self.print_pos(sen)
                # print("ACTIVE")
                # print(sentence)
            if sentence in self.passive_sentences and not isActive:
                passive_correct += 1
                # print("PASSIVE")
                # print(sentence)
            elif sentence in self.passive_sentences and isActive:
                print("Incorrect passive")
                print(sentence)
                self.print_pos(sen)
        print(active_correct / len(self.active_sentences))
        print(passive_correct / len(self.passive_sentences))

    def test_change_voice(self):
        for sentence in self.active_sentences:
            changed = self.change_voice(sentence)
            print(sentence)
            print(changed)
            print()

    def create_pos_and_parent_dicts(self, sen):
        for token in sen:
            self.pos_dict[token.text] = token.dep_
            self.parent_dict[token.text] = token.head.text

    def change_voice(self, sentence):
        en_nlp = spacy.load('en_core_web_sm')
        sen = en_nlp(sentence)
        self.create_pos_and_parent_dicts(sen)
        sentences = list(sen.sents)
        sentence1 = sentences[0]
        # we assume only 1 sentence
        root_node = sentence1.root
        self.tree = self.to_nltk_tree(root_node)
        self.get_phrases()

        if self.isActive(sen):
            # print("this is a active sentence")
            self.find_indirect_object_active()
            self.print_pos(sen)
            self.print_tree(sen)

            verb = root_node.text
            do = []
            # direct object

            objp = []
            # object of preposition
            sub_word = ""
            for token in sen:
                if token.dep_ == "dobj":
                    do.append(token.text)
                if token.dep_ == "pobj":
                    objp.append(token.text)
                if token.dep_ == "nsubj":
                    sub_word = token.text

            if len(do) == 0:
                print("This sentence cannot be converted to passive ")
                print("missing direct object")
                return ""
            do_phrase = []

            objp_phrase = []
            idx = sentence.index(verb)
            subject = sentence[:idx]
            child_list = [self.tree]
            while len(child_list) != 0:
                node = child_list.pop()
                for child in node:
                    if isinstance(child, Tree):
                        for d in do:
                            if child.label() == d:
                                pos = []
                                self.traverse_tree(child, pos)
                                pos.append(child.label())
                                do_phrase.append(pos)
                        for p in objp:
                            if child.label() == p:
                                pos = []
                                self.traverse_tree(child, pos)
                                pos.append(child.label())
                                objp_phrase.append(pos)

                        child_list.append(child)
                    else:
                        for p in objp:
                            if p == child:
                                objp_phrase.append([child])
                        for d in do:
                            if child == d:
                                do_phrase.append([child])
            do_phrase = [" ".join(e) for e in do_phrase]
            objp_phrase = [" ".join(e) for e in objp_phrase]
            io = ""
            verb_idx = sentence.index(verb) + len(verb) + 1
            # +1 account for space
            do_index = 999999
            for do in do_phrase:
                if do in sentence:
                    idx = sentence.index(do)
                    if idx < do_index:
                        do_index = idx
            if do_index > verb_idx + 1 and do_index != 999999:
                io = sentence[verb_idx:do_index]

            # start converting to passive
            result = ""
            # [do] is [verb] to [io] by [sub]
            # [do] is [verb] to [objp] by [sub]
            # check the subject and do pronoun conjugation
            # i did them this way to also take care of modifiers
            if sub_word.lower() == "i":
                subject = "me" + subject.replace(sub_word, "")
            elif sub_word.lower() == "we":
                subject = "us" + subject.replace(sub_word, "")
            elif sub_word.lower() == "he":
                subject = "him" + subject.replace(sub_word, "")
            elif sub_word.lower() == "she":
                subject = "her" + subject.replace(sub_word, "")
            elif sub_word.lower() == "they":
                subject = "them" + subject.replace(sub_word, "")

            if len(do_phrase) > 0:
                do_phrase[0] = do_phrase[0].capitalize()
            #     capitalize first letter
            if io == "" and len(objp_phrase) == 0:
                result = " and ".join(do_phrase) + " is " + verb + " by " + subject
            elif io != "" and len(objp_phrase) == 0:
                result = " and ".join(do_phrase) + " is " + verb + " to " + io + " by " + subject
            elif io == "" and len(objp_phrase) != 0:
                result = " and ".join(do_phrase) + " is " + verb + " to " + " and ".join(objp_phrase) + " by " + subject
            else:
                result = " and ".join(do_phrase) + " is " + verb + " to " + io + " by " + subject + " ".join(
                    objp_phrase)
            result = " ".join(result.split())
            # remove extra space
            return result
        else:
            self.print_pos(sen)
            self.print_tree(sen)
            io = self.find_indirect_object_word_phrases()
            # print("IO: ", io)
            # if word phrases doesn't work
            if io == -1:
                io = self.find_indirect_object_passive_no_word_phrases()
                # print("IO: ",io)

            print("this is a passive sentence")
            # print(io)

            verb = root_node.text
            do = []
            # direct object

            objp = []
            # object of preposition
            sub_word = ""

            # adverbs
            adv = []

            # preposition
            prep = []

            for token in sen:
                if token.dep_ == "dobj":
                    do.append(token.text)
                if token.dep_ == "pobj":
                    objp.append(token.text)
                if token.dep_ == "advmod":
                    adv.append(token.text)
                if token.dep_ == "nsubj":
                    sub_word = token.text
                if token.dep_ == "prep":
                    prep.append(token.text)

            senlist = sentence.split()

            for j, phrase in enumerate(self.phrases):
                for i in range (0, len(senlist)- len(phrase) + 1,1):
                    temp = senlist[i:i+len(phrase)]
                    # print(temp)
                    if all(x in temp for x in phrase):
                        self.phrases[j] = temp

            # print(self.phrases)
            # print(do)
            # print(objp)
            # print(sub_word)
            io_phrase = io

            obj = objp[-1]
            is_plural = False

            if io.lower() == "i":
              io_phrase = 'me'
            elif io.lower() == "we":
              io_phrase = 'us'
            elif io.lower() == "she":
              io_phrase = 'her'
            elif io.lower() == "he":
              io_phrase = 'him'
            elif io.lower() == "they":
              io_phrase = 'them'
            else:
              for phrase in self.phrases:
                if io in phrase:
                    io_phrase = " ".join(phrase)


            # get new subject phrase
            if obj.lower() == "me":
              subject_phrase = 'I'
              is_plural = True
            elif obj.lower() == "us":
              subject_phrase = 'we'
              is_plural = True
            elif obj.lower() == "her":
              subject_phrase = 'she'
            elif obj.lower() == "him":
              subject_phrase = 'he'
            elif obj.lower() == "them":
              subject_phrase = 'they'
              is_plural = True
            else:
              subject_phrase = obj
              for phrase in self.phrases:
                  if obj in phrase:
                      subject_phrase = " ".join(phrase[1:])
              if subject_phrase[-1] == "s":
                is_plural = True

            # get verb phrase
            verb_phrase = WordNetLemmatizer().lemmatize(verb,'v')
            if adv:
                verb_phrase = " ".join(adv) + ' ' +verb_phrase

            # get prep phrase
            prep_phrase = ""
            for p in prep:
                for phrase in self.phrases:
                    if p == phrase[0]:
                        prep_phrase += " ".join(phrase)

            result = ""
            result += subject_phrase
            result += ' '
            result += verb_phrase
            result += ' ' if is_plural else 's '
            result += io_phrase.lower()
            result += ' '
            result += prep_phrase

            # Capitailize first letter
            result = result.capitalize()

            return result

    def get_subtrees(self, tree, pos):
        print(tree)
        for subtree in tree:
            if type(subtree) == nltk.tree.Tree:
                pos.append(subtree)
                print("HERE")
                print(subtree)
                self.get_subtrees(subtree, pos)
            else:
                print("HER2")
                return
        return

    def traverse_tree(self, tree, pos):
        for subtree in tree:
            if type(subtree) == nltk.tree.Tree:
                self.traverse_tree(subtree, pos)
                pos.append(subtree.label())
            else:
                pos.append(subtree)
        return

    def traverse_tree_word(self, tree, pos, word):
        for subtree in tree:
            if type(subtree) == nltk.tree.Tree and subtree.label == word:
                self.traverse_tree(subtree, pos)
                pos.append(subtree.label())
            elif type(subtree) == nltk.tree.Tree:
                self.traverse_tree(subtree, pos)
            elif isinstance(subtree, str):
                pos.append(subtree)
        return

    def to_nltk_tree(self, node):
        if node.n_lefts + node.n_rights > 0:
            return Tree(node.orth_, [self.to_nltk_tree(child) for child in node.children])
        else:
            return node.orth_

    def print_tree(self, sen):
        # print the tree in preety print
        def to_nltk_tree(node):
            if node.n_lefts + node.n_rights > 0:
                return Tree(node.orth_, [to_nltk_tree(child) for child in node.children])
            else:
                return node.orth_

        [to_nltk_tree(sent.root).pretty_print() for sent in sen.sents]

    def print_pos(self, sen):
        # print all atttributes in tabular format
        for token in sen:
            print(f"{token.text:{8}} {token.dep_ + ' =>':{10}}   {token.head.text:{9}}  {spacy.explain(token.dep_)} ")

    def isActive(self, sen):
        isActive = True
        for child in sen:
            if child.dep_ == "auxpass":
                isActive = False

        return isActive

    def find_indirect_object_word_phrases(self):
        for key, value in self.pos_dict.items():
            # find the parent's part of speech and make sure it isn't an agent
            if self.parent_dict[key] == "to" or self.parent_dict[key] == "for":
                if self.pos_dict[self.parent_dict[key]] != "agent":
                    if value == "pobj":
                        # print("Indirect object found: ", key)
                        return key
        return -1

    # if this function is called, the "direct object" is the io.
    def find_indirect_object_passive_no_word_phrases(self):
        # make sure there is an agent in the sentence DONT THINK WE NEED THIS
        # found = False
        # for key, value in self.pos_dict.items():
        #     if self.pos_dict[self.parent_dict[key]] == "agent":
        #         found = True
        # if not found:
        #     print("No agent found")
        #     return -1

        for key, value in self.pos_dict.items():
            if self.pos_dict[key] == "nsubjpass":
                # print("Indirect object found: ", key)
                return key
        return -1

    def find_indirect_object_active(self):
        for key, value in self.pos_dict.items():
            if self.pos_dict[key] == "dative":
                # print("Indirect object found: ", key)
                return key
        return -1

    def get_prep(self, word):
        for key, value in self.pos_dict.items():
            if value != "ROOT" and self.pos_dict[self.parent_dict[word]] == "prep" or self.pos_dict[
                self.parent_dict[word]] == "agent":
                return self.parent_dict[word]

    def traverse_tree_dict(self, parent, attributes):
        for child in parent:
            if isinstance(child, Tree):
                self.traverse_tree_dict(child, attributes)
            else:
                # if self.pos_dict[child] == "amod":
                if parent.label() in attributes:
                    attributes[parent.label()].append(child)
                else:
                    attributes[parent.label()] = [child]

    def get_phrases(self):
        attributes = {}
        for child in self.tree:
            if isinstance(child, Tree):
                self.traverse_tree_dict(child, attributes)
            elif child.isalpha():
                attributes[child] = []

        for key, value in attributes.items():
            if len(value) > 0:
                prep = self.get_prep(key)
                if prep:
                    self.phrases.append([prep])
                    self.phrases[-1].extend(value)
                    self.phrases[-1].extend([key])
                else:
                    self.phrases.append(value)
                    self.phrases[-1].extend([key])

        # for child in self.phrases:
        #     print(child)
