import nltk
from nltk import Tree
import spacy
import random


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
        self.pos_dict = {}  # This assumes that there is only one instance of each word
        self.parent_dict = {} # also assumes the same

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
                active_correct+=1
                print(sentence)
                self.print_pos(sen)
                # print("ACTIVE")
                # print(sentence)
            if sentence in self.passive_sentences and not isActive:
                passive_correct+=1
                # print("PASSIVE")
                # print(sentence)
            elif sentence in self.passive_sentences and isActive:
                print("Incorrect passive")
                print(sentence)
                self.print_pos(sen)
        print(active_correct/ len(self.active_sentences))
        print(passive_correct/len(self.passive_sentences))

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

    def change_voice(self,sentence):
        en_nlp = spacy.load('en_core_web_sm')
        sen = en_nlp(sentence)
        self.create_pos_and_parent_dicts(sen)
        if self.isActive(sen):
            # print("this is a active sentence")
            self.find_indirect_object_active()
            self.print_pos(sen)
            self.print_tree(sen)
            sentences = list(sen.sents)
            sentence = sentences[0]
            # we assume only 1 sentence
            root_node = sentence.root
            tree = self.to_nltk_tree(root_node)
            ans = []
            verb = root_node.text
            subject = ""
            do = ""
            # direct object

            for token in sen:
                if token.dep_ == "nsubj":
                    subject = token.text
                if token.dep_ == "dobj":
                    do = token.text
            if do =="":
                print("This sentence cannot be converted to passive ")
                print("missing direct object")
                return""

            for child in tree:
                if isinstance(child, Tree):
                    # print(child)
                    pos = []
                    self.traverse_tree(child, pos)
                    pos.append(child.label())
                    ans.append(pos)
                else:
                    if child.isalpha():
                        pos = [child]
                        ans.append(pos)
            # group all level 1 subtree together
            subject_phrase = ""
            do_phrase = ""
            io_phrase =""
            for ls in ans:
                if subject in ls:
                    subject_phrase = " ".join(ls)
                elif do in ls:
                    do_phrase = " ".join(ls)
                else:
                    io_phrase = " ".join(ls)


            if io_phrase !="":
                result = do_phrase+ " is "+ verb +" by " +subject_phrase + " to " + io_phrase
            else:
                result = do_phrase+ " is "+ verb +" by " +subject_phrase
            # print(result)
            return result

        # If it is a passive sentence
        else:
            self.print_pos(sen)
            self.print_tree(sen)
            io = self.find_indirect_object_word_phrases()
            # if word phrases doesn't work
            if io == -1:
                io = self.find_indirect_object_passive_no_word_phrases()

            print("this is a passive sentence")
            return ""

    def traverse_tree(self, tree, pos):
        for subtree in tree:
            if type(subtree) == nltk.tree.Tree:
                self.traverse_tree(subtree, pos)
                pos.append(subtree.label())
            else:
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
                        print("Indirect object found: ", key)
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
                print("Indirect object found: ", key)
                return key
        return -1

    def find_indirect_object_active(self):
        for key, value in self.pos_dict.items():
            if self.pos_dict[key] == "dative":
                print("Indirect object found: ", key)
                return key
        return -1
