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

    def test_ap(self):
        combined_list = self.active_sentences + self.passive_sentences
        random.shuffle(combined_list)
        en_nlp = spacy.load('en_core_web_sm')
        active_correct = 0
        passive_correct = 0

        for sentence in combined_list:
            sen = en_nlp(sentence)
            isActive = True
            for child in sen:
                if child.dep_ == "auxpass":
                    isActive = False
            if sentence in self.active_sentences and isActive:
                active_correct+=1
                print("ACTIVE")
                print(sentence)
            if sentence in self.passive_sentences and not isActive:
                passive_correct+=1
                print("PASSIVE")
                print(sentence)
        print(active_correct/ len(self.active_sentences))
        print(passive_correct/len(self.passive_sentences))



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
