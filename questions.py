from itertools import count
import re
import nltk
import sys
import os
import string 
import math


FILE_MATCHES = 1
SENTENCE_MATCHES = 1


def main():

    # Check command-line arguments
    if len(sys.argv) != 2:
        sys.exit("Usage: python questions.py corpus")

    # Calculate IDF values across files
    files = load_files(sys.argv[1])
    file_words = {
        filename: tokenize(files[filename])
        for filename in files
    }
    file_idfs = compute_idfs(file_words)

    # Prompt user for query
    query = set(tokenize(input("Query: ")))

    # Determine top file matches according to TF-IDF
    filenames = top_files(query, file_words, file_idfs, n=FILE_MATCHES)

    # Extract sentences from top files
    sentences = dict()
    for filename in filenames:
        for passage in files[filename].split("\n"):
            for sentence in nltk.sent_tokenize(passage):
                tokens = tokenize(sentence)
                if tokens:
                    sentences[sentence] = tokens

    # Compute IDF values across sentences
    idfs = compute_idfs(sentences)

    # Determine top sentence matches
    matches = top_sentences(query, sentences, idfs, n=SENTENCE_MATCHES)
    for match in matches:
        print(match)


def load_files(directory):
    """
    Given a directory name, return a dictionary mapping the filename of each
    `.txt` file inside that directory to the file's contents as a string.
    """
    dict = {}
    for file in os.listdir(directory):
        with open(os.path.join(directory, file), encoding="utf-8") as The_file:
            dict[file] = The_file.read()  # Maps the contents to the file name and stores it in a dict
    return dict


def tokenize(document):
    """
    Given a document (represented as a string), return a list of all of the
    words in that document, in order.

    Process document by coverting all words to lowercase, and removing any
    punctuation or English stopwords.
    """
    tokenized_words = nltk.tokenize.word_tokenize(document.lower())  # Sperates words and tuens them into lowercase
    processed_document = []
    for letter in tokenized_words:
        if letter not in string.punctuation and letter not in nltk.corpus.stopwords.words("english"):  # Removes punctuation and stopwords
            processed_document.append(letter)  # Writes them to a list 
    return processed_document


def compute_idfs(documents):
    """
    Given a dictionary of `documents` that maps names of documents to a list
    of words, return a dictionary that maps words to their IDF values.

    Any word that appears in at least one of the documents should be in the
    resulting dictionary.
    """
    dict_of_idf = {}
    length_of_doc = len(documents)
    words_with_no_duplicates = set(sum(documents.values(), []))

    for word in words_with_no_duplicates:
        count = 0
        for each_doc in documents.values():
            if word in each_doc:
                count = count + 1
        dict_of_idf[word] = math.log(length_of_doc/count)
    return dict_of_idf



def top_files(query, files, idfs, n):
    """
    Given a `query` (a set of words), `files` (a dictionary mapping names of
    files to a list of their words), and `idfs` (a dictionary mapping words
    to their IDF values), return a list of the filenames of the the `n` top
    files that match the query, ranked according to tf-idf.
    """
    scores_of_files = {}
    for file_name, file_content in files.items():
        file_score = 0
        for word in query:
            if word in file_content:
                file_score += file_content.count(word) * idfs[word]
        if file_score != 0:
            scores_of_files[file_name] = file_score
    
    sorted_by_score = []
    for k, v in sorted(scores_of_files.items(), key=lambda x: x[1], reverse=True):
        sorted_by_score.append(k)
    return sorted_by_score[:n]


def top_sentences(query, sentences, idfs, n):
    """
    Given a `query` (a set of words), `sentences` (a dictionary mapping
    sentences to a list of their words), and `idfs` (a dictionary mapping words
    to their IDF values), return a list of the `n` top sentences that match
    the query, ranked according to idf. If there are ties, preference should
    be given to sentences that have a higher query term density.
    """
    scores_of_sentences = {}
    for sentence, sentence_words in sentences.items():
        sentence_score = 0
        for word in query:
            if word in sentence_words:
                sentence_score += idfs[word]

        if sentence_score != 0:
            density = sum([sentence_words.count(x) for x in query]) / len(sentence_words)
            scores_of_sentences[sentence] = (sentence_score, density)

    sorted_by_score = [k for k, v in sorted(scores_of_sentences.items(), key=lambda x: (x[1][0], x[1][1]), reverse=True)]

    return sorted_by_score[:n]
        


if __name__ == "__main__":
    main()
