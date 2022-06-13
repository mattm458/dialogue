import hyphenate
from nltk.corpus import cmudict


class RateFeatureExtractor:
    def __init__(self):
        self.d = cmudict.dict()

    def _nsyl(self, word):
        """Count the number of syllables in a word."""
        # Remove trailing whitespace and convert to lowercase for dictionary lookup.
        word = word.strip().lower()

        # Special case: Empty string.
        if len(word) == 0:
            return 0

        # Special case: If there is an apostrophe in the word, then it may not be
        # in the dictionary.
        if "'" in word:
            # A common situation is "'s", where the dictionary does not contain the possessive
            # form of all words. If that applies here, remove the "'s" and look up the
            # singular form of the word.
            if word not in self.d and word[-2:] == "'s":
                word = word[:-2]

        # Main syllable lookup functionality.
        if word in self.d:
            # If the word is in the dictionary, extract the syllable count.
            return [len(list(y for y in x if y[-1].isdigit())) for x in self.d[word]][0]
        else:
            # Otherwise, fall back to the hyphenate library for a best (but
            # sometimes inaccurate) guess.
            return len(hyphenate.hyphenate_word(word))

    def __call__(self, context):
        rate = (
            sum([self._nsyl(w) for w in context.get_latest_user_text().split(" ")])
            * 60
            / context.get_latest_feature_value("duration")
        )

        context.add_feature_value("rate", rate)
