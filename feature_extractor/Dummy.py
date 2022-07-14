import json
import random


class DummyFeatureExtractor:
    def __init__(self, config_file, fuzz=0.0):
        self.fuzz=fuzz
        with open(config_file, 'r') as infile:
            self.config = json.load(infile)
        
    def __call__(self, context):
        for feature_key in self.config:
            feature_value = self.config[feature_key]
            fuzz_amount = (random.random() * self.fuzz * 2) - self.fuzz

            feature_fuzzed = feature_value + (feature_value * fuzz_amount)
            context.append_user_feature_value(feature_key, feature_fuzzed)
