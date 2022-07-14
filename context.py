import numpy as np


class Context:
    def __init__(self):
        self.user_audios = []
        self.user_texts = []
        self.response_texts = []
        self.user_duration = []

        self.user_features = {}
        self.transformed_user_features = {}
        self.response_features = {}

        self.partner_entrained_features = {}

    def add_user_audio(self, user_audio):
        self.user_audios.append(user_audio)

    def add_user_text(self, user_text):
        self.user_texts.append(user_text)

    def add_response_text(self, response_text):
        self.response_texts.append(response_text)

    def add_user_duration(self, user_duration):
        self.user_duration.append(user_duration)

    def append_user_feature_value(self, feature, value):
        if feature not in self.user_features:
            self.user_features[feature] = []

        self.user_features[feature].append(value)

    def begin_user_transform(self):
        for feature in self.user_features.keys():
            self.transformed_user_features[feature] = np.array(
                self.user_features[feature]
            )
    
    def get_transformed_user_feature_keys(self):
        return list(self.transformed_user_features.keys())

    def get_transformed_user_feature_values(self, feature):
        return self.transformed_user_features[feature]

    def set_transformed_user_feature_values(self, feature, values):
        self.transformed_user_features[feature] = values

    def set_partner_entrained_feature_value(self, feature, value):
        self.partner_entrained_features[feature] = value

    def get_latest_user_audio(self):
        return self.user_audios[-1]

    def get_latest_user_text(self):
        return self.user_texts[-1]

    def get_latest_response_text(self):
        return self.response_texts[-1]

    def get_feature_keys(self):
        return list(self.user_features.keys())

    def get_entrained_feature_keys(self):
        return list(self.partner_entrained_features.keys())

    def get_latest_feature_value(self, feature):
        return self.user_features[feature][-1]

    def get_user_feature_values(self, feature):
        return self.user_features[feature]

    def get_entrained_feature_value(self, feature):
        return self.partner_entrained_features[feature]

    def has_feature_value(self, feature):
        return feature in self.user_features

    def has_entrained_feature_value(self, feature):
        return feature in self.partner_entrained_features

    def get_text_history(self):
        hist = []
        for i in range(len(self.response_texts)):
            hist.append(self.user_texts[i])
            hist.append(self.response_texts[i])

        hist.append(self.user_texts[-1])

        return hist
