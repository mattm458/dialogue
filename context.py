class Context:
    def __init__(self):
        self.user_audios = []
        self.user_texts = []
        self.response_texts = []
        self.user_duration = []
        self.features = {}
        self.entrained_features={}

    def add_user_audio(self, user_audio):
        self.user_audios.append(user_audio)

    def add_user_text(self, user_text):
        self.user_texts.append(user_text)

    def add_response_text(self, response_text):
        self.response_texts.append(response_text)

    def add_user_duration(self, user_duration):
        self.user_duration.append(user_duration)
    
    def add_feature_value(self, feature, value):
        if feature not in self.features:
            self.features[feature] = []
        
        self.features[feature].append(value)

    def add_entrained_feature_value(self, feature, value):
        self.entrained_features[feature]=value

    def update_entrained_feature_value(self, feature, value):
        self.entrained_features[feature] = value
    
    def remove_entrained_feature_value(self, feature):
        del self.entrained_features[feature]

    def get_latest_user_audio(self):
        return self.user_audios[-1]

    def get_latest_user_text(self):
        return self.user_texts[-1]

    def get_latest_response_text(self):
        return self.response_texts[-1]

    def get_feature_keys(self):
        return list(self.features.keys())

    def get_entrained_feature_keys(self):
        return list(self.entrained_features.keys())

    def get_latest_feature_value(self, feature):
        return self.features[feature][-1]
    
    def get_feature_values(self, feature):
        return self.features[feature]

    def get_entrained_feature_value(self, feature):
        return self.entrained_features[feature]

    def has_feature_value(self, feature):
        return feature in self.features    

    def has_entrained_feature_value(self, feature):
        return feature in self.entrained_features

    def get_text_history(self):
        hist = []
        for i in range(len(self.response_texts)):
            hist.append(self.user_texts[i])
            hist.append(self.response_texts[i])

        hist.append(self.user_texts[-1])

        return hist