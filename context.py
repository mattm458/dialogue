class Context:
    def __init__(self):
        self.user_audios = []
        self.user_texts = []
        self.response_texts = []
        self.user_pitch = []
        self.user_duration = []
        self.user_rate = []
        self.features = {}

    def add_user_audio(self, user_audio):
        self.user_audios.append(user_audio)

    def add_user_text(self, user_text):
        self.user_texts.append(user_text)

    def add_response_text(self, response_text):
        self.response_texts.append(response_text)

    def add_user_pitch(self, user_pitch):
        self.user_pitch.append(user_pitch)
        print(self.user_pitch)

    def add_user_duration(self, user_duration):
        self.user_duration.append(user_duration)

    def add_user_rate(self, user_rate):
        self.user_rate.append(user_rate)
    
    def add_feature_value(self, feature, value):
        if feature not in self.features:
            self.features[feature] = []
        
        self.features[feature].append(value)

    def get_latest_user_audio(self):
        return self.user_audios[-1]

    def get_latest_user_text(self):
        return self.user_texts[-1]

    def get_latest_response_text(self):
        return self.response_texts[-1]

    def get_latest_feature_value(self, feature):
        return self.features[feature][-1]
    
    def get_feature_values(self, feature):
        return self.features[feature]

    def get_text_history(self):
        hist = []
        for i in range(len(self.response_texts)):
            hist.append(self.user_texts[i])
            hist.append(self.response_texts[i])

        hist.append(self.user_texts[-1])

        return hist