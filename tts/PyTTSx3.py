import pyttsx3

class PyTTSx3TTS:
    def __init__(self):
        self.engine = pyttsx3.init()

    def __call__(self, context):
    
        try:
            self.engine.setProperty('pitch', context.get_entrained_feature_value('pitch_mean'))
        except:
            pass
        try:
            self.engine.setProperty('rate', context.get_entrained_feature_value('rate'))
        except:
            pass
        
        # pitch = context.get_latest_user_pitch()/2
        # rate = context.get_latest_user_rate() / 1.5
        
        self.engine.say(context.get_latest_response_text())
        self.engine.runAndWait()