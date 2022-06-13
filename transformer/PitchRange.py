import math


class PitchRangeTransformer:
    def __call__(self, context):
        pitch_95 = context.get_entrained_feature_value("pitch_95")
        pitch_05 = context.get_entrained_feature_value("pitch_05")
        pitch_range = pitch_95 - pitch_05

        print(f"Pitch Range {pitch_range} created")

        context.add_entrained_feature_value("pitch_range", pitch_range)
