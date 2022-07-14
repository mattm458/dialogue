import numpy as np

class PitchRangeTransformer:
    def __call__(self, context):
        pitch_95 = np.array(context.get_transformed_user_feature_values("pitch_95"))
        pitch_05 = np.array(context.get_transformed_user_feature_values("pitch_05"))
        pitch_range = np.abs(pitch_95 - pitch_05)

        print(f"Pitch Range {pitch_range} created")

        context.set_transformed_user_feature_values("pitch_range", pitch_range)
