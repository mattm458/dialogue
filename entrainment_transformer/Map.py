import json
import numpy as np


class MapTransformer:
    def __init__(self, target_config, speaker_config=None):
        if target_config is None:
            raise Exception("MapTransformer target config cannot be None!")

        with open(target_config, "r") as infile:
            self.target_config = json.load(infile)

        if speaker_config is not None:
            with open(speaker_config, "r") as infile:
                self.speaker_config = json.load(infile)
        else:
            self.speaker_config = None

    def __call__(self, context):
        # Step 1: Retrieve the entrained feature keys. If none are available (i.e.,
        # we're using the neutral entrainment strategy), this transformer will
        # do nothing.
        for feature_key in context.get_feature_keys():
            # Get the entrained feature value
            feature_value = context.get_entrained_feature_value(feature_key)

            used_config = False
            # Get the entire original set of feature values across the entire conversation
            if context.has_feature_value(feature_key):
                original_feature_values = np.array(
                    context.get_feature_values(feature_key)
                )

                # Compute the mean and standard deviation for whatever is available
                original_feature_values_mean = original_feature_values.mean()
                original_feature_values_std = original_feature_values.std()
            else:
                original_feature_values_std = 0.0

            # If not enough data is available, the standard deviation will be 0
            if original_feature_values_std == 0:
                # If no speaker config was given, we can remove the feature.
                if self.speaker_config is None:
                    context.remove_entrained_feature_value(feature_key)
                    continue
                else:
                    # If speaker config is available, use it instead of the computed
                    # mean and standard deviation
                    original_feature_values_mean = self.speaker_config["mean"][
                        feature_key
                    ]
                    original_feature_values_std = self.speaker_config["std"][
                        feature_key
                    ]
                    used_config = True

            # Z-score normalize the feature value by speaker
            norm_value = (
                feature_value - original_feature_values_mean
            ) / original_feature_values_std

            # Expand the z-score normalized feature value into the target space
            target_value = (
                norm_value * self.target_config["std"][feature_key]
            ) + self.target_config["mean"][feature_key]

            if target_value > 0:
                context.update_entrained_feature_value(feature_key, target_value)
                print(f"Mapped {feature_key} {feature_value} -> {target_value}")
            else:
                if self.speaker_config is not None:
                    original_feature_values_mean = self.speaker_config["mean"][
                        feature_key
                    ]
                    original_feature_values_std = self.speaker_config["std"][
                        feature_key
                    ]

                    # Z-score normalize the feature value by speaker
                    norm_value = (
                        feature_value - original_feature_values_mean
                    ) / original_feature_values_std

                    # Expand the z-score normalized feature value into the target space
                    target_value = (
                        norm_value * self.target_config["std"][feature_key]
                    ) + self.target_config["mean"][feature_key]
                    context.update_entrained_feature_value(feature_key, target_value)

                    print(
                        f"Mapped with config {feature_key} {feature_value} -> {target_value}"
                    )

                else:
                    context.remove_entrained_feature_value(feature_key)
                    print(f"Removed {feature_key} {feature_value} -> {target_value}")
