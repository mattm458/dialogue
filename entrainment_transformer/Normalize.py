import json


class NormalizeTransformer:
    def __init__(self, config):
        with open(config, "r") as infile:
            self.config = json.load(infile)

    def __call__(self, context):
        for feature_key in context.get_entrained_feature_keys():
            feature_value = context.get_entrained_feature_value(feature_key)
            feature_norm = (
                feature_value - self.config["mean"][feature_key]
            ) / self.config["std"][feature_key]

            print(f"Normalized {feature_key} {feature_value} -> {feature_norm}")

            context.update_entrained_feature_value(feature_key, feature_norm)
