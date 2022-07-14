import json


class ExpandTransformer:
    def __init__(self, config):
        with open(config, "r") as infile:
            self.config = json.load(infile)

    def __call__(self, context):
        for feature_key in context.get_entrained_feature_keys():
            feature_norm = context.get_entrained_feature_value(feature_key)

            feature = (feature_norm * self.config["std"][feature_key]) + self.config["mean"][feature_key]

            print(f"Expanded {feature_key} {feature_norm} -> {feature}")

            context.set_partner_entrained_feature_value(feature_key, feature)
