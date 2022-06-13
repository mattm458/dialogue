import math


class LogTransformer:
    def __call__(self, context):
        for feature_key in context.get_feature_keys():
            feature_value = context.get_entrained_feature_value(feature_key)
            log_feature_value = math.log(feature_value)
            print(f"Log {feature_key} {feature_value} -> {log_feature_value}")
            context.update_entrained_feature_value(feature_key, log_feature_value)
