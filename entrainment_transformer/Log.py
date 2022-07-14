import math

import numpy as np


class LogTransformer:
    def __call__(self, context):
        for feature_key in context.get_transformed_user_feature_keys():
            feature_values = np.array(context.get_transformed_user_feature_values(feature_key))
            log_feature_values = np.log(feature_values)

            print(f"Log {feature_key} {feature_values} -> {log_feature_values}")

            context.set_transformed_user_feature_values(feature_key, log_feature_values)
