import json


class NormalizeUserTransformer:
    def __call__(self, context):
        for feature_key in context.get_transformed_user_feature_keys():
            feature_values = context.get_transformed_user_feature_values(feature_key)

            mean = feature_values.mean()
            std = feature_values.std()
            std = 0.0001 if std == 0 else std

            feature_norms = (
                feature_values
                 - mean
            ) / std
            
            print(f"User normalized {feature_key} {feature_values} -> {feature_norms}")

            context.set_transformed_user_feature_values(feature_key, feature_norms)
