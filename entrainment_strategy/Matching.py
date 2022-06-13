class MatchingEntrainmentStrategy:
    def __call__(self, context):
        for feature in context.get_feature_keys():
            context.add_entrained_feature_value(
                feature, context.get_latest_feature_value(feature)
            )
