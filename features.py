"""Feature definitions for the customer labelling model."""

NUMERIC_FEATURES = [
    "age",
    "browse_duration",
    "purchase_count",
    "avg_order_value",
    "return_rate",
    "coupon_usage",
    "favorite_count",
    "click_through_rate",
    "last_purchase_days",
    "app_install_count",
]

CATEGORICAL_FEATURES = [
    "gender",
    "city_tier",
    "device_type",
    "membership_level",
    "social_media_active",
    "has_children",
    "education_level",
    "income_bracket",
]

LABEL_COLUMN = "label"
