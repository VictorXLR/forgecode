use derive_setters::Setters;
pub use forge_domain::ServiceTier;
use schemars::JsonSchema;
use serde::{Deserialize, Serialize};

/// A type alias for a provider identifier string.
pub type ProviderId = String;

/// A type alias for a model identifier string.
pub type ModelId = String;

/// Pairs a provider and model together for a specific operation.
#[derive(Debug, Setters, Clone, PartialEq, Serialize, Deserialize, JsonSchema, fake::Dummy)]
#[setters(strip_option, into)]
pub struct ModelConfig {
    /// The provider to use for this operation.
    pub provider_id: String,
    /// The model to use for this operation.
    pub model_id: String,
    /// Optional provider service tier for this model.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub service_tier: Option<ServiceTier>,
}

impl ModelConfig {
    /// Creates a new ModelConfig with no service tier override.
    ///
    /// # Arguments
    /// * `provider_id` - The provider identifier
    /// * `model_id` - The model identifier
    pub fn new(provider_id: impl Into<String>, model_id: impl Into<String>) -> Self {
        Self {
            provider_id: provider_id.into(),
            model_id: model_id.into(),
            service_tier: None,
        }
    }
}

#[cfg(test)]
mod tests {
    use pretty_assertions::assert_eq;
    use serde_json::json;

    use super::*;

    #[test]
    fn test_model_config_backward_compatible_serialization() {
        let fixture = json!({"provider_id": "openrouter", "model_id": "test-model"});

        let actual = serde_json::from_value::<ModelConfig>(fixture.clone()).unwrap();
        let expected = ModelConfig::new("openrouter", "test-model");
        assert_eq!(actual, expected);
        assert_eq!(serde_json::to_value(actual).unwrap(), fixture);
    }

    #[test]
    fn test_model_config_service_tier_strings() {
        let fixtures = [
            (ServiceTier::Default, "default"),
            (ServiceTier::Flex, "flex"),
            (ServiceTier::Priority, "priority"),
            (ServiceTier::Ultrafast, "ultrafast"),
        ];

        for (tier, name) in fixtures {
            let fixture = ModelConfig::new("openrouter", "test-model").service_tier(tier);

            let actual = serde_json::to_value(&fixture).unwrap();
            let expected = json!({
                "provider_id": "openrouter",
                "model_id": "test-model",
                "service_tier": name,
            });
            assert_eq!(actual, expected);
            assert_eq!(
                serde_json::from_value::<ModelConfig>(actual).unwrap(),
                fixture
            );
            assert_eq!(tier.to_string(), name);
            assert_eq!(name.parse::<ServiceTier>().unwrap(), tier);
        }
    }

    #[test]
    fn test_model_config_null_service_tier() {
        let fixture = json!({
            "provider_id": "openrouter",
            "model_id": "test-model",
            "service_tier": null,
        });

        let actual = serde_json::from_value::<ModelConfig>(fixture).unwrap();
        let expected = ModelConfig::new("openrouter", "test-model");
        assert_eq!(actual, expected);
    }

    #[test]
    fn test_service_tier_rejects_unknown_strings() {
        let fixture = "unknown";

        let actual = fixture.parse::<ServiceTier>();
        assert!(actual.is_err());
        let actual = serde_json::from_value::<ServiceTier>(json!(fixture));
        assert!(actual.is_err());
    }
}
