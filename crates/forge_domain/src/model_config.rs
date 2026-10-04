use derive_setters::Setters;
use schemars::JsonSchema;
use serde::{Deserialize, Serialize};
use strum_macros::{Display, EnumString};

/// Provider service tier controlling request scheduling.
#[derive(
    Default,
    Debug,
    Clone,
    Copy,
    PartialEq,
    Eq,
    Serialize,
    Deserialize,
    JsonSchema,
    fake::Dummy,
    Display,
    EnumString,
)]
#[serde(rename_all = "snake_case")]
#[strum(serialize_all = "snake_case")]
pub enum ServiceTier {
    /// Standard provider scheduling.
    #[default]
    Default,
    /// Flexible scheduling for requests that can tolerate delays.
    Flex,
    /// Priority scheduling for latency-sensitive requests.
    Priority,
    /// Fastest available provider scheduling.
    Ultrafast,
}

/// Result of discovering service tiers for a provider model.
#[derive(Default, Clone, Debug, PartialEq, Eq, Deserialize, Serialize)]
pub enum ServiceTierDiscovery {
    /// The provider does not support service tier discovery.
    #[default]
    Unsupported,
    /// The discovered endpoints, which may be empty.
    Discovered(Vec<ServiceTierEndpoint>),
}

/// A provider endpoint offering a service tier for a model.
#[derive(Default, Clone, Debug, PartialEq, Eq, Deserialize, Serialize, Setters)]
#[setters(strip_option, into)]
pub struct ServiceTierEndpoint {
    /// Scheduling tier offered by this endpoint.
    pub tier: ServiceTier,
    /// Exact endpoint routing tag reported by the provider, without normalization.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub tag: Option<String>,
    /// Optional endpoint display name.
    pub name: Option<String>,
    /// Optional name of the upstream provider.
    pub provider_name: Option<String>,
    /// Optional pricing reported by the provider.
    pub pricing: Option<ServiceTierPricing>,
}

impl ServiceTierEndpoint {
    /// Creates an endpoint for the given scheduling tier.
    ///
    /// # Arguments
    /// * `tier` - Scheduling tier offered by the endpoint.
    pub fn new(tier: ServiceTier) -> Self {
        Self {
            tier,
            tag: None,
            name: None,
            provider_name: None,
            pricing: None,
        }
    }
}

/// Provider-reported prices preserved as strings without numeric conversion.
#[derive(Default, Clone, Debug, PartialEq, Eq, Deserialize, Serialize, Setters)]
#[setters(strip_option, into)]
pub struct ServiceTierPricing {
    /// Optional price for prompt tokens.
    pub prompt: Option<String>,
    /// Optional price for completion tokens.
    pub completion: Option<String>,
    /// Optional price per request.
    pub request: Option<String>,
    /// Optional price per image.
    pub image: Option<String>,
}

use crate::{ModelId, ProviderId};

/// Domain-level configuration that pairs a provider with a model.
///
/// Used as the unified payload for [`super::ConfigOperation`] variants that
/// configure a provider/model pair (session, commit, suggest). Provider and model
/// are required; the service tier is optional. Use `Option<ModelConfig>` at the
/// call-site when the configuration is optional.
#[derive(Debug, Clone, Serialize, Deserialize, JsonSchema, PartialEq, Setters)]
#[setters(strip_option, into)]
pub struct ModelConfig {
    /// The provider ID (e.g. `"anthropic"`).
    pub provider: ProviderId,

    /// The model ID to use with this provider.
    pub model: ModelId,

    /// Optional provider service tier for this model.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub service_tier: Option<ServiceTier>,
}

impl ModelConfig {
    /// Creates a new [`ModelConfig`] with the given provider and model.
    ///
    /// # Arguments
    /// * `provider` - The provider identifier
    /// * `model` - The model identifier
    pub fn new(provider: impl Into<ProviderId>, model: impl Into<ModelId>) -> Self {
        Self {
            provider: provider.into(),
            model: model.into(),
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
    fn test_service_tier_endpoint_preserves_exact_tag() {
        let fixture = ServiceTierEndpoint::new(ServiceTier::Priority).tag("openai/fast");
        let actual = serde_json::to_value(&fixture).unwrap();
        let expected = json!({
            "tier": "priority", "tag": "openai/fast", "name": null,
            "provider_name": null, "pricing": null,
        });
        assert_eq!(actual, expected);
        let actual = serde_json::from_value::<ServiceTierEndpoint>(actual).unwrap();
        assert_eq!(actual, fixture);
    }

    #[test]
    fn test_model_config_backward_compatible_serialization() {
        let fixture = json!({"provider": "open_router", "model": "test-model"});

        let actual = serde_json::from_value::<ModelConfig>(fixture.clone()).unwrap();
        let expected = ModelConfig::new(ProviderId::OPEN_ROUTER, "test-model");
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
            let fixture =
                ModelConfig::new(ProviderId::OPEN_ROUTER, "test-model").service_tier(tier);

            let actual = serde_json::to_value(&fixture).unwrap();
            let expected = json!({
                "provider": "open_router",
                "model": "test-model",
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
            "provider": "open_router",
            "model": "test-model",
            "service_tier": null,
        });

        let actual = serde_json::from_value::<ModelConfig>(fixture).unwrap();
        let expected = ModelConfig::new(ProviderId::OPEN_ROUTER, "test-model");
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
