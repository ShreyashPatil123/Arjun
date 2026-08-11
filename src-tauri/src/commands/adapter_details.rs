//! What a LoRA adapter actually changes about a model.
//!
//! ## The honesty problem
//!
//! A person choosing an adapter wants one thing: "what will this make the model
//! better at?" HuggingFace rarely states that outright. It is tempting to guess
//! from the repository name and present the guess as fact — "improves coding" —
//! but a name is not evidence, and a confident wrong answer is worse than an
//! uncertain right one.
//!
//! So every claim returned here carries where it came from:
//!
//! * **Stated** — the author's own tags and declared training datasets.
//! * **Suggested** — read out of the repository name, which is a hint only.
//!
//! The UI shows both, labelled. Nothing is invented; when the page says nothing
//! useful, this says so plainly rather than filling the space.

use serde::Serialize;

use crate::model_providers::huggingface::card::{
    categorize, datasets_from_tags, license_from_tags, ModelCategory,
};

/// Where a piece of information came from, so a hint never reads as a fact.
#[derive(Debug, Clone, Copy, Serialize, PartialEq, Eq)]
#[serde(rename_all = "camelCase")]
pub enum Confidence {
    /// The author said so, via tags or declared datasets.
    Stated,
    /// Read out of the repository name — a hint, not a statement.
    Suggested,
}

/// One thing this adapter appears to change.
#[derive(Debug, Clone, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct AdapterEffect {
    /// e.g. "Coding", "Reasoning".
    pub skill: String,
    /// Plain sentence a non-specialist can act on.
    pub description: String,
    pub confidence: Confidence,
}

#[derive(Debug, Clone, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct AdapterDetails {
    pub repo_id: String,
    pub name: String,
    pub author: String,
    pub license: Option<String>,
    pub downloads: u64,
    pub likes: u64,
    /// Total size of the adapter file, when the repository reports it.
    pub size_bytes: u64,
    /// Whether it can be loaded as it stands.
    pub gguf_ready: bool,
    /// Why not, when it cannot.
    pub blocked_reason: Option<String>,
    /// Datasets the author declared training on.
    pub datasets: Vec<String>,
    /// What it appears to change, each labelled with how sure that is.
    pub effects: Vec<AdapterEffect>,
    /// Shown when nothing at all could be determined.
    pub notice: Option<String>,
}

/// A sentence explaining what a skill means for the person using the model.
fn describe(skill: ModelCategory) -> &'static str {
    match skill {
        ModelCategory::Coding => {
            "Writes and explains code more reliably — better at following a language's \
             conventions and producing code that runs."
        }
        ModelCategory::Reasoning => {
            "Works through problems step by step instead of answering immediately, which \
             helps on questions with several stages."
        }
        ModelCategory::Math => "Handles arithmetic and mathematical problems more accurately.",
        ModelCategory::Agentic => {
            "Better at using tools and calling functions — following a structured format \
             so a program can act on the reply."
        }
        ModelCategory::Vision => {
            "Adds or improves the ability to describe and answer questions about images."
        }
        ModelCategory::Multilingual => "Improves answers in languages other than English.",
        ModelCategory::LongContext => {
            "Handles longer documents and conversations before losing track."
        }
        // Matched explicitly rather than with a wildcard, so adding a category
        // to the enum fails to compile here instead of silently describing
        // itself as a vague "general behaviour" change.
        ModelCategory::MixtureOfExperts => {
            "Relates to how the model routes work internally rather than to a skill."
        }
        ModelCategory::SmallAndFast => "Relates to the model's size rather than to a skill.",
        ModelCategory::General | ModelCategory::LoraAdapter => {
            "Adjusts the model's general behaviour."
        }
    }
}

/// Skills the author declared through tags, as opposed to ones merely hinted at
/// by the repository name.
///
/// Shared with [`crate::capability::assign`], which turns the same evidence into
/// a capability slot — so the panel and the assignment can never disagree about
/// what an adapter appears to do.
pub(crate) fn stated_skills(tags: &[String]) -> Vec<ModelCategory> {
    // Categorising the tags alone — with an empty name — means anything found
    // here came from something the author wrote, not from the repo's title.
    categorize("", tags, 0, 0, "")
        .into_iter()
        .filter(|c| !matches!(c, ModelCategory::General | ModelCategory::LoraAdapter))
        .collect()
}

pub(crate) fn suggested_skills(name: &str) -> Vec<ModelCategory> {
    categorize(name, &[], 0, 0, "")
        .into_iter()
        .filter(|c| !matches!(c, ModelCategory::General | ModelCategory::LoraAdapter))
        .collect()
}

/// Looks up one adapter and explains what it does.
#[tauri::command]
pub async fn get_adapter_details(repo_id: String) -> Result<AdapterDetails, String> {
    let token = crate::config::hf_token::get();

    let client = reqwest::Client::builder()
        .user_agent("Sarathi/0.1.0")
        .build()
        .map_err(|e| format!("could not create an HTTP client: {e}"))?;

    let mut req = client.get(format!("https://huggingface.co/api/models/{repo_id}?blobs=true"));
    if let Some(t) = &token {
        req = req.bearer_auth(t);
    }

    let resp = req.send().await.map_err(|e| format!("could not reach HuggingFace: {e}"))?;
    if !resp.status().is_success() {
        return Err(format!("HuggingFace returned {} for {repo_id}", resp.status()));
    }

    let info: serde_json::Value =
        resp.json().await.map_err(|e| format!("unexpected response from HuggingFace: {e}"))?;

    let tags: Vec<String> = info
        .get("tags")
        .and_then(|t| t.as_array())
        .map(|a| a.iter().filter_map(|v| v.as_str().map(String::from)).collect())
        .unwrap_or_default();

    let files: Vec<(String, u64)> = info
        .get("siblings")
        .and_then(|s| s.as_array())
        .map(|arr| {
            arr.iter()
                .filter_map(|f| {
                    let name = f.get("rfilename")?.as_str()?.to_string();
                    let size = f.get("size").and_then(|s| s.as_u64()).unwrap_or(0);
                    Some((name, size))
                })
                .collect()
        })
        .unwrap_or_default();

    let filenames: Vec<String> = files.iter().map(|(n, _)| n.clone()).collect();

    // The same rule the download uses, so the panel and the button never
    // disagree about whether this can be installed.
    // Sized form: the Hub's blob listing is already in hand, and it is what
    // distinguishes a real adapter from a merged model sharing its repository.
    let installable = crate::adapter_manager::store::check_installable_sized(&files);
    let gguf_ready = installable.is_ok();
    let blocked_reason = installable.as_ref().err().map(|e| e.to_string());
    let size_bytes = installable
        .ok()
        .and_then(|chosen| files.iter().find(|(n, _)| *n == chosen).map(|(_, s)| *s))
        .unwrap_or(0);

    let short_name = repo_id.split('/').next_back().unwrap_or(&repo_id);

    // Stated evidence wins: if the author tagged it, the name adds nothing.
    let stated = stated_skills(&tags);
    let mut effects: Vec<AdapterEffect> = stated
        .iter()
        .map(|s| AdapterEffect {
            skill: s.label().to_string(),
            description: describe(*s).to_string(),
            confidence: Confidence::Stated,
        })
        .collect();

    for s in suggested_skills(short_name) {
        if !stated.contains(&s) {
            effects.push(AdapterEffect {
                skill: s.label().to_string(),
                description: describe(s).to_string(),
                confidence: Confidence::Suggested,
            });
        }
    }

    let datasets = datasets_from_tags(&tags);

    let notice = (effects.is_empty() && datasets.is_empty()).then(|| {
        "This adapter's page does not say what it changes, and its name gives no clue. \
         Only its author knows what it was trained for."
            .to_string()
    });

    Ok(AdapterDetails {
        name: short_name.replace(['-', '_'], " "),
        author: repo_id.split('/').next().unwrap_or("").to_string(),
        license: license_from_tags(&tags),
        downloads: info.get("downloads").and_then(|d| d.as_u64()).unwrap_or(0),
        likes: info.get("likes").and_then(|l| l.as_u64()).unwrap_or(0),
        size_bytes,
        gguf_ready,
        blocked_reason,
        datasets,
        effects,
        notice,
        repo_id,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn author_tags_are_reported_as_stated() {
        let skills = stated_skills(&["code".to_string(), "text-generation".to_string()]);
        assert!(skills.contains(&ModelCategory::Coding), "got {skills:?}");
    }

    #[test]
    fn a_name_alone_is_only_a_suggestion() {
        // Nothing in the tags, so the effect must not be presented as stated.
        let stated = stated_skills(&[]);
        let suggested = suggested_skills("llama-3-sql-coder-lora");

        assert!(stated.is_empty(), "no tags means nothing stated: {stated:?}");
        assert!(suggested.contains(&ModelCategory::Coding), "got {suggested:?}");
    }

    #[test]
    fn generic_categories_are_not_presented_as_effects() {
        // "General" says nothing about what changes, and "LoRA adapter" is what
        // the thing *is*, not what it does. Neither belongs in the effect list.
        let skills = suggested_skills("some-random-adapter");
        assert!(!skills.contains(&ModelCategory::General));
        assert!(!skills.contains(&ModelCategory::LoraAdapter));
    }

    #[test]
    fn every_skill_has_a_plain_language_description() {
        for skill in ModelCategory::all() {
            let text = describe(*skill);
            assert!(!text.is_empty(), "{skill:?} has no description");
            assert!(
                text.ends_with('.'),
                "{skill:?} description should be a sentence: {text}"
            );
        }
    }
}
