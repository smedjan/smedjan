//! Phase 6: API Distillation — generate high-quality training data from large models.
//!
//! Uses Claude/OpenAI/local APIs to generate synthetic text that captures
//! the reasoning capability of 100B+ parameter models. Train a small model
//! on this data to "distill" that capability.
//!
//! A 98M model trained on Claude-generated data >>> 98M trained on raw web text.

use std::io::Write;

/// Configuration for API-based data generation.
pub struct DistillConfig {
    pub api_url: String,
    pub api_key: String,
    pub model: String, // e.g. "claude-sonnet-4-20250514", "gpt-4o"
    pub output_path: String,
    pub n_samples: usize,
    pub max_tokens: usize,
    pub temperature: f32,
}

/// A prompt template for generating diverse training data.
pub struct PromptTemplate {
    pub category: String,
    pub system: String,
    pub prompts: Vec<String>,
}

/// Built-in prompt templates for diverse data generation.
pub fn default_templates() -> Vec<PromptTemplate> {
    vec![
        PromptTemplate {
            category: "reasoning".to_string(),
            system: "You are a clear, logical thinker. Explain your reasoning step by step."
                .to_string(),
            prompts: vec![
                "Explain why the sky appears blue.".to_string(),
                "What would happen if the Earth stopped rotating?".to_string(),
                "Why is compound interest powerful for long-term investing?".to_string(),
                "Explain the difference between correlation and causation with examples."
                    .to_string(),
                "What are the trade-offs between microservices and monolithic architecture?"
                    .to_string(),
            ],
        },
        PromptTemplate {
            category: "coding".to_string(),
            system: "You are an expert programmer. Write clean, well-documented code.".to_string(),
            prompts: vec![
                "Write a binary search function in Rust with proper error handling.".to_string(),
                "Implement a simple LRU cache in Python.".to_string(),
                "Explain how hash maps work internally and implement one from scratch.".to_string(),
                "Write a function that detects cycles in a linked list.".to_string(),
                "Implement a basic HTTP server in Rust using only std::net.".to_string(),
            ],
        },
        PromptTemplate {
            category: "analysis".to_string(),
            system: "You are a thoughtful analyst. Consider multiple perspectives.".to_string(),
            prompts: vec![
                "What are the pros and cons of remote work for companies and employees?"
                    .to_string(),
                "Analyze the impact of social media on mental health.".to_string(),
                "Compare the approaches different countries take to healthcare.".to_string(),
                "What are the ethical implications of AI-generated content?".to_string(),
                "Discuss the trade-offs between privacy and security.".to_string(),
            ],
        },
        PromptTemplate {
            category: "creative".to_string(),
            system: "You are a creative writer with vivid imagination.".to_string(),
            prompts: vec![
                "Write a short story about discovering an abandoned space station.".to_string(),
                "Describe a day in the life of the last human on Earth.".to_string(),
                "Write a poem about the beauty of mathematics.".to_string(),
                "Create a dialogue between two AI systems debating consciousness.".to_string(),
                "Write a fairy tale that teaches the concept of recursion.".to_string(),
            ],
        },
        PromptTemplate {
            category: "instruction".to_string(),
            system: "You are a helpful assistant. Give clear, actionable instructions.".to_string(),
            prompts: vec![
                "How do I set up a Rust project with GPU compute?".to_string(),
                "Explain how to configure SSH keys for GitHub.".to_string(),
                "Walk me through deploying a web app on a VPS.".to_string(),
                "How do I optimize a slow SQL query?".to_string(),
                "Explain how to use Git rebase vs merge and when to use each.".to_string(),
            ],
        },
    ]
}

/// Generate a JSONL file of prompt-response pairs using an API.
/// Each line: {"prompt": "...", "response": "...", "category": "..."}
///
/// For local models (Ollama): use api_url = "http://localhost:11434/api/generate"
/// For Claude: use api_url = "https://api.anthropic.com/v1/messages"
/// For OpenAI: use api_url = "https://api.openai.com/v1/chat/completions"
pub fn generate_training_data(config: &DistillConfig) -> std::io::Result<usize> {
    let templates = default_templates();
    let mut output = std::fs::File::create(&config.output_path)?;
    let mut total_pairs = 0;

    eprintln!(
        "Generating training data from {} ({})",
        config.model, config.api_url
    );
    eprintln!(
        "Templates: {} categories, {} prompts each",
        templates.len(),
        templates[0].prompts.len()
    );

    for template in &templates {
        for prompt in &template.prompts {
            if total_pairs >= config.n_samples {
                break;
            }

            // Build API request based on URL pattern
            let response = if config.api_url.contains("anthropic.com") {
                call_claude_api(
                    &config.api_url,
                    &config.api_key,
                    &config.model,
                    &template.system,
                    prompt,
                    config.max_tokens,
                    config.temperature,
                )
            } else if config.api_url.contains("openai.com") {
                call_openai_api(
                    &config.api_url,
                    &config.api_key,
                    &config.model,
                    &template.system,
                    prompt,
                    config.max_tokens,
                    config.temperature,
                )
            } else {
                // Ollama/local API
                call_ollama_api(
                    &config.api_url,
                    &config.model,
                    prompt,
                    config.max_tokens,
                    config.temperature,
                )
            };

            match response {
                Ok(text) => {
                    // Write as JSONL (escape quotes in strings)
                    let prompt_escaped = prompt.replace('\\', "\\\\").replace('"', "\\\"");
                    let text_escaped = text
                        .replace('\\', "\\\\")
                        .replace('"', "\\\"")
                        .replace('\n', "\\n");
                    writeln!(
                        output,
                        "{{\"prompt\":\"{}\",\"response\":\"{}\",\"category\":\"{}\"}}",
                        prompt_escaped, text_escaped, template.category
                    )?;
                    total_pairs += 1;
                    if total_pairs % 5 == 0 {
                        eprintln!("  Generated {}/{} pairs", total_pairs, config.n_samples);
                    }
                }
                Err(e) => {
                    eprintln!(
                        "  API error for '{}': {}",
                        &prompt[..prompt.len().min(40)],
                        e
                    );
                }
            }
        }
    }

    eprintln!(
        "Generated {} training pairs → {}",
        total_pairs, config.output_path
    );
    Ok(total_pairs)
}

/// Call Claude API (Anthropic Messages API).
fn call_claude_api(
    url: &str,
    api_key: &str,
    model: &str,
    system: &str,
    prompt: &str,
    max_tokens: usize,
    temperature: f32,
) -> Result<String, String> {
    // Use std::process::Command to call curl (no HTTP dependency needed)
    let body = format!(
        r#"{{"model":"{}","max_tokens":{},"temperature":{},"system":"{}","messages":[{{"role":"user","content":"{}"}}]}}"#,
        model,
        max_tokens,
        temperature,
        system.replace('"', "\\\""),
        prompt.replace('"', "\\\""),
    );

    let output = std::process::Command::new("curl")
        .args([
            "-s",
            "-X",
            "POST",
            url,
            "-H",
            &format!("x-api-key: {}", api_key),
            "-H",
            "anthropic-version: 2023-06-01",
            "-H",
            "content-type: application/json",
            "-d",
            &body,
        ])
        .output()
        .map_err(|e| format!("curl failed: {}", e))?;

    let response = String::from_utf8_lossy(&output.stdout).to_string();

    // Extract text from Claude response (simple JSON parsing)
    if let Some(start) = response.find("\"text\":\"") {
        let start = start + 8;
        if let Some(end) = find_json_string_end(&response[start..]) {
            return Ok(response[start..start + end].to_string());
        }
    }
    Err(format!(
        "Failed to parse Claude response: {}",
        &response[..response.len().min(200)]
    ))
}

/// Call OpenAI-compatible API.
fn call_openai_api(
    url: &str,
    api_key: &str,
    model: &str,
    system: &str,
    prompt: &str,
    max_tokens: usize,
    temperature: f32,
) -> Result<String, String> {
    let body = format!(
        r#"{{"model":"{}","max_tokens":{},"temperature":{},"messages":[{{"role":"system","content":"{}"}},{{"role":"user","content":"{}"}}]}}"#,
        model,
        max_tokens,
        temperature,
        system.replace('"', "\\\""),
        prompt.replace('"', "\\\""),
    );

    let output = std::process::Command::new("curl")
        .args([
            "-s",
            "-X",
            "POST",
            url,
            "-H",
            &format!("Authorization: Bearer {}", api_key),
            "-H",
            "content-type: application/json",
            "-d",
            &body,
        ])
        .output()
        .map_err(|e| format!("curl failed: {}", e))?;

    let response = String::from_utf8_lossy(&output.stdout).to_string();

    if let Some(start) = response.find("\"content\":\"") {
        let start = start + 11;
        if let Some(end) = find_json_string_end(&response[start..]) {
            return Ok(response[start..start + end].to_string());
        }
    }
    Err(format!(
        "Failed to parse OpenAI response: {}",
        &response[..response.len().min(200)]
    ))
}

/// Call Ollama local API.
fn call_ollama_api(
    url: &str,
    model: &str,
    prompt: &str,
    max_tokens: usize,
    temperature: f32,
) -> Result<String, String> {
    let body = format!(
        r#"{{"model":"{}","prompt":"{}","stream":false,"options":{{"num_predict":{},"temperature":{}}}}}"#,
        model,
        prompt.replace('"', "\\\""),
        max_tokens,
        temperature,
    );

    let output = std::process::Command::new("curl")
        .args(["-s", "-X", "POST", url, "-d", &body])
        .output()
        .map_err(|e| format!("curl failed: {}", e))?;

    let response = String::from_utf8_lossy(&output.stdout).to_string();

    if let Some(start) = response.find("\"response\":\"") {
        let start = start + 12;
        // Walk the string respecting escape sequences to find the real closing quote
        if let Some(end) = find_json_string_end(&response[start..]) {
            return Ok(response[start..start + end].to_string());
        }
    }
    Err(format!(
        "Failed to parse Ollama response: {}",
        &response[..response.len().min(200)]
    ))
}

/// Find the byte offset of the closing quote of a JSON string value,
/// correctly skipping over escaped characters. The input `s` starts
/// immediately after the opening `"`. Returns the byte offset of the
/// closing `"` (i.e., `s[..offset]` is the raw JSON string content).
fn find_json_string_end(s: &str) -> Option<usize> {
    let mut i = 0;
    let bytes = s.as_bytes();
    while i < bytes.len() {
        match bytes[i] {
            b'"' => return Some(i),
            b'\\' => i += 2, // skip escaped character
            _ => i += 1,
        }
    }
    None // unterminated string
}
