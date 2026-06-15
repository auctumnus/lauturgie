// SPDX-License-Identifier: GPL-3.0-or-later
// Copyright (C) Autumn <auctumnus@pm.me>

//! lauturgie-server: an HTTP API wire-compatible with Lexurgy's `scv1`
//! endpoints (everything but `inflectv1`), so existing Lexurgy clients can
//! point at lauturgie. A faithful port of `lexurgy/api` (the Ktor service):
//!
//! - `GET  /` — a friendly banner.
//! - `GET  /version` — the running version string.
//! - `POST /scv1` — apply changes; returns outputs, intermediate romanizer
//!   stages, traces, and per-word errors. With `allowPolling`, long runs go to
//!   the background and return a poll URL (202).
//! - `GET  /scv1/poll/{id}` — poll a backgrounded run.
//! - `POST /scv1/validate` — compile only; returns the rule names.
//!
//! Timeouts (env `SINGLE_STEP_TIMEOUT`/`REQUEST_TIMEOUT`/`TOTAL_TIMEOUT`,
//! defaults 0.1/0.2/0.5s) and `API_KEY`/`PORT` mirror the Ktor service. Single
//! *step* interruption isn't enforced (lauturgie's internal growth/option
//! budgets already keep every word terminating); the request/total budgets are
//! enforced as wall-clock deadlines.

use std::collections::HashMap;
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};

use axum::extract::{DefaultBodyLimit, Path, Request, State};
use axum::http::{header::AUTHORIZATION, StatusCode};
use axum::middleware::Next;
use axum::response::{IntoResponse, Response};
use axum::routing::{get, post};
use axum::{Json, Router};
use serde::{Deserialize, Serialize};
use tokio::sync::Notify;

use lauturgie::compiler::{self, CompileError, CompiledRules};
use lauturgie::session::ChangeOptions;

// ---------------------------------------------------------------------------
// Wire types (JSON shapes mirror lexurgy's kotlinx-serialized data classes).
// ---------------------------------------------------------------------------

#[derive(Deserialize)]
struct ScRequest {
    changes: String,
    #[serde(rename = "inputWords")]
    input_words: Vec<String>,
    #[serde(rename = "traceWords", default)]
    trace_words: Vec<String>,
    #[serde(rename = "startAt", default)]
    start_at: Option<String>,
    #[serde(rename = "stopBefore", default)]
    stop_before: Option<String>,
    #[serde(rename = "allowPolling", default)]
    allow_polling: bool,
}

#[derive(Deserialize)]
struct ValidateRequest {
    changes: String,
}

#[derive(Serialize)]
struct ValidateSuccess {
    #[serde(rename = "ruleNames")]
    rule_names: Vec<String>,
}

/// lexurgy's `SuccessResponse`. Empty maps/lists are omitted (kotlinx
/// `encodeDefaults = false`); field order matches the data class.
#[derive(Serialize, Clone)]
struct SuccessBody {
    #[serde(rename = "ruleNames")]
    rule_names: Vec<String>,
    #[serde(rename = "outputWords")]
    output_words: Vec<String>,
    #[serde(rename = "intermediateWords", skip_serializing_if = "indexmap::IndexMap::is_empty")]
    intermediate_words: indexmap::IndexMap<String, Vec<String>>,
    #[serde(rename = "traces", skip_serializing_if = "indexmap::IndexMap::is_empty")]
    traces: indexmap::IndexMap<String, Vec<TraceStepBody>>,
    #[serde(rename = "errors", skip_serializing_if = "Vec::is_empty")]
    errors: Vec<RuleFailureBody>,
}

#[derive(Serialize, Clone)]
struct TraceStepBody {
    rule: String,
    output: String,
}

#[derive(Serialize, Clone)]
struct RuleFailureBody {
    message: String,
    #[serde(rename = "rule", skip_serializing_if = "Option::is_none")]
    rule: Option<String>,
    #[serde(rename = "originalWord", skip_serializing_if = "Option::is_none")]
    original_word: Option<String>,
    #[serde(rename = "currentWord", skip_serializing_if = "Option::is_none")]
    current_word: Option<String>,
}

/// lexurgy's `ErrorResponse` sealed interface (discriminator field `type`).
#[derive(Serialize, Clone)]
#[serde(tag = "type")]
enum ErrorBody {
    #[serde(rename = "parseError")]
    ParseError {
        message: String,
        #[serde(rename = "lineNumber")]
        line_number: usize,
        #[serde(rename = "columnNumber")]
        column_number: usize,
    },
    #[serde(rename = "invalidExpression")]
    InvalidExpression {
        message: Option<String>,
        rule: String,
        expression: String,
        #[serde(rename = "expressionNumber")]
        expression_number: usize,
    },
    #[serde(rename = "analysisError")]
    AnalysisError { message: String },
    #[serde(rename = "runtimeError")]
    RuntimeError { message: String },
    #[serde(rename = "timeout")]
    Timeout { message: String },
}

/// lexurgy's `PollResponse` sealed interface (discriminator field `status`).
#[derive(Serialize)]
#[serde(tag = "status")]
enum PollBody {
    #[serde(rename = "working")]
    Working,
    #[serde(rename = "done")]
    Done { result: SuccessBody },
    #[serde(rename = "error")]
    DoneError { result: ErrorBody },
    #[serde(rename = "expired")]
    Expired,
}

/// A finished `scv1` run: either a 200 success or a 400 error.
#[derive(Clone)]
enum ScResult {
    Success(SuccessBody),
    Error(ErrorBody),
}

impl IntoResponse for ScResult {
    fn into_response(self) -> Response {
        match self {
            ScResult::Success(b) => Json(b).into_response(),
            ScResult::Error(e) => (StatusCode::BAD_REQUEST, Json(e)).into_response(),
        }
    }
}

// ---------------------------------------------------------------------------
// Server state, background jobs.
// ---------------------------------------------------------------------------

#[derive(Clone, Copy)]
struct Timeouts {
    request: f64,
    total: f64,
}

struct Job {
    /// `None` while the run is in progress.
    result: Mutex<Option<ScResult>>,
    notify: Notify,
    created: Instant,
}

struct AppState {
    jobs: Mutex<HashMap<String, Arc<Job>>>,
    timeouts: Timeouts,
    api_key: Option<String>,
}

// ---------------------------------------------------------------------------
// Compile + run.
// ---------------------------------------------------------------------------

/// Compile a `.lsc` source, mapping failures to the API's error shapes.
fn compile_changes(changes: &str) -> Result<CompiledRules, ErrorBody> {
    let statements = lauturgie::parse(changes).map_err(|e| match e.offset() {
        // A located parse/lex error becomes `parseError` with line/column.
        Some(offset) => {
            let (line, column) = lauturgie::parser::line_col(changes, offset);
            ErrorBody::ParseError {
                message: e.to_string(),
                line_number: line,
                column_number: column,
            }
        }
        // A structural (post-parse) error has no single location.
        None => ErrorBody::AnalysisError {
            message: e.to_string(),
        },
    })?;
    compiler::compile(&statements).map_err(|e| match e {
        // A rule-expression error → `invalidExpression`. The lowered IR has no
        // source text, so `expression` echoes the message (best effort);
        // `expressionNumber` isn't tracked through lowering.
        CompileError::Expression { rule, what } => ErrorBody::InvalidExpression {
            message: Some(what.clone()),
            rule,
            expression: what,
            expression_number: 0,
        },
        // Declaration-level error → `analysisError`.
        other => ErrorBody::AnalysisError {
            message: other.to_string(),
        },
    })
}

fn success_body(o: lauturgie::session::ChangeOutput) -> SuccessBody {
    SuccessBody {
        rule_names: o.rule_names,
        output_words: o.output_words,
        intermediate_words: o.intermediate_words.into_iter().collect(),
        traces: o
            .traces
            .into_iter()
            .map(|(word, steps)| {
                let steps = steps
                    .into_iter()
                    .map(|t| TraceStepBody {
                        rule: t.rule,
                        output: t.output,
                    })
                    .collect();
                (word, steps)
            })
            .collect(),
        errors: o
            .errors
            .into_iter()
            .map(|f| RuleFailureBody {
                message: f.message,
                rule: f.rule,
                original_word: f.original_word,
                current_word: f.current_word,
            })
            .collect(),
    }
}

/// Run a compiled changer over the words on a blocking thread, bounded by a
/// wall-clock `budget`. The work itself can't be cancelled, but lauturgie's
/// internal budgets keep each word terminating, so an overshot deadline just
/// means the thread finishes a little after we've already answered `timeout`.
async fn run_session(
    compiled: CompiledRules,
    words: Vec<String>,
    trace_words: Vec<String>,
    start_at: Option<String>,
    stop_before: Option<String>,
    budget: Duration,
) -> ScResult {
    let handle = tokio::task::spawn_blocking(move || {
        let words: Vec<&str> = words.iter().map(String::as_str).collect();
        let options = ChangeOptions {
            start_at: start_at.as_deref(),
            stop_before: stop_before.as_deref(),
            trace_words: &trace_words,
        };
        compiled
            .change_with_intermediates(&words, &options)
            .map(success_body)
    });
    match tokio::time::timeout(budget, handle).await {
        Ok(Ok(Ok(body))) => ScResult::Success(body),
        // A whole-run error (bad startAt/stopBefore, unparsable input word).
        Ok(Ok(Err(e))) => ScResult::Error(ErrorBody::RuntimeError {
            message: e.to_string(),
        }),
        // The blocking task panicked.
        Ok(Err(_)) => ScResult::Error(ErrorBody::RuntimeError {
            message: "an unknown error occurred".to_string(),
        }),
        Err(_) => ScResult::Error(ErrorBody::Timeout {
            message: "Run timed out".to_string(),
        }),
    }
}

// ---------------------------------------------------------------------------
// Handlers.
// ---------------------------------------------------------------------------

async fn root() -> &'static str {
    "You've reached the Lexurgy API! The root doesn't do anything though."
}

async fn version() -> &'static str {
    env!("CARGO_PKG_VERSION")
}

async fn validate(Json(req): Json<ValidateRequest>) -> Response {
    match compile_changes(&req.changes) {
        Ok(compiled) => Json(ValidateSuccess {
            rule_names: compiled.rule_names(),
        })
        .into_response(),
        Err(e) => (StatusCode::BAD_REQUEST, Json(e)).into_response(),
    }
}

async fn scv1(State(state): State<Arc<AppState>>, Json(req): Json<ScRequest>) -> Response {
    sweep_expired(&state);

    let compiled = match compile_changes(&req.changes) {
        Ok(c) => c,
        Err(e) => return (StatusCode::BAD_REQUEST, Json(e)).into_response(),
    };

    let timeouts = state.timeouts;
    if !req.allow_polling {
        // Synchronous: bound the run by the request budget (lexurgy passes
        // requestTimeoutSeconds as the total here).
        return run_session(
            compiled,
            req.input_words,
            req.trace_words,
            req.start_at,
            req.stop_before,
            Duration::from_secs_f64(timeouts.request),
        )
        .await
        .into_response();
    }

    // Polling: kick the run off in the background (bounded by the *total*
    // budget) and either return its result if it lands within the request
    // budget, or hand back a poll URL (202).
    let job = Arc::new(Job {
        result: Mutex::new(None),
        notify: Notify::new(),
        created: Instant::now(),
    });
    let job_id = uuid::Uuid::new_v4().to_string();
    state.jobs.lock().unwrap().insert(job_id.clone(), job.clone());

    {
        let job = job.clone();
        let total = Duration::from_secs_f64(timeouts.total);
        tokio::spawn(async move {
            let result = run_session(
                compiled,
                req.input_words,
                req.trace_words,
                req.start_at,
                req.stop_before,
                total,
            )
            .await;
            *job.result.lock().unwrap() = Some(result);
            job.notify.notify_waiters();
        });
    }

    let notified = job.notify.notified();
    let _ = tokio::time::timeout(Duration::from_secs_f64(timeouts.request), notified).await;
    // Re-check directly: covers the race where the run finished just after the
    // deadline but before/without us seeing the notification.
    let finished = job.result.lock().unwrap().take();
    match finished {
        Some(result) => {
            state.jobs.lock().unwrap().remove(&job_id);
            result.into_response()
        }
        None => (
            StatusCode::ACCEPTED,
            Json(serde_json::json!({ "url": format!("/scv1/poll/{job_id}") })),
        )
            .into_response(),
    }
}

async fn poll(State(state): State<Arc<AppState>>, Path(job_id): Path<String>) -> Response {
    let job = state.jobs.lock().unwrap().get(&job_id).cloned();
    let Some(job) = job else {
        // Unknown id: the job expired (or never existed).
        return (StatusCode::INTERNAL_SERVER_ERROR, Json(PollBody::Expired)).into_response();
    };
    let result = job.result.lock().unwrap().clone();
    match result {
        None => Json(PollBody::Working).into_response(),
        Some(ScResult::Success(body)) => Json(PollBody::Done { result: body }).into_response(),
        Some(ScResult::Error(err)) => (
            StatusCode::BAD_REQUEST,
            Json(PollBody::DoneError { result: err }),
        )
            .into_response(),
    }
}

/// Drop finished/abandoned jobs once they're well past the total budget, so
/// the map doesn't grow without bound (lexurgy's `removeExpiredSessions`).
fn sweep_expired(state: &AppState) {
    let expiry = Duration::from_secs_f64(state.timeouts.total.max(1.0) * 2.0);
    state
        .jobs
        .lock()
        .unwrap()
        .retain(|_, job| job.created.elapsed() < expiry);
}

// ---------------------------------------------------------------------------
// API-key gate + bootstrap.
// ---------------------------------------------------------------------------

/// Mirror lexurgy's `ApiKeyChecking`: when a key is configured, every path but
/// `/` and `/version` needs a matching `Authorization` header.
async fn api_key_gate(State(state): State<Arc<AppState>>, req: Request, next: Next) -> Response {
    if let Some(key) = &state.api_key {
        let path = req.uri().path();
        if path != "/" && path != "/version" {
            match req.headers().get(AUTHORIZATION).and_then(|v| v.to_str().ok()) {
                None => return (StatusCode::UNAUTHORIZED, "API key missing").into_response(),
                Some(got) if got != key => {
                    return (StatusCode::FORBIDDEN, "API key is incorrect").into_response()
                }
                _ => {}
            }
        }
    }
    next.run(req).await
}

fn env_f64(key: &str, default: f64) -> f64 {
    std::env::var(key)
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(default)
}

#[tokio::main]
async fn main() {
    let state = Arc::new(AppState {
        jobs: Mutex::new(HashMap::new()),
        timeouts: Timeouts {
            request: env_f64("REQUEST_TIMEOUT", 0.2),
            total: env_f64("TOTAL_TIMEOUT", 0.5),
        },
        api_key: std::env::var("API_KEY").ok().filter(|k| !k.is_empty()),
    });

    let app = Router::new()
        .route("/", get(root))
        .route("/version", get(version))
        .route("/scv1", post(scv1))
        .route("/scv1/poll/{job_id}", get(poll))
        .route("/scv1/validate", post(validate))
        // Large word lists are normal here (lexurgy's own tests post 10k
        // words), so lift axum's 2 MB default to a generous ceiling.
        .layer(DefaultBodyLimit::max(64 * 1024 * 1024))
        .layer(axum::middleware::from_fn_with_state(
            state.clone(),
            api_key_gate,
        ))
        .with_state(state);

    let port: u16 = std::env::var("PORT")
        .ok()
        .and_then(|p| p.parse().ok())
        .unwrap_or(8080);
    let addr = std::net::SocketAddr::from(([0, 0, 0, 0], port));
    let listener = tokio::net::TcpListener::bind(addr).await.unwrap();
    println!("lauturgie-server listening on http://{addr}");
    axum::serve(listener, app).await.unwrap();
}
