//! `llmtrim wrap <agent> [-- <args>]` — thin convenience launcher.
//!
//! This is *sugar*, not a provider system. It does two things:
//!
//!   1. Confirm the interceptor is wired (the same `HTTPS_PROXY` mechanism `setup`
//!      installs and `start` checks), so the agent's HTTPS to LLM hosts routes through
//!      llmtrim — there is **no** per-agent quirk handling, no base-URL writing, no
//!      allow-list of "supported" agents. Any binary on PATH works, and on Windows a
//!      `.cmd`/`.bat`/`.ps1` shim is launched the way the shell would launch it (see
//!      `resolve_launch_for_platform`).
//!   2. Exec the named binary as a subprocess that inherits the current environment
//!      (which, post-`setup` + a fresh shell, already carries `HTTPS_PROXY` and the CA
//!      trust vars), forwarding the passthrough args and propagating its exit code.
//!
//! Setup-check behaviour (deliberate, least-surprising): if the env isn't wired we do
//! **not** silently mutate the user's shell profile or env — that's `setup`'s job and
//! doing it from a launcher would be a surprising side effect. We print a clear pointer
//! to `llmtrim setup` and refuse, so the user never gets a wrapped agent that quietly
//! bypasses compression. We *do* start a stopped daemon only when the env is already
//! wired (trivially safe: the contract — port + CA — is already in place, same as `start`).

use anyhow::{Context, Result};
use std::path::PathBuf;

use crate::ui::{self, Tone};

/// A parsed `wrap` invocation: the agent binary to launch and the args to forward to it.
#[derive(Debug, PartialEq, Eq)]
pub struct WrapInvocation {
    /// The agent binary name (or path) to run — free-form, resolved on PATH at launch.
    pub agent: String,
    /// Arguments forwarded verbatim to the agent (everything after `<agent>`/`--`).
    pub args: Vec<String>,
}

/// A few well-known agent names, used *only* to enrich the "not found" hint. This is NOT
/// an allow-list: any binary on PATH is accepted. Kept tiny and advisory on purpose.
const KNOWN_AGENTS: &[&str] = &[
    "claude", "codex", "cursor", "aider", "copilot", "gemini", "dsh",
];

/// Split the raw `wrap` arguments into the agent and its passthrough args. The first token
/// is the agent; everything after it is forwarded as-is. A leading `--` separator (clap
/// convention) is dropped if present. Pure, so it's unit-tested without launching anything.
fn parse_invocation(raw: &[String]) -> Result<WrapInvocation> {
    let mut it = raw.iter();
    let agent = it
        .next()
        .context("`wrap` needs an agent to run, e.g. `llmtrim wrap claude`")?
        .clone();
    let mut args: Vec<String> = it.cloned().collect();
    // Drop a single leading `--` (the conventional end-of-options marker) so
    // `llmtrim wrap claude -- --foo` forwards `--foo`, not `-- --foo`.
    if args.first().map(String::as_str) == Some("--") {
        args.remove(0);
    }
    Ok(WrapInvocation { agent, args })
}

/// Is the interceptor usable for a freshly-launched child? It needs both halves of the
/// contract: a live daemon (so requests have somewhere to go) and the env wired (so the
/// child inherits `HTTPS_PROXY` + CA trust). Returns which half, if any, is missing.
#[derive(Debug, PartialEq, Eq)]
enum Readiness {
    Ready,
    /// Env wired but no daemon listening — trivially fixable by starting it.
    DaemonDown,
    /// Env not wired — needs `setup` (we won't mutate the profile from a launcher).
    EnvUnwired,
}

/// Decide readiness from the two facts `start`/`setup` already expose. Pure seam so the
/// precedence is unit-testable without touching the real daemon or shell profile.
fn readiness(daemon_running: bool, env_wired: bool) -> Readiness {
    match (env_wired, daemon_running) {
        (true, true) => Readiness::Ready,
        (true, false) => Readiness::DaemonDown,
        (false, _) => Readiness::EnvUnwired,
    }
}

/// Does *this* process actually carry an `HTTPS_PROXY` pointing at the local interceptor?
/// This is what matters: the child inherits our live environment, not the shell profile on
/// disk. Checking `profile_has_block()` would pass when `setup` has run but the current
/// shell predates it, launching the agent with no proxy and silently skipping compression.
pub fn https_proxy_is_local() -> bool {
    std::env::var("HTTPS_PROXY")
        .or_else(|_| std::env::var("https_proxy"))
        .map(|v| v.contains("127.0.0.1") || v.contains("localhost"))
        .unwrap_or(false)
}

pub fn run(raw: Vec<String>) -> Result<()> {
    let inv = parse_invocation(&raw)?;
    let color = ui::color_stdout();

    // Resolve the real launch command up front (a Windows shim → `cmd /c` or PowerShell),
    // so readiness hints and the "not found" error talk about the agent the user typed.
    let launch = resolve_launch(&inv.agent, &inv.args, None)?;

    // Reuse the exact helpers `start`/`setup` use — do not reimplement the checks.
    let daemon_running = crate::daemon::running().is_some();
    let env_wired = https_proxy_is_local();

    match readiness(daemon_running, env_wired) {
        Readiness::Ready => {}
        Readiness::DaemonDown => {
            // Env already wired, so the port + CA contract is in place: starting the
            // daemon here is trivially safe and consistent with `llmtrim start`.
            let port = crate::setup::resolve_port(None, None)?;
            let pid = crate::daemon::spawn_detached(port)
                .context("interceptor is down and could not be started")?;
            eprintln!(
                "{}",
                ui::note(
                    ui::color_stderr(),
                    &format!("Started the interceptor (pid {pid} · port {port}).")
                )
            );
        }
        Readiness::EnvUnwired => {
            // Don't silently edit the user's environment from a launcher — point at setup.
            // If setup already ran, the profile has the block but this shell predates it, so
            // tailor the hint instead of telling the user to re-run setup pointlessly.
            let hint = if crate::setup::profile_has_block() {
                "You've run `llmtrim setup`, but this shell started before it. Open a new \
                 shell (or re-source your profile) and try again."
            } else {
                "Run `llmtrim setup` once (then open a new shell), and try again."
            };
            anyhow::bail!(
                "HTTPS_PROXY isn't pointing at llmtrim in this shell, so `{}` wouldn't route \
                 through it.\n{hint}",
                inv.agent
            );
        }
    }

    // The child inherits our environment as-is: post-setup that already contains
    // HTTPS_PROXY + the CA trust vars, which is the entire interception mechanism.
    // When global sub is always-on, also inject a dummy Anthropic auth token so Claude
    // Code skips OAuth (same idea as claude-code-proxy's ANTHROPIC_AUTH_TOKEN=unused).
    eprintln!(
        "{}",
        ui::paint(color, Tone::Dim, &format!("llmtrim wrap → {}", inv.agent))
    );

    exec_agent(&launch, &inv.agent)
}

/// True when the agent binary looks like Claude Code (not Codex/Gemini/etc.).
fn agent_is_claude(agent: &str) -> bool {
    let base = agent
        .rsplit(['/', '\\'])
        .next()
        .unwrap_or(agent)
        .to_ascii_lowercase();
    base == "claude" || base.starts_with("claude-") || base == "claude.exe"
}

/// Extensions to probe, in the order a default Windows install resolves them. `.com` and
/// `.exe` are included so a native binary beside a shim still wins — the shell would run
/// it, and `Command` finds those itself.
const SHIM_CANDIDATES: &[&str] = &["com", "exe", "bat", "cmd", "ps1"];

/// What a name resolves to on Windows.
#[derive(Debug, PartialEq, Eq)]
enum Candidate {
    /// A native executable: leave the name alone and let `Command` resolve it.
    Native,
    /// A script shim that must be launched through its interpreter.
    Script(PathBuf),
}

/// Which candidate the shell would run for `agent`: a path is used as given, a bare name is
/// probed directory by directory along PATH and extension by extension within each, so the
/// first match in the first directory wins. A `.exe` beside a `.cmd` therefore resolves
/// native, which is what typing the bare name would run. `path_env` is a seam for tests.
fn resolve_candidate(agent: &str, path_env: Option<&str>) -> Option<Candidate> {
    let classify = |p: PathBuf| {
        let ext = p
            .extension()
            .and_then(|e| e.to_str())
            .unwrap_or_default()
            .to_ascii_lowercase();
        match ext.as_str() {
            "com" | "exe" => Some(Candidate::Native),
            "bat" | "cmd" | "ps1" => Some(Candidate::Script(p)),
            _ => None,
        }
    };
    if agent.contains(['/', '\\']) {
        let p = PathBuf::from(agent);
        return p.is_file().then(|| classify(p)).flatten();
    }
    let path = match path_env {
        Some(p) => p.to_string(),
        None => std::env::var("PATH").ok()?,
    };
    for dir in std::env::split_paths(&path) {
        for ext in SHIM_CANDIDATES {
            let candidate = dir.join(format!("{agent}.{ext}"));
            if candidate.is_file() {
                return classify(candidate);
            }
        }
    }
    None
}

/// What `exec_agent` should run.
struct Launch {
    program: String,
    args: Vec<String>,
}

/// On Windows a `.cmd`/`.bat` is a script, so the resolved path becomes the program and
/// `Command` wraps it in `cmd.exe` itself — argument escaping included, which it refuses
/// rather than mangles (CVE-2024-24576). `.ps1` has no such handling in std and goes
/// through PowerShell.
fn launch_for_script(shim: PathBuf, args: &[String]) -> Launch {
    let is_ps1 = shim
        .extension()
        .and_then(|e| e.to_str())
        .is_some_and(|e| e.eq_ignore_ascii_case("ps1"));
    if !is_ps1 {
        return Launch {
            program: shim.to_string_lossy().into_owned(),
            args: args.to_vec(),
        };
    }
    let mut final_args = vec![
        "-NoProfile".to_string(),
        "-ExecutionPolicy".to_string(),
        "Bypass".to_string(),
        "-File".to_string(),
        shim.to_string_lossy().into_owned(),
    ];
    final_args.extend(args.iter().cloned());
    Launch {
        program: "powershell".to_string(),
        args: final_args,
    }
}

/// Rewrite a `wrap` invocation into what to launch. `path_env` is a seam for tests;
/// `None` reads the live environment.
fn resolve_launch(agent: &str, args: &[String], path_env: Option<&str>) -> Result<Launch> {
    resolve_launch_for_platform(agent, args, path_env, cfg!(windows))
}

/// On Windows a shim (`.cmd`/`.bat`/`.ps1`) is a script, not an executable, so `Command`
/// cannot run it — and `Command` never consults `PATHEXT`, which is why a bare `dsh`
/// looked for `dsh.exe` and missed `dsh.cmd`. Any agent is handled: `.cmd`/`.bat` go
/// through `cmd /c`, `.ps1` through PowerShell `-File`. Everything else — including a
/// POSIX shim, which is a shebang script — passes through untouched.
fn resolve_launch_for_platform(
    agent: &str,
    args: &[String],
    path_env: Option<&str>,
    is_windows: bool,
) -> Result<Launch> {
    if is_windows && let Some(Candidate::Script(shim)) = resolve_candidate(agent, path_env) {
        return Ok(launch_for_script(shim, args));
    }
    Ok(Launch {
        program: agent.to_string(),
        args: args.to_vec(),
    })
}

/// Launch the resolved program and propagate its exit code. This is the real-IO
/// entrypoint (it spawns a subprocess), so it is left uncovered by unit tests — the
/// testable logic lives in `parse_invocation` / `readiness` / `resolve_launch`.
fn exec_agent(launch: &Launch, display: &str) -> Result<()> {
    let mut cmd = std::process::Command::new(&launch.program);
    cmd.args(&launch.args);
    // Always-sub skip-login: Claude Code must not require a live Anthropic OAuth session.
    // Only inject for Claude-ish binaries — never pollute codex/gemini/etc.
    // Prefer an already-set user value; only inject when missing so a real key still wins.
    if llmtrim_core::config::sub_skip_anthropic_login()
        && agent_is_claude(display)
        && std::env::var_os("ANTHROPIC_AUTH_TOKEN").is_none()
    {
        cmd.env(
            "ANTHROPIC_AUTH_TOKEN",
            crate::statusline::SUB_AUTH_TOKEN_VALUE,
        );
    }
    let status = cmd.status().with_context(|| {
        if KNOWN_AGENTS.contains(&display) {
            format!("failed to launch `{display}`: is it installed and on your PATH?")
        } else {
            format!(
                "failed to launch `{display}`: not found on PATH (pass an installed binary, \
                 e.g. one of: {})",
                KNOWN_AGENTS.join(", ")
            )
        }
    })?;

    // Per the repo's exit-code rule: mirror the child's status so CI/scripts see the truth.
    std::process::exit(status.code().unwrap_or(1));
}

#[cfg(test)]
mod tests {
    use super::*;

    fn s(v: &[&str]) -> Vec<String> {
        v.iter().map(|x| x.to_string()).collect()
    }

    /// Scratch dir for a shim fixture. The existing tests inline this pattern; the shim
    /// tests all want it, so it is named once.
    fn tempdir(tag: &str) -> PathBuf {
        let dir = std::env::temp_dir().join(format!("llmtrim-wrap-{tag}-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&dir);
        std::fs::create_dir_all(&dir).expect("mkdir");
        dir
    }

    #[test]
    fn exe_wins_over_cmd_in_the_same_directory() {
        let dir = tempdir("shadow");
        std::fs::write(dir.join("tool.exe"), b"MZ").expect("exe");
        std::fs::write(dir.join("tool.cmd"), "@echo off\r\n").expect("shim");
        let path = dir.to_string_lossy().into_owned();
        assert_eq!(
            resolve_candidate("tool", Some(&path)),
            Some(Candidate::Native)
        );
    }

    #[test]
    fn earlier_path_directory_wins() {
        let first = tempdir("first");
        let second = tempdir("second");
        std::fs::write(first.join("tool.cmd"), "@echo off\r\n").expect("shim");
        std::fs::write(second.join("tool.cmd"), "@echo off\r\n").expect("other");
        let path = format!("{};{}", first.display(), second.display());
        assert_eq!(
            resolve_candidate("tool", Some(&path)),
            Some(Candidate::Script(first.join("tool.cmd")))
        );
    }

    #[test]
    fn cmd_beats_ps1_within_a_directory() {
        let dir = tempdir("order");
        std::fs::write(dir.join("tool.cmd"), "@echo off\r\n").expect("cmd");
        std::fs::write(dir.join("tool.ps1"), "Write-Output hi\r\n").expect("ps1");
        let path = dir.to_string_lossy().into_owned();
        assert_eq!(
            resolve_candidate("tool", Some(&path)),
            Some(Candidate::Script(dir.join("tool.cmd")))
        );
    }

    #[test]
    fn explicit_script_path_resolves_and_exe_path_does_not() {
        let dir = tempdir("explicit");
        let shim = dir.join("dsh.cmd");
        std::fs::write(&shim, "@echo off\r\n").expect("shim");
        assert_eq!(
            resolve_candidate(&shim.to_string_lossy(), None),
            Some(Candidate::Script(shim.clone()))
        );
        let exe = dir.join("dsh.exe");
        std::fs::write(&exe, b"MZ").expect("exe");
        assert_eq!(
            resolve_candidate(&exe.to_string_lossy(), None),
            Some(Candidate::Native)
        );
    }

    #[test]
    fn missing_name_resolves_to_nothing() {
        let dir = tempdir("missing");
        assert_eq!(
            resolve_candidate("nope", Some(&dir.to_string_lossy())),
            None
        );
    }

    #[test]
    fn windows_cmd_shim_is_launched_by_std() {
        let dir = tempdir("cmd-launch");
        let shim = dir.join("mytool.cmd");
        std::fs::write(&shim, "@echo off\r\n").expect("shim");
        let path = dir.to_string_lossy().into_owned();
        let args = s(&["web"]);
        let launch =
            resolve_launch_for_platform("mytool", &args, Some(&path), true).expect("resolve");
        // The shim path IS the program: std wraps .cmd/.bat in cmd.exe and escapes the args
        // itself (CVE-2024-24576), so we must not build a command line by hand.
        assert_eq!(launch.program, shim.to_string_lossy());
        assert_eq!(launch.args, args);
    }

    #[test]
    fn windows_ps1_launches_through_powershell() {
        let dir = tempdir("ps1-launch");
        let shim = dir.join("mytool.ps1");
        std::fs::write(&shim, "Write-Output hi\r\n").expect("shim");
        let path = dir.to_string_lossy().into_owned();
        let launch =
            resolve_launch_for_platform("mytool", &[], Some(&path), true).expect("resolve");
        assert_eq!(launch.program, "powershell");
        assert_eq!(
            launch.args,
            s(&[
                "-NoProfile",
                "-ExecutionPolicy",
                "Bypass",
                "-File",
                &shim.to_string_lossy(),
            ])
        );
    }

    #[test]
    fn native_and_unknown_agents_pass_through() {
        let dir = tempdir("passthrough");
        std::fs::write(dir.join("mytool.exe"), b"MZ").expect("exe");
        let path = dir.to_string_lossy().into_owned();
        let native =
            resolve_launch_for_platform("mytool", &[], Some(&path), true).expect("resolve");
        assert_eq!(native.program, "mytool");
        let unknown = resolve_launch_for_platform("nope", &[], Some(&path), true).expect("resolve");
        assert_eq!(unknown.program, "nope");
    }

    #[test]
    fn unix_shim_passes_through_untouched() {
        let dir = tempdir("unix-shim");
        std::fs::write(dir.join("mytool.cmd"), "@echo off\r\n").expect("shim");
        let path = dir.to_string_lossy().into_owned();
        let launch =
            resolve_launch_for_platform("mytool", &[], Some(&path), false).expect("resolve");
        assert_eq!(launch.program, "mytool");
    }

    #[test]
    fn parses_agent_with_no_args() {
        let inv = parse_invocation(&s(&["claude"])).expect("agent only");
        assert_eq!(inv.agent, "claude");
        assert!(inv.args.is_empty());
    }

    #[test]
    fn forwards_trailing_args_verbatim() {
        let inv = parse_invocation(&s(&["claude", "chat", "--model", "x"])).expect("with args");
        assert_eq!(inv.agent, "claude");
        assert_eq!(inv.args, s(&["chat", "--model", "x"]));
    }

    #[test]
    fn drops_single_leading_double_dash() {
        let inv = parse_invocation(&s(&["aider", "--", "--foo", "bar"])).expect("dash sep");
        assert_eq!(inv.agent, "aider");
        assert_eq!(inv.args, s(&["--foo", "bar"]));
    }

    #[test]
    fn only_first_double_dash_is_dropped() {
        let inv = parse_invocation(&s(&["x", "--", "--", "y"])).expect("two dashes");
        assert_eq!(inv.args, s(&["--", "y"]));
    }

    #[test]
    fn empty_invocation_is_an_error() {
        assert!(parse_invocation(&[]).is_err());
    }

    #[test]
    fn accepts_any_binary_name_not_just_known_ones() {
        let inv = parse_invocation(&s(&["some-random-tool"])).expect("free-form");
        assert_eq!(inv.agent, "some-random-tool");
    }

    #[test]
    fn readiness_ready_when_both_present() {
        assert_eq!(readiness(true, true), Readiness::Ready);
    }

    #[test]
    fn readiness_daemon_down_when_env_wired_only() {
        assert_eq!(readiness(false, true), Readiness::DaemonDown);
    }

    #[test]
    fn readiness_env_unwired_takes_precedence() {
        assert_eq!(readiness(false, false), Readiness::EnvUnwired);
        assert_eq!(readiness(true, false), Readiness::EnvUnwired);
    }

    #[test]
    fn agent_is_claude_matches_claude_binaries_only() {
        assert!(agent_is_claude("claude"));
        assert!(agent_is_claude("/usr/bin/claude"));
        assert!(agent_is_claude("claude-2"));
        assert!(agent_is_claude(r"C:\Tools\claude.exe"));
        assert!(!agent_is_claude("codex"));
        assert!(!agent_is_claude("gemini"));
        assert!(!agent_is_claude("/usr/bin/aider"));
    }
}
