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
use std::path::{Path, PathBuf};

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

/// Script extensions Rust's `Command` cannot exec directly on Windows.
const SHIM_EXTS: &[&str] = &["cmd", "bat", "ps1"];

/// Resolve `agent` to a Windows script shim, or `None` when it is not one.
///
/// A value containing a path separator is used as given (so `wrap C:\npm\dsh.cmd` works);
/// a bare name is probed on PATH as given first, then with each shim extension. `.exe` and
/// `.com` are deliberately not matched: `Command` finds those unaided, and routing them
/// through `cmd.exe` would add a process and change quoting for every launch. `path_env`
/// is a seam so tests never read the real environment.
fn shim_path(agent: &str, path_env: Option<&str>) -> Option<PathBuf> {
    fn is_shim(p: &Path) -> bool {
        p.is_file()
            && p.extension()
                .and_then(|e| e.to_str())
                .is_some_and(|e| SHIM_EXTS.iter().any(|s| e.eq_ignore_ascii_case(s)))
    }
    if agent.contains(['/', '\\']) {
        let p = PathBuf::from(agent);
        return is_shim(&p).then_some(p);
    }
    let path = match path_env {
        Some(p) => p.to_string(),
        None => std::env::var("PATH").ok()?,
    };
    for dir in std::env::split_paths(&path) {
        let as_given = dir.join(agent);
        if is_shim(&as_given) {
            return Some(as_given);
        }
        for ext in SHIM_EXTS {
            let cand = dir.join(format!("{agent}.{ext}"));
            if cand.is_file() {
                return Some(cand);
            }
        }
    }
    None
}

/// Quote one token for a `cmd.exe` command line. `cmd` parses a *command line*, not an
/// argv array, so Rust's MSVCRT quoting (what `Command::args` applies) is wrong here: we
/// quote each token ourselves and hand the whole line over with `raw_arg`. Interior quotes
/// double, which is how `cmd` escapes them inside a quoted token. `%VAR%`/`!VAR!` still
/// expand — that is what the shell does with the same argument, so it is behaviour, not a
/// hole.
fn cmd_escape(arg: &str) -> String {
    let mut out = String::with_capacity(arg.len() + 2);
    out.push('"');
    for ch in arg.chars() {
        if ch == '"' {
            out.push('"');
        }
        out.push(ch);
    }
    out.push('"');
    out
}

/// The `/c` tail for a `.cmd`/`.bat` shim: the shim path and every forwarded arg escaped.
fn cmd_line(shim: &Path, args: &[String]) -> String {
    let mut line = cmd_escape(&shim.to_string_lossy());
    for a in args {
        line.push(' ');
        line.push_str(&cmd_escape(a));
    }
    line
}

/// What `exec_agent` should run: a program plus either ordinary args, or a pre-built
/// `cmd.exe` command line that must be appended verbatim.
struct Launch {
    program: String,
    args: Vec<String>,
    raw_tail: Option<String>,
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
    if is_windows && let Some(shim) = shim_path(agent, path_env) {
        let is_ps1 = shim
            .extension()
            .and_then(|e| e.to_str())
            .is_some_and(|e| e.eq_ignore_ascii_case("ps1"));
        if is_ps1 {
            let mut final_args = vec![
                "-NoProfile".to_string(),
                "-ExecutionPolicy".to_string(),
                "Bypass".to_string(),
                "-File".to_string(),
                shim.to_string_lossy().into_owned(),
            ];
            final_args.extend(args.iter().cloned());
            return Ok(Launch {
                program: "powershell".to_string(),
                args: final_args,
                raw_tail: None,
            });
        }
        return Ok(Launch {
            program: "cmd".to_string(),
            args: Vec::new(),
            raw_tail: Some(format!("/c {}", cmd_line(&shim, args))),
        });
    }
    Ok(Launch {
        program: agent.to_string(),
        args: args.to_vec(),
        raw_tail: None,
    })
}

/// Launch the resolved program and propagate its exit code. This is the real-IO
/// entrypoint (it spawns a subprocess), so it is left uncovered by unit tests — the
/// testable logic lives in `parse_invocation` / `readiness` / `resolve_launch`.
fn exec_agent(launch: &Launch, display: &str) -> Result<()> {
    let mut cmd = std::process::Command::new(&launch.program);
    cmd.args(&launch.args);
    // `cmd.exe` parses a command line, not an argv array, so the shim tail is escaped for
    // it and appended verbatim rather than re-quoted by `Command::args`.
    #[cfg(windows)]
    if let Some(tail) = &launch.raw_tail {
        use std::os::windows::process::CommandExt;
        cmd.raw_arg(format!(" {tail}"));
    }
    #[cfg(not(windows))]
    let _ = &launch.raw_tail;
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

    use std::path::Path;

    #[test]
    fn cmd_escape_quotes_and_doubles_interior_quotes() {
        assert_eq!(cmd_escape("plain"), "\"plain\"");
        assert_eq!(cmd_escape("has space"), "\"has space\"");
        assert_eq!(cmd_escape("say \"hi\""), "\"say \"\"hi\"\"\"");
    }

    #[test]
    fn cmd_line_quotes_shim_and_every_arg() {
        let line = cmd_line(Path::new("C:\\npm\\dsh.cmd"), &s(&["web", "a b"]));
        assert_eq!(line, "\"C:\\npm\\dsh.cmd\" \"web\" \"a b\"");
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
    fn shim_path_probes_extensions_on_path() {
        let dir = tempdir("probe");
        std::fs::write(dir.join("mytool.cmd"), "@echo off\r\n").expect("shim");
        let path = dir.to_string_lossy().into_owned();
        assert_eq!(
            shim_path("mytool", Some(&path)),
            Some(dir.join("mytool.cmd"))
        );
        assert_eq!(
            shim_path("mytool.cmd", Some(&path)),
            Some(dir.join("mytool.cmd"))
        );
        assert_eq!(shim_path("missing", Some(&path)), None);
    }

    #[test]
    fn shim_path_accepts_an_explicit_path() {
        let dir = tempdir("explicit");
        let shim = dir.join("dsh.cmd");
        std::fs::write(&shim, "@echo off\r\n").expect("shim");
        let given = shim.to_string_lossy().into_owned();
        assert_eq!(shim_path(&given, None), Some(shim));
    }

    #[test]
    fn shim_path_ignores_exe() {
        let dir = tempdir("exe");
        std::fs::write(dir.join("mytool.exe"), b"MZ").expect("exe");
        assert_eq!(shim_path("mytool", Some(&dir.to_string_lossy())), None);
    }

    #[test]
    fn windows_shim_launches_through_cmd() {
        let dir = tempdir("cmd-launch");
        std::fs::write(dir.join("mytool.cmd"), "@echo off\r\n").expect("shim");
        let path = dir.to_string_lossy().into_owned();
        let args = s(&["web"]);
        let launch =
            resolve_launch_for_platform("mytool", &args, Some(&path), true).expect("resolve");
        assert_eq!(launch.program, "cmd");
        assert_eq!(
            launch.raw_tail,
            Some(format!("/c {}", cmd_line(&dir.join("mytool.cmd"), &args)))
        );
        assert!(launch.args.is_empty());
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
        assert_eq!(launch.raw_tail, None);
    }

    #[test]
    fn unix_shim_passes_through_untouched() {
        let dir = tempdir("unix-shim");
        std::fs::write(dir.join("mytool.cmd"), "@echo off\r\n").expect("shim");
        let path = dir.to_string_lossy().into_owned();
        let launch =
            resolve_launch_for_platform("mytool", &[], Some(&path), false).expect("resolve");
        assert_eq!(launch.program, "mytool");
        assert_eq!(launch.raw_tail, None);
    }

    #[test]
    fn exe_and_unknown_agents_pass_through() {
        let dir = tempdir("passthrough");
        std::fs::write(dir.join("mytool.exe"), b"MZ").expect("exe");
        let path = dir.to_string_lossy().into_owned();
        let exe = resolve_launch_for_platform("mytool", &[], Some(&path), true).expect("resolve");
        assert_eq!(exe.program, "mytool");
        let unknown = resolve_launch_for_platform("nope", &[], Some(&path), true).expect("resolve");
        assert_eq!(unknown.program, "nope");
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
