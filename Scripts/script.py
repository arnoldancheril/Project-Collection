"""
paddock_setup.py
Run inside a RHEL8 WSL instance AFTER:  wsl -d RHEL8
- Prompts for setup folder path (Windows or /mnt/... path), resolves to WSL path
- Verifies required files exist
- Prompts for Nexus username/password (password hidden)
- Updates esf_nexus_8.repo username/password (lines 66/67)
- Copies repo + corporate certs into system locations and updates CA trust
- Runs dnf update
- Installs Python 3.12 + pip, upgrades pip
- Copies pip.conf into ~/.config/pip
- Installs Conan (user install) and configures Conan remote + login using Nexus creds
- Installs paddock_main.rpm and hatchery rpm
- Sets RAPTOR_HOME (folder containing config.ini) and QT_PLUGIN_PATH, persists to ~/.bashrc
"""

import getpass
import os
import re
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple


REQUIRED_ALWAYS = [
    "esf_nexus_8.repo",
    "ng-certificate-chain.cer",
    "NorthGrumCorporate-G2.pem",
    "NorthGrumMult-G2.pem",
    "pip.conf",
    "paddock_main.rpm",
    "config.ini",
]

HATCHERY_CANDIDATES = [
    "hatchery.rpm",
    "hatchery_file.rpm",
    "hatchery_file.rpm".replace("_file", ""),
]


BASHRC_MARK_BEGIN = "# Paddock environment (added by paddock_setup.py)"
BASHRC_MARK_END = "# End Paddock environment"


def die(msg: str, code: int = 1) -> None:
    print(f"\nERROR: {msg}\n", file=sys.stderr)
    sys.exit(code)


def run_cmd(
    cmd: List[str],
    *,
    sudo: bool = False,
    env: Optional[Dict[str, str]] = None,
    check: bool = True,
    display: Optional[str] = None,
) -> subprocess.CompletedProcess:
    full_cmd = (["sudo"] + cmd) if sudo else cmd

    shown = display if display is not None else " ".join(full_cmd)
    print(f"\n→ {shown}")

    merged_env = os.environ.copy()
    if env:
        merged_env.update(env)

    return subprocess.run(full_cmd, env=merged_env, check=check)


def normalize_path_input(raw: str) -> str:
    s = raw.strip().strip('"').strip("'").strip()
    return s


def windows_to_wsl_path(win_path: str) -> str:
    """
    Convert:
      C:\paddock   -> /mnt/c/paddock
      C:/paddock   -> /mnt/c/paddock
    If already starts with '/', return as-is.
    """
    p = normalize_path_input(win_path)
    if p.startswith("/"):
        return p

    p = p.replace("\\", "/")

    m = re.match(r"^([A-Za-z]):(.*)$", p)
    if not m:
        die(
            f"Could not parse Windows path '{win_path}'. "
            "Use 'C:\\paddock' or '/mnt/c/paddock'."
        )

    drive = m.group(1).lower()
    rest = m.group(2)
    if not rest.startswith("/"):
        rest = "/" + rest

    return f"/mnt/{drive}{rest}"


# Get the path where the setup files live
def prompt_setup_dir() -> Path:
    print("Paddock Environment Setup\n")
    print("Enter the path to the folder containing the setup files.")
    print("You can enter either a Windows path or an already-converted WSL path:")
    print(r"  Windows example : C:\paddock")
    print("  WSL example     : /mnt/c/paddock")
    raw = input("Path: ").strip()

    wsl_path = windows_to_wsl_path(raw)
    p = Path(wsl_path)

    print(f"\nResolved path: {p}")

    if not p.exists() or not p.is_dir():
        die(
            f"Directory '{p}' does not exist or is not accessible.\n"
            "Make sure the Windows path is correct and the drive is mounted in WSL."
        )

    return p


# Find hatchery RPM
def find_hatchery_rpm(setup_dir: Path) -> str:
    for name in HATCHERY_CANDIDATES:
        if (setup_dir / name).is_file():
            return name
    die("Could not find a hatchery RPM. Looked for: " + ", ".join(HATCHERY_CANDIDATES))
    return ""  # unreachable


# Verify required files exist
def verify_required_files(setup_dir: Path) -> Tuple[Path, Path]:
    print("\nChecking for required files...")
    for f in REQUIRED_ALWAYS:
        full = setup_dir / f
        if not full.is_file():
            die(f"Required file not found: {full}")
        print(f"  [OK] {f}")

    hatchery_name = find_hatchery_rpm(setup_dir)
    print(f"  [OK] {hatchery_name}")

    repo_file = setup_dir / "esf_nexus_8.repo"
    config_ini = setup_dir / "config.ini"
    return repo_file, config_ini


# Prompt for Nexus credentials
def prompt_nexus_creds() -> Tuple[str, str]:
    print("\nEnter your Nexus Repository credentials.")
    print("WARNING: Your Nexus password will NOT appear on the screen while typing.")
    user = input("Nexus Username: ").strip()
    if not user:
        die("Nexus username cannot be empty.")
    pw = getpass.getpass("Nexus Password: ")
    if not pw:
        die("Nexus password cannot be empty.")
    return user, pw


# Update username/password in repo file
def update_repo_credentials(repo_file: Path, nexus_user: str, nexus_pass: str) -> None:
    print("\nUpdating credentials in esf_nexus_8.repo...")

    text = repo_file.read_text(encoding="utf-8", errors="replace")
    lines = text.splitlines(True)  # keep line endings

    def set_key(lines_: List[str], key: str, value: str, preferred_idx_0: int) -> None:
        new_line = f"{key}={value}\n"

        if 0 <= preferred_idx_0 < len(lines_):
            if re.match(rf"^\s*{re.escape(key)}\s*=", lines_[preferred_idx_0]):
                lines_[preferred_idx_0] = new_line
                return

        for i, ln in enumerate(lines_):
            if re.match(rf"^\s*{re.escape(key)}\s*=", ln):
                lines_[i] = new_line
                return

        if lines_ and not lines_[-1].endswith("\n"):
            lines_[-1] += "\n"
        lines_.append(new_line)

    # line 66 and 67 (1-indexed) => indices 65 and 66
    set_key(lines, "username", nexus_user, 65)
    set_key(lines, "password", nexus_pass, 66)

    repo_file.write_text("".join(lines), encoding="utf-8")
    print("  [OK] Credentials written to repo file.")


# Configure pip
def ensure_pip_config(setup_dir: Path) -> None:
    print("\nSetting up pip...")
    pip_dir = Path.home() / ".config" / "pip"
    pip_dir.mkdir(parents=True, exist_ok=True)

    src = setup_dir / "pip.conf"
    dst = pip_dir / "pip.conf"
    shutil.copy2(src, dst)
    print(f"  [OK] pip.conf copied to {dst}")


def which_or_localbin(exe: str) -> Optional[str]:
    p = shutil.which(exe)
    if p:
        return p
    candidate = str(Path.home() / ".local" / "bin" / exe)
    if Path(candidate).exists():
        return candidate
    return None


# Persist environment variables to ~/.bashrc
def write_bashrc_exports(raptor_home: str, qt_plugin_path: str) -> None:
    bashrc = Path.home() / ".bashrc"
    existing = ""
    if bashrc.exists():
        existing = bashrc.read_text(encoding="utf-8", errors="replace")

    pattern = re.compile(
        re.escape(BASHRC_MARK_BEGIN) + r".*?" + re.escape(BASHRC_MARK_END) + r"\n?",
        re.DOTALL,
    )
    cleaned = re.sub(pattern, "", existing)

    block = (
        "\n"
        f"{BASHRC_MARK_BEGIN}\n"
        'export PATH="$HOME/.local/bin:$PATH"\n'
        f'export RAPTOR_HOME="{raptor_home}"\n'
        f'export QT_PLUGIN_PATH="{qt_plugin_path}"\n'
        f"{BASHRC_MARK_END}\n"
    )

    bashrc.write_text(cleaned + block, encoding="utf-8")
    print("  [OK] Environment variables appended to ~/.bashrc (idempotent).")


def main() -> int:
    setup_dir = prompt_setup_dir()
    repo_file, config_ini = verify_required_files(setup_dir)
    hatchery_name = find_hatchery_rpm(setup_dir)

    nexus_user, nexus_pass = prompt_nexus_creds()
    update_repo_credentials(repo_file, nexus_user, nexus_pass)

    # YUM/DNF setup
    print("\nSetting up YUM/DNF (requires sudo)...")

    run_cmd(
        ["cp", str(repo_file), "/etc/yum.repos.d/esf_nexus_8.repo"],
        sudo=True,
        display="sudo cp <repo> /etc/yum.repos.d/esf_nexus_8.repo",
    )
    print("  [OK] Repo file copied.")

    run_cmd(
        ["cp", str(setup_dir / "NorthGrumCorporate-G2.pem"), "/etc/pki/ca-trust/source/anchors/"],
        sudo=True,
        display="sudo cp NorthGrumCorporate-G2.pem /etc/pki/ca-trust/source/anchors/",
    )
    run_cmd(
        ["cp", str(setup_dir / "NorthGrumMult-G2.pem"), "/etc/pki/ca-trust/source/anchors/"],
        sudo=True,
        display="sudo cp NorthGrumMult-G2.pem /etc/pki/ca-trust/source/anchors/",
    )
    print("  [OK] CA certificates copied.")

    run_cmd(["update-ca-trust", "extract"], sudo=True)
    print("  [OK] CA trust updated.")

    # dnf update
    print("\nRunning dnf update...")
    run_cmd(["dnf", "update", "-y", "--nobest", "--skip-broken"], sudo=True)

    # Install Python 3.12 & upgrade pip
    print("\nInstalling Python 3.12 & upgrading pip...")
    run_cmd(["dnf", "install", "-y", "python3.12", "python3.12-pip", "python3.12-devel"], sudo=True)

    try:
        run_cmd(["python3.12", "-m", "pip", "install", "--upgrade", "--user", "pip"])
    except subprocess.CalledProcessError:
        run_cmd(["python3.12", "-m", "pip", "install", "--upgrade", "pip"], check=True)

    print("  [OK] Python 3.12 installed and pip upgraded.")

    # pip setup
    ensure_pip_config(setup_dir)

    # Install Conan
    print("\nInstalling Conan...")

    cert_path = str(setup_dir / "ng-certificate-chain.cer")
    env = {"REQUESTS_CA_BUNDLE": cert_path}
    print(f"  REQUESTS_CA_BUNDLE set to: {cert_path}")

    run_cmd(["python3.12", "-m", "pip", "install", "--user", "conan>=2.0,<3.0"], env=env)

    conan_exe = which_or_localbin("conan")
    if not conan_exe:
        die("Conan installed but 'conan' not found in PATH or ~/.local/bin. Try restarting shell or check ~/.local/bin.")
    print(f"  [OK] Conan installed: {conan_exe}")

    # Configure Conan remotes
    print("\nConfiguring Conan remotes...")

    try:
        run_cmd([conan_exe, "remote", "remove", "conancenter"], env=env, check=True)
    except subprocess.CalledProcessError:
        print("  (conancenter remote did not exist; continuing)")

    run_cmd([conan_exe, "profile", "detect", "--force"], env=env)

    try:
        run_cmd(
            [
                conan_exe,
                "remote",
                "add",
                "conan-v2-center-proxy",
                "https://nexus-repository.northgrum.com/repository/conan-v2-center-proxy/",
            ],
            env=env,
            check=True,
        )
    except subprocess.CalledProcessError:
        print("  (remote already exists; continuing)")

    run_cmd(
        [conan_exe, "remote", "login", "conan-v2-center-proxy", nexus_user, "-p", nexus_pass],
        env=env,
        display=f"{conan_exe} remote login conan-v2-center-proxy {nexus_user} -p ********",
    )
    print("  [OK] Conan remotes configured.")

    # Install RPMs
    print("\nInstalling paddock_main.rpm...")
    run_cmd(
        ["dnf", "install", "-y", "--nobest", str(setup_dir / "paddock_main.rpm")],
        sudo=True,
    )
    print("  [OK] paddock_main.rpm installed.")

    print(f"\nInstalling {hatchery_name}...")
    run_cmd(
        ["dnf", "install", "-y", "--nobest", str(setup_dir / hatchery_name)],
        sudo=True,
    )
    print(f"  [OK] {hatchery_name} installed.")

    # Set environment variables
    print("\nSetting environment variables...")

    raptor_home = str(config_ini.parent)
    qt_plugin_path = "/opt/raptorb/hatchery/plugins"

    os.environ["RAPTOR_HOME"] = raptor_home
    os.environ["QT_PLUGIN_PATH"] = qt_plugin_path

    write_bashrc_exports(raptor_home, qt_plugin_path)

    print(f"  RAPTOR_HOME={raptor_home}")
    print(f"  QT_PLUGIN_PATH={qt_plugin_path}")

    print("\nPaddock setup completed successfully!\n")
    print("Next: open a new shell (or run: source ~/.bashrc) so RAPTOR_HOME/QT_PLUGIN_PATH are loaded.")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except KeyboardInterrupt:
        die("Interrupted by user.", code=130)
