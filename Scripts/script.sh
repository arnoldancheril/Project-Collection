#!/bin/bash
set -e

# ─────────────────────────────────────────────
# Paddock Setup Script
# Run inside a RHEL8 WSL instance
# ─────────────────────────────────────────────

echo "============================================"
echo "         Paddock Environment Setup"
echo "============================================"
echo ""

# ── 1. Get the Windows folder path where the setup files live ──────────────────
echo "Enter the Windows path to the folder containing the setup files."
echo "Example: C:\\paddock   or   C:\\Users\\you\\Downloads\\paddock"
read -rp "Windows folder path: " WIN_PATH

# Convert Windows path to WSL mount path:
#   C:\paddock  ->  /mnt/c/paddock
#   Handles both forward- and back-slash separators.
WIN_PATH_NORMALIZED="${WIN_PATH//\\//}"          # backslashes -> forward slashes

# Extract drive letter (first character, lowercased)
DRIVE_LETTER=$(echo "${WIN_PATH_NORMALIZED:0:1}" | tr '[:upper:]' '[:lower:]')

# Strip the drive letter and colon from the front
REST_OF_PATH="${WIN_PATH_NORMALIZED:2}"          # remove "C:"

WSL_PATH="/mnt/${DRIVE_LETTER}${REST_OF_PATH}"

echo ""
echo "Resolved WSL path: ${WSL_PATH}"

# Verify the folder exists
if [ ! -d "${WSL_PATH}" ]; then
    echo "ERROR: Directory '${WSL_PATH}' does not exist or is not accessible."
    echo "Make sure the Windows path is correct and the drive is mounted in WSL."
    exit 1
fi

# Verify all required files are present
REQUIRED_FILES=(
    "esf_nexus_8.repo"
    "ng-certificate-chain.cer"
    "NorthGrumCorporate-G2.pem"
    "NorthGrumMult-G2.pem"
    "pip.conf"
)

echo ""
echo "Checking for required files..."
for f in "${REQUIRED_FILES[@]}"; do
    FULL_PATH="${WSL_PATH}/${f}"
    if [ ! -f "${FULL_PATH}" ]; then
        echo "ERROR: Required file not found: ${FULL_PATH}"
        exit 1
    fi
    echo "  [OK] ${f}"
done

# ── 2. Collect Nexus credentials ─────────────────────────────────────────────
echo ""
echo "============================================"
echo "Enter your Nexus Repository credentials."
echo "============================================"
read -rp "Nexus Username: " NEXUS_USER
read -rsp "Nexus Password: " NEXUS_PASS
echo ""   # newline after hidden password input

# ── 3. Update esf_nexus_8.repo with credentials ───────────────────────────────
REPO_FILE="${WSL_PATH}/esf_nexus_8.repo"

echo ""
echo "Updating credentials in esf_nexus_8.repo (lines 66–67)..."

# Use sed to replace the username= and password= lines in-place.
# The lines are expected to be exactly "username=" and "password=".
sed -i "66s|.*|username=${NEXUS_USER}|" "${REPO_FILE}"
sed -i "67s|.*|password=${NEXUS_PASS}|" "${REPO_FILE}"

echo "  [OK] Credentials written."

# ── 4. pip setup ─────────────────────────────────────────────────────────────
echo ""
echo "============================================"
echo "Setting up pip..."
echo "============================================"

mkdir -p ~/.config/pip
cp "${WSL_PATH}/pip.conf" ~/.config/pip/
echo "  [OK] pip.conf copied to ~/.config/pip/"

# ── 5. Install Conan ──────────────────────────────────────────────────────────
echo ""
echo "============================================"
echo "Installing Conan..."
echo "============================================"

CERT_PATH="${WSL_PATH}/ng-certificate-chain.cer"

export REQUESTS_CA_BUNDLE="${CERT_PATH}"
echo "  REQUESTS_CA_BUNDLE set to: ${REQUESTS_CA_BUNDLE}"

pip3 install "conan>=2.0,<3.0"
echo "  [OK] Conan installed."

# ── 6. Configure Conan remotes ────────────────────────────────────────────────
echo ""
echo "============================================"
echo "Configuring Conan remotes..."
echo "============================================"

conan remote remove conancenter || true   # tolerate if it does not exist
conan profile detect --force
conan remote add conan-v2-center-proxy \
    https://nexus-repository.northgrum.com/repository/conan-v2-center-proxy/ \
    || true   # tolerate if remote already exists
conan remote login conan-v2-center-proxy "${NEXUS_USER}" -p "${NEXUS_PASS}"
echo "  [OK] Conan remotes configured."

# ── 7. YUM/DNF setup ─────────────────────────────────────────────────────────
echo ""
echo "============================================"
echo "Setting up YUM/DNF (requires sudo)..."
echo "============================================"

sudo cp "${REPO_FILE}" /etc/yum.repos.d/esf_nexus_8.repo
echo "  [OK] Repo file copied to /etc/yum.repos.d/"

sudo cp "${WSL_PATH}/NorthGrumCorporate-G2.pem" /etc/pki/ca-trust/source/anchors/
sudo cp "${WSL_PATH}/NorthGrumMult-G2.pem"      /etc/pki/ca-trust/source/anchors/
echo "  [OK] CA certificates copied."

sudo update-ca-trust extract
echo "  [OK] CA trust updated."

# ── 8. Update & install Paddock ───────────────────────────────────────────────
echo ""
echo "============================================"
echo "Running dnf update..."
echo "============================================"
sudo dnf update -y

echo ""
echo "============================================"
echo "Installing paddock_main.rpm..."
echo "============================================"
sudo dnf install -y paddock_main.rpm

# ── Done ──────────────────────────────────────────────────────────────────────
echo ""
echo "============================================"
echo "  Paddock setup completed successfully!"
echo "============================================"
