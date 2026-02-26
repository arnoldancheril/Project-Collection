set -e

# Paddock Setup Script
# Run inside a RHEL8 WSL instance

echo "Paddock Environment Setup"
echo ""

# Get the path where the setup files live
echo "Enter the path to the folder containing the setup files."
echo "You can enter either a Windows path or an already-converted WSL path:"
echo "  Windows example : C:\\paddock"
echo "  WSL example     : /mnt/c/paddock"
read -rp "Path: " WIN_PATH

# Check for Linux/WSL path (starts with /),
# Or convert a Windows path (C:\paddock) to a WSL mount path (/mnt/c/paddock)
if [[ "${WIN_PATH}" == /* ]]; then
    WSL_PATH="${WIN_PATH}"
else
    WIN_PATH_NORMALIZED="${WIN_PATH//\\//}"

    # Extract drive letter (first character, lowercased) and strip "X:"
    DRIVE_LETTER=$(echo "${WIN_PATH_NORMALIZED:0:1}" | tr '[:upper:]' '[:lower:]')
    REST_OF_PATH="${WIN_PATH_NORMALIZED:2}"

    WSL_PATH="/mnt/${DRIVE_LETTER}${REST_OF_PATH}"
fi

echo ""
echo "Resolved path: ${WSL_PATH}"

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
    "paddock_main.rpm"
    "config.ini"
    "hatchery_file.rpm"
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

# Collect Nexus credentials
echo ""
echo "Enter your Nexus Repository credentials."
echo "WARNING: When typing your Nexus password, it will NOT appear on screen (no characters will be shown)."
read -rp "Nexus Username: " NEXUS_USER
read -rsp "Nexus Password: " NEXUS_PASS
echo ""   # newline after hidden password input

# Update esf_nexus_8.repo with credentials
REPO_FILE="${WSL_PATH}/esf_nexus_8.repo"

echo ""
echo "Updating credentials in esf_nexus_8.repo (lines 66–67)..."

# Use sed to replace the username= and password= lines in-place
# The lines are expected to be exactly "username=" and "password="
sed -i "66s|.*|username=${NEXUS_USER}|" "${REPO_FILE}"
sed -i "67s|.*|password=${NEXUS_PASS}|" "${REPO_FILE}"

echo "  [OK] Credentials written."

# YUM/DNF setup (requires sudo)
echo ""
echo "Setting up YUM/DNF (requires sudo)..."

sudo cp "${REPO_FILE}" /etc/yum.repos.d/esf_nexus_8.repo
echo "  [OK] Repo file copied to /etc/yum.repos.d/"

sudo cp "${WSL_PATH}/NorthGrumCorporate-G2.pem" /etc/pki/ca-trust/source/anchors/
sudo cp "${WSL_PATH}/NorthGrumMult-G2.pem"      /etc/pki/ca-trust/source/anchors/
echo "  [OK] CA certificates copied."

sudo update-ca-trust extract
echo "  [OK] CA trust updated."

# dnf update
echo ""
echo "Running dnf update..."
sudo dnf update -y --nobest --skip-broken

# Install Python 3.12 & upgrade pip
echo ""
echo "Installing Python 3.12 & upgrading pip..."

sudo dnf install -y python3.12 python3.12-pip python3.12-devel
echo "  [OK] Python 3.12 installed."

python3.12 -m pip install --upgrade pip
echo "  [OK] pip upgraded."

# pip setup
echo ""
echo "Setting up pip..."

mkdir -p ~/.config/pip
cp "${WSL_PATH}/pip.conf" ~/.config/pip/
echo "  [OK] pip.conf copied to ~/.config/pip/"

# Install Conan
echo ""
echo "Installing Conan..."

CERT_PATH="${WSL_PATH}/ng-certificate-chain.cer"

export REQUESTS_CA_BUNDLE="${CERT_PATH}"
echo "  REQUESTS_CA_BUNDLE set to: ${REQUESTS_CA_BUNDLE}"

python3.12 -m pip install "conan>=2.0,<3.0"
echo "  [OK] Conan installed."

# Configure Conan remotes
echo ""
echo "Configuring Conan remotes..."

conan remote remove conancenter || true
conan profile detect --force
conan remote add conan-v2-center-proxy \
    https://nexus-repository.northgrum.com/repository/conan-v2-center-proxy/ \
    || true
conan remote login conan-v2-center-proxy "${NEXUS_USER}" -p "${NEXUS_PASS}"
echo "  [OK] Conan remotes configured."

# Install Paddock
echo ""
echo "Installing paddock_main.rpm..."
sudo dnf install -y --nobest "${WSL_PATH}/paddock_main.rpm"
echo "  [OK] paddock_main.rpm installed."

# Install Hatchery
echo ""
echo "Installing hatchery_file.rpm..."
sudo dnf install -y --nobest "${WSL_PATH}/hatchery_file.rpm"
echo "  [OK] hatchery_file.rpm installed."

# Set environment variables
echo ""
echo "Setting environment variables..."

export RAPTOR_HOME="${WSL_PATH}"
export QT_PLUGIN_PATH="/opt/raptorb/hatchery/plugins"

# Persist to ~/.bashrc so they survive across sessions
{
    echo ""
    echo "# Paddock environment (added by setup.sh)"
    echo "export RAPTOR_HOME=\"${WSL_PATH}\""
    echo "export QT_PLUGIN_PATH=\"/opt/raptorb/hatchery/plugins\""
} >> ~/.bashrc

echo "  RAPTOR_HOME=${RAPTOR_HOME}"
echo "  QT_PLUGIN_PATH=${QT_PLUGIN_PATH}"
echo "  [OK] Environment variables set and appended to ~/.bashrc"

# Done
echo ""
echo "  Paddock setup completed successfully!"
