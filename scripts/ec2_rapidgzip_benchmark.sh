#!/bin/bash
#
# EC2 Rapidgzip Benchmark Script
#
# This script:
# 1. Launches a compute-optimized EC2 instance
# 2. Builds gzippy, rapidgzip, pigz, and igzip from source
# 3. Downloads the Silesia corpus
# 4. Runs comprehensive compression and decompression benchmarks
# 5. Collects and reports results
#
# Prerequisites:
# - AWS CLI configured with credentials (via ENV vars or ~/.aws/credentials)
# - SSH key pair available in AWS
#
# Usage:
#   export AWS_ACCESS_KEY_ID=...
#   export AWS_SECRET_ACCESS_KEY=...
#   export AWS_DEFAULT_REGION=us-east-1
#   ./scripts/ec2_rapidgzip_benchmark.sh
#

set -euo pipefail

# Configuration
INSTANCE_TYPE="${INSTANCE_TYPE:-c6i.16xlarge}"  # 64 vCPUs, 128GB RAM
AMI_ID="${AMI_ID:-}"  # Will auto-detect latest Amazon Linux 2023
KEY_NAME="${KEY_NAME:-gzippy-benchmark}"
SECURITY_GROUP="${SECURITY_GROUP:-gzippy-benchmark-sg}"
REGION="${AWS_DEFAULT_REGION:-us-east-1}"
INSTANCE_NAME="gzippy-rapidgzip-benchmark"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

log_info() { echo -e "${GREEN}[INFO]${NC} $*"; }
log_warn() { echo -e "${YELLOW}[WARN]${NC} $*"; }
log_error() { echo -e "${RED}[ERROR]${NC} $*"; }

# Check prerequisites
check_prerequisites() {
    log_info "Checking prerequisites..."
    
    if ! command -v aws &> /dev/null; then
        log_error "AWS CLI not found. Install with: brew install awscli"
        exit 1
    fi
    
    if [[ -z "${AWS_ACCESS_KEY_ID:-}" ]] && [[ ! -f ~/.aws/credentials ]]; then
        log_error "AWS credentials not found. Set AWS_ACCESS_KEY_ID/AWS_SECRET_ACCESS_KEY or configure ~/.aws/credentials"
        exit 1
    fi
    
    # Test AWS access
    if ! aws sts get-caller-identity &> /dev/null; then
        log_error "AWS credentials invalid or expired"
        exit 1
    fi
    
    log_info "AWS credentials valid: $(aws sts get-caller-identity --query 'Arn' --output text)"
}

# Get latest Amazon Linux 2023 AMI
get_ami_id() {
    if [[ -n "$AMI_ID" ]]; then
        echo "$AMI_ID"
        return
    fi
    
    log_info "Finding latest Amazon Linux 2023 AMI..."
    aws ec2 describe-images \
        --owners amazon \
        --filters "Name=name,Values=al2023-ami-*-x86_64" \
                  "Name=state,Values=available" \
        --query 'sort_by(Images, &CreationDate)[-1].ImageId' \
        --output text \
        --region "$REGION"
}

# Create or get security group
setup_security_group() {
    log_info "Setting up security group..."
    
    # Check if security group exists
    SG_ID=$(aws ec2 describe-security-groups \
        --filters "Name=group-name,Values=$SECURITY_GROUP" \
        --query 'SecurityGroups[0].GroupId' \
        --output text \
        --region "$REGION" 2>/dev/null || echo "None")
    
    if [[ "$SG_ID" == "None" ]] || [[ -z "$SG_ID" ]]; then
        log_info "Creating security group $SECURITY_GROUP..."
        SG_ID=$(aws ec2 create-security-group \
            --group-name "$SECURITY_GROUP" \
            --description "Security group for gzippy benchmarks" \
            --region "$REGION" \
            --query 'GroupId' \
            --output text)
        
        # Allow SSH from anywhere (you may want to restrict this)
        aws ec2 authorize-security-group-ingress \
            --group-id "$SG_ID" \
            --protocol tcp \
            --port 22 \
            --cidr 0.0.0.0/0 \
            --region "$REGION"
        
        log_info "Created security group: $SG_ID"
    else
        log_info "Using existing security group: $SG_ID"
    fi
    
    echo "$SG_ID"
}

# Create or get key pair
setup_key_pair() {
    log_info "Setting up key pair..."
    
    KEY_FILE="$HOME/.ssh/${KEY_NAME}.pem"
    
    # Check if key pair exists in AWS
    if aws ec2 describe-key-pairs --key-names "$KEY_NAME" --region "$REGION" &>/dev/null; then
        if [[ -f "$KEY_FILE" ]]; then
            log_info "Using existing key pair: $KEY_NAME"
            echo "$KEY_FILE"
            return
        else
            log_warn "Key pair exists in AWS but local file missing. Deleting and recreating..."
            aws ec2 delete-key-pair --key-name "$KEY_NAME" --region "$REGION"
        fi
    fi
    
    log_info "Creating key pair $KEY_NAME..."
    aws ec2 create-key-pair \
        --key-name "$KEY_NAME" \
        --query 'KeyMaterial' \
        --output text \
        --region "$REGION" > "$KEY_FILE"
    
    chmod 600 "$KEY_FILE"
    log_info "Created key pair: $KEY_FILE"
    echo "$KEY_FILE"
}

# Generate the benchmark script to run on EC2
generate_benchmark_script() {
    cat << 'BENCHMARK_SCRIPT'
#!/bin/bash
set -euo pipefail

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m'

log_info() { echo -e "${GREEN}[INFO]${NC} $*"; }
log_warn() { echo -e "${YELLOW}[WARN]${NC} $*"; }
log_error() { echo -e "${RED}[ERROR]${NC} $*"; }

WORK_DIR="$HOME/benchmark"
RESULTS_DIR="$WORK_DIR/results"
RUNS=5

mkdir -p "$WORK_DIR" "$RESULTS_DIR"
cd "$WORK_DIR"

log_info "=== System Information ==="
echo "CPU: $(lscpu | grep 'Model name' | cut -d: -f2 | xargs)"
echo "Cores: $(nproc)"
echo "Memory: $(free -h | grep Mem | awk '{print $2}')"
echo "Kernel: $(uname -r)"

log_info "=== Installing Dependencies ==="
sudo dnf install -y git gcc gcc-c++ make cmake nasm autoconf automake libtool \
    python3 python3-pip python3-devel zlib-devel curl wget tar

# Install Rust
if ! command -v cargo &> /dev/null; then
    log_info "Installing Rust..."
    curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y
    source "$HOME/.cargo/env"
fi

log_info "=== Building gzippy ==="
if [[ ! -d gzippy ]]; then
    git clone --depth 1 https://github.com/JackDanger/gzippy.git
fi
cd gzippy
git pull
cargo build --release
GZIPPY="$PWD/target/release/gzippy"
log_info "gzippy built: $GZIPPY"
cd "$WORK_DIR"

log_info "=== Building pigz ==="
if [[ ! -d pigz ]]; then
    git clone --depth 1 https://github.com/madler/pigz.git
fi
cd pigz
make clean || true
make -j$(nproc)
PIGZ="$PWD/pigz"
log_info "pigz built: $PIGZ"
cd "$WORK_DIR"

log_info "=== Building igzip (ISA-L) ==="
if [[ ! -d isa-l ]]; then
    git clone --depth 1 https://github.com/intel/isa-l.git
fi
cd isa-l
./autogen.sh
./configure
make -j$(nproc)
IGZIP="$PWD/igzip"
log_info "igzip built: $IGZIP"
cd "$WORK_DIR"

log_info "=== Building rapidgzip ==="
pip3 install --user rapidgzip
RAPIDGZIP="$HOME/.local/bin/rapidgzip"
if [[ ! -f "$RAPIDGZIP" ]]; then
    # Try alternative location
    RAPIDGZIP=$(python3 -c "import rapidgzip; import os; print(os.path.dirname(rapidgzip.__file__))")/../../../bin/rapidgzip
fi
# Verify rapidgzip works
python3 -c "import rapidgzip; print(f'rapidgzip version: {rapidgzip.__version__}')"
log_info "rapidgzip installed"

log_info "=== Downloading Silesia Corpus ==="
SILESIA_URL="https://sun.aei.polsl.pl/~sdeor/corpus/silesia.zip"
if [[ ! -f silesia.zip ]]; then
    wget -q "$SILESIA_URL" -O silesia.zip
fi
if [[ ! -d silesia ]]; then
    unzip -q silesia.zip -d silesia
fi
# Create a single tarball of the corpus
if [[ ! -f silesia.tar ]]; then
    tar -cf silesia.tar silesia/
fi
SILESIA_SIZE=$(stat -c%s silesia.tar)
log_info "Silesia corpus: $(du -h silesia.tar | cut -f1)"

# Create enlarged test files for scaling benchmarks
log_info "=== Creating Enlarged Test Files ==="
CORES=$(nproc)

# Create a file that's ~2GB uncompressed for scaling tests
SCALE_FILE="$WORK_DIR/silesia-scaled.tar"
if [[ ! -f "$SCALE_FILE" ]]; then
    log_info "Creating scaled test file (~2GB)..."
    for i in $(seq 1 10); do
        cat silesia.tar
    done > "$SCALE_FILE"
fi
SCALE_SIZE=$(stat -c%s "$SCALE_FILE")
log_info "Scaled test file: $(du -h "$SCALE_FILE" | cut -f1)"

# Function to benchmark compression
benchmark_compress() {
    local tool=$1
    local level=$2
    local threads=$3
    local input=$4
    local output=$5
    local runs=$6
    
    local times=()
    local sizes=()
    
    for i in $(seq 1 $runs); do
        rm -f "$output"
        
        local start=$(date +%s.%N)
        case $tool in
            gzippy)
                "$GZIPPY" -$level -p$threads -c "$input" > "$output" 2>/dev/null
                ;;
            pigz)
                "$PIGZ" -$level -p $threads -c "$input" > "$output" 2>/dev/null
                ;;
            igzip)
                # igzip levels 0-3, map appropriately
                local igzip_level=$(( (level - 1) / 3 ))
                igzip_level=$((igzip_level > 3 ? 3 : igzip_level))
                "$IGZIP" -$igzip_level -T $threads -c "$input" > "$output" 2>/dev/null
                ;;
            gzip)
                gzip -$level -c "$input" > "$output" 2>/dev/null
                ;;
        esac
        local end=$(date +%s.%N)
        
        local duration=$(echo "$end - $start" | bc)
        local size=$(stat -c%s "$output")
        times+=("$duration")
        sizes+=("$size")
    done
    
    # Calculate median time
    local sorted_times=($(printf '%s\n' "${times[@]}" | sort -n))
    local median_idx=$((runs / 2))
    local median_time=${sorted_times[$median_idx]}
    local final_size=${sizes[0]}
    
    echo "$median_time $final_size"
}

# Function to benchmark decompression
benchmark_decompress() {
    local tool=$1
    local threads=$2
    local input=$3
    local runs=$4
    
    local times=()
    
    for i in $(seq 1 $runs); do
        local start=$(date +%s.%N)
        case $tool in
            gzippy)
                "$GZIPPY" -d -p$threads -c "$input" > /dev/null 2>/dev/null
                ;;
            pigz)
                "$PIGZ" -d -p $threads -c "$input" > /dev/null 2>/dev/null
                ;;
            igzip)
                "$IGZIP" -d -T $threads -c "$input" > /dev/null 2>/dev/null
                ;;
            rapidgzip)
                python3 -c "
import rapidgzip
import sys
with rapidgzip.open('$input', parallelization=$threads) as f:
    while True:
        chunk = f.read(1024*1024)
        if not chunk:
            break
" 2>/dev/null
                ;;
            gzip)
                gzip -d -c "$input" > /dev/null 2>/dev/null
                ;;
        esac
        local end=$(date +%s.%N)
        
        local duration=$(echo "$end - $start" | bc)
        times+=("$duration")
    done
    
    # Calculate median time
    local sorted_times=($(printf '%s\n' "${times[@]}" | sort -n))
    local median_idx=$((runs / 2))
    echo "${sorted_times[$median_idx]}"
}

log_info "=== Running Compression Benchmarks ==="

echo ""
echo "## Compression Benchmark Results" | tee "$RESULTS_DIR/compression.md"
echo "" | tee -a "$RESULTS_DIR/compression.md"
echo "Input: silesia.tar ($(du -h silesia.tar | cut -f1))" | tee -a "$RESULTS_DIR/compression.md"
echo "Cores: $(nproc)" | tee -a "$RESULTS_DIR/compression.md"
echo "" | tee -a "$RESULTS_DIR/compression.md"

for level in 1 6 9; do
    echo "### Compression Level $level" | tee -a "$RESULTS_DIR/compression.md"
    echo "" | tee -a "$RESULTS_DIR/compression.md"
    echo "| Tool | Threads | Time (s) | Size | Ratio | Bandwidth (MB/s) |" | tee -a "$RESULTS_DIR/compression.md"
    echo "|------|---------|----------|------|-------|-----------------|" | tee -a "$RESULTS_DIR/compression.md"
    
    for threads in 1 $((CORES/4)) $((CORES/2)) $CORES; do
        for tool in gzip pigz igzip gzippy; do
            # gzip is always single-threaded
            if [[ "$tool" == "gzip" ]] && [[ "$threads" != "1" ]]; then
                continue
            fi
            
            output="$WORK_DIR/test-${tool}-l${level}-t${threads}.gz"
            result=$(benchmark_compress "$tool" "$level" "$threads" "silesia.tar" "$output" $RUNS)
            time=$(echo "$result" | cut -d' ' -f1)
            size=$(echo "$result" | cut -d' ' -f2)
            ratio=$(echo "scale=2; $SILESIA_SIZE / $size" | bc)
            bandwidth=$(echo "scale=1; $SILESIA_SIZE / $time / 1024 / 1024" | bc)
            
            printf "| %-7s | %7d | %8.3f | %s | %5.2fx | %15.1f |\n" \
                "$tool" "$threads" "$time" "$(numfmt --to=iec $size)" "$ratio" "$bandwidth" | tee -a "$RESULTS_DIR/compression.md"
        done
    done
    echo "" | tee -a "$RESULTS_DIR/compression.md"
done

log_info "=== Running Decompression Benchmarks ==="

echo ""
echo "## Decompression Benchmark Results" | tee "$RESULTS_DIR/decompression.md"
echo "" | tee -a "$RESULTS_DIR/decompression.md"

# Create test files compressed with different tools
log_info "Creating test files for decompression..."
"$GZIPPY" -9 -p$CORES -c silesia.tar > silesia-gzippy.tar.gz
"$PIGZ" -9 -p $CORES -c silesia.tar > silesia-pigz.tar.gz
gzip -9 -c silesia.tar > silesia-gzip.tar.gz

for compressed_by in gzippy pigz gzip; do
    input="silesia-${compressed_by}.tar.gz"
    input_size=$(stat -c%s "$input")
    
    echo "### Decompressing file compressed by $compressed_by" | tee -a "$RESULTS_DIR/decompression.md"
    echo "" | tee -a "$RESULTS_DIR/decompression.md"
    echo "Input: $input ($(du -h "$input" | cut -f1))" | tee -a "$RESULTS_DIR/decompression.md"
    echo "" | tee -a "$RESULTS_DIR/decompression.md"
    echo "| Tool | Threads | Time (s) | Bandwidth (MB/s) | Speedup vs gzip |" | tee -a "$RESULTS_DIR/decompression.md"
    echo "|------|---------|----------|------------------|-----------------|" | tee -a "$RESULTS_DIR/decompression.md"
    
    # Baseline: single-threaded gzip
    gzip_time=$(benchmark_decompress "gzip" 1 "$input" $RUNS)
    gzip_bandwidth=$(echo "scale=1; $SILESIA_SIZE / $gzip_time / 1024 / 1024" | bc)
    printf "| %-10s | %7d | %8.3f | %16.1f | %15.1fx |\n" \
        "gzip" "1" "$gzip_time" "$gzip_bandwidth" "1.0" | tee -a "$RESULTS_DIR/decompression.md"
    
    for threads in 1 $((CORES/4)) $((CORES/2)) $CORES; do
        for tool in pigz igzip gzippy rapidgzip; do
            time=$(benchmark_decompress "$tool" "$threads" "$input" $RUNS)
            bandwidth=$(echo "scale=1; $SILESIA_SIZE / $time / 1024 / 1024" | bc)
            speedup=$(echo "scale=1; $gzip_time / $time" | bc)
            
            printf "| %-10s | %7d | %8.3f | %16.1f | %15.1fx |\n" \
                "$tool" "$threads" "$time" "$bandwidth" "$speedup" | tee -a "$RESULTS_DIR/decompression.md"
        done
    done
    echo "" | tee -a "$RESULTS_DIR/decompression.md"
done

log_info "=== Scaling Benchmark (Large File) ==="

echo ""
echo "## Scaling Benchmark (2GB file)" | tee "$RESULTS_DIR/scaling.md"
echo "" | tee -a "$RESULTS_DIR/scaling.md"
echo "Input: silesia-scaled.tar ($(du -h "$SCALE_FILE" | cut -f1))" | tee -a "$RESULTS_DIR/scaling.md"
echo "" | tee -a "$RESULTS_DIR/scaling.md"

# Compress the scaled file
log_info "Compressing scaled file..."
"$GZIPPY" -6 -p$CORES -c "$SCALE_FILE" > silesia-scaled-gzippy.tar.gz
"$PIGZ" -6 -p $CORES -c "$SCALE_FILE" > silesia-scaled-pigz.tar.gz

echo "### Compression Scaling" | tee -a "$RESULTS_DIR/scaling.md"
echo "" | tee -a "$RESULTS_DIR/scaling.md"
echo "| Tool | Threads | Time (s) | Bandwidth (MB/s) |" | tee -a "$RESULTS_DIR/scaling.md"
echo "|------|---------|----------|------------------|" | tee -a "$RESULTS_DIR/scaling.md"

for threads in 1 2 4 8 16 32 $CORES; do
    if [[ $threads -gt $CORES ]]; then
        continue
    fi
    
    for tool in pigz gzippy; do
        output="$WORK_DIR/scale-${tool}-t${threads}.gz"
        result=$(benchmark_compress "$tool" 6 "$threads" "$SCALE_FILE" "$output" 3)
        time=$(echo "$result" | cut -d' ' -f1)
        bandwidth=$(echo "scale=1; $SCALE_SIZE / $time / 1024 / 1024" | bc)
        
        printf "| %-7s | %7d | %8.3f | %16.1f |\n" \
            "$tool" "$threads" "$time" "$bandwidth" | tee -a "$RESULTS_DIR/scaling.md"
    done
done

echo "" | tee -a "$RESULTS_DIR/scaling.md"
echo "### Decompression Scaling" | tee -a "$RESULTS_DIR/scaling.md"
echo "" | tee -a "$RESULTS_DIR/scaling.md"
echo "| Tool | Threads | Time (s) | Bandwidth (MB/s) |" | tee -a "$RESULTS_DIR/scaling.md"
echo "|------|---------|----------|------------------|" | tee -a "$RESULTS_DIR/scaling.md"

for threads in 1 2 4 8 16 32 $CORES; do
    if [[ $threads -gt $CORES ]]; then
        continue
    fi
    
    for tool in pigz igzip gzippy rapidgzip; do
        time=$(benchmark_decompress "$tool" "$threads" "silesia-scaled-gzippy.tar.gz" 3)
        bandwidth=$(echo "scale=1; $SCALE_SIZE / $time / 1024 / 1024" | bc)
        
        printf "| %-10s | %7d | %8.3f | %16.1f |\n" \
            "$tool" "$threads" "$time" "$bandwidth" | tee -a "$RESULTS_DIR/scaling.md"
    done
done

log_info "=== Summary ==="
echo ""
cat "$RESULTS_DIR/compression.md"
echo ""
cat "$RESULTS_DIR/decompression.md"
echo ""
cat "$RESULTS_DIR/scaling.md"

log_info "Results saved to $RESULTS_DIR/"
log_info "Benchmark complete!"
BENCHMARK_SCRIPT
}

# Launch EC2 instance
launch_instance() {
    local ami_id=$1
    local sg_id=$2
    local key_file=$3
    
    log_info "Launching EC2 instance..."
    log_info "  Instance type: $INSTANCE_TYPE"
    log_info "  AMI: $ami_id"
    
    # Launch instance
    INSTANCE_ID=$(aws ec2 run-instances \
        --image-id "$ami_id" \
        --instance-type "$INSTANCE_TYPE" \
        --key-name "$KEY_NAME" \
        --security-group-ids "$sg_id" \
        --block-device-mappings '[{"DeviceName":"/dev/xvda","Ebs":{"VolumeSize":100,"VolumeType":"gp3"}}]' \
        --tag-specifications "ResourceType=instance,Tags=[{Key=Name,Value=$INSTANCE_NAME}]" \
        --query 'Instances[0].InstanceId' \
        --output text \
        --region "$REGION")
    
    log_info "Instance launched: $INSTANCE_ID"
    
    # Wait for instance to be running
    log_info "Waiting for instance to be running..."
    aws ec2 wait instance-running --instance-ids "$INSTANCE_ID" --region "$REGION"
    
    # Get public IP
    PUBLIC_IP=$(aws ec2 describe-instances \
        --instance-ids "$INSTANCE_ID" \
        --query 'Reservations[0].Instances[0].PublicIpAddress' \
        --output text \
        --region "$REGION")
    
    log_info "Instance running at: $PUBLIC_IP"
    
    # Wait for SSH to be available
    log_info "Waiting for SSH to be available..."
    local retries=30
    while [[ $retries -gt 0 ]]; do
        if ssh -i "$key_file" -o StrictHostKeyChecking=no -o ConnectTimeout=5 ec2-user@"$PUBLIC_IP" "echo 'SSH ready'" 2>/dev/null; then
            break
        fi
        retries=$((retries - 1))
        sleep 10
    done
    
    if [[ $retries -eq 0 ]]; then
        log_error "Timeout waiting for SSH"
        exit 1
    fi
    
    echo "$INSTANCE_ID $PUBLIC_IP"
}

# Run benchmark on instance
run_benchmark() {
    local public_ip=$1
    local key_file=$2
    
    log_info "Uploading benchmark script..."
    generate_benchmark_script > /tmp/benchmark.sh
    chmod +x /tmp/benchmark.sh
    scp -i "$key_file" -o StrictHostKeyChecking=no /tmp/benchmark.sh ec2-user@"$public_ip":/tmp/
    
    log_info "Running benchmark (this will take a while)..."
    ssh -i "$key_file" -o StrictHostKeyChecking=no ec2-user@"$public_ip" "bash /tmp/benchmark.sh" 2>&1 | tee benchmark-output.log
    
    log_info "Downloading results..."
    scp -i "$key_file" -o StrictHostKeyChecking=no -r ec2-user@"$public_ip":~/benchmark/results ./benchmark-results/
    
    log_info "Results saved to ./benchmark-results/"
}

# Cleanup
cleanup() {
    local instance_id=$1
    
    log_info "Terminating instance $instance_id..."
    aws ec2 terminate-instances --instance-ids "$instance_id" --region "$REGION" > /dev/null
    aws ec2 wait instance-terminated --instance-ids "$instance_id" --region "$REGION"
    log_info "Instance terminated"
}

# Main
main() {
    check_prerequisites
    
    AMI_ID=$(get_ami_id)
    log_info "Using AMI: $AMI_ID"
    
    SG_ID=$(setup_security_group)
    KEY_FILE=$(setup_key_pair)
    
    mkdir -p benchmark-results
    
    read INSTANCE_ID PUBLIC_IP <<< $(launch_instance "$AMI_ID" "$SG_ID" "$KEY_FILE")
    
    # Trap to cleanup on exit
    trap "cleanup $INSTANCE_ID" EXIT
    
    run_benchmark "$PUBLIC_IP" "$KEY_FILE"
    
    log_info "=== Benchmark Complete ==="
    log_info "Results are in ./benchmark-results/"
    log_info "Full output log: benchmark-output.log"
    
    # Show summary
    echo ""
    echo "=== SUMMARY ==="
    cat benchmark-results/*.md
}

main "$@"
