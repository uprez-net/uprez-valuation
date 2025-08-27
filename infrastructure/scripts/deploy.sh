#!/bin/bash

# IPO Valuation Platform Deployment Script
# Comprehensive deployment automation with safety checks

set -euo pipefail

# Configuration
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
TIMESTAMP=$(date +%Y%m%d-%H%M%S)
LOG_FILE="/tmp/deploy-${TIMESTAMP}.log"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Default values
ENVIRONMENT=""
PROJECT_ID=""
REGION="us-central1"
CLUSTER_NAME=""
SKIP_TESTS=false
DRY_RUN=false
FORCE_DEPLOY=false
GIT_SHA=""
BUILD_NUMBER=""

# Functions
log() {
    echo -e "${BLUE}[$(date +'%Y-%m-%d %H:%M:%S')]${NC} $1" | tee -a "$LOG_FILE"
}

error() {
    echo -e "${RED}[ERROR]${NC} $1" | tee -a "$LOG_FILE"
    exit 1
}

warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1" | tee -a "$LOG_FILE"
}

success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1" | tee -a "$LOG_FILE"
}

usage() {
    cat << EOF
Usage: $0 [OPTIONS]

Deploy the IPO Valuation Platform to Google Cloud Platform

OPTIONS:
    -e, --environment ENVIRONMENT    Target environment (dev|staging|prod)
    -p, --project PROJECT_ID         GCP project ID
    -r, --region REGION              GCP region (default: us-central1)
    -c, --cluster CLUSTER_NAME       GKE cluster name
    -s, --skip-tests                 Skip test execution
    -d, --dry-run                    Perform dry run without actual deployment
    -f, --force                      Force deployment without confirmation
    --git-sha GIT_SHA               Git SHA for image tagging
    --build-number BUILD_NUMBER      CI build number
    -h, --help                       Show this help message

EXAMPLES:
    # Deploy to staging environment
    $0 -e staging -p my-project-id -c staging-cluster

    # Deploy to production with confirmation
    $0 -e prod -p my-project-id -c prod-cluster

    # Dry run for production
    $0 -e prod -p my-project-id -c prod-cluster --dry-run

EOF
}

# Parse command line arguments
parse_args() {
    while [[ $# -gt 0 ]]; do
        case $1 in
            -e|--environment)
                ENVIRONMENT="$2"
                shift 2
                ;;
            -p|--project)
                PROJECT_ID="$2"
                shift 2
                ;;
            -r|--region)
                REGION="$2"
                shift 2
                ;;
            -c|--cluster)
                CLUSTER_NAME="$2"
                shift 2
                ;;
            -s|--skip-tests)
                SKIP_TESTS=true
                shift
                ;;
            -d|--dry-run)
                DRY_RUN=true
                shift
                ;;
            -f|--force)
                FORCE_DEPLOY=true
                shift
                ;;
            --git-sha)
                GIT_SHA="$2"
                shift 2
                ;;
            --build-number)
                BUILD_NUMBER="$2"
                shift 2
                ;;
            -h|--help)
                usage
                exit 0
                ;;
            *)
                error "Unknown option: $1"
                ;;
        esac
    done

    # Validate required parameters
    if [[ -z "$ENVIRONMENT" ]]; then
        error "Environment is required. Use -e or --environment"
    fi

    if [[ -z "$PROJECT_ID" ]]; then
        error "Project ID is required. Use -p or --project"
    fi

    if [[ -z "$CLUSTER_NAME" ]]; then
        CLUSTER_NAME="uprez-valuation-${ENVIRONMENT}-cluster"
    fi

    # Set git sha if not provided
    if [[ -z "$GIT_SHA" ]]; then
        GIT_SHA=$(git rev-parse HEAD)
    fi

    # Set build number if not provided
    if [[ -z "$BUILD_NUMBER" ]]; then
        BUILD_NUMBER=$(date +%Y%m%d%H%M%S)
    fi
}

# Pre-flight checks
preflight_checks() {
    log "Running pre-flight checks..."

    # Check required tools
    local tools=("kubectl" "gcloud" "docker" "kustomize" "helm")
    for tool in "${tools[@]}"; do
        if ! command -v "$tool" &> /dev/null; then
            error "$tool is required but not installed"
        fi
    done

    # Check GCP authentication
    if ! gcloud auth list --filter=status:ACTIVE --format="value(account)" | grep -q "@"; then
        error "Not authenticated with gcloud. Run 'gcloud auth login'"
    fi

    # Validate environment
    if [[ ! "$ENVIRONMENT" =~ ^(dev|staging|prod)$ ]]; then
        error "Environment must be one of: dev, staging, prod"
    fi

    # Check if running on production
    if [[ "$ENVIRONMENT" == "prod" ]] && [[ "$FORCE_DEPLOY" != true ]] && [[ "$DRY_RUN" != true ]]; then
        warning "Deploying to PRODUCTION environment!"
        read -p "Are you sure you want to continue? (yes/no): " -r
        if [[ ! $REPLY =~ ^[Yy][Ee][Ss]$ ]]; then
            log "Deployment cancelled by user"
            exit 0
        fi
    fi

    # Check git status
    if [[ -n "$(git status --porcelain)" ]] && [[ "$ENVIRONMENT" == "prod" ]]; then
        warning "Working directory has uncommitted changes"
        if [[ "$FORCE_DEPLOY" != true ]]; then
            read -p "Continue with uncommitted changes? (yes/no): " -r
            if [[ ! $REPLY =~ ^[Yy][Ee][Ss]$ ]]; then
                log "Deployment cancelled"
                exit 0
            fi
        fi
    fi

    success "Pre-flight checks completed"
}

# Set up GCP and Kubernetes context
setup_context() {
    log "Setting up GCP and Kubernetes context..."

    # Set GCP project
    gcloud config set project "$PROJECT_ID"

    # Get GKE credentials
    gcloud container clusters get-credentials "$CLUSTER_NAME" \
        --region "$REGION" \
        --project "$PROJECT_ID"

    # Verify kubectl context
    local current_context
    current_context=$(kubectl config current-context)
    log "Using Kubernetes context: $current_context"

    # Verify we can connect to the cluster
    kubectl cluster-info --request-timeout=10s > /dev/null || error "Cannot connect to Kubernetes cluster"

    success "Context setup completed"
}

# Build and push Docker images
build_images() {
    log "Building and pushing Docker images..."

    local services=("backend" "frontend" "admin")
    local registry="gcr.io/$PROJECT_ID"

    # Configure Docker for GCR
    gcloud auth configure-docker --quiet

    for service in "${services[@]}"; do
        local image_name="$registry/uprez-valuation-$service"
        local tag="$GIT_SHA"
        local full_image="$image_name:$tag"

        log "Building $service image: $full_image"

        if [[ "$DRY_RUN" == true ]]; then
            log "DRY RUN: Would build and push $full_image"
            continue
        fi

        # Build image
        docker build \
            --tag "$full_image" \
            --tag "$image_name:latest" \
            --file "$PROJECT_ROOT/infrastructure/docker/Dockerfile.$service" \
            --build-arg BUILD_DATE="$(date -u +'%Y-%m-%dT%H:%M:%SZ')" \
            --build-arg VCS_REF="$GIT_SHA" \
            --build-arg VERSION="$BUILD_NUMBER" \
            "$PROJECT_ROOT/src/$service"

        # Push image
        docker push "$full_image"
        docker push "$image_name:latest"

        log "Image $full_image pushed successfully"
    done

    success "Docker images built and pushed"
}

# Run tests
run_tests() {
    if [[ "$SKIP_TESTS" == true ]]; then
        warning "Skipping tests as requested"
        return 0
    fi

    log "Running test suite..."

    # Run unit tests
    log "Running unit tests..."
    cd "$PROJECT_ROOT/src/backend"
    npm test -- --coverage --watchAll=false || error "Backend unit tests failed"

    cd "$PROJECT_ROOT/src/frontend"
    npm test -- --coverage --watchAll=false || error "Frontend unit tests failed"

    # Run integration tests
    log "Running integration tests..."
    cd "$PROJECT_ROOT/tests/integration"
    npm test || error "Integration tests failed"

    # Security scan
    log "Running security scans..."
    cd "$PROJECT_ROOT"
    npm audit --audit-level high || error "Security audit failed"

    success "All tests passed"
}

# Deploy infrastructure with Terraform
deploy_infrastructure() {
    log "Deploying infrastructure with Terraform..."

    cd "$PROJECT_ROOT/infrastructure/terraform"

    # Initialize Terraform
    terraform init \
        -backend-config="bucket=${PROJECT_ID}-terraform-state" \
        -backend-config="prefix=environments/${ENVIRONMENT}"

    # Plan deployment
    log "Planning Terraform deployment..."
    terraform plan \
        -var="project_id=$PROJECT_ID" \
        -var="environment=$ENVIRONMENT" \
        -var="region=$REGION" \
        -var-file="environments/${ENVIRONMENT}/terraform.tfvars" \
        -out="tfplan"

    if [[ "$DRY_RUN" == true ]]; then
        log "DRY RUN: Terraform plan completed, skipping apply"
        return 0
    fi

    # Apply infrastructure changes
    log "Applying Terraform configuration..."
    terraform apply -auto-approve tfplan

    success "Infrastructure deployment completed"
}

# Deploy applications to Kubernetes
deploy_applications() {
    log "Deploying applications to Kubernetes..."

    cd "$PROJECT_ROOT/infrastructure/kubernetes"

    # Update image tags in kustomization
    local overlay_dir="overlays/$ENVIRONMENT"
    
    if [[ ! -d "$overlay_dir" ]]; then
        error "Overlay directory not found: $overlay_dir"
    fi

    cd "$overlay_dir"

    # Update image tags
    kustomize edit set image \
        "frontend-image=gcr.io/$PROJECT_ID/uprez-valuation-frontend:$GIT_SHA" \
        "backend-image=gcr.io/$PROJECT_ID/uprez-valuation-backend:$GIT_SHA" \
        "admin-image=gcr.io/$PROJECT_ID/uprez-valuation-admin:$GIT_SHA"

    if [[ "$DRY_RUN" == true ]]; then
        log "DRY RUN: Would deploy applications with the following manifests:"
        kustomize build .
        return 0
    fi

    # Apply Kubernetes manifests
    log "Applying Kubernetes manifests..."
    kubectl apply -k .

    # Wait for rollout to complete
    log "Waiting for deployment rollout..."
    kubectl rollout status deployment/backend-deployment -n "uprez-valuation-$ENVIRONMENT" --timeout=600s
    kubectl rollout status deployment/frontend-deployment -n "uprez-valuation-$ENVIRONMENT" --timeout=600s
    kubectl rollout status deployment/admin-deployment -n "uprez-valuation-$ENVIRONMENT" --timeout=600s

    success "Application deployment completed"
}

# Run health checks
health_checks() {
    log "Running health checks..."

    local namespace="uprez-valuation-$ENVIRONMENT"
    local max_attempts=30
    local attempt=1

    # Wait for pods to be ready
    while [[ $attempt -le $max_attempts ]]; do
        local ready_pods
        ready_pods=$(kubectl get pods -n "$namespace" -o json | jq -r '.items[] | select(.status.phase == "Running" and (.status.containerStatuses[]?.ready == true)) | .metadata.name' | wc -l)
        local total_pods
        total_pods=$(kubectl get pods -n "$namespace" -o json | jq -r '.items[] | .metadata.name' | wc -l)

        log "Pods ready: $ready_pods/$total_pods (attempt $attempt/$max_attempts)"

        if [[ $ready_pods -eq $total_pods ]] && [[ $total_pods -gt 0 ]]; then
            break
        fi

        if [[ $attempt -eq $max_attempts ]]; then
            error "Health check failed: Not all pods are ready"
        fi

        sleep 10
        ((attempt++))
    done

    # Test application endpoints
    if [[ "$ENVIRONMENT" == "prod" ]]; then
        local base_url="https://uprez-valuation.com"
    elif [[ "$ENVIRONMENT" == "staging" ]]; then
        local base_url="https://staging.uprez-valuation.com"
    else
        local base_url="https://dev.uprez-valuation.com"
    fi

    # Test backend health
    log "Testing backend health endpoint..."
    if curl -f --max-time 30 "$base_url/api/health" > /dev/null 2>&1; then
        success "Backend health check passed"
    else
        error "Backend health check failed"
    fi

    # Test frontend
    log "Testing frontend..."
    if curl -f --max-time 30 "$base_url" > /dev/null 2>&1; then
        success "Frontend health check passed"
    else
        error "Frontend health check failed"
    fi

    success "All health checks passed"
}

# Smoke tests
run_smoke_tests() {
    log "Running smoke tests..."

    if [[ "$DRY_RUN" == true ]]; then
        log "DRY RUN: Skipping smoke tests"
        return 0
    fi

    cd "$PROJECT_ROOT/tests/smoke"
    
    # Run environment-specific smoke tests
    case $ENVIRONMENT in
        "prod")
            npm run test:production
            ;;
        "staging")
            npm run test:staging
            ;;
        "dev")
            npm run test:development
            ;;
    esac

    success "Smoke tests completed"
}

# Cleanup old deployments
cleanup() {
    log "Cleaning up old deployments..."

    if [[ "$DRY_RUN" == true ]]; then
        log "DRY RUN: Skipping cleanup"
        return 0
    fi

    # Clean up old Docker images (keep last 10)
    local services=("backend" "frontend" "admin")
    for service in "${services[@]}"; do
        log "Cleaning up old $service images..."
        gcloud container images list-tags "gcr.io/$PROJECT_ID/uprez-valuation-$service" \
            --limit=999 --sort-by=TIMESTAMP --format="get(digest)" \
            | tail -n +11 \
            | xargs -I {} gcloud container images delete "gcr.io/$PROJECT_ID/uprez-valuation-$service@{}" --force-delete-tags --quiet || true
    done

    success "Cleanup completed"
}

# Send notifications
send_notifications() {
    log "Sending deployment notifications..."

    local status="SUCCESS"
    local message="Deployment to $ENVIRONMENT completed successfully"
    
    if [[ "$DRY_RUN" == true ]]; then
        message="DRY RUN: Deployment to $ENVIRONMENT would succeed"
    fi

    # Slack notification (if webhook configured)
    if [[ -n "${SLACK_WEBHOOK_URL:-}" ]]; then
        curl -X POST -H 'Content-type: application/json' \
            --data "{\"text\":\"🚀 $message\n\nEnvironment: $ENVIRONMENT\nGit SHA: $GIT_SHA\nBuild: $BUILD_NUMBER\"}" \
            "$SLACK_WEBHOOK_URL" || warning "Failed to send Slack notification"
    fi

    # Email notification (if configured)
    if [[ -n "${NOTIFICATION_EMAIL:-}" ]]; then
        echo "$message" | mail -s "Deployment Notification - $ENVIRONMENT" "$NOTIFICATION_EMAIL" || warning "Failed to send email notification"
    fi

    success "Notifications sent"
}

# Main deployment function
main() {
    log "Starting deployment of IPO Valuation Platform"
    log "Environment: $ENVIRONMENT"
    log "Project ID: $PROJECT_ID"
    log "Region: $REGION"
    log "Cluster: $CLUSTER_NAME"
    log "Git SHA: $GIT_SHA"
    log "Build Number: $BUILD_NUMBER"
    log "Dry Run: $DRY_RUN"
    log "Log File: $LOG_FILE"

    # Execute deployment steps
    preflight_checks
    setup_context
    run_tests
    build_images
    deploy_infrastructure
    deploy_applications
    health_checks
    run_smoke_tests
    cleanup
    send_notifications

    success "🎉 Deployment completed successfully!"
    log "Deployment summary:"
    log "  - Environment: $ENVIRONMENT"
    log "  - Git SHA: $GIT_SHA"
    log "  - Build Number: $BUILD_NUMBER"
    log "  - Deployment Time: $(date)"
    log "  - Log File: $LOG_FILE"
}

# Error handling
trap 'error "Deployment failed at line $LINENO"' ERR

# Parse arguments and run main function
parse_args "$@"
main