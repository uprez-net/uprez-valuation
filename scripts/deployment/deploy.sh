#!/bin/bash

# Uprez IPO Valuation Platform Deployment Script
# This script handles complete deployment of the platform to different environments

set -e  # Exit on any error
set -u  # Exit on undefined variables

# Script configuration
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")

# Default values
ENVIRONMENT=${ENVIRONMENT:-"staging"}
VERSION=${VERSION:-"latest"}
DRY_RUN=${DRY_RUN:-"false"}
SKIP_TESTS=${SKIP_TESTS:-"false"}
SKIP_BUILD=${SKIP_BUILD:-"false"}
HEALTH_CHECK_TIMEOUT=${HEALTH_CHECK_TIMEOUT:-300}
ROLLBACK_ON_FAILURE=${ROLLBACK_ON_FAILURE:-"true"}

# Color codes for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Logging functions
log() {
    echo -e "${BLUE}[$(date +'%Y-%m-%d %H:%M:%S')] INFO: $1${NC}"
}

warn() {
    echo -e "${YELLOW}[$(date +'%Y-%m-%d %H:%M:%S')] WARN: $1${NC}"
}

error() {
    echo -e "${RED}[$(date +'%Y-%m-%d %H:%M:%S')] ERROR: $1${NC}"
}

success() {
    echo -e "${GREEN}[$(date +'%Y-%m-%d %H:%M:%S')] SUCCESS: $1${NC}"
}

# Function to show usage
usage() {
    cat << EOF
Usage: $0 [OPTIONS]

Deploy Uprez IPO Valuation Platform

OPTIONS:
    -e, --environment ENVIRONMENT    Target environment (staging|production) [default: staging]
    -v, --version VERSION           Version tag to deploy [default: latest]
    -d, --dry-run                   Perform dry run without actual deployment
    -s, --skip-tests                Skip running tests
    -b, --skip-build                Skip building images
    -t, --timeout SECONDS           Health check timeout in seconds [default: 300]
    -r, --no-rollback               Don't rollback on failure
    -h, --help                      Show this help message

EXAMPLES:
    # Deploy to staging
    $0 -e staging -v v1.2.3

    # Production deployment with specific version
    $0 -e production -v v1.2.3

    # Dry run deployment
    $0 -e staging --dry-run

    # Skip tests and build (use existing images)
    $0 -e staging -s -b

ENVIRONMENT VARIABLES:
    ENVIRONMENT                     Target environment
    VERSION                         Version to deploy
    DRY_RUN                         Perform dry run (true|false)
    SKIP_TESTS                      Skip tests (true|false)
    SKIP_BUILD                      Skip build (true|false)
    HEALTH_CHECK_TIMEOUT            Health check timeout in seconds
    ROLLBACK_ON_FAILURE             Rollback on failure (true|false)
    KUBECONFIG                      Path to kubeconfig file
    DOCKER_REGISTRY                 Docker registry URL
    SLACK_WEBHOOK                   Slack webhook for notifications

EOF
}

# Parse command line arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        -e|--environment)
            ENVIRONMENT="$2"
            shift 2
            ;;
        -v|--version)
            VERSION="$2"
            shift 2
            ;;
        -d|--dry-run)
            DRY_RUN="true"
            shift
            ;;
        -s|--skip-tests)
            SKIP_TESTS="true"
            shift
            ;;
        -b|--skip-build)
            SKIP_BUILD="true"
            shift
            ;;
        -t|--timeout)
            HEALTH_CHECK_TIMEOUT="$2"
            shift 2
            ;;
        -r|--no-rollback)
            ROLLBACK_ON_FAILURE="false"
            shift
            ;;
        -h|--help)
            usage
            exit 0
            ;;
        *)
            error "Unknown option: $1"
            usage
            exit 1
            ;;
    esac
done

# Validate environment
if [[ "$ENVIRONMENT" != "staging" && "$ENVIRONMENT" != "production" ]]; then
    error "Invalid environment: $ENVIRONMENT. Must be 'staging' or 'production'"
    exit 1
fi

# Set environment-specific variables
case $ENVIRONMENT in
    staging)
        NAMESPACE="uprez-staging"
        REGISTRY="${DOCKER_REGISTRY:-gcr.io/uprez-staging}"
        INGRESS_HOST="staging.uprez.com"
        ;;
    production)
        NAMESPACE="uprez-production"
        REGISTRY="${DOCKER_REGISTRY:-gcr.io/uprez-production}"
        INGRESS_HOST="api.uprez.com"
        ;;
esac

# Services to deploy
SERVICES=(
    "api-service"
    "ml-service"
    "frontend-service"
    "data-pipeline"
)

# Function to check prerequisites
check_prerequisites() {
    log "Checking prerequisites..."
    
    # Check required commands
    local required_commands=("kubectl" "docker" "helm" "jq")
    for cmd in "${required_commands[@]}"; do
        if ! command -v "$cmd" &> /dev/null; then
            error "$cmd is not installed or not in PATH"
            exit 1
        fi
    done
    
    # Check kubectl access
    if ! kubectl cluster-info &> /dev/null; then
        error "Cannot connect to Kubernetes cluster"
        exit 1
    fi
    
    # Check namespace exists
    if ! kubectl get namespace "$NAMESPACE" &> /dev/null; then
        warn "Namespace $NAMESPACE does not exist, creating it..."
        if [[ "$DRY_RUN" != "true" ]]; then
            kubectl create namespace "$NAMESPACE"
        fi
    fi
    
    # Check Docker registry access
    if ! docker pull "$REGISTRY/api-service:$VERSION" &> /dev/null; then
        warn "Cannot pull images from registry, will attempt to build"
        SKIP_BUILD="false"
    fi
    
    success "Prerequisites check completed"
}

# Function to run tests
run_tests() {
    if [[ "$SKIP_TESTS" == "true" ]]; then
        warn "Skipping tests (--skip-tests flag provided)"
        return 0
    fi
    
    log "Running test suite..."
    
    cd "$PROJECT_ROOT"
    
    # Run unit tests
    log "Running unit tests..."
    if [[ "$DRY_RUN" != "true" ]]; then
        python -m pytest tests/unit/ -v --cov=src/ --cov-report=xml || {
            error "Unit tests failed"
            return 1
        }
    else
        log "DRY RUN: Would run unit tests"
    fi
    
    # Run integration tests
    log "Running integration tests..."
    if [[ "$DRY_RUN" != "true" ]]; then
        docker-compose -f docker-compose.test.yml up -d
        sleep 30  # Wait for services to start
        
        python -m pytest tests/integration/ -v || {
            error "Integration tests failed"
            docker-compose -f docker-compose.test.yml down
            return 1
        }
        
        docker-compose -f docker-compose.test.yml down
    else
        log "DRY RUN: Would run integration tests"
    fi
    
    success "All tests passed"
}

# Function to build and push images
build_and_push_images() {
    if [[ "$SKIP_BUILD" == "true" ]]; then
        warn "Skipping image build (--skip-build flag provided)"
        return 0
    fi
    
    log "Building and pushing container images..."
    
    cd "$PROJECT_ROOT"
    
    # Build images
    for service in "${SERVICES[@]}"; do
        local image_name="$REGISTRY/$service:$VERSION"
        local dockerfile_path="docker/$service/Dockerfile"
        
        log "Building $service image..."
        
        if [[ "$DRY_RUN" != "true" ]]; then
            docker build \
                -f "$dockerfile_path" \
                -t "$image_name" \
                --build-arg VERSION="$VERSION" \
                --build-arg BUILD_DATE="$(date -u +'%Y-%m-%dT%H:%M:%SZ')" \
                --build-arg VCS_REF="$(git rev-parse HEAD)" \
                . || {
                    error "Failed to build $service image"
                    return 1
                }
            
            # Tag as latest if this is the latest version
            if [[ "$VERSION" != "latest" ]]; then
                docker tag "$image_name" "$REGISTRY/$service:latest"
            fi
            
            # Push image
            log "Pushing $service image..."
            docker push "$image_name" || {
                error "Failed to push $service image"
                return 1
            }
            
            if [[ "$VERSION" != "latest" ]]; then
                docker push "$REGISTRY/$service:latest"
            fi
        else
            log "DRY RUN: Would build and push $service image"
        fi
    done
    
    success "All images built and pushed successfully"
}

# Function to create deployment backup
create_backup() {
    log "Creating deployment backup..."
    
    local backup_dir="$PROJECT_ROOT/backups/$ENVIRONMENT/$TIMESTAMP"
    mkdir -p "$backup_dir"
    
    # Backup current deployments
    for service in "${SERVICES[@]}"; do
        if kubectl get deployment "$service" -n "$NAMESPACE" &> /dev/null; then
            kubectl get deployment "$service" -n "$NAMESPACE" -o yaml > "$backup_dir/$service-deployment.yaml"
        fi
    done
    
    # Backup configmaps and secrets
    kubectl get configmaps -n "$NAMESPACE" -o yaml > "$backup_dir/configmaps.yaml"
    kubectl get secrets -n "$NAMESPACE" -o yaml > "$backup_dir/secrets.yaml"
    
    success "Backup created at $backup_dir"
    echo "$backup_dir" > /tmp/uprez_backup_path
}

# Function to update configurations
update_configurations() {
    log "Updating configurations..."
    
    cd "$PROJECT_ROOT"
    
    # Apply ConfigMaps
    log "Applying ConfigMaps..."
    if [[ "$DRY_RUN" != "true" ]]; then
        envsubst < "k8s/configs/api-config.yaml" | kubectl apply -f - || {
            error "Failed to apply API config"
            return 1
        }
        envsubst < "k8s/configs/ml-service-config.yaml" | kubectl apply -f - || {
            error "Failed to apply ML service config"
            return 1
        }
    else
        log "DRY RUN: Would apply ConfigMaps"
    fi
    
    # Apply Secrets (if they don't exist)
    if ! kubectl get secret database-secrets -n "$NAMESPACE" &> /dev/null; then
        log "Creating database secrets..."
        if [[ "$DRY_RUN" != "true" ]]; then
            kubectl create secret generic database-secrets \
                --from-env-file="configs/$ENVIRONMENT/database.env" \
                -n "$NAMESPACE" || {
                error "Failed to create database secrets"
                return 1
            }
        else
            log "DRY RUN: Would create database secrets"
        fi
    fi
    
    success "Configurations updated"
}

# Function to deploy services
deploy_services() {
    log "Deploying services..."
    
    cd "$PROJECT_ROOT"
    
    # Deploy each service
    for service in "${SERVICES[@]}"; do
        log "Deploying $service..."
        
        local deployment_file="k8s/deployments/$service-deployment.yaml"
        
        if [[ "$DRY_RUN" != "true" ]]; then
            # Update image tag in deployment
            sed "s|IMAGE_TAG|$VERSION|g" "$deployment_file" | \
            sed "s|REGISTRY|$REGISTRY|g" | \
            kubectl apply -f - || {
                error "Failed to deploy $service"
                return 1
            }
            
            # Wait for rollout to complete
            log "Waiting for $service rollout to complete..."
            kubectl rollout status deployment "$service" -n "$NAMESPACE" --timeout=300s || {
                error "$service deployment failed"
                return 1
            }
        else
            log "DRY RUN: Would deploy $service"
        fi
    done
    
    success "All services deployed successfully"
}

# Function to perform health checks
health_check() {
    log "Performing health checks..."
    
    local start_time=$(date +%s)
    local timeout=$HEALTH_CHECK_TIMEOUT
    
    # Wait for all pods to be ready
    for service in "${SERVICES[@]}"; do
        log "Checking health of $service..."
        
        if [[ "$DRY_RUN" != "true" ]]; then
            local ready=false
            local elapsed=0
            
            while [[ $elapsed -lt $timeout ]]; do
                local ready_replicas=$(kubectl get deployment "$service" -n "$NAMESPACE" -o jsonpath='{.status.readyReplicas}' 2>/dev/null || echo "0")
                local desired_replicas=$(kubectl get deployment "$service" -n "$NAMESPACE" -o jsonpath='{.spec.replicas}' 2>/dev/null || echo "1")
                
                if [[ "$ready_replicas" == "$desired_replicas" && "$ready_replicas" != "0" ]]; then
                    ready=true
                    break
                fi
                
                log "Waiting for $service to be ready... ($ready_replicas/$desired_replicas)"
                sleep 10
                elapsed=$(($(date +%s) - start_time))
            done
            
            if [[ "$ready" != "true" ]]; then
                error "$service failed health check (timeout after ${timeout}s)"
                return 1
            fi
        else
            log "DRY RUN: Would check health of $service"
        fi
    done
    
    # Test API endpoints
    if [[ "$DRY_RUN" != "true" ]]; then
        log "Testing API endpoints..."
        
        # Test health endpoint
        local health_url="https://$INGRESS_HOST/health"
        if ! curl -f -s "$health_url" > /dev/null; then
            error "Health check endpoint failed: $health_url"
            return 1
        fi
        
        # Test ML service endpoint
        local ml_health_url="https://$INGRESS_HOST/api/v1/ml/health"
        if ! curl -f -s "$ml_health_url" > /dev/null; then
            error "ML service health check failed: $ml_health_url"
            return 1
        fi
    else
        log "DRY RUN: Would test API endpoints"
    fi
    
    success "All health checks passed"
}

# Function to run smoke tests
run_smoke_tests() {
    log "Running smoke tests..."
    
    if [[ "$DRY_RUN" != "true" ]]; then
        cd "$PROJECT_ROOT"
        
        # Set environment variables for smoke tests
        export SMOKE_TEST_BASE_URL="https://$INGRESS_HOST"
        export SMOKE_TEST_ENVIRONMENT="$ENVIRONMENT"
        
        # Run smoke tests
        python -m pytest tests/smoke/ -v || {
            error "Smoke tests failed"
            return 1
        }
    else
        log "DRY RUN: Would run smoke tests"
    fi
    
    success "Smoke tests passed"
}

# Function to rollback deployment
rollback_deployment() {
    if [[ "$ROLLBACK_ON_FAILURE" != "true" ]]; then
        warn "Rollback disabled (--no-rollback flag provided)"
        return 0
    fi
    
    error "Deployment failed, initiating rollback..."
    
    # Get backup path
    local backup_path
    if [[ -f "/tmp/uprez_backup_path" ]]; then
        backup_path=$(cat /tmp/uprez_backup_path)
    else
        error "No backup path found, cannot rollback"
        return 1
    fi
    
    # Rollback each service
    for service in "${SERVICES[@]}"; do
        local backup_file="$backup_path/$service-deployment.yaml"
        
        if [[ -f "$backup_file" ]]; then
            log "Rolling back $service..."
            kubectl apply -f "$backup_file" || {
                error "Failed to rollback $service"
            }
            
            # Wait for rollback to complete
            kubectl rollout status deployment "$service" -n "$NAMESPACE" --timeout=300s || {
                error "$service rollback failed"
            }
        else
            warn "No backup found for $service"
        fi
    done
    
    warn "Rollback completed"
}

# Function to send notifications
send_notification() {
    local message="$1"
    local status="$2"  # success, warning, error
    
    # Send Slack notification if webhook is configured
    if [[ -n "${SLACK_WEBHOOK:-}" ]]; then
        local color
        case $status in
            success) color="good" ;;
            warning) color="warning" ;;
            error) color="danger" ;;
            *) color="warning" ;;
        esac
        
        local payload=$(cat <<EOF
{
    "attachments": [
        {
            "color": "$color",
            "title": "Uprez Deployment - $ENVIRONMENT",
            "fields": [
                {
                    "title": "Environment",
                    "value": "$ENVIRONMENT",
                    "short": true
                },
                {
                    "title": "Version",
                    "value": "$VERSION",
                    "short": true
                },
                {
                    "title": "Status",
                    "value": "$message",
                    "short": false
                },
                {
                    "title": "Deployed by",
                    "value": "${USER:-unknown}",
                    "short": true
                },
                {
                    "title": "Timestamp",
                    "value": "$(date)",
                    "short": true
                }
            ]
        }
    ]
}
EOF
        )
        
        curl -X POST -H 'Content-type: application/json' \
             --data "$payload" \
             "$SLACK_WEBHOOK" &> /dev/null || {
            warn "Failed to send Slack notification"
        }
    fi
}

# Function to cleanup temporary files
cleanup() {
    log "Cleaning up temporary files..."
    
    # Remove backup path file
    rm -f /tmp/uprez_backup_path
    
    # Clean up docker images older than 7 days (only in CI)
    if [[ -n "${CI:-}" ]]; then
        docker image prune -a --force --filter "until=168h" &> /dev/null || true
    fi
}

# Main deployment function
main() {
    log "Starting deployment of Uprez IPO Valuation Platform"
    log "Environment: $ENVIRONMENT"
    log "Version: $VERSION"
    log "Namespace: $NAMESPACE"
    log "Registry: $REGISTRY"
    
    if [[ "$DRY_RUN" == "true" ]]; then
        warn "Running in DRY RUN mode - no actual changes will be made"
    fi
    
    # Set up error handling
    trap 'error "Deployment failed"; rollback_deployment; send_notification "Deployment failed" "error"; cleanup; exit 1' ERR
    
    # Run deployment steps
    check_prerequisites
    
    if [[ "$ENVIRONMENT" == "production" ]]; then
        log "Production deployment detected - additional checks required"
        read -p "Are you sure you want to deploy to production? (yes/no): " -r
        if [[ ! $REPLY =~ ^[Yy][Ee][Ss]$ ]]; then
            log "Deployment cancelled by user"
            exit 0
        fi
    fi
    
    create_backup
    run_tests
    build_and_push_images
    update_configurations
    deploy_services
    health_check
    run_smoke_tests
    
    success "Deployment completed successfully!"
    send_notification "Deployment completed successfully" "success"
    
    # Show deployment summary
    log "Deployment Summary:"
    log "  Environment: $ENVIRONMENT"
    log "  Version: $VERSION"
    log "  Namespace: $NAMESPACE"
    log "  Services Deployed: ${SERVICES[*]}"
    log "  Health Check URL: https://$INGRESS_HOST/health"
    
    cleanup
}

# Run main function
main "$@"