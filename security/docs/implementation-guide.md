# IPO Valuation SaaS - Security Framework Implementation Guide

## Overview

This guide provides comprehensive instructions for implementing enterprise-grade security for the IPO Valuation SaaS platform on Google Cloud Platform. The security framework addresses financial services compliance requirements including SOC 2, ISO 27001, Australian Privacy Principles, and APRA CPS 234.

## Architecture Overview

### Security Layers

1. **Network Security Layer**
   - Multi-tier VPC architecture with isolated subnets
   - Cloud Armor WAF with OWASP Top 10 protection
   - VPC Service Controls for API access
   - DDoS protection and rate limiting

2. **Identity & Access Layer**
   - Zero Trust security model
   - Multi-factor authentication enforcement
   - Role-based access control (RBAC)
   - Conditional access policies

3. **Data Protection Layer**
   - End-to-end encryption (at rest, in transit, in use)
   - Customer-managed encryption keys (CMEK)
   - Data Loss Prevention (DLP)
   - Confidential Computing

4. **Monitoring & Response Layer**
   - Security Command Center integration
   - Real-time threat detection
   - Automated incident response
   - Comprehensive audit logging

## Prerequisites

### Required Permissions

Your GCP account needs the following roles:
- `Organization Administrator` (for organization-level policies)
- `Project Owner` or `Project Editor` with additional IAM roles:
  - `Security Admin`
  - `Network Admin`
  - `Storage Admin`
  - `Cloud KMS Admin`
  - `BigQuery Admin`

### Required Tools

```bash
# Install Google Cloud SDK
curl https://sdk.cloud.google.com | bash
exec -l $SHELL

# Install Terraform
wget https://releases.hashicorp.com/terraform/1.6.0/terraform_1.6.0_linux_amd64.zip
unzip terraform_1.6.0_linux_amd64.zip
sudo mv terraform /usr/local/bin/

# Authenticate with Google Cloud
gcloud auth login
gcloud config set project YOUR_PROJECT_ID
```

### Environment Variables

Set the following environment variables:

```bash
export PROJECT_ID="your-ipo-valuation-project"
export ORGANIZATION_ID="your-organization-id"
export TERRAFORM_STATE_BUCKET="your-terraform-state-bucket"
export REGION="australia-southeast1"
export DOMAIN="uprez.com"
```

## Implementation Steps

### Phase 1: Foundation Setup (Week 1)

#### 1.1 Enable APIs and Create State Bucket

```bash
# Run the deployment script
cd security/scripts
chmod +x deployment-script.sh
./deployment-script.sh
```

Or manually execute individual steps:

```bash
# Enable required APIs
gcloud services enable compute.googleapis.com \
  container.googleapis.com \
  cloudsql.googleapis.com \
  storage.googleapis.com \
  bigquery.googleapis.com \
  cloudkms.googleapis.com \
  securitycenter.googleapis.com

# Create Terraform state bucket
gsutil mb -p $PROJECT_ID -l $REGION gs://$TERRAFORM_STATE_BUCKET
gsutil versioning set on gs://$TERRAFORM_STATE_BUCKET
```

#### 1.2 Deploy Core Infrastructure

```bash
cd security/configs
terraform init -backend-config="bucket=$TERRAFORM_STATE_BUCKET"
terraform plan -var="project_id=$PROJECT_ID" -var="organization_id=$ORGANIZATION_ID"
terraform apply
```

### Phase 2: Network Security (Week 2)

#### 2.1 VPC Configuration

The Terraform deployment creates:
- Production VPC with three-tier architecture
- Application subnet (10.1.0.0/24)
- Database subnet (10.1.1.0/24)  
- Management subnet (10.1.2.0/24)
- Cloud NAT for controlled outbound access

#### 2.2 Firewall Rules

Key firewall rules implemented:
```bash
# Verify firewall rules
gcloud compute firewall-rules list --filter="network:ipo-valuation-prod-vpc"

# Test connectivity (from bastion host)
curl -I https://app.uprez.com
```

#### 2.3 Cloud Armor Configuration

Verify WAF deployment:
```bash
gcloud compute security-policies describe ipo-valuation-security-policy --global
```

### Phase 3: Encryption & Key Management (Week 3)

#### 3.1 KMS Key Management

Verify key creation:
```bash
# List key rings
gcloud kms keyrings list --location=$REGION

# List keys in financial-data keyring
gcloud kms keys list --keyring=financial-data --location=$REGION
```

#### 3.2 Storage Encryption

Test encrypted storage:
```bash
# Upload test file to encrypted bucket
echo "sensitive financial data" > /tmp/test-file.txt
gsutil cp /tmp/test-file.txt gs://financial-data-prod-encrypted/
```

#### 3.3 Database Encryption

Verify Cloud SQL encryption:
```bash
gcloud sql instances describe ipo-valuation-db-prod --format="value(diskEncryptionConfiguration.kmsKeyName)"
```

### Phase 4: Identity & Access Management (Week 4)

#### 4.1 Cloud Identity Setup

1. Configure Cloud Identity:
   - Go to [Cloud Identity Console](https://admin.google.com)
   - Set up organizational units
   - Create security groups

2. Create user groups:
```bash
# Create security groups (via Cloud Console or API)
# - ipo-valuation-analysts@uprez.com
# - ipo-data-managers@uprez.com
# - ipo-security-team@uprez.com
# - ipo-executives@uprez.com
```

#### 4.2 Multi-Factor Authentication

1. Enable 2-Step Verification:
   - Go to Security → 2-Step Verification
   - Set enforcement to "On"
   - Configure allowed methods: Security Key, Google Authenticator

2. Configure security key policy:
```bash
# This is typically done through the Cloud Console
# Admin Console → Security → 2-Step Verification → Security key settings
```

#### 4.3 Identity-Aware Proxy Setup

```bash
# Create OAuth brand
gcloud iap oauth-brands create --support_email=support@uprez.com --application_title="IPO Valuation Platform"

# Create OAuth client
gcloud iap oauth-clients create BRAND_ID --display_name="IPO Web Application"
```

### Phase 5: Data Protection & DLP (Week 5)

#### 5.1 Data Loss Prevention

Configure DLP inspection:
```bash
# Create custom info types for IPO data
cat > dlp-config.json << 'EOF'
{
  "displayName": "IPO Financial Inspector",
  "inspectConfig": {
    "customInfoTypes": [
      {
        "infoType": {"name": "IPO_DOCUMENT_ID"},
        "regex": {"pattern": "IPO-\\d{6}-\\d{4}"}
      },
      {
        "infoType": {"name": "VALUATION_AMOUNT"},
        "regex": {"pattern": "\\$[0-9,]+(\\.[0-9]{2})?"}
      }
    ]
  }
}
EOF

gcloud dlp inspect-templates create --source=dlp-config.json --location=$REGION
```

#### 5.2 Confidential Computing

Deploy confidential GKE cluster:
```bash
gcloud container clusters create ipo-confidential-cluster \
    --zone=$ZONE \
    --enable-network-policy \
    --enable-confidential-nodes \
    --machine-type=n2d-standard-4 \
    --num-nodes=3
```

### Phase 6: Monitoring & Incident Response (Week 6)

#### 6.1 Security Command Center

Configure SCC:
```bash
# Enable Security Command Center
gcloud scc settings update \
    --organization=$ORGANIZATION_ID \
    --enable-asset-discovery \
    --enable-security-health-analytics
```

#### 6.2 Logging Configuration

Verify log sinks:
```bash
gcloud logging sinks list
```

Test log export:
```bash
# Generate test log entry
gcloud logging write test-log '{"message":"security test","severity":"INFO"}'

# Query BigQuery for logs
bq query --use_legacy_sql=false '
SELECT timestamp, severity, jsonPayload.message
FROM `'$PROJECT_ID'.security_logs.cloudaudit_googleapis_com_activity`
LIMIT 10'
```

#### 6.3 Monitoring Alerts

Create notification channels:
```bash
# Create email notification channel
cat > notification-channel.json << 'EOF'
{
  "type": "email",
  "displayName": "Security Team",
  "labels": {
    "email_address": "security@uprez.com"
  }
}
EOF

gcloud alpha monitoring channels create --channel-content-from-file=notification-channel.json
```

### Phase 7: Compliance Configuration (Week 7)

#### 7.1 Access Context Manager

Verify VPC Service Controls:
```bash
gcloud access-context-manager policies list --organization=$ORGANIZATION_ID
```

#### 7.2 Audit Configuration

Configure audit logs retention:
```bash
# Set organization policy for audit log retention
cat > audit-retention-policy.yaml << 'EOF'
constraint: constraints/gcp.resourceLocations
listPolicy:
  allowedValues:
  - in:australia-southeast1-locations
EOF

gcloud resource-manager org-policies set-policy audit-retention-policy.yaml --organization=$ORGANIZATION_ID
```

## Testing & Validation

### Security Control Testing

#### 1. Network Security Testing

```bash
# Test firewall rules
nmap -sS -O target-ip

# Test Cloud Armor (should be blocked)
curl -H "User-Agent: sqlmap" https://app.uprez.com
```

#### 2. Encryption Testing

```bash
# Verify encryption at rest
gsutil stat gs://financial-data-prod-encrypted/test-file.txt

# Test database encryption
gcloud sql instances describe ipo-valuation-db-prod | grep kmsKeyName
```

#### 3. Access Control Testing

```bash
# Test IAM conditional access
gcloud auth activate-service-account --key-file=test-sa-key.json
gcloud storage ls gs://financial-data-prod-encrypted/  # Should fail outside business hours
```

### Compliance Validation

#### SOC 2 Control Testing

1. **CC6.1 - Access Controls**: Verify IAM policies and conditional access
2. **CC6.7 - Encryption**: Validate encryption in transit and at rest
3. **CC7.1 - Monitoring**: Test security event detection and alerting

#### APRA CPS 234 Testing

1. **Information Security Capability**: Document security organization and roles
2. **Control Testing**: Execute quarterly security control effectiveness testing
3. **Incident Response**: Test incident response procedures

## Maintenance & Operations

### Daily Operations

```bash
# Daily security dashboard check
gcloud scc findings list --organization=$ORGANIZATION_ID --filter="state=ACTIVE"

# Check key rotation status  
gcloud kms keys list --keyring=financial-data --location=$REGION --format="table(name,nextRotationTime)"
```

### Weekly Tasks

1. Review Security Command Center findings
2. Analyze access logs for anomalies
3. Update threat intelligence feeds
4. Review and approve access requests

### Monthly Tasks

1. Conduct access reviews
2. Update security policies
3. Review compliance metrics
4. Test incident response procedures

### Quarterly Tasks

1. Penetration testing
2. Vulnerability assessments
3. Security control effectiveness testing
4. Compliance audit preparation

## Troubleshooting

### Common Issues

#### 1. KMS Permission Errors
```bash
# Grant KMS permissions to service account
gcloud kms keys add-iam-policy-binding financial-key \
    --keyring=financial-data \
    --location=$REGION \
    --member=serviceAccount:ipo-app@$PROJECT_ID.iam.gserviceaccount.com \
    --role=roles/cloudkms.cryptoKeyEncrypterDecrypter
```

#### 2. VPC Service Controls Access Denied
```bash
# Check access level membership
gcloud access-context-manager levels describe ACCESS_LEVEL_NAME --policy=POLICY_NAME
```

#### 3. Cloud Armor Blocking Legitimate Traffic
```bash
# Review Cloud Armor logs
gcloud logging read "resource.type=http_load_balancer AND jsonPayload.enforcedSecurityPolicy.outcome=DENY" --limit=50
```

### Emergency Procedures

#### Security Incident Response

1. **Immediate containment**:
```bash
# Disable compromised user account
gcloud identity groups memberships delete --group-email=ipo-valuation-analysts@uprez.com --member-email=compromised-user@uprez.com

# Block suspicious IP in Cloud Armor
gcloud compute security-policies rules create 100 \
    --security-policy=ipo-valuation-security-policy \
    --expression="origin.ip == 'SUSPICIOUS_IP'" \
    --action=deny-403
```

2. **Investigation**:
```bash
# Query audit logs for suspicious activity
bq query --use_legacy_sql=false '
SELECT timestamp, protoPayload.authenticationInfo.principalEmail, protoPayload.methodName, protoPayload.resourceName
FROM `'$PROJECT_ID'.security_logs.cloudaudit_googleapis_com_activity`
WHERE protoPayload.authenticationInfo.principalEmail = "suspicious-user@domain.com"
ORDER BY timestamp DESC'
```

## Security Metrics & KPIs

### Key Performance Indicators

1. **Security Event Response Time**: Target < 15 minutes for critical incidents
2. **Vulnerability Remediation**: Critical vulnerabilities resolved within 48 hours
3. **Compliance Score**: Maintain >95% compliance posture
4. **Security Training**: 100% of users complete annual security training

### Monitoring Dashboards

Access security dashboards at:
- **Security Overview**: [Cloud Console Security Dashboard](https://console.cloud.google.com/security)
- **Compliance Dashboard**: Custom dashboard in Cloud Monitoring
- **Incident Response**: Integration with your ITSM system

## Support & Resources

### Internal Contacts
- **Security Team**: security@uprez.com
- **Infrastructure Team**: infrastructure@uprez.com  
- **Compliance Officer**: compliance@uprez.com

### External Resources
- [GCP Security Best Practices](https://cloud.google.com/security/best-practices)
- [APRA CPS 234 Guidance](https://www.apra.gov.au/information-security-management)
- [ISO 27001 Controls](https://www.iso.org/isoiec-27001-information-security.html)

### Emergency Contacts
- **Google Cloud Support**: Create support case in Cloud Console
- **APRA Cyber Incident Reporting**: cyber.incidents@apra.gov.au (for material incidents)
- **Australian Cyber Security Centre**: 1300 292 371

---

**Document Version**: 1.0  
**Last Updated**: $(date)  
**Next Review**: $(date -d '+3 months')  

This implementation guide should be reviewed and updated quarterly or after any major security changes.