/**
 * Security Monitoring and Incident Response System
 * Comprehensive monitoring solution for IPO Valuation SaaS
 */

class SecurityMonitoringSystem {
  constructor() {
    this.monitoringConfig = new Map();
    this.alertingRules = new Map();
    this.incidentResponse = new Map();
    this.complianceReporting = new Map();
    this.initializeMonitoring();
  }

  initializeMonitoring() {
    this.setupSecurityCommandCenter();
    this.setupCloudLogging();
    this.setupCloudMonitoring();
    this.setupIncidentResponse();
    this.setupComplianceReporting();
    this.setupAutomatedResponse();
  }

  // Security Command Center Configuration
  setupSecurityCommandCenter() {
    const sccConfig = {
      // Organization-level configuration
      organizationSettings: {
        enableAssetDiscovery: true,
        assetDiscoveryConfig: {
          projectIds: ['ipo-valuation-prod', 'ipo-valuation-staging', 'ipo-valuation-dev'],
          inclusionMode: 'INCLUDE_SPECIFIED',
          folderIds: [],
          organizationIds: []
        },
        enableSecurityHealthAnalytics: true
      },

      // Security sources configuration
      securitySources: {
        'Web Security Scanner': {
          enabled: true,
          configuration: {
            scanTargets: [
              'https://app.uprez.com',
              'https://api.uprez.com',
              'https://admin.uprez.com'
            ],
            scanFrequency: 'weekly',
            authenticationConfig: {
              googleAccount: {
                username: 'security-scanner@uprez.com',
                password: 'stored-in-secret-manager'
              }
            },
            excludedUrls: [
              'https://app.uprez.com/logout',
              'https://api.uprez.com/admin/reset'
            ]
          }
        },
        'Container Analysis': {
          enabled: true,
          configuration: {
            scanOnPush: true,
            vulnerabilityScanning: true,
            policyAnalysis: true
          }
        },
        'Binary Authorization': {
          enabled: true,
          attestationAuthority: 'projects/PROJECT_ID/attestors/security-attestor'
        }
      },

      // Custom security findings
      customSecurityMarks: {
        'financial_data_access': {
          description: 'Resources containing financial data',
          markKey: 'data_classification',
          markValues: ['financial', 'restricted', 'confidential']
        },
        'compliance_scope': {
          description: 'Resources in compliance scope',
          markKey: 'compliance',
          markValues: ['sox', 'pci', 'iso27001', 'apra']
        }
      },

      // Notification configurations
      notificationConfigs: {
        'critical-security-findings': {
          description: 'Critical security findings notification',
          pubsubTopic: 'projects/PROJECT_ID/topics/security-alerts',
          streamingConfig: {
            filter: `
              severity="CRITICAL" OR 
              (severity="HIGH" AND category="FINANCIAL_DATA_EXPOSURE")
            `
          }
        },
        'compliance-violations': {
          description: 'Compliance violation notifications',
          pubsubTopic: 'projects/PROJECT_ID/topics/compliance-alerts',
          streamingConfig: {
            filter: `
              finding_class="VIOLATION" AND 
              (security_marks.marks.compliance="sox" OR 
               security_marks.marks.compliance="apra")
            `
          }
        }
      }
    };

    this.monitoringConfig.set('securityCommandCenter', sccConfig);
  }

  // Cloud Logging Configuration
  setupCloudLogging() {
    const loggingConfig = {
      // Log sinks for different security events
      logSinks: {
        'security-events-bigquery': {
          name: 'security-events-bigquery-sink',
          destination: 'bigquery.googleapis.com/projects/PROJECT_ID/datasets/security_logs',
          filter: `
            protoPayload.authenticationInfo.principalEmail!="" OR
            severity>=ERROR OR
            protoPayload.serviceName="iap.googleapis.com" OR
            protoPayload.serviceName="cloudkms.googleapis.com" OR
            protoPayload.serviceName="secretmanager.googleapis.com" OR
            protoPayload.serviceName="cloudsql.googleapis.com" OR
            jsonPayload.message=~"SECURITY_EVENT"
          `,
          uniqueWriterIdentity: true,
          bigqueryOptions: {
            usePartitionedTables: true,
            writeMetadata: true
          }
        },
        'audit-logs-storage': {
          name: 'audit-logs-storage-sink',
          destination: 'storage.googleapis.com/audit-logs-bucket-encrypted',
          filter: `
            logName:"cloudaudit.googleapis.com" AND
            NOT protoPayload.methodName="storage.objects.list" AND
            NOT protoPayload.methodName="logging.entries.list"
          `,
          uniqueWriterIdentity: true
        },
        'security-siem-export': {
          name: 'security-siem-export',
          destination: 'pubsub.googleapis.com/projects/PROJECT_ID/topics/siem-integration',
          filter: `
            severity>=WARNING AND (
              protoPayload.serviceName="iap.googleapis.com" OR
              protoPayload.serviceName="cloudkms.googleapis.com" OR
              protoPayload.authenticationInfo.principalEmail!="" OR
              jsonPayload.eventType="SECURITY_INCIDENT"
            )
          `
        }
      },

      // Log-based metrics for security events
      logBasedMetrics: {
        'failed_authentication_rate': {
          name: 'failed_authentication_rate',
          description: 'Rate of failed authentication attempts',
          filter: `
            protoPayload.authenticationInfo.principalEmail="" AND
            protoPayload.authorizationInfo.granted=false
          `,
          metricDescriptor: {
            metricKind: 'GAUGE',
            valueType: 'INT64',
            displayName: 'Failed Authentication Rate'
          },
          labelExtractors: {
            'source_ip': 'EXTRACT(protoPayload.requestMetadata.callerIp)',
            'user_agent': 'EXTRACT(protoPayload.requestMetadata.callerSuppliedUserAgent)',
            'resource': 'EXTRACT(protoPayload.resourceName)'
          }
        },
        'suspicious_data_access': {
          name: 'suspicious_data_access',
          description: 'Suspicious data access patterns',
          filter: `
            protoPayload.serviceName="storage.googleapis.com" AND
            protoPayload.methodName="storage.objects.get" AND
            protoPayload.resourceName=~"financial-data|ipo-documents"
          `,
          metricDescriptor: {
            metricKind: 'COUNTER',
            valueType: 'INT64'
          },
          labelExtractors: {
            'user': 'EXTRACT(protoPayload.authenticationInfo.principalEmail)',
            'bucket': 'EXTRACT(protoPayload.resourceName)',
            'time_of_day': 'EXTRACT(timestamp)'
          }
        },
        'privilege_escalation_attempts': {
          name: 'privilege_escalation_attempts',
          description: 'Attempts to escalate privileges',
          filter: `
            protoPayload.serviceName="cloudresourcemanager.googleapis.com" AND
            protoPayload.methodName="SetIamPolicy" AND
            protoPayload.authorizationInfo.granted=false
          `,
          metricDescriptor: {
            metricKind: 'COUNTER',
            valueType: 'INT64'
          }
        }
      },

      // Log retention policies
      retentionPolicies: {
        'security-events': {
          retentionDays: 2555, // 7 years for financial compliance
          lockedRetentionPolicy: true
        },
        'audit-logs': {
          retentionDays: 2555, // 7 years
          lockedRetentionPolicy: true
        },
        'application-logs': {
          retentionDays: 90
        },
        'debug-logs': {
          retentionDays: 7
        }
      }
    };

    this.monitoringConfig.set('logging', loggingConfig);
  }

  // Cloud Monitoring Configuration
  setupCloudMonitoring() {
    const monitoringConfig = {
      // Alert policies for security events
      alertPolicies: [
        {
          displayName: 'Critical Security Finding',
          documentation: {
            content: 'Critical security finding detected in Security Command Center',
            mimeType: 'text/markdown'
          },
          conditions: [
            {
              displayName: 'Critical SCC Finding',
              conditionThreshold: {
                filter: `
                  resource.type="global" AND
                  metric.type="securitycenter.googleapis.com/finding/count" AND
                  metric.labels.category="FINANCIAL_DATA_EXPOSURE"
                `,
                comparison: 'COMPARISON_GREATER_THAN',
                thresholdValue: 0,
                duration: '60s',
                aggregations: [
                  {
                    alignmentPeriod: '60s',
                    perSeriesAligner: 'ALIGN_RATE',
                    crossSeriesReducer: 'REDUCE_SUM'
                  }
                ]
              }
            }
          ],
          notificationChannels: [
            'projects/PROJECT_ID/notificationChannels/security-team-pager',
            'projects/PROJECT_ID/notificationChannels/security-team-slack'
          ],
          alertStrategy: {
            autoClose: '43200s' // 12 hours
          }
        },
        {
          displayName: 'Unusual Data Access Pattern',
          conditions: [
            {
              displayName: 'High volume data access',
              conditionThreshold: {
                filter: `
                  resource.type="gcs_bucket" AND
                  metric.type="storage.googleapis.com/network/sent_bytes_count" AND
                  resource.labels.bucket_name=~"financial-data|ipo-documents"
                `,
                comparison: 'COMPARISON_GREATER_THAN',
                thresholdValue: 1073741824, // 1GB
                duration: '300s'
              }
            }
          ]
        },
        {
          displayName: 'Failed Authentication Spike',
          conditions: [
            {
              displayName: 'Authentication failure rate',
              conditionThreshold: {
                filter: 'metric.type="logging.googleapis.com/user/failed_authentication_rate"',
                comparison: 'COMPARISON_GREATER_THAN',
                thresholdValue: 10,
                duration: '300s'
              }
            }
          ]
        },
        {
          displayName: 'KMS Key Unusual Access',
          conditions: [
            {
              displayName: 'KMS key access from unusual location',
              conditionThreshold: {
                filter: `
                  resource.type="cloudkms_key" AND
                  metric.type="logging.googleapis.com/user/kms_key_access"
                `,
                comparison: 'COMPARISON_GREATER_THAN',
                thresholdValue: 0,
                duration: '60s'
              }
            }
          ]
        }
      ],

      // Custom dashboards
      dashboards: {
        'security-overview': {
          displayName: 'Security Overview Dashboard',
          mosaicLayout: {
            tiles: [
              {
                width: 6,
                height: 4,
                widget: {
                  title: 'Security Command Center Findings',
                  xyChart: {
                    dataSets: [
                      {
                        timeSeriesQuery: {
                          timeSeriesFilter: {
                            filter: 'metric.type="securitycenter.googleapis.com/finding/count"',
                            aggregation: {
                              alignmentPeriod: '3600s',
                              perSeriesAligner: 'ALIGN_SUM'
                            }
                          }
                        },
                        plotType: 'STACKED_AREA'
                      }
                    ],
                    timeshiftDuration: '0s',
                    yAxis: {
                      label: 'Finding Count',
                      scale: 'LINEAR'
                    }
                  }
                }
              },
              {
                width: 6,
                height: 4,
                widget: {
                  title: 'Authentication Events',
                  xyChart: {
                    dataSets: [
                      {
                        timeSeriesQuery: {
                          timeSeriesFilter: {
                            filter: 'metric.type="logging.googleapis.com/user/failed_authentication_rate"'
                          }
                        }
                      }
                    ]
                  }
                }
              },
              {
                width: 12,
                height: 4,
                widget: {
                  title: 'Data Access Patterns',
                  xyChart: {
                    dataSets: [
                      {
                        timeSeriesQuery: {
                          timeSeriesFilter: {
                            filter: 'metric.type="logging.googleapis.com/user/suspicious_data_access"'
                          }
                        }
                      }
                    ]
                  }
                }
              }
            ]
          }
        },
        'compliance-dashboard': {
          displayName: 'Compliance Monitoring Dashboard',
          mosaicLayout: {
            tiles: [
              {
                width: 4,
                height: 4,
                widget: {
                  title: 'SOC 2 Compliance Score',
                  scorecard: {
                    timeSeriesQuery: {
                      timeSeriesFilter: {
                        filter: 'metric.type="custom.googleapis.com/compliance/soc2_score"'
                      }
                    },
                    sparkChartView: {
                      sparkChartType: 'SPARK_LINE'
                    }
                  }
                }
              },
              {
                width: 4,
                height: 4,
                widget: {
                  title: 'APRA CPS 234 Status',
                  scorecard: {
                    timeSeriesQuery: {
                      timeSeriesFilter: {
                        filter: 'metric.type="custom.googleapis.com/compliance/apra_cps234_status"'
                      }
                    }
                  }
                }
              }
            ]
          }
        }
      },

      // Uptime checks for security endpoints
      uptimeCheckConfigs: [
        {
          displayName: 'Security Command Center API Health',
          httpCheck: {
            useSsl: true,
            path: '/v1/organizations/ORGANIZATION_ID/sources',
            port: 443
          },
          monitoredResource: {
            type: 'uptime_url',
            labels: {
              project_id: 'PROJECT_ID',
              host: 'securitycenter.googleapis.com'
            }
          },
          timeout: '10s',
          period: '300s'
        }
      ],

      // Service Level Objectives
      serviceLevelObjectives: [
        {
          displayName: 'Security Alert Response Time SLO',
          serviceLevelIndicator: {
            basicSli: {
              latency: {
                threshold: '300s' // 5 minutes
              }
            }
          },
          goal: 0.99, // 99% of security alerts processed within 5 minutes
          rollingPeriod: '30d'
        }
      ]
    };

    this.monitoringConfig.set('monitoring', monitoringConfig);
  }

  // Incident Response Configuration
  setupIncidentResponse() {
    const incidentConfig = {
      // Incident response playbooks
      playbooks: {
        'data_breach_response': {
          title: 'Data Breach Incident Response',
          description: 'Response procedures for potential data breaches',
          severity: 'CRITICAL',
          steps: [
            {
              step: 1,
              title: 'Immediate Containment',
              actions: [
                'Isolate affected systems',
                'Preserve evidence',
                'Assess scope of breach',
                'Notify incident commander'
              ],
              automation: {
                enabled: true,
                cloudFunction: 'projects/PROJECT_ID/locations/australia-southeast1/functions/containment-response'
              }
            },
            {
              step: 2,
              title: 'Assessment and Investigation',
              actions: [
                'Analyze logs and forensic evidence',
                'Determine cause and attack vector',
                'Assess impact and affected data',
                'Document findings'
              ],
              timeframe: '4 hours'
            },
            {
              step: 3,
              title: 'Notification and Reporting',
              actions: [
                'Notify APRA (within 72 hours if material)',
                'Notify affected customers',
                'Report to board of directors',
                'File regulatory notifications'
              ],
              timeframe: '72 hours'
            },
            {
              step: 4,
              title: 'Recovery and Remediation',
              actions: [
                'Implement security improvements',
                'Restore services securely',
                'Monitor for further incidents',
                'Update security controls'
              ]
            }
          ]
        },
        'unauthorized_access_response': {
          title: 'Unauthorized Access Response',
          description: 'Response to unauthorized system access',
          severity: 'HIGH',
          steps: [
            {
              step: 1,
              title: 'Account Security',
              actions: [
                'Disable compromised accounts',
                'Reset credentials',
                'Review access logs',
                'Check for privilege escalation'
              ],
              automation: {
                enabled: true,
                cloudFunction: 'projects/PROJECT_ID/locations/australia-southeast1/functions/account-lockdown'
              }
            },
            {
              step: 2,
              title: 'System Analysis',
              actions: [
                'Analyze authentication logs',
                'Check for lateral movement',
                'Review data access patterns',
                'Assess system integrity'
              ]
            }
          ]
        }
      },

      // Automated response actions
      automatedResponses: {
        'suspicious_login_blocking': {
          trigger: 'Multiple failed login attempts from same IP',
          action: 'Block IP address in Cloud Armor',
          automation: {
            cloudFunction: 'projects/PROJECT_ID/locations/australia-southeast1/functions/block-suspicious-ip',
            timeout: '30s'
          }
        },
        'malware_detection_isolation': {
          trigger: 'Malware detected on compute instance',
          action: 'Isolate instance and preserve for forensics',
          automation: {
            cloudFunction: 'projects/PROJECT_ID/locations/australia-southeast1/functions/isolate-infected-instance'
          }
        },
        'data_exfiltration_alert': {
          trigger: 'Unusual large data downloads',
          action: 'Alert security team and monitor user session',
          automation: {
            pubsubTopic: 'projects/PROJECT_ID/topics/security-incidents',
            slackWebhook: 'https://hooks.slack.com/services/...'
          }
        }
      },

      // Escalation procedures
      escalationMatrix: {
        'LOW': {
          timeToEscalate: '24h',
          escalateTo: 'Security Team Lead',
          notificationChannels: ['email', 'slack']
        },
        'MEDIUM': {
          timeToEscalate: '4h',
          escalateTo: 'CISO',
          notificationChannels: ['email', 'slack', 'sms']
        },
        'HIGH': {
          timeToEscalate: '1h',
          escalateTo: 'CISO + CTO',
          notificationChannels: ['email', 'slack', 'sms', 'phone']
        },
        'CRITICAL': {
          timeToEscalate: '15m',
          escalateTo: 'Executive Team',
          notificationChannels: ['pager', 'phone', 'sms', 'slack']
        }
      },

      // Communication templates
      communicationTemplates: {
        'internal_security_alert': {
          subject: '[SECURITY ALERT] {severity} - {incident_type}',
          body: `
            Security Alert: {incident_type}
            Severity: {severity}
            Detection Time: {detection_time}
            Affected Systems: {affected_systems}
            
            Immediate Actions Taken:
            {actions_taken}
            
            Next Steps:
            {next_steps}
            
            Incident Commander: {incident_commander}
          `
        },
        'customer_notification': {
          subject: 'Important Security Update - IPO Valuation Platform',
          body: `
            Dear Valued Customer,
            
            We are writing to inform you of a security incident that may have affected your account...
            
            [Details appropriate for customer communication]
          `
        }
      }
    };

    this.incidentResponse.set('procedures', incidentConfig);
  }

  // Compliance Reporting Configuration
  setupComplianceReporting() {
    const reportingConfig = {
      // Automated compliance reports
      complianceReports: {
        'soc2_monthly_report': {
          name: 'SOC 2 Monthly Compliance Report',
          schedule: '0 9 1 * *', // First day of month at 9 AM
          recipients: ['compliance@uprez.com', 'audit@uprez.com'],
          sections: [
            'Access Control Effectiveness',
            'Encryption Status',
            'Incident Summary',
            'Control Testing Results',
            'Exception Reports'
          ],
          dataSource: 'bigquery.PROJECT_ID.compliance_data.soc2_metrics'
        },
        'apra_quarterly_report': {
          name: 'APRA CPS 234 Quarterly Report',
          schedule: '0 9 1 */3 *', // First day of quarter
          recipients: ['risk@uprez.com', 'board@uprez.com'],
          sections: [
            'Information Security Capability Assessment',
            'Control Testing Summary',
            'Incident Response Metrics',
            'Third Party Risk Assessment',
            'Vulnerability Management Status'
          ]
        },
        'security_metrics_weekly': {
          name: 'Weekly Security Metrics Report',
          schedule: '0 9 * * 1', // Monday at 9 AM
          recipients: ['security@uprez.com', 'ciso@uprez.com'],
          sections: [
            'Security Event Summary',
            'Threat Detection Metrics',
            'Vulnerability Status',
            'Compliance Posture',
            'Key Performance Indicators'
          ]
        }
      },

      // Compliance metrics and KPIs
      complianceMetrics: {
        'control_effectiveness': {
          description: 'Percentage of security controls operating effectively',
          target: 95,
          calculation: 'effective_controls / total_controls * 100',
          frequency: 'daily'
        },
        'incident_response_time': {
          description: 'Average time to respond to security incidents',
          target: '15m',
          calculation: 'avg(response_time)',
          frequency: 'real-time'
        },
        'vulnerability_remediation_time': {
          description: 'Average time to remediate critical vulnerabilities',
          target: '7d',
          calculation: 'avg(remediation_time) WHERE severity=CRITICAL',
          frequency: 'weekly'
        },
        'compliance_score': {
          description: 'Overall compliance posture score',
          target: 90,
          calculation: 'weighted_average(framework_scores)',
          frequency: 'monthly'
        }
      },

      // Audit trail requirements
      auditTrails: {
        'admin_activities': {
          description: 'All administrative activities',
          retention: '7 years',
          immutableStorage: true,
          logSources: [
            'Cloud Audit Logs',
            'IAM Admin Activity',
            'KMS Admin Activity',
            'Security Command Center'
          ]
        },
        'data_access': {
          description: 'Access to sensitive financial data',
          retention: '7 years',
          immutableStorage: true,
          logSources: [
            'Cloud Storage Access Logs',
            'BigQuery Audit Logs',
            'Cloud SQL Audit Logs'
          ]
        },
        'security_events': {
          description: 'All security-related events',
          retention: '7 years',
          immutableStorage: true,
          logSources: [
            'Security Command Center',
            'Cloud Logging',
            'Identity-Aware Proxy Logs'
          ]
        }
      }
    };

    this.complianceReporting.set('configuration', reportingConfig);
  }

  // Automated Response System
  setupAutomatedResponse() {
    const automationConfig = {
      // SOAR (Security Orchestration, Automation, and Response) workflows
      soarWorkflows: {
        'phishing_email_response': {
          trigger: 'Phishing email detected in Gmail',
          workflow: [
            'Extract email metadata and attachments',
            'Analyze URLs and attachments for threats',
            'Block malicious URLs in web proxy',
            'Quarantine affected mailboxes',
            'Notify security team',
            'Generate incident ticket'
          ],
          automation: {
            platform: 'Cloud Functions',
            runtime: 'python39'
          }
        },
        'malware_incident_response': {
          trigger: 'Malware detected on endpoint',
          workflow: [
            'Isolate affected system from network',
            'Preserve system state for forensics',
            'Scan connected systems',
            'Update threat intelligence',
            'Notify incident response team',
            'Create investigation case'
          ]
        }
      },

      // Threat intelligence integration
      threatIntelligence: {
        feeds: [
          'Google Threat Intelligence',
          'Commercial threat feeds',
          'Industry-specific IoCs',
          'Government security advisories'
        ],
        automation: {
          enabled: true,
          updateFrequency: '1h',
          autoBlocking: {
            enabled: true,
            confidenceThreshold: 0.8
          }
        }
      },

      // Machine learning for anomaly detection
      mlAnomalyDetection: {
        models: {
          'user_behavior_analysis': {
            description: 'Detect anomalous user behavior patterns',
            features: [
              'Login times',
              'Data access patterns',
              'Geographic locations',
              'Device usage'
            ],
            training: {
              dataset: 'security_logs.user_activity',
              algorithm: 'isolation_forest',
              retrainFrequency: 'weekly'
            }
          },
          'network_traffic_analysis': {
            description: 'Detect anomalous network traffic',
            features: [
              'Traffic volume',
              'Connection patterns',
              'Protocol usage',
              'Geographic distribution'
            ]
          }
        }
      }
    };

    this.monitoringConfig.set('automation', automationConfig);
  }

  // Generate monitoring deployment script
  generateDeploymentScript() {
    return `#!/bin/bash
# Security Monitoring Deployment Script for IPO Valuation SaaS

set -e

PROJECT_ID="${PROJECT_ID}"
ORGANIZATION_ID="${ORGANIZATION_ID}"
REGION="australia-southeast1"

echo "Deploying security monitoring infrastructure..."

# 1. Enable required APIs
echo "Enabling required APIs..."
gcloud services enable securitycenter.googleapis.com
gcloud services enable logging.googleapis.com
gcloud services enable monitoring.googleapis.com
gcloud services enable cloudfunctions.googleapis.com
gcloud services enable pubsub.googleapis.com
gcloud services enable bigquery.googleapis.com

# 2. Create BigQuery dataset for security logs
echo "Creating BigQuery datasets..."
bq mk --dataset --location=$REGION $PROJECT_ID:security_logs
bq mk --dataset --location=$REGION $PROJECT_ID:compliance_data

# 3. Create Pub/Sub topics for security alerts
echo "Creating Pub/Sub topics..."
gcloud pubsub topics create security-alerts
gcloud pubsub topics create compliance-alerts
gcloud pubsub topics create incident-response
gcloud pubsub topics create siem-integration

# 4. Create Cloud Storage buckets for audit logs
echo "Creating encrypted storage buckets for audit logs..."
gsutil mb -l $REGION gs://audit-logs-bucket-encrypted
gsutil kms encryption -k projects/$PROJECT_ID/locations/$REGION/keyRings/audit-logs/cryptoKeys/audit-key gs://audit-logs-bucket-encrypted

# 5. Set up log sinks
echo "Creating log sinks..."

# Security events to BigQuery
gcloud logging sinks create security-events-bigquery-sink \\
    bigquery.googleapis.com/projects/$PROJECT_ID/datasets/security_logs \\
    --log-filter='protoPayload.authenticationInfo.principalEmail!="" OR severity>=ERROR OR protoPayload.serviceName="iap.googleapis.com"' \\
    --use-partitioned-tables

# Audit logs to Cloud Storage
gcloud logging sinks create audit-logs-storage-sink \\
    storage.googleapis.com/audit-logs-bucket-encrypted \\
    --log-filter='logName:"cloudaudit.googleapis.com"'

# 6. Create log-based metrics
echo "Creating log-based metrics..."
gcloud logging metrics create failed_authentication_rate \\
    --description="Rate of failed authentication attempts" \\
    --log-filter='protoPayload.authenticationInfo.principalEmail="" AND protoPayload.authorizationInfo.granted=false'

# 7. Set up Security Command Center notifications
echo "Setting up Security Command Center..."
# Note: This typically requires manual configuration in the console

# 8. Create monitoring alerts
echo "Creating monitoring alert policies..."
cat > critical-security-alert.yaml << EOF
displayName: "Critical Security Finding"
conditions:
  - displayName: "Critical SCC Finding"
    conditionThreshold:
      filter: 'resource.type="global" AND metric.type="securitycenter.googleapis.com/finding/count"'
      comparison: COMPARISON_GREATER_THAN
      thresholdValue: 0
      duration: 60s
notificationChannels:
  - projects/$PROJECT_ID/notificationChannels/security-team-email
EOF

gcloud alpha monitoring policies create --policy-from-file=critical-security-alert.yaml

# 9. Deploy automated response functions
echo "Deploying Cloud Functions for automated response..."

# Create source code for containment response
mkdir -p functions/containment-response
cat > functions/containment-response/main.py << 'EOF'
import functions_framework
from google.cloud import compute_v1
from google.cloud import logging

@functions_framework.cloud_event
def containment_response(cloud_event):
    """Automated containment response for security incidents."""
    
    client = logging.Client()
    logger = client.logger("security-response")
    
    # Log the incident
    logger.log_text(f"Security incident triggered: {cloud_event.data}")
    
    # Implement containment logic here
    # This is a simplified example
    
    return "Containment response executed"
EOF

cat > functions/containment-response/requirements.txt << 'EOF'
google-cloud-compute==1.14.1
google-cloud-logging==3.8.0
functions-framework==3.*
EOF

# Deploy the function
cd functions/containment-response
gcloud functions deploy containment-response \\
    --runtime python39 \\
    --trigger-topic security-alerts \\
    --region $REGION \\
    --memory 256MB \\
    --timeout 300s

cd ../..

# 10. Set up monitoring dashboards
echo "Creating monitoring dashboards..."
# This would typically use the Monitoring API or be done through the console

echo "Security monitoring infrastructure deployment complete!"
echo ""
echo "Next steps:"
echo "1. Configure Security Command Center sources in the console"
echo "2. Set up notification channels for alerts"
echo "3. Customize alert policies for your specific requirements"
echo "4. Configure SIEM integration if required"
echo "5. Test incident response procedures"
`;
  }

  // Get monitoring system summary
  getMonitoringSummary() {
    return {
      coreSystems: [
        'Security Command Center for unified security management',
        'Cloud Logging with 7-year retention for compliance',
        'Cloud Monitoring with custom security metrics',
        'Automated incident response workflows',
        'Compliance reporting and audit trails'
      ],
      securityEventSources: [
        'Identity and Access Management logs',
        'Cloud Key Management Service access',
        'Data access patterns in Cloud Storage',
        'Application security events',
        'Network security monitoring'
      ],
      incidentResponseCapabilities: [
        'Automated threat detection and containment',
        'Playbook-driven incident response',
        'Real-time security team notifications',
        'Forensic data preservation',
        'Regulatory notification workflows'
      ],
      complianceFeatures: [
        'SOC 2 Type II reporting automation',
        'APRA CPS 234 compliance tracking',
        'ISO 27001 control monitoring',
        'Australian Privacy Principles audit trails',
        'Immutable audit log storage'
      ]
    };
  }
}

module.exports = { SecurityMonitoringSystem };