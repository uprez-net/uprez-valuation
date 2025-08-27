/**
 * GCP Security Architecture for IPO Valuation SaaS
 * Comprehensive security framework with financial services compliance
 */

class GCPSecurityArchitecture {
  constructor() {
    this.gcpServices = new Map();
    this.securityControls = new Map();
    this.complianceFrameworks = new Set(['SOC2', 'ISO27001', 'APP', 'CPS234', 'PCI_DSS']);
    this.initializeArchitecture();
  }

  initializeArchitecture() {
    this.setupCoreSecurityServices();
    this.setupNetworkSecurity();
    this.setupDataProtection();
    this.setupIdentityManagement();
    this.setupMonitoring();
  }

  // Core GCP Security Services Configuration
  setupCoreSecurityServices() {
    const securityServices = {
      // Security Command Center - Unified security management
      securityCommandCenter: {
        service: 'securitycenter.googleapis.com',
        configuration: {
          organizationId: process.env.GCP_ORGANIZATION_ID,
          findings: {
            autoRemediation: true,
            notificationChannels: ['email', 'slack', 'pagerduty'],
            severityThresholds: {
              critical: 'immediate',
              high: '15_minutes',
              medium: '1_hour',
              low: '24_hours'
            }
          },
          assetDiscovery: {
            enabled: true,
            scanInterval: '1_hour',
            includedProjects: 'all',
            assetTypes: [
              'compute.googleapis.com/Instance',
              'storage.googleapis.com/Bucket',
              'cloudsql.googleapis.com/DatabaseInstance',
              'container.googleapis.com/Cluster'
            ]
          }
        }
      },

      // Cloud Key Management Service
      keyManagementService: {
        service: 'cloudkms.googleapis.com',
        configuration: {
          keyRings: {
            'ipo-valuation-keys': {
              location: 'australia-southeast1',
              keys: {
                'database-encryption-key': {
                  purpose: 'ENCRYPT_DECRYPT',
                  algorithm: 'GOOGLE_SYMMETRIC_ENCRYPTION',
                  rotationPeriod: '2592000s', // 30 days
                  nextRotationTime: 'auto'
                },
                'application-secrets-key': {
                  purpose: 'ENCRYPT_DECRYPT',
                  algorithm: 'GOOGLE_SYMMETRIC_ENCRYPTION',
                  rotationPeriod: '2592000s'
                },
                'backup-encryption-key': {
                  purpose: 'ENCRYPT_DECRYPT',
                  algorithm: 'GOOGLE_SYMMETRIC_ENCRYPTION',
                  rotationPeriod: '5184000s' // 60 days
                }
              }
            }
          },
          hsm: {
            enabled: true,
            protectionLevel: 'HSM',
            attestation: true
          }
        }
      },

      // Data Loss Prevention API
      dataLossPrevention: {
        service: 'dlp.googleapis.com',
        configuration: {
          inspectTemplates: {
            'financial-data-inspector': {
              displayName: 'Financial Data Inspector',
              description: 'Detects financial and PII data in IPO documents',
              inspectConfig: {
                infoTypes: [
                  { name: 'CREDIT_CARD_NUMBER' },
                  { name: 'BANK_ACCOUNT_NUMBER' },
                  { name: 'AU_TAX_FILE_NUMBER' },
                  { name: 'AU_MEDICARE_NUMBER' },
                  { name: 'EMAIL_ADDRESS' },
                  { name: 'PHONE_NUMBER' },
                  { name: 'PERSON_NAME' }
                ],
                customInfoTypes: [
                  {
                    infoType: { name: 'IPO_DOCUMENT_ID' },
                    regex: { pattern: 'IPO-\\d{6}-\\d{4}' }
                  },
                  {
                    infoType: { name: 'VALUATION_AMOUNT' },
                    regex: { pattern: '\\$[0-9,]+(\\.[0-9]{2})?' }
                  }
                ],
                minLikelihood: 'POSSIBLE',
                limits: {
                  maxFindingsPerInfoType: 100,
                  maxFindingsPerRequest: 1000
                }
              }
            }
          },
          deidentifyTemplates: {
            'financial-data-deidentifier': {
              displayName: 'Financial Data De-identifier',
              deidentifyConfig: {
                infoTypeTransformations: {
                  transformations: [
                    {
                      infoTypes: [{ name: 'CREDIT_CARD_NUMBER' }],
                      primitiveTransformation: {
                        cryptoHashConfig: {
                          cryptoKey: {
                            kmsWrapped: {
                              wrappedKey: 'encrypted-key-data',
                              cryptoKeyName: 'projects/PROJECT_ID/locations/australia-southeast1/keyRings/ipo-valuation-keys/cryptoKeys/dlp-key'
                            }
                          }
                        }
                      }
                    }
                  ]
                }
              }
            }
          }
        }
      },

      // Binary Authorization for container security
      binaryAuthorization: {
        service: 'binaryauthorization.googleapis.com',
        configuration: {
          defaultAdmissionRule: {
            requireAttestationsBy: [
              'projects/PROJECT_ID/attestors/security-attestor',
              'projects/PROJECT_ID/attestors/quality-attestor'
            ],
            enforcementMode: 'ENFORCED_BLOCK_AND_AUDIT_LOG'
          },
          clusterAdmissionRules: {
            'australia-southeast1-a.ipo-valuation-cluster': {
              requireAttestationsBy: [
                'projects/PROJECT_ID/attestors/security-attestor'
              ],
              enforcementMode: 'ENFORCED_BLOCK_AND_AUDIT_LOG'
            }
          },
          attestors: {
            'security-attestor': {
              description: 'Security vulnerability scanning attestor',
              userOwnedGrafeasNote: {
                noteReference: 'projects/PROJECT_ID/notes/security-note',
                publicKeys: [
                  {
                    asciiArmoredPgpPublicKey: 'PGP_PUBLIC_KEY_DATA',
                    comment: 'Security team key'
                  }
                ]
              }
            }
          }
        }
      }
    };

    this.gcpServices.set('security', securityServices);
  }

  // Network Security Configuration
  setupNetworkSecurity() {
    const networkSecurity = {
      // VPC Security Controls
      vpcSecurityControls: {
        accessPolicies: {
          'ipo-valuation-policy': {
            title: 'IPO Valuation Access Policy',
            scopes: ['projects/PROJECT_ID'],
            accessLevels: {
              'trusted-networks': {
                title: 'Trusted Networks',
                basic: {
                  conditions: [
                    {
                      ipSubnetworks: [
                        '10.0.0.0/8', // Internal VPC
                        '203.0.113.0/24' // Office network
                      ]
                    },
                    {
                      devicePolicy: {
                        requireScreenLock: true,
                        requireAdminApproval: true,
                        allowedDeviceManagementLevels: ['COMPLETE']
                      }
                    }
                  ]
                }
              },
              'high-security': {
                title: 'High Security Access',
                basic: {
                  conditions: [
                    {
                      members: [
                        'user:admin@uprez.com',
                        'group:security-team@uprez.com'
                      ]
                    },
                    {
                      devicePolicy: {
                        requireCorpOwned: true,
                        requireScreenLock: true
                      }
                    }
                  ],
                  combiningFunction: 'AND'
                }
              }
            },
            servicePerimeters: {
              'ipo-valuation-perimeter': {
                title: 'IPO Valuation Service Perimeter',
                perimeterType: 'PERIMETER_TYPE_REGULAR',
                resources: [
                  'projects/ipo-valuation-prod',
                  'projects/ipo-valuation-staging'
                ],
                restrictedServices: [
                  'storage.googleapis.com',
                  'cloudsql.googleapis.com',
                  'secretmanager.googleapis.com'
                ],
                accessLevels: ['trusted-networks'],
                vpcAccessibleServices: {
                  enableRestriction: true,
                  allowedServices: [
                    'storage.googleapis.com',
                    'cloudsql.googleapis.com'
                  ]
                }
              }
            }
          }
        }
      },

      // Firewall Rules
      firewallRules: {
        'allow-https-internal': {
          direction: 'INGRESS',
          priority: 1000,
          sourceRanges: ['10.0.0.0/8'],
          allowed: [
            { IPProtocol: 'tcp', ports: ['443'] }
          ],
          targetTags: ['https-server']
        },
        'allow-ssh-bastion': {
          direction: 'INGRESS',
          priority: 1000,
          sourceRanges: ['203.0.113.0/24'], // Office IP
          allowed: [
            { IPProtocol: 'tcp', ports: ['22'] }
          ],
          targetTags: ['bastion-host']
        },
        'deny-all-external': {
          direction: 'INGRESS',
          priority: 65534,
          sourceRanges: ['0.0.0.0/0'],
          denied: [
            { IPProtocol: 'all' }
          ],
          targetTags: ['no-external-access']
        }
      },

      // Cloud Armor Security Policies
      cloudArmor: {
        securityPolicies: {
          'ipo-valuation-security-policy': {
            description: 'Security policy for IPO valuation application',
            rules: [
              {
                priority: 1000,
                description: 'Rate limiting rule',
                match: {
                  versionedExpr: 'SRC_IPS_V1',
                  config: {
                    srcIpRanges: ['*']
                  }
                },
                action: 'rate_based_ban',
                rateLimitOptions: {
                  rateLimitThreshold: {
                    count: 100,
                    intervalSec: 60
                  },
                  banThreshold: {
                    count: 1000,
                    intervalSec: 600
                  },
                  banDurationSec: 3600
                }
              },
              {
                priority: 2000,
                description: 'Block malicious IPs',
                match: {
                  versionedExpr: 'SRC_IPS_V1',
                  config: {
                    srcIpRanges: ['0.0.0.0/0']
                  }
                },
                action: 'deny(403)',
                preview: false
              }
            ]
          }
        }
      }
    };

    this.gcpServices.set('network', networkSecurity);
  }

  // Data Protection Configuration
  setupDataProtection() {
    const dataProtection = {
      // Encryption Configuration
      encryption: {
        atRest: {
          cloudSQL: {
            encryptionType: 'CUSTOMER_MANAGED_ENCRYPTION_KEY',
            kmsKeyName: 'projects/PROJECT_ID/locations/australia-southeast1/keyRings/ipo-valuation-keys/cryptoKeys/database-encryption-key'
          },
          cloudStorage: {
            defaultKmsKeyName: 'projects/PROJECT_ID/locations/australia-southeast1/keyRings/ipo-valuation-keys/cryptoKeys/storage-encryption-key',
            bucketEncryption: {
              'ipo-documents-bucket': {
                kmsKeyName: 'projects/PROJECT_ID/locations/australia-southeast1/keyRings/ipo-valuation-keys/cryptoKeys/document-encryption-key'
              }
            }
          },
          computeEngine: {
            diskEncryption: {
              type: 'CUSTOMER_MANAGED_ENCRYPTION_KEY',
              kmsKeyName: 'projects/PROJECT_ID/locations/australia-southeast1/keyRings/ipo-valuation-keys/cryptoKeys/disk-encryption-key'
            }
          }
        },
        inTransit: {
          tlsVersion: '1.3',
          cipherSuites: [
            'TLS_AES_256_GCM_SHA384',
            'TLS_CHACHA20_POLY1305_SHA256',
            'TLS_AES_128_GCM_SHA256'
          ],
          certificateManagement: {
            provider: 'GOOGLE_MANAGED_CERTS',
            autoRenewal: true,
            domains: [
              'api.uprez.com',
              'app.uprez.com',
              'admin.uprez.com'
            ]
          }
        }
      },

      // Confidential Computing
      confidentialComputing: {
        enabled: true,
        confidentialInstanceConfig: {
          enableConfidentialCompute: true,
          confidentialInstanceType: 'SEV'
        },
        gkeConfidentialNodes: {
          enabled: true,
          nodeConfig: {
            confidentialNodes: {
              enabled: true
            }
          }
        }
      },

      // Backup and Recovery
      backupRecovery: {
        cloudSQL: {
          backupConfiguration: {
            enabled: true,
            startTime: '02:00',
            location: 'australia-southeast1',
            pointInTimeRecoveryEnabled: true,
            transactionLogRetentionDays: 7,
            backupRetentionSettings: {
              retainedBackups: 30,
              retentionUnit: 'COUNT'
            }
          }
        },
        cloudStorage: {
          lifecycleRules: [
            {
              action: { type: 'SetStorageClass', storageClass: 'NEARLINE' },
              condition: { age: 30 }
            },
            {
              action: { type: 'SetStorageClass', storageClass: 'COLDLINE' },
              condition: { age: 365 }
            },
            {
              action: { type: 'Delete' },
              condition: { age: 2555 } // 7 years for financial records
            }
          ],
          versioning: {
            enabled: true
          }
        }
      }
    };

    this.gcpServices.set('dataProtection', dataProtection);
  }

  // Identity and Access Management
  setupIdentityManagement() {
    const identityManagement = {
      // IAM Policies and Roles
      iamPolicies: {
        customRoles: {
          'ipo.valuationAnalyst': {
            title: 'IPO Valuation Analyst',
            description: 'Can view and analyze IPO valuation data',
            stage: 'GA',
            includedPermissions: [
              'cloudsql.instances.connect',
              'storage.objects.get',
              'storage.objects.list',
              'logging.entries.list',
              'monitoring.timeSeries.list'
            ]
          },
          'ipo.dataManager': {
            title: 'IPO Data Manager',
            description: 'Can manage IPO data and documents',
            stage: 'GA',
            includedPermissions: [
              'cloudsql.instances.connect',
              'storage.objects.create',
              'storage.objects.delete',
              'storage.objects.get',
              'storage.objects.list',
              'storage.objects.update'
            ]
          },
          'ipo.securityAuditor': {
            title: 'IPO Security Auditor',
            description: 'Can audit security controls and access logs',
            stage: 'GA',
            includedPermissions: [
              'logging.entries.list',
              'logging.logEntries.list',
              'monitoring.timeSeries.list',
              'securitycenter.findings.list',
              'securitycenter.sources.list'
            ]
          }
        },
        bindings: {
          'projects/PROJECT_ID': [
            {
              role: 'projects/PROJECT_ID/roles/ipo.valuationAnalyst',
              members: [
                'group:valuation-analysts@uprez.com'
              ],
              condition: {
                title: 'Business hours only',
                description: 'Access limited to business hours',
                expression: 'request.time.getHours() >= 9 && request.time.getHours() <= 17'
              }
            },
            {
              role: 'projects/PROJECT_ID/roles/ipo.dataManager',
              members: [
                'group:data-managers@uprez.com'
              ]
            }
          ]
        }
      },

      // Identity-Aware Proxy
      identityAwareProxy: {
        enabled: true,
        oauthBrand: {
          applicationTitle: 'IPO Valuation Platform',
          supportEmail: 'support@uprez.com',
          privacyPolicyUrl: 'https://uprez.com/privacy',
          termsOfServiceUrl: 'https://uprez.com/terms'
        },
        oauthClients: {
          'ipo-valuation-web': {
            displayName: 'IPO Valuation Web Application',
            origins: [
              'https://app.uprez.com',
              'https://admin.uprez.com'
            ]
          }
        },
        accessPolicies: {
          'ipo-app-access': {
            name: 'IPO Application Access',
            members: [
              'group:ipo-users@uprez.com'
            ],
            accessLevels: ['trusted-networks']
          }
        }
      },

      // Multi-factor Authentication
      multiFactorAuth: {
        enforced: true,
        methods: [
          'TOTP',
          'SMS',
          'SECURITY_KEY'
        ],
        gracePerod: 0, // No grace period for financial data
        exemptions: [] // No exemptions
      }
    };

    this.gcpServices.set('identity', identityManagement);
  }

  // Monitoring and Logging Configuration
  setupMonitoring() {
    const monitoring = {
      // Cloud Logging Configuration
      logging: {
        logSinks: [
          {
            name: 'security-events-sink',
            destination: 'storage.googleapis.com/security-logs-bucket',
            filter: 'protoPayload.authenticationInfo.principalEmail!="" OR severity>=ERROR',
            uniqueWriterIdentity: true
          },
          {
            name: 'audit-logs-sink',
            destination: 'bigquery.googleapis.com/projects/PROJECT_ID/datasets/audit_logs',
            filter: 'logName:"cloudaudit.googleapis.com"',
            uniqueWriterIdentity: true
          }
        ],
        exclusions: [
          {
            name: 'exclude-gke-noise',
            description: 'Exclude noisy GKE container logs',
            filter: 'resource.type="k8s_container" AND severity<WARNING'
          }
        ],
        retentionPolicies: {
          'security-events': {
            retentionDays: 2555 // 7 years for compliance
          },
          'audit-logs': {
            retentionDays: 2555
          },
          'application-logs': {
            retentionDays: 90
          }
        }
      },

      // Cloud Monitoring Configuration
      monitoring: {
        alertingPolicies: [
          {
            displayName: 'Unauthorized Access Attempt',
            conditions: [
              {
                displayName: 'Failed authentication rate',
                conditionThreshold: {
                  filter: 'resource.type="gce_instance" AND metric.type="logging.googleapis.com/user/failed_auth"',
                  comparison: 'COMPARISON_GREATER_THAN',
                  thresholdValue: 5,
                  duration: '300s'
                }
              }
            ],
            notificationChannels: [
              'projects/PROJECT_ID/notificationChannels/CHANNEL_ID'
            ],
            alertStrategy: {
              autoClose: '86400s'
            }
          },
          {
            displayName: 'Data Exfiltration Alert',
            conditions: [
              {
                displayName: 'Unusual data egress',
                conditionThreshold: {
                  filter: 'resource.type="gcs_bucket" AND metric.type="storage.googleapis.com/network/sent_bytes_count"',
                  comparison: 'COMPARISON_GREATER_THAN',
                  thresholdValue: 1073741824, // 1GB
                  duration: '600s'
                }
              }
            ]
          }
        ],
        dashboards: {
          'security-dashboard': {
            displayName: 'Security Monitoring Dashboard',
            widgets: [
              {
                title: 'Authentication Events',
                xyChart: {
                  dataSets: [
                    {
                      timeSeriesQuery: {
                        timeSeriesFilter: {
                          filter: 'resource.type="gce_instance" AND metric.type="logging.googleapis.com/user/authentication"'
                        }
                      }
                    }
                  ]
                }
              },
              {
                title: 'Security Command Center Findings',
                scorecard: {
                  timeSeriesQuery: {
                    timeSeriesFilter: {
                      filter: 'resource.type="global" AND metric.type="securitycenter.googleapis.com/finding/count"'
                    }
                  }
                }
              }
            ]
          }
        }
      },

      // Error Reporting
      errorReporting: {
        enabled: true,
        notificationChannels: [
          'projects/PROJECT_ID/notificationChannels/error-alerts'
        ]
      }
    };

    this.gcpServices.set('monitoring', monitoring);
  }

  // Generate Terraform configuration
  generateTerraformConfig() {
    const config = {
      terraform: {
        required_providers: {
          google: {
            source: 'hashicorp/google',
            version: '~> 4.0'
          },
          google-beta: {
            source: 'hashicorp/google-beta',
            version: '~> 4.0'
          }
        }
      },
      provider: {
        google: {
          project: 'var.project_id',
          region: 'australia-southeast1',
          zone: 'australia-southeast1-a'
        }
      },
      variables: {
        project_id: {
          description: 'GCP Project ID',
          type: 'string'
        },
        organization_id: {
          description: 'GCP Organization ID',
          type: 'string'
        }
      }
    };

    return JSON.stringify(config, null, 2);
  }

  // Get security architecture summary
  getArchitectureSummary() {
    return {
      services: Array.from(this.gcpServices.keys()),
      complianceFrameworks: Array.from(this.complianceFrameworks),
      securityLayers: [
        'Network Security (VPC, Cloud Armor, Firewall)',
        'Identity & Access (IAM, IAP, MFA)',
        'Data Protection (KMS, DLP, Encryption)',
        'Application Security (Binary Auth, Confidential Computing)',
        'Monitoring & Response (Security Command Center, Cloud Logging)'
      ],
      keyFeatures: [
        'Zero Trust Architecture',
        'End-to-End Encryption',
        'Continuous Compliance Monitoring',
        'Automated Threat Detection',
        'Incident Response Automation'
      ]
    };
  }
}

module.exports = { GCPSecurityArchitecture };