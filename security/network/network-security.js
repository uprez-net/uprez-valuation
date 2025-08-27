/**
 * Network Security Implementation for IPO Valuation SaaS
 * Comprehensive network security controls and monitoring
 */

class NetworkSecurityImplementation {
  constructor() {
    this.networkConfig = new Map();
    this.firewallRules = new Map();
    this.cloudArmorPolicies = new Map();
    this.vpcConfig = new Map();
    this.ddosProtection = new Map();
    this.initializeNetworkSecurity();
  }

  initializeNetworkSecurity() {
    this.setupVPCArchitecture();
    this.setupFirewallRules();
    this.setupCloudArmor();
    this.setupDDoSProtection();
    this.setupNetworkMonitoring();
    this.setupPrivateConnectivity();
  }

  // VPC Network Architecture
  setupVPCArchitecture() {
    const vpcConfig = {
      // Primary production VPC
      productionVPC: {
        name: 'ipo-valuation-prod-vpc',
        autoCreateSubnetworks: false,
        description: 'Production VPC for IPO Valuation SaaS',
        routingMode: 'REGIONAL',
        subnets: {
          'application-tier': {
            name: 'ipo-app-subnet',
            region: 'australia-southeast1',
            ipCidrRange: '10.1.0.0/24',
            description: 'Application tier subnet',
            privateIpGoogleAccess: true,
            enableFlowLogs: true,
            flowLogsConfig: {
              aggregationInterval: 'INTERVAL_5_SEC',
              flowSampling: 0.5,
              metadata: 'INCLUDE_ALL_METADATA',
              metadataFields: [
                'src_instance',
                'dest_instance',
                'project_id',
                'vpc_name',
                'subnet_name'
              ]
            },
            secondaryIpRanges: [
              {
                rangeName: 'pod-range',
                ipCidrRange: '10.11.0.0/16' // For GKE pods
              },
              {
                rangeName: 'service-range',
                ipCidrRange: '10.12.0.0/16' // For GKE services
              }
            ]
          },
          'database-tier': {
            name: 'ipo-db-subnet',
            region: 'australia-southeast1',
            ipCidrRange: '10.1.1.0/24',
            description: 'Database tier subnet (private)',
            privateIpGoogleAccess: true,
            enableFlowLogs: true,
            purpose: 'PRIVATE_SERVICE_CONNECT'
          },
          'management-tier': {
            name: 'ipo-mgmt-subnet',
            region: 'australia-southeast1',
            ipCidrRange: '10.1.2.0/24',
            description: 'Management and monitoring subnet',
            privateIpGoogleAccess: true,
            enableFlowLogs: true
          }
        },
        routes: [
          {
            name: 'default-internet-route',
            destRange: '0.0.0.0/0',
            priority: 1000,
            nextHopGateway: 'default-internet-gateway',
            tags: ['allow-internet']
          },
          {
            name: 'private-google-access-route',
            destRange: '199.36.153.8/30',
            priority: 1000,
            nextHopGateway: 'default-internet-gateway'
          }
        ]
      },

      // Staging VPC (isolated from production)
      stagingVPC: {
        name: 'ipo-valuation-staging-vpc',
        autoCreateSubnetworks: false,
        description: 'Staging environment VPC',
        routingMode: 'REGIONAL',
        subnets: {
          'staging-subnet': {
            name: 'ipo-staging-subnet',
            region: 'australia-southeast1',
            ipCidrRange: '10.2.0.0/24',
            privateIpGoogleAccess: true,
            enableFlowLogs: true
          }
        }
      },

      // Management VPC (for bastion hosts and monitoring)
      managementVPC: {
        name: 'ipo-management-vpc',
        autoCreateSubnetworks: false,
        description: 'Management VPC for bastion and monitoring',
        routingMode: 'REGIONAL',
        subnets: {
          'bastion-subnet': {
            name: 'ipo-bastion-subnet',
            region: 'australia-southeast1',
            ipCidrRange: '10.3.0.0/28', // Small subnet for bastion hosts
            privateIpGoogleAccess: false, // Bastion needs external access
            enableFlowLogs: true
          }
        }
      },

      // VPC Peering connections
      vpcPeering: {
        'prod-to-mgmt': {
          name: 'prod-to-mgmt-peering',
          network: 'ipo-valuation-prod-vpc',
          peerNetwork: 'ipo-management-vpc',
          autoCreateRoutes: true,
          importCustomRoutes: false,
          exportCustomRoutes: false
        }
      },

      // Cloud NAT for outbound internet access
      cloudNAT: {
        'prod-nat': {
          name: 'ipo-prod-nat',
          region: 'australia-southeast1',
          router: 'ipo-prod-router',
          natIpAllocateOption: 'MANUAL_ONLY',
          natIps: ['ipo-prod-nat-ip-1', 'ipo-prod-nat-ip-2'], // Static IPs for whitelisting
          sourceSubnetworkIpRangesToNat: 'LIST_OF_SUBNETWORKS',
          subnetworks: [
            {
              name: 'ipo-app-subnet',
              sourceIpRangesToNat: 'PRIMARY_IP_RANGE'
            }
          ],
          logConfig: {
            enable: true,
            filter: 'ALL'
          }
        }
      }
    };

    this.vpcConfig.set('architecture', vpcConfig);
  }

  // Comprehensive Firewall Rules
  setupFirewallRules() {
    const firewallConfig = {
      // Ingress rules (incoming traffic)
      ingressRules: [
        {
          name: 'allow-health-checks',
          description: 'Allow Google Cloud health checks',
          direction: 'INGRESS',
          priority: 1000,
          sourceRanges: [
            '130.211.0.0/22',  // Google Cloud Load Balancer health checks
            '35.191.0.0/16',   // Google Cloud Load Balancer health checks
            '209.85.152.0/22', // Google services
            '209.85.204.0/22'  // Google services
          ],
          allowed: [
            { IPProtocol: 'tcp', ports: ['80', '443', '8080'] }
          ],
          targetTags: ['http-server', 'https-server'],
          logConfig: {
            enable: true,
            metadata: 'INCLUDE_ALL_METADATA'
          }
        },
        {
          name: 'allow-load-balancer-ingress',
          description: 'Allow traffic from load balancer to application servers',
          direction: 'INGRESS',
          priority: 1000,
          sourceRanges: ['10.1.0.0/24'], // Application subnet
          allowed: [
            { IPProtocol: 'tcp', ports: ['80', '443', '8080'] }
          ],
          targetTags: ['app-server'],
          logConfig: {
            enable: true
          }
        },
        {
          name: 'allow-ssh-from-bastion',
          description: 'Allow SSH from bastion hosts only',
          direction: 'INGRESS',
          priority: 1000,
          sourceRanges: ['10.3.0.0/28'], // Bastion subnet
          allowed: [
            { IPProtocol: 'tcp', ports: ['22'] }
          ],
          targetTags: ['ssh-access'],
          logConfig: {
            enable: true
          }
        },
        {
          name: 'allow-database-access-internal',
          description: 'Allow database access from application tier only',
          direction: 'INGRESS',
          priority: 1000,
          sourceRanges: ['10.1.0.0/24'], // Application subnet
          allowed: [
            { IPProtocol: 'tcp', ports: ['5432', '3306'] } // PostgreSQL, MySQL
          ],
          targetTags: ['database-server'],
          logConfig: {
            enable: true
          }
        },
        {
          name: 'allow-monitoring-internal',
          description: 'Allow monitoring traffic between internal subnets',
          direction: 'INGRESS',
          priority: 1100,
          sourceRanges: ['10.1.0.0/16'], // All production subnets
          allowed: [
            { IPProtocol: 'tcp', ports: ['9090', '9100', '3000'] } // Prometheus, Grafana
          ],
          targetTags: ['monitoring'],
          logConfig: {
            enable: true
          }
        }
      ],

      // Egress rules (outgoing traffic)
      egressRules: [
        {
          name: 'deny-all-egress-default',
          description: 'Default deny all egress traffic',
          direction: 'EGRESS',
          priority: 65534,
          destinationRanges: ['0.0.0.0/0'],
          denied: [
            { IPProtocol: 'all' }
          ],
          targetTags: ['no-external-access'],
          logConfig: {
            enable: true
          }
        },
        {
          name: 'allow-google-apis-egress',
          description: 'Allow access to Google APIs',
          direction: 'EGRESS',
          priority: 1000,
          destinationRanges: [
            '199.36.153.8/30', // Google APIs
            '199.36.153.4/30'  // Google APIs
          ],
          allowed: [
            { IPProtocol: 'tcp', ports: ['443'] }
          ],
          targetTags: ['google-api-access']
        },
        {
          name: 'allow-external-https-egress',
          description: 'Allow HTTPS to external services (controlled)',
          direction: 'EGRESS',
          priority: 1000,
          destinationRanges: ['0.0.0.0/0'],
          allowed: [
            { IPProtocol: 'tcp', ports: ['443'] }
          ],
          targetTags: ['external-https-access'],
          logConfig: {
            enable: true
          }
        },
        {
          name: 'allow-internal-egress',
          description: 'Allow all traffic between internal subnets',
          direction: 'EGRESS',
          priority: 1000,
          destinationRanges: ['10.0.0.0/8'],
          allowed: [
            { IPProtocol: 'all' }
          ],
          targetTags: ['internal-communication']
        },
        {
          name: 'allow-dns-egress',
          description: 'Allow DNS queries',
          direction: 'EGRESS',
          priority: 1000,
          destinationRanges: ['0.0.0.0/0'],
          allowed: [
            { IPProtocol: 'tcp', ports: ['53'] },
            { IPProtocol: 'udp', ports: ['53'] }
          ],
          targetTags: ['dns-access']
        }
      ],

      // Security groups (represented as network tags)
      securityGroups: {
        'web-tier': ['http-server', 'https-server', 'google-api-access', 'dns-access'],
        'app-tier': ['app-server', 'internal-communication', 'google-api-access', 'dns-access'],
        'db-tier': ['database-server', 'internal-communication', 'google-api-access'],
        'bastion-tier': ['ssh-access', 'external-https-access', 'dns-access'],
        'monitoring-tier': ['monitoring', 'internal-communication', 'google-api-access']
      }
    };

    this.firewallRules.set('configuration', firewallConfig);
  }

  // Cloud Armor Web Application Firewall
  setupCloudArmor() {
    const armorConfig = {
      // Main security policy for IPO application
      securityPolicies: {
        'ipo-valuation-security-policy': {
          description: 'Security policy for IPO valuation web application',
          rules: [
            {
              priority: 1000,
              description: 'Rate limiting - requests per minute',
              match: {
                versionedExpr: 'SRC_IPS_V1',
                config: {
                  srcIpRanges: ['*']
                }
              },
              action: 'rate_based_ban',
              rateLimitOptions: {
                conformAction: 'allow',
                exceedAction: 'deny(429)',
                enforceOnKey: 'IP',
                rateLimitThreshold: {
                  count: 100, // 100 requests
                  intervalSec: 60 // per minute
                },
                banThreshold: {
                  count: 500, // Ban after 500 requests
                  intervalSec: 300 // in 5 minutes
                },
                banDurationSec: 1800 // Ban for 30 minutes
              }
            },
            {
              priority: 2000,
              description: 'Block known malicious IP ranges',
              match: {
                versionedExpr: 'SRC_IPS_V1',
                config: {
                  srcIpRanges: [
                    // These would be populated from threat intelligence feeds
                    '192.0.2.0/24', // Example malicious range
                    '198.51.100.0/24' // Example malicious range
                  ]
                }
              },
              action: 'deny(403)',
              preview: false
            },
            {
              priority: 3000,
              description: 'Block common attack patterns',
              match: {
                expr: {
                  expression: `
                    request.url_map =~ ".*\\.(php|asp|jsp)$" || 
                    request.url_map =~ ".*(union|select|insert|delete).*" ||
                    request.url_map =~ ".*(<script|javascript:|onerror=).*" ||
                    request.headers['user-agent'] =~ ".*(nikto|sqlmap|nmap|masscan).*"
                  `
                }
              },
              action: 'deny(403)',
              preview: false
            },
            {
              priority: 4000,
              description: 'Geographic restriction - Australia and NZ only',
              match: {
                expr: {
                  expression: 'origin.region_code != "AU" && origin.region_code != "NZ"'
                }
              },
              action: 'deny(403)',
              preview: false
            },
            {
              priority: 5000,
              description: 'Allow trusted office networks',
              match: {
                versionedExpr: 'SRC_IPS_V1',
                config: {
                  srcIpRanges: [
                    '203.0.113.0/24' // Office network range
                  ]
                }
              },
              action: 'allow',
              preview: false
            },
            {
              priority: 10000,
              description: 'OWASP Top 10 protection',
              match: {
                expr: {
                  expression: `
                    evaluatePreconfiguredExpr('sqli-stable') ||
                    evaluatePreconfiguredExpr('xss-stable') ||
                    evaluatePreconfiguredExpr('lfi-stable') ||
                    evaluatePreconfiguredExpr('rfi-stable') ||
                    evaluatePreconfiguredExpr('rce-stable') ||
                    evaluatePreconfiguredExpr('methodenforcement-stable') ||
                    evaluatePreconfiguredExpr('scannerdetection-stable') ||
                    evaluatePreconfiguredExpr('protocolattack-stable') ||
                    evaluatePreconfiguredExpr('sessionfixation-stable')
                  `
                }
              },
              action: 'deny(403)',
              preview: false
            },
            {
              priority: 2147483647, // Lowest priority (default rule)
              description: 'Default allow rule',
              match: {
                versionedExpr: 'SRC_IPS_V1',
                config: {
                  srcIpRanges: ['*']
                }
              },
              action: 'allow',
              preview: false
            }
          ],
          adaptiveProtectionConfig: {
            layer7DdosDefenseConfig: {
              enable: true,
              ruleVisibility: 'STANDARD'
            }
          }
        },

        // API-specific security policy
        'ipo-api-security-policy': {
          description: 'Security policy for IPO API endpoints',
          rules: [
            {
              priority: 1000,
              description: 'API rate limiting - stricter limits',
              match: {
                versionedExpr: 'SRC_IPS_V1',
                config: {
                  srcIpRanges: ['*']
                }
              },
              action: 'rate_based_ban',
              rateLimitOptions: {
                conformAction: 'allow',
                exceedAction: 'deny(429)',
                enforceOnKey: 'IP',
                rateLimitThreshold: {
                  count: 50, // 50 API calls
                  intervalSec: 60 // per minute
                },
                banThreshold: {
                  count: 200,
                  intervalSec: 300
                },
                banDurationSec: 3600 // 1 hour ban
              }
            },
            {
              priority: 2000,
              description: 'Block requests without proper API authentication',
              match: {
                expr: {
                  expression: '!has(request.headers["authorization"]) && !has(request.headers["x-api-key"])'
                }
              },
              action: 'deny(401)',
              preview: false
            },
            {
              priority: 3000,
              description: 'Block non-HTTPS requests to API',
              match: {
                expr: {
                  expression: 'request.url_map =~ "^http://.*"'
                }
              },
              action: 'redirect',
              redirectOptions: {
                type: 'HTTPS_REDIRECT'
              }
            }
          ]
        }
      },

      // Bot Management
      botManagement: {
        enabled: true,
        configuration: {
          'challenge-page': {
            description: 'Challenge suspicious bots',
            action: 'challenge',
            criteria: [
              'Unusual request patterns',
              'Missing or suspicious user agents',
              'High frequency requests',
              'Suspicious JavaScript behavior'
            ]
          },
          'allow-good-bots': {
            description: 'Allow legitimate search engine bots',
            userAgents: [
              'Googlebot',
              'Bingbot',
              'Slurp', // Yahoo
              'DuckDuckBot'
            ],
            action: 'allow'
          }
        }
      }
    };

    this.cloudArmorPolicies.set('policies', armorConfig);
  }

  // DDoS Protection Configuration
  setupDDoSProtection() {
    const ddosConfig = {
      // Cloud Armor DDoS protection
      cloudArmorDDoS: {
        enabled: true,
        adaptiveProtection: {
          layer7Protection: true,
          autoDeployConfig: {
            loadThreshold: 0.8, // Deploy when 80% capacity reached
            confidenceThreshold: 0.9, // High confidence before auto-deployment
            impactedBaselineThreshold: 0.1 // 10% impact threshold
          }
        },
        ddosProtectionConfig: {
          ddosProtection: 'ADVANCED' // Advanced DDoS protection
        }
      },

      // Load Balancer DDoS mitigation
      loadBalancerProtection: {
        globalLoadBalancer: {
          enabled: true,
          backendConfig: {
            timeoutSec: 30,
            connectionDrainingTimeoutSec: 300,
            maxConnections: 10000,
            maxConnectionsPerInstance: 1000,
            maxConnectionsPerEndpoint: 100,
            maxRate: 1000, // Requests per second
            maxRatePerInstance: 100,
            maxRatePerEndpoint: 10
          },
          circuitBreaker: {
            maxRequestsPerConnection: 10,
            maxRequests: 1000,
            maxPendingRequests: 100,
            maxRetries: 3,
            intervalMs: 30000, // 30 seconds
            baseEjectionTimeMs: 30000,
            maxEjectionPercent: 50
          }
        }
      },

      // Network-level DDoS protection
      networkProtection: {
        packetMirroring: {
          enabled: true,
          configuration: {
            name: 'ddos-packet-mirror',
            region: 'australia-southeast1',
            network: 'ipo-valuation-prod-vpc',
            priority: 1000,
            mirroredResources: {
              subnetworks: ['ipo-app-subnet'],
              tags: ['web-tier', 'app-tier']
            },
            collectorIlb: 'ddos-analysis-lb',
            filter: {
              direction: 'INGRESS',
              ipProtocols: ['TCP', 'UDP', 'ICMP']
            }
          }
        },
        trafficShaping: {
          enabled: true,
          policies: [
            {
              name: 'web-traffic-shaping',
              priority: 100,
              rules: [
                {
                  description: 'Limit HTTP requests per source IP',
                  sourceIpRanges: ['0.0.0.0/0'],
                  rateLimitPps: 1000, // Packets per second
                  burstSize: 5000
                }
              ]
            }
          ]
        }
      },

      // Monitoring and alerting for DDoS
      ddosMonitoring: {
        alerts: [
          {
            displayName: 'DDoS Attack Detected',
            conditions: [
              {
                displayName: 'High request rate',
                conditionThreshold: {
                  filter: 'resource.type="http_load_balancer" AND metric.type="loadbalancing.googleapis.com/https/request_count"',
                  comparison: 'COMPARISON_GREATER_THAN',
                  thresholdValue: 10000, // 10k requests per minute
                  duration: '300s'
                }
              }
            ],
            notificationChannels: [
              'projects/PROJECT_ID/notificationChannels/ddos-alerts'
            ]
          }
        ],
        dashboards: {
          'ddos-monitoring-dashboard': {
            displayName: 'DDoS Protection Dashboard',
            widgets: [
              'Request rate over time',
              'Error rate by response code',
              'Geographic distribution of requests',
              'Cloud Armor policy hits',
              'Backend instance health'
            ]
          }
        }
      }
    };

    this.ddosProtection.set('configuration', ddosConfig);
  }

  // Network Monitoring and Logging
  setupNetworkMonitoring() {
    const monitoringConfig = {
      // VPC Flow Logs
      flowLogsConfig: {
        enabled: true,
        aggregationInterval: 'INTERVAL_5_SEC',
        flowSampling: 1.0, // 100% sampling for security
        metadata: 'INCLUDE_ALL_METADATA',
        metadataFields: [
          'src_instance',
          'dest_instance',
          'src_vpc',
          'dest_vpc',
          'src_gke_details',
          'dest_gke_details'
        ],
        filterExpr: 'true', // Log all traffic
        exportSettings: {
          destination: 'bigquery',
          dataset: 'network_security_logs',
          table: 'vpc_flow_logs'
        }
      },

      // Firewall rule logging
      firewallLogging: {
        enabled: true,
        metadata: 'INCLUDE_ALL_METADATA',
        exportSettings: {
          destination: 'cloud-logging',
          logName: 'firewall-rules'
        }
      },

      // Cloud Armor logging
      cloudArmorLogging: {
        enabled: true,
        logLevel: 'VERBOSE',
        rateLimitOptions: {
          enabled: true,
          rateLimitThreshold: {
            count: 1000,
            intervalSec: 60
          }
        },
        exportSettings: {
          destination: 'bigquery',
          dataset: 'security_logs',
          table: 'cloud_armor_logs'
        }
      },

      // Network security monitoring alerts
      securityAlerts: [
        {
          name: 'Suspicious Network Activity',
          description: 'Alert on unusual network traffic patterns',
          condition: {
            filter: `
              resource.type="vpc_flow_log" AND
              (jsonPayload.bytes_sent > 1073741824 OR  -- 1GB
               jsonPayload.packets_sent > 1000000)     -- 1M packets
            `,
            threshold: 1,
            duration: '300s'
          }
        },
        {
          name: 'Port Scanning Detection',
          description: 'Detect potential port scanning activities',
          condition: {
            filter: `
              resource.type="vpc_flow_log" AND
              jsonPayload.connection_state="SYN_SENT" AND
              jsonPayload.dest_port != 80 AND
              jsonPayload.dest_port != 443
            `,
            threshold: 50, // 50 connection attempts
            duration: '300s'
          }
        },
        {
          name: 'Cloud Armor Rule Violations',
          description: 'High number of Cloud Armor rule violations',
          condition: {
            filter: 'resource.type="http_load_balancer" AND jsonPayload.enforcedSecurityPolicy.outcome="DENY"',
            threshold: 100,
            duration: '300s'
          }
        }
      ],

      // Network topology discovery
      topologyDiscovery: {
        enabled: true,
        scanFrequency: 'daily',
        includeExternalConnections: true,
        includeGKEServices: true,
        exportResults: {
          destination: 'cloud-asset-inventory'
        }
      }
    };

    this.networkConfig.set('monitoring', monitoringConfig);
  }

  // Private Connectivity Configuration
  setupPrivateConnectivity() {
    const privateConnectivityConfig = {
      // Private Google Access
      privateGoogleAccess: {
        enabled: true,
        subnets: [
          'ipo-app-subnet',
          'ipo-db-subnet',
          'ipo-mgmt-subnet'
        ],
        customRoutes: [
          {
            name: 'private-google-apis-route',
            destRange: '199.36.153.8/30',
            nextHop: 'default-internet-gateway'
          }
        ]
      },

      // Private Service Connect
      privateServiceConnect: {
        endpoints: [
          {
            name: 'storage-psc-endpoint',
            target: 'storage.googleapis.com',
            network: 'ipo-valuation-prod-vpc',
            subnet: 'ipo-app-subnet',
            ipAddress: '10.1.0.100'
          },
          {
            name: 'bigquery-psc-endpoint',
            target: 'bigquery.googleapis.com',
            network: 'ipo-valuation-prod-vpc',
            subnet: 'ipo-app-subnet',
            ipAddress: '10.1.0.101'
          }
        ]
      },

      // VPC Service Controls integration
      vpcServiceControls: {
        perimeter: 'ipo-production-perimeter',
        bridgeRules: [
          {
            name: 'allow-storage-access',
            ingressFrom: {
              identities: ['serviceAccount:ipo-app@PROJECT_ID.iam.gserviceaccount.com']
            },
            ingressTo: {
              resources: ['projects/PROJECT_ID'],
              operations: [
                {
                  serviceName: 'storage.googleapis.com',
                  methodSelectors: [
                    { method: 'google.storage.objects.get' },
                    { method: 'google.storage.objects.create' }
                  ]
                }
              ]
            }
          }
        ]
      },

      // Cloud Interconnect (for enterprise customers)
      cloudInterconnect: {
        type: 'DEDICATED', // or 'PARTNER' for smaller connections
        configuration: {
          linkType: '10G_LR',
          location: 'sydney-nextdc-s1',
          customerName: 'Uprez IPO Valuation',
          description: 'Dedicated interconnect for enterprise customers'
        },
        vlan: {
          vlanTag: 100,
          ipRange: '169.254.1.0/30',
          bgp: {
            asn: 65001,
            advertisedIpRanges: ['10.1.0.0/16']
          }
        }
      }
    };

    this.networkConfig.set('privateConnectivity', privateConnectivityConfig);
  }

  // Generate Terraform configuration for network security
  generateTerraformConfig() {
    return `
# Network Security Implementation for IPO Valuation SaaS

# Production VPC
resource "google_compute_network" "prod_vpc" {
  name                    = "ipo-valuation-prod-vpc"
  auto_create_subnetworks = false
  description            = "Production VPC for IPO Valuation SaaS"
  routing_mode          = "REGIONAL"
}

# Application tier subnet
resource "google_compute_subnetwork" "app_subnet" {
  name                     = "ipo-app-subnet"
  ip_cidr_range           = "10.1.0.0/24"
  region                  = "australia-southeast1"
  network                 = google_compute_network.prod_vpc.id
  private_ip_google_access = true
  
  log_config {
    aggregation_interval = "INTERVAL_5_SEC"
    flow_sampling       = 0.5
    metadata           = "INCLUDE_ALL_METADATA"
    metadata_fields    = ["src_instance", "dest_instance", "project_id"]
  }

  secondary_ip_range {
    range_name    = "pod-range"
    ip_cidr_range = "10.11.0.0/16"
  }

  secondary_ip_range {
    range_name    = "service-range"
    ip_cidr_range = "10.12.0.0/16"
  }
}

# Database tier subnet
resource "google_compute_subnetwork" "db_subnet" {
  name                     = "ipo-db-subnet"
  ip_cidr_range           = "10.1.1.0/24"
  region                  = "australia-southeast1"
  network                 = google_compute_network.prod_vpc.id
  private_ip_google_access = true
  
  log_config {
    aggregation_interval = "INTERVAL_5_SEC"
    flow_sampling       = 1.0
    metadata           = "INCLUDE_ALL_METADATA"
  }
}

# Cloud Router for NAT
resource "google_compute_router" "prod_router" {
  name    = "ipo-prod-router"
  region  = "australia-southeast1"
  network = google_compute_network.prod_vpc.id
}

# Cloud NAT
resource "google_compute_router_nat" "prod_nat" {
  name                               = "ipo-prod-nat"
  router                            = google_compute_router.prod_router.name
  region                            = "australia-southeast1"
  nat_ip_allocate_option            = "MANUAL_ONLY"
  nat_ips                           = [google_compute_address.nat_ip.self_link]
  source_subnetwork_ip_ranges_to_nat = "LIST_OF_SUBNETWORKS"

  subnetwork {
    name                    = google_compute_subnetwork.app_subnet.id
    source_ip_ranges_to_nat = ["PRIMARY_IP_RANGE"]
  }

  log_config {
    enable = true
    filter = "ALL"
  }
}

# Static IP for NAT
resource "google_compute_address" "nat_ip" {
  name   = "ipo-prod-nat-ip"
  region = "australia-southeast1"
}

# Firewall Rules
resource "google_compute_firewall" "allow_health_checks" {
  name    = "allow-health-checks"
  network = google_compute_network.prod_vpc.name

  allow {
    protocol = "tcp"
    ports    = ["80", "443", "8080"]
  }

  source_ranges = [
    "130.211.0.0/22",
    "35.191.0.0/16"
  ]

  target_tags = ["http-server", "https-server"]

  log_config {
    metadata = "INCLUDE_ALL_METADATA"
  }
}

resource "google_compute_firewall" "allow_ssh_bastion" {
  name    = "allow-ssh-from-bastion"
  network = google_compute_network.prod_vpc.name

  allow {
    protocol = "tcp"
    ports    = ["22"]
  }

  source_ranges = ["10.3.0.0/28"]
  target_tags   = ["ssh-access"]

  log_config {
    metadata = "INCLUDE_ALL_METADATA"
  }
}

resource "google_compute_firewall" "deny_all_egress" {
  name      = "deny-all-egress-default"
  network   = google_compute_network.prod_vpc.name
  direction = "EGRESS"
  priority  = 65534

  deny {
    protocol = "all"
  }

  destination_ranges = ["0.0.0.0/0"]
  target_tags        = ["no-external-access"]

  log_config {
    metadata = "INCLUDE_ALL_METADATA"
  }
}

# Cloud Armor Security Policy
resource "google_compute_security_policy" "ipo_security_policy" {
  name        = "ipo-valuation-security-policy"
  description = "Security policy for IPO valuation web application"

  rule {
    action   = "rate_based_ban"
    priority = "1000"
    match {
      versioned_expr = "SRC_IPS_V1"
      config {
        src_ip_ranges = ["*"]
      }
    }
    rate_limit_options {
      conform_action = "allow"
      exceed_action  = "deny(429)"
      enforce_on_key = "IP"
      rate_limit_threshold {
        count        = 100
        interval_sec = 60
      }
      ban_threshold {
        count        = 500
        interval_sec = 300
      }
      ban_duration_sec = 1800
    }
    description = "Rate limiting - requests per minute"
  }

  rule {
    action   = "deny(403)"
    priority = "2000"
    match {
      expr {
        expression = <<-EOT
          evaluatePreconfiguredExpr('sqli-stable') ||
          evaluatePreconfiguredExpr('xss-stable') ||
          evaluatePreconfiguredExpr('lfi-stable') ||
          evaluatePreconfiguredExpr('rce-stable')
        EOT
      }
    }
    description = "OWASP Top 10 protection"
  }

  rule {
    action   = "allow"
    priority = "2147483647"
    match {
      versioned_expr = "SRC_IPS_V1"
      config {
        src_ip_ranges = ["*"]
      }
    }
    description = "Default allow rule"
  }

  adaptive_protection_config {
    layer_7_ddos_defense_config {
      enable          = true
      rule_visibility = "STANDARD"
    }
  }
}

# Global HTTP Load Balancer with Cloud Armor
resource "google_compute_global_forwarding_rule" "https_forwarding_rule" {
  name       = "ipo-https-forwarding-rule"
  target     = google_compute_target_https_proxy.https_proxy.id
  port_range = "443"
  ip_address = google_compute_global_address.lb_ip.id
}

resource "google_compute_target_https_proxy" "https_proxy" {
  name             = "ipo-https-proxy"
  url_map          = google_compute_url_map.url_map.id
  ssl_certificates = [google_compute_managed_ssl_certificate.ssl_cert.id]
}

resource "google_compute_url_map" "url_map" {
  name            = "ipo-url-map"
  default_service = google_compute_backend_service.backend_service.id
}

resource "google_compute_backend_service" "backend_service" {
  name                  = "ipo-backend-service"
  protocol             = "HTTP"
  port_name            = "http"
  load_balancing_scheme = "EXTERNAL"
  security_policy      = google_compute_security_policy.ipo_security_policy.id
  
  backend {
    group           = google_compute_instance_group.app_instance_group.id
    balancing_mode  = "UTILIZATION"
    capacity_scaler = 1.0
  }

  health_checks = [google_compute_health_check.health_check.id]
}

# Managed SSL Certificate
resource "google_compute_managed_ssl_certificate" "ssl_cert" {
  name = "ipo-ssl-cert"

  managed {
    domains = ["app.uprez.com", "api.uprez.com"]
  }
}

# Global IP Address
resource "google_compute_global_address" "lb_ip" {
  name         = "ipo-lb-ip"
  address_type = "EXTERNAL"
}

# Health Check
resource "google_compute_health_check" "health_check" {
  name               = "ipo-health-check"
  check_interval_sec = 30
  timeout_sec        = 5

  http_health_check {
    port         = 8080
    request_path = "/health"
  }
}

# Variables
variable "project_id" {
  description = "GCP Project ID"
  type        = string
}

variable "region" {
  description = "GCP Region"
  type        = string
  default     = "australia-southeast1"
}

variable "office_ip_range" {
  description = "Office IP range for trusted access"
  type        = string
  default     = "203.0.113.0/24"
}
`;
  }

  // Get network security summary
  getNetworkSecuritySummary() {
    return {
      networkArchitecture: [
        'Multi-tier VPC with isolated subnets',
        'Private Google Access for secure API connectivity',
        'Cloud NAT for controlled outbound internet access',
        'VPC peering for management connectivity',
        'Private Service Connect for Google services'
      ],
      securityLayers: [
        'VPC firewall rules with default deny policies',
        'Cloud Armor WAF with OWASP protection',
        'DDoS protection with adaptive policies',
        'Rate limiting and bot management',
        'Geographic access restrictions'
      ],
      monitoringCapabilities: [
        'VPC Flow Logs with 100% sampling',
        'Firewall rule logging and analysis',
        'Cloud Armor security event logging',
        'Real-time network anomaly detection',
        'Comprehensive security dashboards'
      ],
      complianceFeatures: [
        'Network segmentation for data protection',
        'Encrypted network communications',
        'Audit trails for all network access',
        'Geographic data residency controls',
        'Financial services security standards'
      ]
    };
  }
}

module.exports = { NetworkSecurityImplementation };