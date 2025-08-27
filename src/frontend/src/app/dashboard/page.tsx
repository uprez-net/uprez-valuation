'use client';

import React from 'react';
import { motion } from 'framer-motion';
import { 
  TrendingUp, 
  FolderOpen, 
  FileText, 
  BarChart3,
  Users,
  Clock,
  DollarSign,
  Activity,
  Plus,
  ArrowUpRight,
  ArrowDownRight,
  Minus
} from 'lucide-react';
import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/card';
import { Button } from '@/components/ui/button';
import { ValuationMetricCard } from '@/components/valuation/ValuationMetricCard';
import { InteractiveValuationChart } from '@/components/valuation/InteractiveValuationChart';
import { WaterfallChart } from '@/components/charts/WaterfallChart';
import { formatCurrency, formatPercentage } from '@/lib/utils';

const mockMetrics = {
  totalProjects: 24,
  activeProjects: 8,
  completedValuations: 16,
  totalValue: 2400000000, // $2.4B
  avgValuation: 150000000, // $150M
  successRate: 0.85,
};

const mockChartData = [
  { label: 'Jan', value: 120000000, date: '2024-01-01' },
  { label: 'Feb', value: 135000000, date: '2024-02-01' },
  { label: 'Mar', value: 158000000, date: '2024-03-01' },
  { label: 'Apr', value: 142000000, date: '2024-04-01' },
  { label: 'May', value: 165000000, date: '2024-05-01' },
  { label: 'Jun', value: 180000000, date: '2024-06-01' },
];

const mockWaterfallData = [
  { name: 'Base Valuation', value: 100000000, cumulative: 100000000, type: 'total' as const },
  { name: 'Revenue Growth', value: 25000000, cumulative: 125000000, type: 'positive' as const },
  { name: 'Market Expansion', value: 15000000, cumulative: 140000000, type: 'positive' as const },
  { name: 'Risk Discount', value: -10000000, cumulative: 130000000, type: 'negative' as const },
  { name: 'Competitive Pressure', value: -8000000, cumulative: 122000000, type: 'negative' as const },
  { name: 'Final Valuation', value: 122000000, cumulative: 122000000, type: 'total' as const },
];

const recentProjects = [
  {
    id: '1',
    name: 'TechCorp IPO Analysis',
    company: 'TechCorp Inc.',
    status: 'completed',
    value: 180000000,
    updatedAt: '2024-01-15T10:30:00Z',
    progress: 100,
  },
  {
    id: '2',
    name: 'DataFlow Valuation',
    company: 'DataFlow Systems',
    status: 'in_progress',
    value: 95000000,
    updatedAt: '2024-01-14T16:45:00Z',
    progress: 75,
  },
  {
    id: '3',
    name: 'CloudTech Assessment',
    company: 'CloudTech Solutions',
    status: 'draft',
    value: 0,
    updatedAt: '2024-01-13T09:15:00Z',
    progress: 25,
  },
];

const teamActivity = [
  {
    id: '1',
    user: 'John Smith',
    action: 'completed valuation for',
    target: 'TechCorp IPO',
    time: '2 hours ago',
    type: 'completion',
  },
  {
    id: '2',
    user: 'Sarah Johnson',
    action: 'uploaded prospectus for',
    target: 'DataFlow Systems',
    time: '4 hours ago',
    type: 'upload',
  },
  {
    id: '3',
    user: 'Mike Chen',
    action: 'created new project',
    target: 'GreenEnergy IPO',
    time: '6 hours ago',
    type: 'creation',
  },
];

export default function DashboardPage() {
  return (
    <div className="space-y-6">
      {/* Header */}
      <div className="flex items-center justify-between">
        <div>
          <h1 className="text-3xl font-bold tracking-tight">Dashboard</h1>
          <p className="text-muted-foreground mt-1">
            Welcome back! Here&apos;s what&apos;s happening with your valuations.
          </p>
        </div>
        <Button>
          <Plus className="h-4 w-4 mr-2" />
          New Project
        </Button>
      </div>

      {/* Key Metrics */}
      <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-6">
        <ValuationMetricCard
          title="Total Projects"
          value={mockMetrics.totalProjects}
          previousValue={20}
          format="number"
          icon={<FolderOpen className="h-5 w-5" />}
          description="Active and completed projects"
          showTrend
        />
        
        <ValuationMetricCard
          title="Portfolio Value"
          value={mockMetrics.totalValue}
          previousValue={2100000000}
          format="currency"
          icon={<DollarSign className="h-5 w-5" />}
          description="Total valuation across all projects"
          showTrend
          confidenceLevel={0.85}
        />
        
        <ValuationMetricCard
          title="Avg. Valuation"
          value={mockMetrics.avgValuation}
          previousValue={140000000}
          format="currency"
          icon={<BarChart3 className="h-5 w-5" />}
          description="Average per project"
          showTrend
        />
        
        <ValuationMetricCard
          title="Success Rate"
          value={mockMetrics.successRate}
          previousValue={0.80}
          format="percentage"
          icon={<TrendingUp className="h-5 w-5" />}
          description="Completed vs total projects"
          showTrend
        />
      </div>

      {/* Charts Row */}
      <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
        {/* Valuation Trends */}
        <InteractiveValuationChart
          data={mockChartData}
          title="Monthly Valuation Trends"
          type="area"
          showBrush
          height={350}
        />

        {/* Valuation Bridge */}
        <WaterfallChart
          data={mockWaterfallData}
          title="Latest Valuation Bridge"
          height={350}
        />
      </div>

      {/* Content Grid */}
      <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
        {/* Recent Projects */}
        <Card className="lg:col-span-2">
          <CardHeader>
            <div className="flex items-center justify-between">
              <CardTitle className="flex items-center gap-2">
                <FolderOpen className="h-5 w-5" />
                Recent Projects
              </CardTitle>
              <Button variant="outline" size="sm">
                View All
              </Button>
            </div>
          </CardHeader>
          <CardContent>
            <div className="space-y-4">
              {recentProjects.map((project, index) => (
                <motion.div
                  key={project.id}
                  initial={{ opacity: 0, y: 20 }}
                  animate={{ opacity: 1, y: 0 }}
                  transition={{ duration: 0.3, delay: index * 0.1 }}
                  className="flex items-center justify-between p-4 border rounded-lg hover:shadow-md transition-shadow cursor-pointer"
                >
                  <div className="flex items-center space-x-4">
                    <div className={`w-3 h-3 rounded-full ${
                      project.status === 'completed' 
                        ? 'bg-green-500' 
                        : project.status === 'in_progress' 
                          ? 'bg-yellow-500' 
                          : 'bg-gray-400'
                    }`} />
                    <div>
                      <h4 className="font-medium">{project.name}</h4>
                      <p className="text-sm text-muted-foreground">{project.company}</p>
                    </div>
                  </div>
                  <div className="text-right">
                    <div className="font-medium">
                      {project.value > 0 ? formatCurrency(project.value) : 'Draft'}
                    </div>
                    <div className="text-sm text-muted-foreground flex items-center">
                      <Clock className="h-3 w-3 mr-1" />
                      {new Date(project.updatedAt).toLocaleDateString()}
                    </div>
                  </div>
                </motion.div>
              ))}
            </div>
          </CardContent>
        </Card>

        {/* Team Activity */}
        <Card>
          <CardHeader>
            <CardTitle className="flex items-center gap-2">
              <Activity className="h-5 w-5" />
              Team Activity
            </CardTitle>
          </CardHeader>
          <CardContent>
            <div className="space-y-4">
              {teamActivity.map((activity, index) => (
                <motion.div
                  key={activity.id}
                  initial={{ opacity: 0, x: 20 }}
                  animate={{ opacity: 1, x: 0 }}
                  transition={{ duration: 0.3, delay: index * 0.1 }}
                  className="flex items-start space-x-3"
                >
                  <div className="flex-shrink-0 w-8 h-8 bg-primary/10 rounded-full flex items-center justify-center">
                    {activity.type === 'completion' && <TrendingUp className="h-4 w-4 text-green-600" />}
                    {activity.type === 'upload' && <FileText className="h-4 w-4 text-blue-600" />}
                    {activity.type === 'creation' && <Plus className="h-4 w-4 text-purple-600" />}
                  </div>
                  <div className="flex-1 min-w-0">
                    <p className="text-sm">
                      <span className="font-medium">{activity.user}</span>{' '}
                      {activity.action}{' '}
                      <span className="font-medium">{activity.target}</span>
                    </p>
                    <p className="text-xs text-muted-foreground mt-1">
                      {activity.time}
                    </p>
                  </div>
                </motion.div>
              ))}
            </div>
          </CardContent>
        </Card>
      </div>

      {/* Quick Actions */}
      <Card>
        <CardHeader>
          <CardTitle>Quick Actions</CardTitle>
        </CardHeader>
        <CardContent>
          <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
            <Button variant="outline" className="h-20 flex flex-col items-center justify-center space-y-2">
              <FileText className="h-6 w-6" />
              <span>Upload Prospectus</span>
            </Button>
            <Button variant="outline" className="h-20 flex flex-col items-center justify-center space-y-2">
              <BarChart3 className="h-6 w-6" />
              <span>Start Valuation</span>
            </Button>
            <Button variant="outline" className="h-20 flex flex-col items-center justify-center space-y-2">
              <Users className="h-6 w-6" />
              <span>Invite Team</span>
            </Button>
          </div>
        </CardContent>
      </Card>
    </div>
  );
}