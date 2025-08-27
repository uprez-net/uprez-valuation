'use client';

import React, { useState, useMemo } from 'react';
import { motion } from 'framer-motion';
import {
  LineChart,
  Line,
  AreaChart,
  Area,
  BarChart,
  Bar,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  ResponsiveContainer,
  ReferenceLine,
  Brush,
  Legend,
} from 'recharts';
import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/card';
import { Button } from '@/components/ui/button';
import * as Select from '@radix-ui/react-select';
import { ChevronDown, Download, Maximize2, TrendingUp } from 'lucide-react';
import { formatCurrency, formatPercentage, cn } from '@/lib/utils';
import { ChartDataPoint } from '@/types';

interface InteractiveValuationChartProps {
  data: ChartDataPoint[];
  title?: string;
  type?: 'line' | 'area' | 'bar';
  showBrush?: boolean;
  showLegend?: boolean;
  showGrid?: boolean;
  height?: number;
  className?: string;
  onDataPointClick?: (dataPoint: ChartDataPoint) => void;
  onExport?: (format: 'png' | 'pdf' | 'csv') => void;
  scenarios?: {
    name: string;
    data: ChartDataPoint[];
    color: string;
  }[];
  confidenceIntervals?: {
    upper: number[];
    lower: number[];
  };
}

const CustomTooltip = ({ active, payload, label }: any) => {
  if (active && payload && payload.length) {
    return (
      <div className="bg-popover border rounded-lg p-3 shadow-lg">
        <p className="font-medium text-sm mb-2">{label}</p>
        {payload.map((item: any, index: number) => (
          <div key={index} className="flex items-center text-sm">
            <div
              className="w-3 h-3 rounded mr-2"
              style={{ backgroundColor: item.color }}
            />
            <span className="text-muted-foreground mr-2">{item.name}:</span>
            <span className="font-medium">
              {typeof item.value === 'number'
                ? item.name.toLowerCase().includes('percentage') || item.name.toLowerCase().includes('rate')
                  ? formatPercentage(item.value / 100)
                  : formatCurrency(item.value)
                : item.value}
            </span>
          </div>
        ))}
      </div>
    );
  }
  return null;
};

export function InteractiveValuationChart({
  data,
  title = 'Valuation Analysis',
  type = 'line',
  showBrush = false,
  showLegend = true,
  showGrid = true,
  height = 400,
  className,
  onDataPointClick,
  onExport,
  scenarios,
  confidenceIntervals,
}: InteractiveValuationChartProps) {
  const [selectedTimeRange, setSelectedTimeRange] = useState('all');
  const [selectedScenario, setSelectedScenario] = useState('all');
  const [chartType, setChartType] = useState(type);

  const timeRanges = [
    { value: 'all', label: 'All Time' },
    { value: '1y', label: 'Last Year' },
    { value: '6m', label: 'Last 6 Months' },
    { value: '3m', label: 'Last 3 Months' },
    { value: '1m', label: 'Last Month' },
  ];

  const chartTypes = [
    { value: 'line', label: 'Line Chart' },
    { value: 'area', label: 'Area Chart' },
    { value: 'bar', label: 'Bar Chart' },
  ];

  const filteredData = useMemo(() => {
    if (selectedTimeRange === 'all') return data;

    const now = new Date();
    const ranges: Record<string, number> = {
      '1y': 365,
      '6m': 180,
      '3m': 90,
      '1m': 30,
    };

    const daysBack = ranges[selectedTimeRange];
    const cutoffDate = new Date(now.getTime() - daysBack * 24 * 60 * 60 * 1000);

    return data.filter(item => {
      if (item.date) {
        return new Date(item.date) >= cutoffDate;
      }
      return true;
    });
  }, [data, selectedTimeRange]);

  const displayScenarios = useMemo(() => {
    if (!scenarios) return [];
    
    if (selectedScenario === 'all') {
      return scenarios;
    }
    
    return scenarios.filter(s => s.name === selectedScenario);
  }, [scenarios, selectedScenario]);

  const renderChart = () => {
    const commonProps = {
      data: filteredData,
      margin: { top: 20, right: 30, left: 20, bottom: 5 },
      onMouseDown: (data: any) => {
        if (onDataPointClick && data?.activePayload?.[0]?.payload) {
          onDataPointClick(data.activePayload[0].payload);
        }
      },
    };

    const ChartComponent = {
      line: LineChart,
      area: AreaChart,
      bar: BarChart,
    }[chartType];

    return (
      <ResponsiveContainer width="100%" height={height}>
        <ChartComponent {...commonProps}>
          {showGrid && <CartesianGrid strokeDasharray="3 3" opacity={0.3} />}
          <XAxis 
            dataKey="label" 
            tick={{ fontSize: 12 }}
            tickLine={false}
            axisLine={false}
          />
          <YAxis 
            tick={{ fontSize: 12 }}
            tickLine={false}
            axisLine={false}
            tickFormatter={(value) => formatCurrency(value, 'USD')}
          />
          <Tooltip content={<CustomTooltip />} />
          {showLegend && <Legend />}

          {/* Confidence Intervals */}
          {confidenceIntervals && chartType === 'area' && (
            <Area
              type="monotone"
              dataKey="upper"
              data={confidenceIntervals.upper.map((val, index) => ({
                ...filteredData[index],
                upper: val,
                lower: confidenceIntervals.lower[index],
              }))}
              fill="rgba(59, 130, 246, 0.1)"
              stroke="none"
            />
          )}

          {/* Main Data Line/Area/Bar */}
          {chartType === 'line' && (
            <Line
              type="monotone"
              dataKey="value"
              stroke="#0ea5e9"
              strokeWidth={2}
              dot={{ fill: '#0ea5e9', strokeWidth: 2, r: 4 }}
              activeDot={{ r: 6, stroke: '#0ea5e9', strokeWidth: 2 }}
            />
          )}

          {chartType === 'area' && (
            <Area
              type="monotone"
              dataKey="value"
              stroke="#0ea5e9"
              strokeWidth={2}
              fill="url(#colorGradient)"
              fillOpacity={0.6}
            />
          )}

          {chartType === 'bar' && (
            <Bar
              dataKey="value"
              fill="#0ea5e9"
              radius={[4, 4, 0, 0]}
            />
          )}

          {/* Scenario Lines */}
          {displayScenarios.map((scenario, index) => (
            <Line
              key={scenario.name}
              type="monotone"
              dataKey={`scenario_${index}`}
              data={scenario.data}
              stroke={scenario.color}
              strokeWidth={2}
              strokeDasharray={index > 0 ? '5,5' : undefined}
              name={scenario.name}
            />
          ))}

          {/* Reference Lines */}
          <ReferenceLine 
            y={Math.max(...filteredData.map(d => d.value))} 
            stroke="#ef4444" 
            strokeDasharray="3 3" 
            label="High"
          />
          <ReferenceLine 
            y={Math.min(...filteredData.map(d => d.value))} 
            stroke="#10b981" 
            strokeDasharray="3 3" 
            label="Low"
          />

          {showBrush && <Brush dataKey="label" height={30} stroke="#0ea5e9" />}

          {/* Gradient Definition */}
          <defs>
            <linearGradient id="colorGradient" x1="0" y1="0" x2="0" y2="1">
              <stop offset="5%" stopColor="#0ea5e9" stopOpacity={0.8} />
              <stop offset="95%" stopColor="#0ea5e9" stopOpacity={0.1} />
            </linearGradient>
          </defs>
        </ChartComponent>
      </ResponsiveContainer>
    );
  };

  return (
    <motion.div
      initial={{ opacity: 0, y: 20 }}
      animate={{ opacity: 1, y: 0 }}
      transition={{ duration: 0.3 }}
    >
      <Card className={className}>
        <CardHeader>
          <div className="flex items-center justify-between">
            <CardTitle className="flex items-center gap-2">
              <TrendingUp className="h-5 w-5" />
              {title}
            </CardTitle>
            
            <div className="flex items-center gap-2">
              {/* Time Range Selector */}
              <Select.Root value={selectedTimeRange} onValueChange={setSelectedTimeRange}>
                <Select.Trigger className="inline-flex items-center gap-2 px-3 py-2 text-sm border rounded-md hover:bg-accent">
                  <Select.Value />
                  <ChevronDown className="h-4 w-4" />
                </Select.Trigger>
                <Select.Portal>
                  <Select.Content className="bg-popover border rounded-md shadow-lg z-50">
                    <Select.Viewport className="p-1">
                      {timeRanges.map((range) => (
                        <Select.Item
                          key={range.value}
                          value={range.value}
                          className="px-3 py-2 text-sm cursor-pointer hover:bg-accent rounded"
                        >
                          <Select.ItemText>{range.label}</Select.ItemText>
                        </Select.Item>
                      ))}
                    </Select.Viewport>
                  </Select.Content>
                </Select.Portal>
              </Select.Root>

              {/* Chart Type Selector */}
              <Select.Root value={chartType} onValueChange={setChartType}>
                <Select.Trigger className="inline-flex items-center gap-2 px-3 py-2 text-sm border rounded-md hover:bg-accent">
                  <Select.Value />
                  <ChevronDown className="h-4 w-4" />
                </Select.Trigger>
                <Select.Portal>
                  <Select.Content className="bg-popover border rounded-md shadow-lg z-50">
                    <Select.Viewport className="p-1">
                      {chartTypes.map((type) => (
                        <Select.Item
                          key={type.value}
                          value={type.value}
                          className="px-3 py-2 text-sm cursor-pointer hover:bg-accent rounded"
                        >
                          <Select.ItemText>{type.label}</Select.ItemText>
                        </Select.Item>
                      ))}
                    </Select.Viewport>
                  </Select.Content>
                </Select.Portal>
              </Select.Root>

              {/* Scenario Selector */}
              {scenarios && (
                <Select.Root value={selectedScenario} onValueChange={setSelectedScenario}>
                  <Select.Trigger className="inline-flex items-center gap-2 px-3 py-2 text-sm border rounded-md hover:bg-accent">
                    <Select.Value />
                    <ChevronDown className="h-4 w-4" />
                  </Select.Trigger>
                  <Select.Portal>
                    <Select.Content className="bg-popover border rounded-md shadow-lg z-50">
                      <Select.Viewport className="p-1">
                        <Select.Item
                          value="all"
                          className="px-3 py-2 text-sm cursor-pointer hover:bg-accent rounded"
                        >
                          <Select.ItemText>All Scenarios</Select.ItemText>
                        </Select.Item>
                        {scenarios.map((scenario) => (
                          <Select.Item
                            key={scenario.name}
                            value={scenario.name}
                            className="px-3 py-2 text-sm cursor-pointer hover:bg-accent rounded"
                          >
                            <Select.ItemText>{scenario.name}</Select.ItemText>
                          </Select.Item>
                        ))}
                      </Select.Viewport>
                    </Select.Content>
                  </Select.Portal>
                </Select.Root>
              )}

              {/* Export Button */}
              {onExport && (
                <Button
                  variant="outline"
                  size="sm"
                  onClick={() => onExport('png')}
                >
                  <Download className="h-4 w-4 mr-2" />
                  Export
                </Button>
              )}

              {/* Fullscreen Button */}
              <Button variant="outline" size="sm">
                <Maximize2 className="h-4 w-4" />
              </Button>
            </div>
          </div>
        </CardHeader>

        <CardContent>
          {renderChart()}
        </CardContent>
      </Card>
    </motion.div>
  );
}