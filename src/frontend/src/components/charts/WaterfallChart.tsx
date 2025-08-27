'use client';

import React, { useMemo } from 'react';
import { motion } from 'framer-motion';
import { 
  ComposedChart, 
  Bar, 
  Line, 
  XAxis, 
  YAxis, 
  CartesianGrid, 
  Tooltip, 
  ResponsiveContainer,
  Cell
} from 'recharts';
import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/card';
import { formatCurrency, cn } from '@/lib/utils';
import { WaterfallDataPoint } from '@/types';
import { TrendingUp, ArrowDown, ArrowUp, Minus } from 'lucide-react';

interface WaterfallChartProps {
  data: WaterfallDataPoint[];
  title?: string;
  height?: number;
  className?: string;
  showConnectors?: boolean;
  colorScheme?: {
    positive: string;
    negative: string;
    total: string;
    connector: string;
  };
}

const defaultColorScheme = {
  positive: '#10b981',
  negative: '#ef4444',
  total: '#0ea5e9',
  connector: '#6b7280',
};

const CustomTooltip = ({ active, payload, label }: any) => {
  if (active && payload && payload.length) {
    const data = payload[0].payload;
    const value = payload[0].value;
    
    return (
      <div className="bg-popover border rounded-lg p-3 shadow-lg">
        <p className="font-medium text-sm mb-2">{label}</p>
        <div className="space-y-1">
          <div className="flex items-center text-sm">
            <div
              className="w-3 h-3 rounded mr-2"
              style={{ backgroundColor: payload[0].fill }}
            />
            <span className="text-muted-foreground mr-2">Value:</span>
            <span className="font-medium">{formatCurrency(Math.abs(value))}</span>
          </div>
          <div className="flex items-center text-sm">
            <span className="text-muted-foreground mr-2">Cumulative:</span>
            <span className="font-medium">{formatCurrency(data.cumulative)}</span>
          </div>
          <div className="flex items-center text-sm">
            <span className="text-muted-foreground mr-2">Type:</span>
            <span className="capitalize">{data.type}</span>
          </div>
        </div>
      </div>
    );
  }
  return null;
};

export function WaterfallChart({
  data,
  title = 'Valuation Bridge',
  height = 400,
  className,
  showConnectors = true,
  colorScheme = defaultColorScheme,
}: WaterfallChartProps) {
  const processedData = useMemo(() => {
    let runningTotal = 0;
    const processed = data.map((item, index) => {
      const isTotal = item.type === 'total';
      const value = item.value;
      
      // For waterfall display, we need to position bars correctly
      const barStart = isTotal ? 0 : runningTotal;
      const barValue = isTotal ? item.cumulative : Math.abs(value);
      
      if (!isTotal) {
        runningTotal += value;
      } else {
        runningTotal = item.cumulative;
      }
      
      return {
        ...item,
        barStart,
        barValue,
        displayValue: value,
        cumulative: runningTotal,
        color: item.type === 'positive' 
          ? colorScheme.positive 
          : item.type === 'negative' 
            ? colorScheme.negative 
            : colorScheme.total,
        previousCumulative: index > 0 ? processed[index - 1]?.cumulative || 0 : 0,
      };
    });

    return processed;
  }, [data, colorScheme]);

  const maxValue = Math.max(...processedData.map(d => Math.max(d.cumulative, d.barStart + d.barValue)));
  const minValue = Math.min(...processedData.map(d => Math.min(0, d.barStart)));

  const getIcon = (type: WaterfallDataPoint['type']) => {
    switch (type) {
      case 'positive':
        return <ArrowUp className="h-4 w-4 text-green-600" />;
      case 'negative':
        return <ArrowDown className="h-4 w-4 text-red-600" />;
      case 'total':
        return <Minus className="h-4 w-4 text-blue-600" />;
      default:
        return null;
    }
  };

  return (
    <motion.div
      initial={{ opacity: 0, y: 20 }}
      animate={{ opacity: 1, y: 0 }}
      transition={{ duration: 0.3 }}
    >
      <Card className={className}>
        <CardHeader>
          <CardTitle className="flex items-center gap-2">
            <TrendingUp className="h-5 w-5" />
            {title}
          </CardTitle>
        </CardHeader>

        <CardContent>
          <ResponsiveContainer width="100%" height={height}>
            <ComposedChart
              data={processedData}
              margin={{ top: 20, right: 30, left: 20, bottom: 60 }}
            >
              <CartesianGrid strokeDasharray="3 3" opacity={0.3} />
              <XAxis 
                dataKey="name" 
                angle={-45}
                textAnchor="end"
                height={80}
                tick={{ fontSize: 12 }}
                interval={0}
              />
              <YAxis 
                tick={{ fontSize: 12 }}
                tickFormatter={(value) => formatCurrency(value, 'USD')}
                domain={[minValue * 1.1, maxValue * 1.1]}
              />
              <Tooltip content={<CustomTooltip />} />

              {/* Waterfall Bars */}
              <Bar dataKey="barValue" stackId="waterfall">
                {processedData.map((entry, index) => (
                  <Cell key={`cell-${index}`} fill={entry.color} />
                ))}
              </Bar>

              {/* Connector Lines */}
              {showConnectors && (
                <Line
                  type="stepAfter"
                  dataKey="cumulative"
                  stroke={colorScheme.connector}
                  strokeWidth={1}
                  strokeDasharray="2,2"
                  dot={false}
                  connectNulls={false}
                />
              )}
            </ComposedChart>
          </ResponsiveContainer>

          {/* Legend */}
          <div className="flex flex-wrap justify-center gap-6 mt-4 pt-4 border-t">
            {Object.entries({
              'Positive Impact': { color: colorScheme.positive, icon: <ArrowUp className="h-3 w-3" /> },
              'Negative Impact': { color: colorScheme.negative, icon: <ArrowDown className="h-3 w-3" /> },
              'Total/Subtotal': { color: colorScheme.total, icon: <Minus className="h-3 w-3" /> },
            }).map(([label, { color, icon }]) => (
              <div key={label} className="flex items-center gap-2 text-sm">
                <div 
                  className="w-3 h-3 rounded" 
                  style={{ backgroundColor: color }}
                />
                {icon}
                <span className="text-muted-foreground">{label}</span>
              </div>
            ))}
          </div>

          {/* Summary Statistics */}
          <div className="grid grid-cols-1 md:grid-cols-3 gap-4 mt-6 p-4 bg-muted/50 rounded-lg">
            <div className="text-center">
              <div className="text-lg font-semibold text-green-600">
                {formatCurrency(
                  processedData
                    .filter(d => d.type === 'positive')
                    .reduce((sum, d) => sum + d.displayValue, 0)
                )}
              </div>
              <div className="text-sm text-muted-foreground">Total Positive</div>
            </div>
            <div className="text-center">
              <div className="text-lg font-semibold text-red-600">
                {formatCurrency(
                  Math.abs(processedData
                    .filter(d => d.type === 'negative')
                    .reduce((sum, d) => sum + d.displayValue, 0))
                )}
              </div>
              <div className="text-sm text-muted-foreground">Total Negative</div>
            </div>
            <div className="text-center">
              <div className="text-lg font-semibold text-primary">
                {formatCurrency(
                  processedData[processedData.length - 1]?.cumulative || 0
                )}
              </div>
              <div className="text-sm text-muted-foreground">Net Result</div>
            </div>
          </div>

          {/* Detailed Breakdown */}
          <div className="mt-6">
            <h4 className="text-sm font-medium mb-3">Detailed Breakdown</h4>
            <div className="space-y-2">
              {processedData.map((item, index) => (
                <motion.div
                  key={item.name}
                  initial={{ opacity: 0, x: -20 }}
                  animate={{ opacity: 1, x: 0 }}
                  transition={{ duration: 0.2, delay: index * 0.1 }}
                  className="flex items-center justify-between py-2 px-3 rounded-lg hover:bg-muted/50 transition-colors"
                >
                  <div className="flex items-center gap-3">
                    {getIcon(item.type)}
                    <span className="text-sm font-medium">{item.name}</span>
                  </div>
                  <div className="flex items-center gap-4">
                    <span className={cn(
                      "text-sm font-medium",
                      item.type === 'positive' && "text-green-600",
                      item.type === 'negative' && "text-red-600",
                      item.type === 'total' && "text-primary"
                    )}>
                      {item.type === 'positive' ? '+' : item.type === 'negative' ? '-' : ''}
                      {formatCurrency(Math.abs(item.displayValue))}
                    </span>
                    <span className="text-sm text-muted-foreground min-w-24 text-right">
                      {formatCurrency(item.cumulative)}
                    </span>
                  </div>
                </motion.div>
              ))}
            </div>
          </div>
        </CardContent>
      </Card>
    </motion.div>
  );
}