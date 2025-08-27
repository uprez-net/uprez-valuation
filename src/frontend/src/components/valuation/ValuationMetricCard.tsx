'use client';

import React from 'react';
import { motion } from 'framer-motion';
import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/card';
import { formatCurrency, formatPercentage, cn } from '@/lib/utils';
import { TrendingUp, TrendingDown, Minus, Info } from 'lucide-react';
import * as Tooltip from '@radix-ui/react-tooltip';

interface ValuationMetricCardProps {
  title: string;
  value: number;
  previousValue?: number;
  format?: 'currency' | 'percentage' | 'number';
  currency?: string;
  precision?: number;
  icon?: React.ReactNode;
  description?: string;
  isLoading?: boolean;
  className?: string;
  size?: 'sm' | 'md' | 'lg';
  showTrend?: boolean;
  trendPeriod?: string;
  confidenceLevel?: number;
  onClick?: () => void;
}

export function ValuationMetricCard({
  title,
  value,
  previousValue,
  format = 'currency',
  currency = 'USD',
  precision = 0,
  icon,
  description,
  isLoading = false,
  className,
  size = 'md',
  showTrend = true,
  trendPeriod = '30d',
  confidenceLevel,
  onClick,
}: ValuationMetricCardProps) {
  const formatValue = (val: number) => {
    switch (format) {
      case 'currency':
        return formatCurrency(val, currency);
      case 'percentage':
        return formatPercentage(val, precision);
      case 'number':
        return new Intl.NumberFormat('en-US', {
          minimumFractionDigits: precision,
          maximumFractionDigits: precision,
        }).format(val);
      default:
        return val.toString();
    }
  };

  const calculateTrend = () => {
    if (!previousValue || previousValue === 0) return null;
    
    const change = value - previousValue;
    const percentageChange = (change / previousValue) * 100;
    
    return {
      change,
      percentageChange,
      direction: change > 0 ? 'up' : change < 0 ? 'down' : 'neutral',
    };
  };

  const trend = showTrend && previousValue !== undefined ? calculateTrend() : null;

  const sizeClasses = {
    sm: 'p-4',
    md: 'p-6',
    lg: 'p-8',
  };

  const titleSizes = {
    sm: 'text-sm',
    md: 'text-base',
    lg: 'text-lg',
  };

  const valueSizes = {
    sm: 'text-xl',
    md: 'text-2xl',
    lg: 'text-3xl',
  };

  if (isLoading) {
    return (
      <Card className={cn('animate-pulse', className)}>
        <CardHeader className={sizeClasses[size]}>
          <div className="flex items-center justify-between">
            <div className="h-4 bg-muted rounded w-24"></div>
            {icon && <div className="h-5 w-5 bg-muted rounded"></div>}
          </div>
        </CardHeader>
        <CardContent className={cn(sizeClasses[size], 'pt-0')}>
          <div className={cn('h-8 bg-muted rounded w-32', valueSizes[size])}></div>
          {showTrend && (
            <div className="flex items-center mt-2">
              <div className="h-3 bg-muted rounded w-16"></div>
            </div>
          )}
        </CardContent>
      </Card>
    );
  }

  const TrendIcon = () => {
    if (!trend) return null;
    
    switch (trend.direction) {
      case 'up':
        return <TrendingUp className="h-3 w-3 text-green-500" />;
      case 'down':
        return <TrendingDown className="h-3 w-3 text-red-500" />;
      default:
        return <Minus className="h-3 w-3 text-gray-500" />;
    }
  };

  const getTrendColor = () => {
    if (!trend) return 'text-muted-foreground';
    
    switch (trend.direction) {
      case 'up':
        return 'text-green-600';
      case 'down':
        return 'text-red-600';
      default:
        return 'text-muted-foreground';
    }
  };

  return (
    <motion.div
      initial={{ opacity: 0, y: 20 }}
      animate={{ opacity: 1, y: 0 }}
      transition={{ duration: 0.3 }}
    >
      <Card 
        className={cn(
          'transition-all duration-200 hover:shadow-md',
          onClick && 'cursor-pointer hover:shadow-lg',
          className
        )}
        onClick={onClick}
      >
        <CardHeader className={sizeClasses[size]}>
          <div className="flex items-center justify-between">
            <CardTitle className={cn('font-medium text-muted-foreground', titleSizes[size])}>
              {title}
            </CardTitle>
            <div className="flex items-center gap-2">
              {confidenceLevel && (
                <Tooltip.Provider>
                  <Tooltip.Root>
                    <Tooltip.Trigger asChild>
                      <div className="flex items-center text-xs text-muted-foreground">
                        <Info className="h-3 w-3 mr-1" />
                        {formatPercentage(confidenceLevel, 0)}
                      </div>
                    </Tooltip.Trigger>
                    <Tooltip.Portal>
                      <Tooltip.Content className="bg-popover text-popover-foreground px-3 py-2 rounded-md shadow-md text-sm max-w-xs">
                        Confidence Level: This metric has a {formatPercentage(confidenceLevel, 1)} confidence interval
                        <Tooltip.Arrow className="fill-popover" />
                      </Tooltip.Content>
                    </Tooltip.Portal>
                  </Tooltip.Root>
                </Tooltip.Provider>
              )}
              {icon && <div className="text-muted-foreground">{icon}</div>}
            </div>
          </div>
          {description && (
            <p className="text-xs text-muted-foreground mt-1">{description}</p>
          )}
        </CardHeader>
        
        <CardContent className={cn(sizeClasses[size], 'pt-0')}>
          <motion.div
            key={value}
            initial={{ scale: 1.1, opacity: 0.8 }}
            animate={{ scale: 1, opacity: 1 }}
            transition={{ duration: 0.2 }}
            className={cn('font-semibold tracking-tight', valueSizes[size])}
          >
            {formatValue(value)}
          </motion.div>
          
          {trend && (
            <div className={cn('flex items-center mt-2 text-sm', getTrendColor())}>
              <TrendIcon />
              <span className="ml-1">
                {Math.abs(trend.percentageChange).toFixed(1)}% vs {trendPeriod} ago
              </span>
            </div>
          )}
        </CardContent>
      </Card>
    </motion.div>
  );
}