'use client';

import React, { useState } from 'react';
import { motion, AnimatePresence } from 'framer-motion';
import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/card';
import { Button } from '@/components/ui/button';
import { Input } from '@/components/ui/input';
import * as Slider from '@radix-ui/react-slider';
import * as Select from '@radix-ui/react-select';
import * as Tabs from '@radix-ui/react-tabs';
import { 
  Plus, 
  Trash2, 
  Copy, 
  TrendingUp, 
  TrendingDown, 
  BarChart3,
  Calculator,
  ChevronDown,
  Save,
  Play
} from 'lucide-react';
import { formatCurrency, formatPercentage, cn } from '@/lib/utils';
import { ValuationScenario } from '@/types';
import { InteractiveValuationChart } from './InteractiveValuationChart';

interface ScenarioModelerProps {
  scenarios: ValuationScenario[];
  onScenariosChange: (scenarios: ValuationScenario[]) => void;
  onRunValuation: (scenarios: ValuationScenario[]) => void;
  baseMetrics: {
    currentRevenue: number;
    currentEbitda: number;
    sharesOutstanding: number;
  };
  isCalculating?: boolean;
  className?: string;
}

const scenarioTypes = [
  { value: 'base', label: 'Base Case', color: '#0ea5e9' },
  { value: 'bull', label: 'Bull Case', color: '#10b981' },
  { value: 'bear', label: 'Bear Case', color: '#ef4444' },
  { value: 'custom', label: 'Custom', color: '#8b5cf6' },
];

const defaultScenario: Omit<ValuationScenario, 'id'> = {
  name: 'New Scenario',
  type: 'custom',
  assumptions: {
    revenueGrowth: 0.15,
    ebitdaMargin: 0.20,
    discountRate: 0.10,
    terminalGrowthRate: 0.03,
    multipleRange: {
      min: 8,
      max: 12,
    },
  },
  results: {
    dcfValue: 0,
    comparableValue: 0,
    impliedValue: 0,
    confidence: 0.75,
  },
};

export function ScenarioModeler({
  scenarios,
  onScenariosChange,
  onRunValuation,
  baseMetrics,
  isCalculating = false,
  className,
}: ScenarioModelerProps) {
  const [activeScenario, setActiveScenario] = useState(scenarios[0]?.id || '');
  const [isExpanded, setIsExpanded] = useState(true);

  const addScenario = (type: ValuationScenario['type'] = 'custom') => {
    const presetAssumptions = {
      base: {
        revenueGrowth: 0.15,
        ebitdaMargin: 0.20,
        discountRate: 0.10,
        terminalGrowthRate: 0.03,
        multipleRange: { min: 8, max: 12 },
      },
      bull: {
        revenueGrowth: 0.25,
        ebitdaMargin: 0.25,
        discountRate: 0.08,
        terminalGrowthRate: 0.04,
        multipleRange: { min: 12, max: 16 },
      },
      bear: {
        revenueGrowth: 0.08,
        ebitdaMargin: 0.15,
        discountRate: 0.12,
        terminalGrowthRate: 0.02,
        multipleRange: { min: 6, max: 9 },
      },
      custom: defaultScenario.assumptions,
    };

    const newScenario: ValuationScenario = {
      id: `scenario_${Date.now()}`,
      name: `${scenarioTypes.find(s => s.value === type)?.label || 'Custom'} Scenario`,
      type,
      assumptions: presetAssumptions[type],
      results: {
        dcfValue: 0,
        comparableValue: 0,
        impliedValue: 0,
        confidence: 0.75,
      },
    };

    const updatedScenarios = [...scenarios, newScenario];
    onScenariosChange(updatedScenarios);
    setActiveScenario(newScenario.id);
  };

  const updateScenario = (id: string, updates: Partial<ValuationScenario>) => {
    const updatedScenarios = scenarios.map(scenario =>
      scenario.id === id ? { ...scenario, ...updates } : scenario
    );
    onScenariosChange(updatedScenarios);
  };

  const duplicateScenario = (id: string) => {
    const scenario = scenarios.find(s => s.id === id);
    if (!scenario) return;

    const duplicated: ValuationScenario = {
      ...scenario,
      id: `scenario_${Date.now()}`,
      name: `${scenario.name} (Copy)`,
      type: 'custom',
    };

    const updatedScenarios = [...scenarios, duplicated];
    onScenariosChange(updatedScenarios);
    setActiveScenario(duplicated.id);
  };

  const deleteScenario = (id: string) => {
    if (scenarios.length <= 1) return;
    
    const updatedScenarios = scenarios.filter(s => s.id !== id);
    onScenariosChange(updatedScenarios);
    
    if (activeScenario === id) {
      setActiveScenario(updatedScenarios[0]?.id || '');
    }
  };

  const activeScenarioData = scenarios.find(s => s.id === activeScenario);

  const getScenarioColor = (type: ValuationScenario['type']) => {
    return scenarioTypes.find(s => s.value === type)?.color || '#8b5cf6';
  };

  const calculateImpliedValue = (scenario: ValuationScenario) => {
    const { assumptions } = scenario;
    const projectedRevenue = baseMetrics.currentRevenue * (1 + assumptions.revenueGrowth);
    const projectedEbitda = projectedRevenue * assumptions.ebitdaMargin;
    
    // Simple DCF calculation (normally would be more complex)
    const terminalValue = projectedEbitda * (1 + assumptions.terminalGrowthRate) / 
      (assumptions.discountRate - assumptions.terminalGrowthRate);
    
    const dcfValue = terminalValue / Math.pow(1 + assumptions.discountRate, 5);
    
    // Comparable multiple valuation
    const averageMultiple = (assumptions.multipleRange.min + assumptions.multipleRange.max) / 2;
    const comparableValue = projectedEbitda * averageMultiple;
    
    // Weighted average
    const impliedValue = (dcfValue * 0.6) + (comparableValue * 0.4);
    
    return {
      dcfValue,
      comparableValue,
      impliedValue: impliedValue / baseMetrics.sharesOutstanding, // Per share
    };
  };

  const chartData = scenarios.map(scenario => {
    const values = calculateImpliedValue(scenario);
    return {
      label: scenario.name,
      value: values.impliedValue,
      category: scenario.type,
    };
  });

  const scenarioChartData = scenarios.map((scenario, index) => ({
    name: scenario.name,
    data: chartData,
    color: getScenarioColor(scenario.type),
  }));

  return (
    <motion.div
      initial={{ opacity: 0, y: 20 }}
      animate={{ opacity: 1, y: 0 }}
      transition={{ duration: 0.3 }}
      className={className}
    >
      <Card>
        <CardHeader>
          <div className="flex items-center justify-between">
            <CardTitle className="flex items-center gap-2">
              <Calculator className="h-5 w-5" />
              Scenario Modeling
            </CardTitle>
            
            <div className="flex items-center gap-2">
              <Select.Root onValueChange={(value) => addScenario(value as ValuationScenario['type'])}>
                <Select.Trigger asChild>
                  <Button variant="outline" size="sm">
                    <Plus className="h-4 w-4 mr-2" />
                    Add Scenario
                    <ChevronDown className="h-4 w-4 ml-2" />
                  </Button>
                </Select.Trigger>
                <Select.Portal>
                  <Select.Content className="bg-popover border rounded-md shadow-lg z-50">
                    <Select.Viewport className="p-1">
                      {scenarioTypes.map((type) => (
                        <Select.Item
                          key={type.value}
                          value={type.value}
                          className="px-3 py-2 text-sm cursor-pointer hover:bg-accent rounded flex items-center gap-2"
                        >
                          <div 
                            className="w-3 h-3 rounded-full" 
                            style={{ backgroundColor: type.color }}
                          />
                          <Select.ItemText>{type.label}</Select.ItemText>
                        </Select.Item>
                      ))}
                    </Select.Viewport>
                  </Select.Content>
                </Select.Portal>
              </Select.Root>

              <Button
                onClick={() => onRunValuation(scenarios)}
                disabled={isCalculating}
                className="bg-primary hover:bg-primary/90"
              >
                {isCalculating ? (
                  <>
                    <motion.div
                      animate={{ rotate: 360 }}
                      transition={{ duration: 1, repeat: Infinity, ease: "linear" }}
                    >
                      <Calculator className="h-4 w-4 mr-2" />
                    </motion.div>
                    Calculating...
                  </>
                ) : (
                  <>
                    <Play className="h-4 w-4 mr-2" />
                    Run Valuation
                  </>
                )}
              </Button>
            </div>
          </div>
        </CardHeader>

        <CardContent>
          <Tabs.Root value={activeScenario} onValueChange={setActiveScenario}>
            {/* Scenario Tabs */}
            <Tabs.List className="grid w-full grid-cols-1 sm:grid-cols-2 lg:grid-cols-4 gap-2 mb-6">
              {scenarios.map((scenario) => (
                <Tabs.Trigger
                  key={scenario.id}
                  value={scenario.id}
                  asChild
                >
                  <motion.div
                    whileHover={{ scale: 1.02 }}
                    whileTap={{ scale: 0.98 }}
                    className={cn(
                      "p-4 rounded-lg border-2 cursor-pointer transition-all",
                      activeScenario === scenario.id
                        ? "border-primary bg-primary/10"
                        : "border-muted hover:border-primary/50"
                    )}
                  >
                    <div className="flex items-center justify-between mb-2">
                      <div
                        className="w-3 h-3 rounded-full"
                        style={{ backgroundColor: getScenarioColor(scenario.type) }}
                      />
                      <div className="flex items-center gap-1">
                        <Button
                          variant="ghost"
                          size="icon"
                          className="h-6 w-6"
                          onClick={(e) => {
                            e.stopPropagation();
                            duplicateScenario(scenario.id);
                          }}
                        >
                          <Copy className="h-3 w-3" />
                        </Button>
                        <Button
                          variant="ghost"
                          size="icon"
                          className="h-6 w-6 text-red-600 hover:text-red-700"
                          onClick={(e) => {
                            e.stopPropagation();
                            deleteScenario(scenario.id);
                          }}
                          disabled={scenarios.length <= 1}
                        >
                          <Trash2 className="h-3 w-3" />
                        </Button>
                      </div>
                    </div>
                    
                    <h4 className="font-medium text-sm mb-1">{scenario.name}</h4>
                    <p className="text-xs text-muted-foreground mb-2">
                      {scenarioTypes.find(s => s.value === scenario.type)?.label}
                    </p>
                    
                    {scenario.results.impliedValue > 0 && (
                      <div className="text-sm font-semibold">
                        {formatCurrency(scenario.results.impliedValue)}
                      </div>
                    )}
                  </motion.div>
                </Tabs.Trigger>
              ))}
            </Tabs.List>

            {/* Scenario Configuration */}
            <AnimatePresence mode="wait">
              {scenarios.map((scenario) => (
                <Tabs.Content key={scenario.id} value={scenario.id} asChild>
                  <motion.div
                    initial={{ opacity: 0, y: 20 }}
                    animate={{ opacity: 1, y: 0 }}
                    exit={{ opacity: 0, y: -20 }}
                    transition={{ duration: 0.2 }}
                    className="space-y-6"
                  >
                    {/* Scenario Name */}
                    <div>
                      <label className="text-sm font-medium mb-2 block">
                        Scenario Name
                      </label>
                      <Input
                        value={scenario.name}
                        onChange={(e) => updateScenario(scenario.id, { name: e.target.value })}
                        className="max-w-sm"
                      />
                    </div>

                    {/* Assumptions Grid */}
                    <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-6">
                      {/* Revenue Growth */}
                      <div className="space-y-3">
                        <label className="text-sm font-medium flex items-center gap-2">
                          <TrendingUp className="h-4 w-4" />
                          Revenue Growth Rate
                        </label>
                        <div className="space-y-2">
                          <Slider.Root
                            className="relative flex items-center select-none touch-none w-full h-5"
                            value={[scenario.assumptions.revenueGrowth * 100]}
                            onValueChange={([value]) =>
                              updateScenario(scenario.id, {
                                assumptions: {
                                  ...scenario.assumptions,
                                  revenueGrowth: value / 100,
                                },
                              })
                            }
                            max={50}
                            min={-10}
                            step={1}
                          >
                            <Slider.Track className="bg-muted relative grow rounded-full h-[3px]">
                              <Slider.Range className="absolute bg-primary rounded-full h-full" />
                            </Slider.Track>
                            <Slider.Thumb className="block w-5 h-5 bg-primary rounded-full shadow-lg focus:outline-none focus:shadow-xl" />
                          </Slider.Root>
                          <div className="flex justify-between text-xs text-muted-foreground">
                            <span>-10%</span>
                            <span className="font-medium">
                              {formatPercentage(scenario.assumptions.revenueGrowth)}
                            </span>
                            <span>50%</span>
                          </div>
                        </div>
                      </div>

                      {/* EBITDA Margin */}
                      <div className="space-y-3">
                        <label className="text-sm font-medium flex items-center gap-2">
                          <BarChart3 className="h-4 w-4" />
                          EBITDA Margin
                        </label>
                        <div className="space-y-2">
                          <Slider.Root
                            className="relative flex items-center select-none touch-none w-full h-5"
                            value={[scenario.assumptions.ebitdaMargin * 100]}
                            onValueChange={([value]) =>
                              updateScenario(scenario.id, {
                                assumptions: {
                                  ...scenario.assumptions,
                                  ebitdaMargin: value / 100,
                                },
                              })
                            }
                            max={40}
                            min={0}
                            step={0.5}
                          >
                            <Slider.Track className="bg-muted relative grow rounded-full h-[3px]">
                              <Slider.Range className="absolute bg-primary rounded-full h-full" />
                            </Slider.Track>
                            <Slider.Thumb className="block w-5 h-5 bg-primary rounded-full shadow-lg focus:outline-none focus:shadow-xl" />
                          </Slider.Root>
                          <div className="flex justify-between text-xs text-muted-foreground">
                            <span>0%</span>
                            <span className="font-medium">
                              {formatPercentage(scenario.assumptions.ebitdaMargin)}
                            </span>
                            <span>40%</span>
                          </div>
                        </div>
                      </div>

                      {/* Discount Rate */}
                      <div className="space-y-3">
                        <label className="text-sm font-medium flex items-center gap-2">
                          <TrendingDown className="h-4 w-4" />
                          Discount Rate (WACC)
                        </label>
                        <div className="space-y-2">
                          <Slider.Root
                            className="relative flex items-center select-none touch-none w-full h-5"
                            value={[scenario.assumptions.discountRate * 100]}
                            onValueChange={([value]) =>
                              updateScenario(scenario.id, {
                                assumptions: {
                                  ...scenario.assumptions,
                                  discountRate: value / 100,
                                },
                              })
                            }
                            max={20}
                            min={5}
                            step={0.1}
                          >
                            <Slider.Track className="bg-muted relative grow rounded-full h-[3px]">
                              <Slider.Range className="absolute bg-primary rounded-full h-full" />
                            </Slider.Track>
                            <Slider.Thumb className="block w-5 h-5 bg-primary rounded-full shadow-lg focus:outline-none focus:shadow-xl" />
                          </Slider.Root>
                          <div className="flex justify-between text-xs text-muted-foreground">
                            <span>5%</span>
                            <span className="font-medium">
                              {formatPercentage(scenario.assumptions.discountRate)}
                            </span>
                            <span>20%</span>
                          </div>
                        </div>
                      </div>

                      {/* Terminal Growth Rate */}
                      <div className="space-y-3">
                        <label className="text-sm font-medium">
                          Terminal Growth Rate
                        </label>
                        <div className="space-y-2">
                          <Slider.Root
                            className="relative flex items-center select-none touch-none w-full h-5"
                            value={[scenario.assumptions.terminalGrowthRate * 100]}
                            onValueChange={([value]) =>
                              updateScenario(scenario.id, {
                                assumptions: {
                                  ...scenario.assumptions,
                                  terminalGrowthRate: value / 100,
                                },
                              })
                            }
                            max={6}
                            min={0}
                            step={0.1}
                          >
                            <Slider.Track className="bg-muted relative grow rounded-full h-[3px]">
                              <Slider.Range className="absolute bg-primary rounded-full h-full" />
                            </Slider.Track>
                            <Slider.Thumb className="block w-5 h-5 bg-primary rounded-full shadow-lg focus:outline-none focus:shadow-xl" />
                          </Slider.Root>
                          <div className="flex justify-between text-xs text-muted-foreground">
                            <span>0%</span>
                            <span className="font-medium">
                              {formatPercentage(scenario.assumptions.terminalGrowthRate)}
                            </span>
                            <span>6%</span>
                          </div>
                        </div>
                      </div>

                      {/* Multiple Range */}
                      <div className="space-y-3">
                        <label className="text-sm font-medium">
                          EV/EBITDA Multiple Range
                        </label>
                        <div className="space-y-2">
                          <Slider.Root
                            className="relative flex items-center select-none touch-none w-full h-5"
                            value={[scenario.assumptions.multipleRange.min, scenario.assumptions.multipleRange.max]}
                            onValueChange={([min, max]) =>
                              updateScenario(scenario.id, {
                                assumptions: {
                                  ...scenario.assumptions,
                                  multipleRange: { min, max },
                                },
                              })
                            }
                            max={25}
                            min={3}
                            step={0.5}
                          >
                            <Slider.Track className="bg-muted relative grow rounded-full h-[3px]">
                              <Slider.Range className="absolute bg-primary rounded-full h-full" />
                            </Slider.Track>
                            <Slider.Thumb className="block w-5 h-5 bg-primary rounded-full shadow-lg focus:outline-none focus:shadow-xl" />
                            <Slider.Thumb className="block w-5 h-5 bg-primary rounded-full shadow-lg focus:outline-none focus:shadow-xl" />
                          </Slider.Root>
                          <div className="flex justify-between text-xs text-muted-foreground">
                            <span>3x</span>
                            <span className="font-medium">
                              {scenario.assumptions.multipleRange.min}x - {scenario.assumptions.multipleRange.max}x
                            </span>
                            <span>25x</span>
                          </div>
                        </div>
                      </div>
                    </div>

                    {/* Results Display */}
                    {scenario.results.impliedValue > 0 && (
                      <motion.div
                        initial={{ opacity: 0, y: 10 }}
                        animate={{ opacity: 1, y: 0 }}
                        className="grid grid-cols-1 md:grid-cols-3 gap-4 p-4 bg-muted/50 rounded-lg"
                      >
                        <div className="text-center">
                          <div className="text-2xl font-bold">
                            {formatCurrency(scenario.results.dcfValue)}
                          </div>
                          <div className="text-sm text-muted-foreground">DCF Value</div>
                        </div>
                        <div className="text-center">
                          <div className="text-2xl font-bold">
                            {formatCurrency(scenario.results.comparableValue)}
                          </div>
                          <div className="text-sm text-muted-foreground">Comparable Value</div>
                        </div>
                        <div className="text-center">
                          <div className="text-2xl font-bold text-primary">
                            {formatCurrency(scenario.results.impliedValue)}
                          </div>
                          <div className="text-sm text-muted-foreground">
                            Implied Value (per share)
                          </div>
                        </div>
                      </motion.div>
                    )}
                  </motion.div>
                </Tabs.Content>
              ))}
            </AnimatePresence>
          </Tabs.Root>

          {/* Scenario Comparison Chart */}
          {scenarios.length > 1 && scenarios.some(s => s.results.impliedValue > 0) && (
            <div className="mt-8">
              <InteractiveValuationChart
                data={chartData}
                title="Scenario Comparison"
                type="bar"
                scenarios={scenarioChartData}
                height={300}
              />
            </div>
          )}
        </CardContent>
      </Card>
    </motion.div>
  );
}