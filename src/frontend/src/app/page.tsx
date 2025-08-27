'use client';

import React, { useEffect } from 'react';
import { useRouter } from 'next/navigation';
import { motion } from 'framer-motion';
import Link from 'next/link';
import { 
  TrendingUp, 
  BarChart3, 
  FileText, 
  Shield, 
  Zap, 
  Users,
  ArrowRight,
  CheckCircle,
  Star,
  Globe,
  MessageCircle
} from 'lucide-react';
import { Button } from '@/components/ui/button';
import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/card';
import { useAuthStore } from '@/stores/auth';

const features = [
  {
    icon: <BarChart3 className="h-8 w-8" />,
    title: 'AI-Powered Analysis',
    description: 'Leverage advanced AI algorithms to analyze IPO prospectuses and generate comprehensive valuations automatically.',
  },
  {
    icon: <FileText className="h-8 w-8" />,
    title: 'Document Processing',
    description: 'Upload and process prospectuses, financial statements, and research reports with intelligent data extraction.',
  },
  {
    icon: <TrendingUp className="h-8 w-8" />,
    title: 'Scenario Modeling',
    description: 'Create multiple valuation scenarios with customizable assumptions and sensitivity analysis.',
  },
  {
    icon: <Users className="h-8 w-8" />,
    title: 'Real-time Collaboration',
    description: 'Work together with your team in real-time with live updates and collaborative editing features.',
  },
  {
    icon: <Shield className="h-8 w-8" />,
    title: 'Enterprise Security',
    description: 'Bank-grade security with SOC 2 compliance, end-to-end encryption, and audit trails.',
  },
  {
    icon: <Zap className="h-8 w-8" />,
    title: 'Fast & Accurate',
    description: 'Generate detailed valuation reports in minutes, not days, with 95%+ accuracy compared to manual analysis.',
  },
];

const testimonials = [
  {
    name: 'Sarah Johnson',
    role: 'VP of Investments',
    company: 'Goldman Sachs',
    content: 'Uprez has transformed how we evaluate IPO opportunities. The AI-driven insights are incredibly accurate.',
    rating: 5,
  },
  {
    name: 'Michael Chen',
    role: 'Senior Analyst',
    company: 'Morgan Stanley',
    content: 'The platform reduced our IPO analysis time by 70% while improving the quality of our valuations.',
    rating: 5,
  },
  {
    name: 'Emily Rodriguez',
    role: 'Portfolio Manager',
    company: 'BlackRock',
    content: 'Outstanding collaboration features and real-time updates. Our team productivity has increased significantly.',
    rating: 5,
  },
];

const stats = [
  { value: '500+', label: 'IPOs Analyzed' },
  { value: '95%', label: 'Accuracy Rate' },
  { value: '70%', label: 'Time Saved' },
  { value: '50+', label: 'Enterprise Clients' },
];

export default function LandingPage() {
  const router = useRouter();
  const { isAuthenticated } = useAuthStore();

  useEffect(() => {
    if (isAuthenticated) {
      router.push('/dashboard');
    }
  }, [isAuthenticated, router]);

  if (isAuthenticated) {
    return null; // Will redirect
  }

  return (
    <div className="min-h-screen bg-gradient-to-br from-blue-50 via-white to-cyan-50 dark:from-gray-900 dark:via-gray-800 dark:to-gray-900">
      {/* Navigation */}
      <nav className="border-b border-border/40 bg-background/80 backdrop-blur supports-[backdrop-filter]:bg-background/60">
        <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
          <div className="flex justify-between items-center py-4">
            <div className="flex items-center space-x-2">
              <div className="h-8 w-8 bg-primary rounded-lg flex items-center justify-center">
                <TrendingUp className="h-5 w-5 text-primary-foreground" />
              </div>
              <span className="text-xl font-bold">Uprez</span>
            </div>
            
            <div className="hidden md:flex items-center space-x-8">
              <Link href="#features" className="text-muted-foreground hover:text-foreground">Features</Link>
              <Link href="#pricing" className="text-muted-foreground hover:text-foreground">Pricing</Link>
              <Link href="#about" className="text-muted-foreground hover:text-foreground">About</Link>
              <Link href="#contact" className="text-muted-foreground hover:text-foreground">Contact</Link>
            </div>
            
            <div className="flex items-center space-x-4">
              <Link href="/auth/login">
                <Button variant="ghost">Sign In</Button>
              </Link>
              <Link href="/auth/register">
                <Button>Get Started</Button>
              </Link>
            </div>
          </div>
        </div>
      </nav>

      {/* Hero Section */}
      <section className="relative py-20 lg:py-32">
        <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 text-center">
          <motion.div
            initial={{ opacity: 0, y: 20 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ duration: 0.6 }}
          >
            <h1 className="text-4xl lg:text-6xl font-bold tracking-tight mb-6">
              AI-Powered IPO{' '}
              <span className="text-primary">Valuation Platform</span>
            </h1>
            <p className="text-xl text-muted-foreground mb-8 max-w-3xl mx-auto">
              Transform your IPO analysis with cutting-edge AI. Upload prospectuses, 
              generate comprehensive valuations, and collaborate with your team in real-time.
            </p>
            <div className="flex flex-col sm:flex-row gap-4 justify-center">
              <Link href="/auth/register">
                <Button size="lg" className="text-lg px-8 py-3">
                  Start Free Trial
                  <ArrowRight className="ml-2 h-5 w-5" />
                </Button>
              </Link>
              <Button variant="outline" size="lg" className="text-lg px-8 py-3">
                Watch Demo
              </Button>
            </div>
          </motion.div>

          {/* Stats */}
          <motion.div
            initial={{ opacity: 0, y: 20 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ duration: 0.6, delay: 0.3 }}
            className="grid grid-cols-2 lg:grid-cols-4 gap-8 mt-16"
          >
            {stats.map((stat, index) => (
              <div key={index} className="text-center">
                <div className="text-3xl lg:text-4xl font-bold text-primary mb-2">
                  {stat.value}
                </div>
                <div className="text-muted-foreground">{stat.label}</div>
              </div>
            ))}
          </motion.div>
        </div>
      </section>

      {/* Features Section */}
      <section id="features" className="py-20 bg-background/50">
        <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
          <div className="text-center mb-16">
            <h2 className="text-3xl lg:text-4xl font-bold mb-4">
              Everything you need for IPO analysis
            </h2>
            <p className="text-xl text-muted-foreground max-w-2xl mx-auto">
              Powerful features designed for institutional investors and financial analysts
            </p>
          </div>

          <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-8">
            {features.map((feature, index) => (
              <motion.div
                key={index}
                initial={{ opacity: 0, y: 20 }}
                animate={{ opacity: 1, y: 0 }}
                transition={{ duration: 0.6, delay: index * 0.1 }}
              >
                <Card className="h-full hover:shadow-lg transition-shadow">
                  <CardHeader>
                    <div className="text-primary mb-4">{feature.icon}</div>
                    <CardTitle className="text-xl">{feature.title}</CardTitle>
                  </CardHeader>
                  <CardContent>
                    <p className="text-muted-foreground">{feature.description}</p>
                  </CardContent>
                </Card>
              </motion.div>
            ))}
          </div>
        </div>
      </section>

      {/* Testimonials */}
      <section className="py-20">
        <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
          <div className="text-center mb-16">
            <h2 className="text-3xl lg:text-4xl font-bold mb-4">
              Trusted by leading financial institutions
            </h2>
            <p className="text-xl text-muted-foreground">
              See what our clients say about Uprez
            </p>
          </div>

          <div className="grid grid-cols-1 md:grid-cols-3 gap-8">
            {testimonials.map((testimonial, index) => (
              <motion.div
                key={index}
                initial={{ opacity: 0, y: 20 }}
                animate={{ opacity: 1, y: 0 }}
                transition={{ duration: 0.6, delay: index * 0.1 }}
              >
                <Card className="h-full">
                  <CardContent className="pt-6">
                    <div className="flex mb-4">
                      {[...Array(testimonial.rating)].map((_, i) => (
                        <Star key={i} className="h-5 w-5 text-yellow-400 fill-current" />
                      ))}
                    </div>
                    <blockquote className="text-muted-foreground mb-6">
                      "{testimonial.content}"
                    </blockquote>
                    <div>
                      <div className="font-semibold">{testimonial.name}</div>
                      <div className="text-sm text-muted-foreground">
                        {testimonial.role} at {testimonial.company}
                      </div>
                    </div>
                  </CardContent>
                </Card>
              </motion.div>
            ))}
          </div>
        </div>
      </section>

      {/* CTA Section */}
      <section className="py-20 bg-primary text-primary-foreground">
        <div className="max-w-4xl mx-auto px-4 sm:px-6 lg:px-8 text-center">
          <motion.div
            initial={{ opacity: 0, y: 20 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ duration: 0.6 }}
          >
            <h2 className="text-3xl lg:text-4xl font-bold mb-6">
              Ready to transform your IPO analysis?
            </h2>
            <p className="text-xl mb-8 opacity-90">
              Join hundreds of analysts who trust Uprez for accurate, fast IPO valuations.
            </p>
            <div className="flex flex-col sm:flex-row gap-4 justify-center">
              <Link href="/auth/register">
                <Button size="lg" variant="secondary" className="text-lg px-8 py-3">
                  Start Free Trial
                  <ArrowRight className="ml-2 h-5 w-5" />
                </Button>
              </Link>
              <Button size="lg" variant="outline" className="text-lg px-8 py-3 border-white text-white hover:bg-white hover:text-primary">
                <MessageCircle className="mr-2 h-5 w-5" />
                Talk to Sales
              </Button>
            </div>
          </motion.div>
        </div>
      </section>

      {/* Footer */}
      <footer className="bg-muted py-12">
        <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
          <div className="grid grid-cols-1 md:grid-cols-4 gap-8">
            <div className="col-span-1 md:col-span-2">
              <div className="flex items-center space-x-2 mb-4">
                <div className="h-8 w-8 bg-primary rounded-lg flex items-center justify-center">
                  <TrendingUp className="h-5 w-5 text-primary-foreground" />
                </div>
                <span className="text-xl font-bold">Uprez</span>
              </div>
              <p className="text-muted-foreground mb-4 max-w-sm">
                The most advanced AI-powered IPO valuation platform for institutional investors.
              </p>
              <div className="flex items-center space-x-4">
                <Globe className="h-5 w-5 text-muted-foreground" />
                <span className="text-sm text-muted-foreground">
                  Enterprise-grade security • SOC 2 Compliant
                </span>
              </div>
            </div>
            
            <div>
              <h3 className="font-semibold mb-4">Product</h3>
              <ul className="space-y-2 text-muted-foreground">
                <li><Link href="#features" className="hover:text-foreground">Features</Link></li>
                <li><Link href="#pricing" className="hover:text-foreground">Pricing</Link></li>
                <li><Link href="#security" className="hover:text-foreground">Security</Link></li>
                <li><Link href="#api" className="hover:text-foreground">API</Link></li>
              </ul>
            </div>
            
            <div>
              <h3 className="font-semibold mb-4">Company</h3>
              <ul className="space-y-2 text-muted-foreground">
                <li><Link href="#about" className="hover:text-foreground">About</Link></li>
                <li><Link href="#contact" className="hover:text-foreground">Contact</Link></li>
                <li><Link href="#careers" className="hover:text-foreground">Careers</Link></li>
                <li><Link href="#blog" className="hover:text-foreground">Blog</Link></li>
              </ul>
            </div>
          </div>
          
          <div className="border-t border-border mt-8 pt-8 flex flex-col md:flex-row justify-between items-center">
            <p className="text-muted-foreground text-sm">
              © 2024 Uprez. All rights reserved.
            </p>
            <div className="flex space-x-6 mt-4 md:mt-0">
              <Link href="/privacy" className="text-muted-foreground hover:text-foreground text-sm">
                Privacy Policy
              </Link>
              <Link href="/terms" className="text-muted-foreground hover:text-foreground text-sm">
                Terms of Service
              </Link>
            </div>
          </div>
        </div>
      </footer>
    </div>
  );
}