'use client';

import React, { useState } from 'react';
import { useRouter } from 'next/navigation';
import Link from 'next/link';
import { motion } from 'framer-motion';
import { useForm } from 'react-hook-form';
import { zodResolver } from '@hookform/resolvers/zod';
import { z } from 'zod';
import { 
  Eye, 
  EyeOff, 
  Mail, 
  Lock, 
  User, 
  Building2,
  TrendingUp, 
  ArrowRight,
  Loader2,
  Check
} from 'lucide-react';
import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/card';
import { Button } from '@/components/ui/button';
import { Input } from '@/components/ui/input';
import { cn } from '@/lib/utils';
import { useAuthStore } from '@/stores/auth';
import { useToast } from '@/components/ui/toaster';

const registerSchema = z.object({
  name: z.string().min(2, 'Name must be at least 2 characters'),
  email: z.string().email('Please enter a valid email address'),
  password: z.string().min(8, 'Password must be at least 8 characters')
    .regex(/^(?=.*[a-z])(?=.*[A-Z])(?=.*\d)/, 'Password must contain at least one uppercase letter, one lowercase letter, and one number'),
  confirmPassword: z.string(),
  organizationName: z.string().min(2, 'Organization name must be at least 2 characters'),
  agreeToTerms: z.boolean().refine(val => val === true, 'You must agree to the terms and conditions'),
}).refine((data) => data.password === data.confirmPassword, {
  message: "Passwords don't match",
  path: ["confirmPassword"],
});

type RegisterFormData = z.infer<typeof registerSchema>;

const passwordRequirements = [
  { test: (password: string) => password.length >= 8, text: 'At least 8 characters' },
  { test: (password: string) => /[A-Z]/.test(password), text: 'One uppercase letter' },
  { test: (password: string) => /[a-z]/.test(password), text: 'One lowercase letter' },
  { test: (password: string) => /\d/.test(password), text: 'One number' },
];

export default function RegisterPage() {
  const [showPassword, setShowPassword] = useState(false);
  const [showConfirmPassword, setShowConfirmPassword] = useState(false);
  const router = useRouter();
  const { register: registerUser, isLoading, error, clearError } = useAuthStore();
  const { toast } = useToast();

  const {
    register,
    handleSubmit,
    watch,
    formState: { errors, isSubmitting },
  } = useForm<RegisterFormData>({
    resolver: zodResolver(registerSchema),
    defaultValues: {
      name: '',
      email: '',
      password: '',
      confirmPassword: '',
      organizationName: '',
      agreeToTerms: false,
    },
  });

  const password = watch('password');

  const onSubmit = async (data: RegisterFormData) => {
    try {
      clearError();
      await registerUser({
        name: data.name,
        email: data.email,
        password: data.password,
        organizationName: data.organizationName,
      });
      
      toast({
        title: 'Account Created!',
        description: 'Welcome to Uprez Valuation. Your account has been successfully created.',
        variant: 'success',
      });
      
      router.push('/dashboard');
    } catch (error: any) {
      toast({
        title: 'Registration Failed',
        description: error.message || 'Please check your information and try again.',
        variant: 'destructive',
      });
    }
  };

  return (
    <div className="min-h-screen flex items-center justify-center bg-gradient-to-br from-blue-50 via-white to-cyan-50 dark:from-gray-900 dark:via-gray-800 dark:to-gray-900 p-4">
      <motion.div
        initial={{ opacity: 0, y: 20 }}
        animate={{ opacity: 1, y: 0 }}
        transition={{ duration: 0.5 }}
        className="w-full max-w-md"
      >
        {/* Logo and Brand */}
        <div className="text-center mb-8">
          <motion.div
            initial={{ scale: 0.8 }}
            animate={{ scale: 1 }}
            transition={{ duration: 0.5, delay: 0.1 }}
            className="inline-flex items-center justify-center w-16 h-16 bg-primary rounded-xl shadow-lg mb-4"
          >
            <TrendingUp className="h-8 w-8 text-primary-foreground" />
          </motion.div>
          <h1 className="text-2xl font-bold text-foreground">Join Uprez</h1>
          <p className="text-muted-foreground mt-2">
            Create your account to start analyzing IPO valuations
          </p>
        </div>

        <Card className="shadow-xl border-0">
          <CardHeader className="space-y-1 pb-4">
            <CardTitle className="text-xl text-center">Create Account</CardTitle>
          </CardHeader>
          <CardContent>
            <form onSubmit={handleSubmit(onSubmit)} className="space-y-4">
              {/* Name Field */}
              <div className="space-y-2">
                <label htmlFor="name" className="text-sm font-medium">
                  Full Name
                </label>
                <div className="relative">
                  <User className="absolute left-3 top-1/2 transform -translate-y-1/2 h-4 w-4 text-muted-foreground" />
                  <Input
                    id="name"
                    type="text"
                    placeholder="John Doe"
                    className={cn(
                      'pl-10',
                      errors.name && 'border-red-500 focus:border-red-500'
                    )}
                    {...register('name')}
                  />
                </div>
                {errors.name && (
                  <p className="text-sm text-red-600">{errors.name.message}</p>
                )}
              </div>

              {/* Email Field */}
              <div className="space-y-2">
                <label htmlFor="email" className="text-sm font-medium">
                  Email Address
                </label>
                <div className="relative">
                  <Mail className="absolute left-3 top-1/2 transform -translate-y-1/2 h-4 w-4 text-muted-foreground" />
                  <Input
                    id="email"
                    type="email"
                    placeholder="you@company.com"
                    className={cn(
                      'pl-10',
                      errors.email && 'border-red-500 focus:border-red-500'
                    )}
                    {...register('email')}
                  />
                </div>
                {errors.email && (
                  <p className="text-sm text-red-600">{errors.email.message}</p>
                )}
              </div>

              {/* Organization Name */}
              <div className="space-y-2">
                <label htmlFor="organizationName" className="text-sm font-medium">
                  Organization Name
                </label>
                <div className="relative">
                  <Building2 className="absolute left-3 top-1/2 transform -translate-y-1/2 h-4 w-4 text-muted-foreground" />
                  <Input
                    id="organizationName"
                    type="text"
                    placeholder="Your company name"
                    className={cn(
                      'pl-10',
                      errors.organizationName && 'border-red-500 focus:border-red-500'
                    )}
                    {...register('organizationName')}
                  />
                </div>
                {errors.organizationName && (
                  <p className="text-sm text-red-600">{errors.organizationName.message}</p>
                )}
              </div>

              {/* Password Field */}
              <div className="space-y-2">
                <label htmlFor="password" className="text-sm font-medium">
                  Password
                </label>
                <div className="relative">
                  <Lock className="absolute left-3 top-1/2 transform -translate-y-1/2 h-4 w-4 text-muted-foreground" />
                  <Input
                    id="password"
                    type={showPassword ? 'text' : 'password'}
                    placeholder="Create a strong password"
                    className={cn(
                      'pl-10 pr-10',
                      errors.password && 'border-red-500 focus:border-red-500'
                    )}
                    {...register('password')}
                  />
                  <button
                    type="button"
                    onClick={() => setShowPassword(!showPassword)}
                    className="absolute right-3 top-1/2 transform -translate-y-1/2 text-muted-foreground hover:text-foreground"
                  >
                    {showPassword ? (
                      <EyeOff className="h-4 w-4" />
                    ) : (
                      <Eye className="h-4 w-4" />
                    )}
                  </button>
                </div>
                
                {/* Password Requirements */}
                {password && (
                  <div className="space-y-1">
                    {passwordRequirements.map((req, index) => (
                      <div key={index} className="flex items-center text-xs">
                        <div className={cn(
                          'mr-2 h-3 w-3 rounded-full flex items-center justify-center',
                          req.test(password) 
                            ? 'bg-green-500' 
                            : 'bg-muted'
                        )}>
                          {req.test(password) && <Check className="h-2 w-2 text-white" />}
                        </div>
                        <span className={cn(
                          req.test(password) 
                            ? 'text-green-600' 
                            : 'text-muted-foreground'
                        )}>
                          {req.text}
                        </span>
                      </div>
                    ))}
                  </div>
                )}
                
                {errors.password && (
                  <p className="text-sm text-red-600">{errors.password.message}</p>
                )}
              </div>

              {/* Confirm Password Field */}
              <div className="space-y-2">
                <label htmlFor="confirmPassword" className="text-sm font-medium">
                  Confirm Password
                </label>
                <div className="relative">
                  <Lock className="absolute left-3 top-1/2 transform -translate-y-1/2 h-4 w-4 text-muted-foreground" />
                  <Input
                    id="confirmPassword"
                    type={showConfirmPassword ? 'text' : 'password'}
                    placeholder="Confirm your password"
                    className={cn(
                      'pl-10 pr-10',
                      errors.confirmPassword && 'border-red-500 focus:border-red-500'
                    )}
                    {...register('confirmPassword')}
                  />
                  <button
                    type="button"
                    onClick={() => setShowConfirmPassword(!showConfirmPassword)}
                    className="absolute right-3 top-1/2 transform -translate-y-1/2 text-muted-foreground hover:text-foreground"
                  >
                    {showConfirmPassword ? (
                      <EyeOff className="h-4 w-4" />
                    ) : (
                      <Eye className="h-4 w-4" />
                    )}
                  </button>
                </div>
                {errors.confirmPassword && (
                  <p className="text-sm text-red-600">{errors.confirmPassword.message}</p>
                )}
              </div>

              {/* Terms and Conditions */}
              <div className="flex items-start">
                <input
                  id="agreeToTerms"
                  type="checkbox"
                  className={cn(
                    'h-4 w-4 text-primary focus:ring-primary border-gray-300 rounded mt-1',
                    errors.agreeToTerms && 'border-red-500'
                  )}
                  {...register('agreeToTerms')}
                />
                <label htmlFor="agreeToTerms" className="ml-2 block text-sm text-muted-foreground">
                  I agree to the{' '}
                  <Link href="/terms" className="text-primary hover:text-primary/80">
                    Terms of Service
                  </Link>{' '}
                  and{' '}
                  <Link href="/privacy" className="text-primary hover:text-primary/80">
                    Privacy Policy
                  </Link>
                </label>
              </div>
              {errors.agreeToTerms && (
                <p className="text-sm text-red-600">{errors.agreeToTerms.message}</p>
              )}

              {/* Error Message */}
              {error && (
                <motion.div
                  initial={{ opacity: 0, y: -10 }}
                  animate={{ opacity: 1, y: 0 }}
                  className="bg-red-50 dark:bg-red-900/20 text-red-600 dark:text-red-400 p-3 rounded-lg text-sm"
                >
                  {error}
                </motion.div>
              )}

              {/* Submit Button */}
              <Button
                type="submit"
                className="w-full"
                disabled={isSubmitting || isLoading}
              >
                {isSubmitting || isLoading ? (
                  <>
                    <Loader2 className="mr-2 h-4 w-4 animate-spin" />
                    Creating account...
                  </>
                ) : (
                  <>
                    Create Account
                    <ArrowRight className="ml-2 h-4 w-4" />
                  </>
                )}
              </Button>
            </form>

            {/* Divider */}
            <div className="relative my-6">
              <div className="absolute inset-0 flex items-center">
                <span className="w-full border-t border-muted" />
              </div>
              <div className="relative flex justify-center text-xs uppercase">
                <span className="bg-card px-2 text-muted-foreground">
                  Already have an account?
                </span>
              </div>
            </div>

            {/* Sign In Link */}
            <div className="text-center">
              <Link
                href="/auth/login"
                className="text-primary hover:text-primary/80 text-sm font-medium"
              >
                Sign in instead →
              </Link>
            </div>
          </CardContent>
        </Card>
      </motion.div>
    </div>
  );
}