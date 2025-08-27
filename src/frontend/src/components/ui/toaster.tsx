'use client';

import React from 'react';
import * as Toast from '@radix-ui/react-toast';
import { cn } from '@/lib/utils';
import { X, CheckCircle, AlertCircle, Info, AlertTriangle } from 'lucide-react';

export interface ToastProps extends React.ComponentPropsWithoutRef<typeof Toast.Root> {
  title?: string;
  description?: string;
  variant?: 'default' | 'destructive' | 'success' | 'warning';
}

const toastVariants = {
  default: {
    className: 'bg-background text-foreground border border-border',
    icon: Info,
  },
  destructive: {
    className: 'bg-destructive text-destructive-foreground border-destructive',
    icon: AlertCircle,
  },
  success: {
    className: 'bg-green-500 text-white border-green-500',
    icon: CheckCircle,
  },
  warning: {
    className: 'bg-yellow-500 text-white border-yellow-500',
    icon: AlertTriangle,
  },
};

const ToastComponent = React.forwardRef<
  React.ElementRef<typeof Toast.Root>,
  ToastProps
>(({ className, variant = 'default', title, description, ...props }, ref) => {
  const { className: variantClassName, icon: Icon } = toastVariants[variant];

  return (
    <Toast.Root
      ref={ref}
      className={cn(
        'group pointer-events-auto relative flex w-full items-center justify-between space-x-2 overflow-hidden rounded-md p-4 pr-6 shadow-lg transition-all',
        'data-[swipe=cancel]:translate-x-0 data-[swipe=end]:translate-x-[var(--radix-toast-swipe-end-x)] data-[swipe=move]:translate-x-[var(--radix-toast-swipe-move-x)] data-[swipe=move]:transition-none',
        'data-[state=open]:animate-in data-[state=open]:slide-in-from-right-full data-[state=open]:sm:slide-in-from-bottom-full',
        'data-[state=closed]:animate-out data-[state=closed]:fade-out-80 data-[state=closed]:slide-out-to-right-full data-[state=closed]:sm:slide-out-to-bottom-full',
        variantClassName,
        className
      )}
      {...props}
    >
      <div className="flex items-start space-x-3">
        <Icon className="h-5 w-5 mt-0.5 flex-shrink-0" />
        <div className="flex-1">
          {title && (
            <Toast.Title className="text-sm font-semibold">
              {title}
            </Toast.Title>
          )}
          {description && (
            <Toast.Description className="text-sm opacity-90 mt-1">
              {description}
            </Toast.Description>
          )}
        </div>
      </div>
      <Toast.Close className="absolute right-1 top-1 rounded-md p-1 text-foreground/50 opacity-0 transition-opacity hover:text-foreground focus:opacity-100 focus:outline-none focus:ring-1 group-hover:opacity-100 group-[.destructive]:text-red-300 group-[.destructive]:hover:text-red-50">
        <X className="h-4 w-4" />
      </Toast.Close>
    </Toast.Root>
  );
});

ToastComponent.displayName = Toast.Root.displayName;

export { ToastComponent as Toast };

// Hook for using toast notifications
export function useToast() {
  const [toasts, setToasts] = React.useState<ToastProps[]>([]);

  const addToast = React.useCallback((toast: ToastProps) => {
    const id = Math.random().toString(36).substr(2, 9);
    setToasts(prev => [...prev, { ...toast, key: id }]);
    
    // Auto remove after 5 seconds
    setTimeout(() => {
      setToasts(prev => prev.filter(t => t.key !== id));
    }, 5000);
  }, []);

  const removeToast = React.useCallback((key: string) => {
    setToasts(prev => prev.filter(t => t.key !== key));
  }, []);

  const toast = React.useCallback(
    (props: ToastProps) => addToast(props),
    [addToast]
  );

  return {
    toast,
    toasts,
    removeToast,
  };
}

export function Toaster() {
  const { toasts, removeToast } = useToast();

  return (
    <>
      {toasts.map((toast) => (
        <ToastComponent
          key={toast.key}
          {...toast}
          onOpenChange={(open) => {
            if (!open && toast.key) {
              removeToast(toast.key);
            }
          }}
        />
      ))}
    </>
  );
}