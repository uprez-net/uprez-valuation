'use client';

import React, { useState } from 'react';
import Link from 'next/link';
import { useRouter, usePathname } from 'next/navigation';
import { motion, AnimatePresence } from 'framer-motion';
import { 
  Home, 
  FolderOpen, 
  BarChart3, 
  FileText, 
  Settings, 
  Users, 
  HelpCircle,
  ChevronLeft,
  ChevronRight,
  TrendingUp,
  Calculator,
  Upload,
  Download,
  Bell,
  User,
  LogOut
} from 'lucide-react';
import { cn } from '@/lib/utils';
import { useAuthStore } from '@/stores/auth';
import { useThemeStore } from '@/stores/theme';
import { Button } from '@/components/ui/button';
import * as Avatar from '@radix-ui/react-avatar';
import * as Popover from '@radix-ui/react-popover';
import * as Separator from '@radix-ui/react-separator';

interface NavigationItem {
  name: string;
  href: string;
  icon: React.ComponentType<{ className?: string }>;
  badge?: number;
  children?: NavigationItem[];
}

const navigation: NavigationItem[] = [
  {
    name: 'Dashboard',
    href: '/dashboard',
    icon: Home,
  },
  {
    name: 'Projects',
    href: '/projects',
    icon: FolderOpen,
    children: [
      { name: 'All Projects', href: '/projects', icon: FolderOpen },
      { name: 'Create New', href: '/projects/new', icon: Upload },
    ],
  },
  {
    name: 'Valuations',
    href: '/valuations',
    icon: Calculator,
    children: [
      { name: 'With Prospectus', href: '/valuations/prospectus', icon: FileText },
      { name: 'Without Prospectus', href: '/valuations/manual', icon: TrendingUp },
    ],
  },
  {
    name: 'Reports',
    href: '/reports',
    icon: BarChart3,
    children: [
      { name: 'Generate Report', href: '/reports/generate', icon: FileText },
      { name: 'Export Data', href: '/reports/export', icon: Download },
    ],
  },
  {
    name: 'Documents',
    href: '/documents',
    icon: FileText,
  },
];

const bottomNavigation: NavigationItem[] = [
  {
    name: 'Settings',
    href: '/settings',
    icon: Settings,
  },
  {
    name: 'Help',
    href: '/help',
    icon: HelpCircle,
  },
];

interface SidebarProps {
  className?: string;
}

export function Sidebar({ className }: SidebarProps) {
  const [isCollapsed, setIsCollapsed] = useState(false);
  const [expandedItems, setExpandedItems] = useState<string[]>([]);
  const pathname = usePathname();
  const router = useRouter();
  
  const { user, logout } = useAuthStore();
  const { branding } = useThemeStore();

  const toggleExpanded = (itemName: string) => {
    setExpandedItems(prev =>
      prev.includes(itemName)
        ? prev.filter(name => name !== itemName)
        : [...prev, itemName]
    );
  };

  const handleLogout = async () => {
    await logout();
    router.push('/auth/login');
  };

  const renderNavigationItem = (item: NavigationItem, depth = 0) => {
    const isActive = pathname === item.href || pathname.startsWith(item.href + '/');
    const isExpanded = expandedItems.includes(item.name);
    const hasChildren = item.children && item.children.length > 0;

    return (
      <div key={item.name}>
        <div
          className={cn(
            'relative flex items-center group',
            depth > 0 && 'ml-4 pl-4 border-l border-muted'
          )}
        >
          {hasChildren ? (
            <button
              onClick={() => toggleExpanded(item.name)}
              className={cn(
                'flex items-center w-full px-3 py-2 text-sm rounded-lg transition-all duration-200',
                'hover:bg-accent hover:text-accent-foreground',
                isActive && 'bg-primary text-primary-foreground',
                isCollapsed && 'justify-center px-2'
              )}
            >
              <item.icon className={cn('h-5 w-5', !isCollapsed && 'mr-3')} />
              {!isCollapsed && (
                <>
                  <span className="flex-1 text-left">{item.name}</span>
                  <motion.div
                    animate={{ rotate: isExpanded ? 90 : 0 }}
                    transition={{ duration: 0.2 }}
                  >
                    <ChevronRight className="h-4 w-4" />
                  </motion.div>
                </>
              )}
              {item.badge && !isCollapsed && (
                <span className="ml-auto bg-primary text-primary-foreground text-xs rounded-full px-2 py-1">
                  {item.badge}
                </span>
              )}
            </button>
          ) : (
            <Link
              href={item.href}
              className={cn(
                'flex items-center w-full px-3 py-2 text-sm rounded-lg transition-all duration-200',
                'hover:bg-accent hover:text-accent-foreground',
                isActive && 'bg-primary text-primary-foreground',
                isCollapsed && 'justify-center px-2'
              )}
            >
              <item.icon className={cn('h-5 w-5', !isCollapsed && 'mr-3')} />
              {!isCollapsed && <span>{item.name}</span>}
              {item.badge && !isCollapsed && (
                <span className="ml-auto bg-primary text-primary-foreground text-xs rounded-full px-2 py-1">
                  {item.badge}
                </span>
              )}
            </Link>
          )}

          {/* Tooltip for collapsed state */}
          {isCollapsed && (
            <div className="absolute left-full ml-2 px-2 py-1 bg-popover text-popover-foreground text-sm rounded-md shadow-lg opacity-0 group-hover:opacity-100 transition-opacity duration-200 z-50 whitespace-nowrap">
              {item.name}
            </div>
          )}
        </div>

        {/* Children */}
        <AnimatePresence>
          {hasChildren && isExpanded && !isCollapsed && (
            <motion.div
              initial={{ opacity: 0, height: 0 }}
              animate={{ opacity: 1, height: 'auto' }}
              exit={{ opacity: 0, height: 0 }}
              transition={{ duration: 0.2 }}
              className="overflow-hidden"
            >
              <div className="py-1">
                {item.children?.map(child => renderNavigationItem(child, depth + 1))}
              </div>
            </motion.div>
          )}
        </AnimatePresence>
      </div>
    );
  };

  return (
    <motion.aside
      initial={{ width: isCollapsed ? '4rem' : '16rem' }}
      animate={{ width: isCollapsed ? '4rem' : '16rem' }}
      transition={{ duration: 0.3, ease: 'easeInOut' }}
      className={cn(
        'flex flex-col h-screen bg-card border-r border-border',
        className
      )}
    >
      {/* Header */}
      <div className="flex items-center justify-between p-4 border-b border-border">
        {!isCollapsed && (
          <motion.div
            initial={{ opacity: 0 }}
            animate={{ opacity: 1 }}
            exit={{ opacity: 0 }}
            className="flex items-center gap-2"
          >
            {branding.logo ? (
              <img src={branding.logo} alt="Logo" className="h-8 w-8" />
            ) : (
              <div className="h-8 w-8 bg-primary rounded-lg flex items-center justify-center">
                <TrendingUp className="h-5 w-5 text-primary-foreground" />
              </div>
            )}
            <span className="text-lg font-semibold">{branding.companyName}</span>
          </motion.div>
        )}
        
        <Button
          variant="ghost"
          size="icon"
          onClick={() => setIsCollapsed(!isCollapsed)}
          className="ml-auto"
        >
          {isCollapsed ? (
            <ChevronRight className="h-4 w-4" />
          ) : (
            <ChevronLeft className="h-4 w-4" />
          )}
        </Button>
      </div>

      {/* Navigation */}
      <nav className="flex-1 p-4 overflow-y-auto">
        <div className="space-y-2">
          {navigation.map(item => renderNavigationItem(item))}
        </div>
      </nav>

      {/* Bottom Section */}
      <div className="p-4 border-t border-border">
        {/* Bottom Navigation */}
        <div className="space-y-2 mb-4">
          {bottomNavigation.map(item => renderNavigationItem(item))}
        </div>

        {/* User Profile */}
        <Popover.Root>
          <Popover.Trigger asChild>
            <button
              className={cn(
                'flex items-center w-full p-2 rounded-lg hover:bg-accent transition-colors',
                isCollapsed && 'justify-center'
              )}
            >
              <Avatar.Root className="h-8 w-8 rounded-full">
                <Avatar.Image
                  src={user?.avatar}
                  alt={user?.name}
                  className="rounded-full"
                />
                <Avatar.Fallback className="bg-primary text-primary-foreground text-sm rounded-full">
                  {user?.name?.charAt(0)?.toUpperCase() || 'U'}
                </Avatar.Fallback>
              </Avatar.Root>
              
              {!isCollapsed && (
                <div className="ml-3 text-left flex-1">
                  <p className="text-sm font-medium">{user?.name}</p>
                  <p className="text-xs text-muted-foreground capitalize">{user?.role}</p>
                </div>
              )}
            </button>
          </Popover.Trigger>
          
          <Popover.Portal>
            <Popover.Content
              className="bg-popover border rounded-lg shadow-lg p-1 w-48 z-50"
              side="top"
              align="end"
            >
              <div className="p-2">
                <p className="text-sm font-medium">{user?.name}</p>
                <p className="text-xs text-muted-foreground">{user?.email}</p>
              </div>
              
              <Separator.Root className="h-px bg-border my-1" />
              
              <Link
                href="/profile"
                className="flex items-center w-full px-2 py-2 text-sm rounded hover:bg-accent"
              >
                <User className="h-4 w-4 mr-2" />
                Profile
              </Link>
              
              <Link
                href="/notifications"
                className="flex items-center w-full px-2 py-2 text-sm rounded hover:bg-accent"
              >
                <Bell className="h-4 w-4 mr-2" />
                Notifications
              </Link>
              
              <Separator.Root className="h-px bg-border my-1" />
              
              <button
                onClick={handleLogout}
                className="flex items-center w-full px-2 py-2 text-sm rounded hover:bg-accent text-red-600"
              >
                <LogOut className="h-4 w-4 mr-2" />
                Logout
              </button>
            </Popover.Content>
          </Popover.Portal>
        </Popover.Root>
      </div>
    </motion.aside>
  );
}