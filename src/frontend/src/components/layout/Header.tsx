'use client';

import React, { useState } from 'react';
import { useRouter } from 'next/navigation';
import { motion } from 'framer-motion';
import { 
  Search, 
  Bell, 
  Settings, 
  Moon, 
  Sun,
  Plus,
  Filter,
  Download,
  User
} from 'lucide-react';
import { Button } from '@/components/ui/button';
import { Input } from '@/components/ui/input';
import * as Avatar from '@radix-ui/react-avatar';
import * as Popover from '@radix-ui/react-popover';
import * as DropdownMenu from '@radix-ui/react-dropdown-menu';
import * as Badge from '@radix-ui/react-badge';
import { cn } from '@/lib/utils';
import { useAuthStore } from '@/stores/auth';
import { useThemeStore } from '@/stores/theme';
import { useCollaborationStore } from '@/stores/collaboration';

export function Header() {
  const [searchQuery, setSearchQuery] = useState('');
  const [isSearchFocused, setIsSearchFocused] = useState(false);
  const router = useRouter();
  
  const { user, logout } = useAuthStore();
  const { theme, setTheme } = useThemeStore();
  const { connectedUsers, isConnected } = useCollaborationStore();

  const handleSearch = (e: React.FormEvent) => {
    e.preventDefault();
    if (searchQuery.trim()) {
      router.push(`/search?q=${encodeURIComponent(searchQuery.trim())}`);
    }
  };

  const handleLogout = async () => {
    await logout();
    router.push('/auth/login');
  };

  const notifications = [
    {
      id: '1',
      title: 'Document Processed',
      message: 'Prospectus for TechCorp IPO has been analyzed',
      time: '2 minutes ago',
      unread: true,
    },
    {
      id: '2',
      title: 'Valuation Complete',
      message: 'Bull case scenario for DataFlow Inc. is ready',
      time: '15 minutes ago',
      unread: true,
    },
    {
      id: '3',
      title: 'Team Collaboration',
      message: 'John added comments to your valuation model',
      time: '1 hour ago',
      unread: false,
    },
  ];

  const unreadCount = notifications.filter(n => n.unread).length;

  return (
    <header className="bg-card border-b border-border px-6 py-4">
      <div className="flex items-center justify-between">
        {/* Left Section - Search */}
        <div className="flex items-center space-x-4 flex-1 max-w-md">
          <motion.form
            initial={{ width: '200px' }}
            animate={{ width: isSearchFocused ? '300px' : '200px' }}
            transition={{ duration: 0.2 }}
            onSubmit={handleSearch}
            className="relative"
          >
            <Search className="absolute left-3 top-1/2 transform -translate-y-1/2 h-4 w-4 text-muted-foreground" />
            <Input
              type="text"
              placeholder="Search projects, documents..."
              value={searchQuery}
              onChange={(e) => setSearchQuery(e.target.value)}
              onFocus={() => setIsSearchFocused(true)}
              onBlur={() => setIsSearchFocused(false)}
              className="pl-10 w-full"
            />
          </motion.form>

          <Button variant="outline" size="sm">
            <Filter className="h-4 w-4 mr-2" />
            Filters
          </Button>
        </div>

        {/* Right Section - Actions & User */}
        <div className="flex items-center space-x-2">
          {/* Quick Actions */}
          <DropdownMenu.Root>
            <DropdownMenu.Trigger asChild>
              <Button variant="outline" size="sm">
                <Plus className="h-4 w-4 mr-2" />
                New
              </Button>
            </DropdownMenu.Trigger>
            <DropdownMenu.Portal>
              <DropdownMenu.Content className="bg-popover border rounded-md shadow-lg p-1 z-50">
                <DropdownMenu.Item 
                  className="px-3 py-2 text-sm cursor-pointer hover:bg-accent rounded"
                  onClick={() => router.push('/projects/new')}
                >
                  New Project
                </DropdownMenu.Item>
                <DropdownMenu.Item 
                  className="px-3 py-2 text-sm cursor-pointer hover:bg-accent rounded"
                  onClick={() => router.push('/valuations/prospectus')}
                >
                  Valuation with Prospectus
                </DropdownMenu.Item>
                <DropdownMenu.Item 
                  className="px-3 py-2 text-sm cursor-pointer hover:bg-accent rounded"
                  onClick={() => router.push('/valuations/manual')}
                >
                  Manual Valuation
                </DropdownMenu.Item>
                <DropdownMenu.Separator className="h-px bg-border my-1" />
                <DropdownMenu.Item 
                  className="px-3 py-2 text-sm cursor-pointer hover:bg-accent rounded"
                  onClick={() => router.push('/reports/generate')}
                >
                  Generate Report
                </DropdownMenu.Item>
              </DropdownMenu.Content>
            </DropdownMenu.Portal>
          </DropdownMenu.Root>

          <Button variant="outline" size="sm">
            <Download className="h-4 w-4 mr-2" />
            Export
          </Button>

          {/* Theme Toggle */}
          <Button
            variant="ghost"
            size="icon"
            onClick={() => setTheme(theme === 'light' ? 'dark' : 'light')}
          >
            {theme === 'light' ? (
              <Moon className="h-4 w-4" />
            ) : (
              <Sun className="h-4 w-4" />
            )}
          </Button>

          {/* Notifications */}
          <Popover.Root>
            <Popover.Trigger asChild>
              <Button variant="ghost" size="icon" className="relative">
                <Bell className="h-4 w-4" />
                {unreadCount > 0 && (
                  <span className="absolute -top-1 -right-1 h-5 w-5 bg-red-500 text-white text-xs rounded-full flex items-center justify-center">
                    {unreadCount}
                  </span>
                )}
              </Button>
            </Popover.Trigger>
            <Popover.Portal>
              <Popover.Content className="bg-popover border rounded-lg shadow-lg w-80 z-50" side="bottom" align="end">
                <div className="p-4 border-b border-border">
                  <h3 className="font-semibold text-sm">Notifications</h3>
                </div>
                <div className="max-h-96 overflow-y-auto">
                  {notifications.map((notification) => (
                    <div
                      key={notification.id}
                      className={cn(
                        'p-4 border-b border-border last:border-b-0 hover:bg-accent/50 cursor-pointer',
                        notification.unread && 'bg-accent/20'
                      )}
                    >
                      <div className="flex justify-between items-start mb-1">
                        <h4 className="font-medium text-sm">{notification.title}</h4>
                        {notification.unread && (
                          <div className="w-2 h-2 bg-blue-500 rounded-full mt-1" />
                        )}
                      </div>
                      <p className="text-sm text-muted-foreground mb-1">
                        {notification.message}
                      </p>
                      <span className="text-xs text-muted-foreground">
                        {notification.time}
                      </span>
                    </div>
                  ))}
                </div>
                <div className="p-2 border-t border-border">
                  <Button variant="ghost" size="sm" className="w-full">
                    View All Notifications
                  </Button>
                </div>
              </Popover.Content>
            </Popover.Portal>
          </Popover.Root>

          {/* Collaboration Status */}
          {isConnected && connectedUsers.length > 0 && (
            <Popover.Root>
              <Popover.Trigger asChild>
                <Button variant="ghost" size="sm" className="px-2">
                  <div className="flex -space-x-2">
                    {connectedUsers.slice(0, 3).map((collaborator, index) => (
                      <Avatar.Root 
                        key={collaborator.id}
                        className="h-6 w-6 rounded-full border-2 border-background"
                      >
                        <Avatar.Image
                          src={collaborator.avatar}
                          alt={collaborator.name}
                          className="rounded-full"
                        />
                        <Avatar.Fallback className="bg-primary text-primary-foreground text-xs rounded-full">
                          {collaborator.name.charAt(0).toUpperCase()}
                        </Avatar.Fallback>
                      </Avatar.Root>
                    ))}
                    {connectedUsers.length > 3 && (
                      <div className="h-6 w-6 rounded-full bg-muted border-2 border-background flex items-center justify-center text-xs">
                        +{connectedUsers.length - 3}
                      </div>
                    )}
                  </div>
                  <span className="ml-2 text-sm">{connectedUsers.length} online</span>
                </Button>
              </Popover.Trigger>
              <Popover.Portal>
                <Popover.Content className="bg-popover border rounded-lg shadow-lg w-64 z-50" side="bottom" align="end">
                  <div className="p-3">
                    <h4 className="font-medium text-sm mb-3">Active Collaborators</h4>
                    <div className="space-y-2">
                      {connectedUsers.map((collaborator) => (
                        <div key={collaborator.id} className="flex items-center space-x-2">
                          <Avatar.Root className="h-8 w-8 rounded-full">
                            <Avatar.Image
                              src={collaborator.avatar}
                              alt={collaborator.name}
                              className="rounded-full"
                            />
                            <Avatar.Fallback className="bg-primary text-primary-foreground text-xs rounded-full">
                              {collaborator.name.charAt(0).toUpperCase()}
                            </Avatar.Fallback>
                          </Avatar.Root>
                          <div className="flex-1">
                            <p className="text-sm font-medium">{collaborator.name}</p>
                            <p className="text-xs text-muted-foreground">
                              Active {collaborator.lastSeen}
                            </p>
                          </div>
                          <div className="w-2 h-2 bg-green-500 rounded-full" />
                        </div>
                      ))}
                    </div>
                  </div>
                </Popover.Content>
              </Popover.Portal>
            </Popover.Root>
          )}

          {/* User Menu */}
          <DropdownMenu.Root>
            <DropdownMenu.Trigger asChild>
              <Button variant="ghost" className="p-2">
                <div className="flex items-center space-x-2">
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
                  <div className="text-left hidden sm:block">
                    <p className="text-sm font-medium">{user?.name}</p>
                    <p className="text-xs text-muted-foreground capitalize">{user?.role}</p>
                  </div>
                </div>
              </Button>
            </DropdownMenu.Trigger>
            <DropdownMenu.Portal>
              <DropdownMenu.Content className="bg-popover border rounded-md shadow-lg p-1 w-48 z-50" side="bottom" align="end">
                <div className="px-2 py-2 border-b border-border">
                  <p className="text-sm font-medium">{user?.name}</p>
                  <p className="text-xs text-muted-foreground">{user?.email}</p>
                </div>
                <DropdownMenu.Item 
                  className="px-2 py-2 text-sm cursor-pointer hover:bg-accent rounded flex items-center"
                  onClick={() => router.push('/profile')}
                >
                  <User className="h-4 w-4 mr-2" />
                  Profile
                </DropdownMenu.Item>
                <DropdownMenu.Item 
                  className="px-2 py-2 text-sm cursor-pointer hover:bg-accent rounded flex items-center"
                  onClick={() => router.push('/settings')}
                >
                  <Settings className="h-4 w-4 mr-2" />
                  Settings
                </DropdownMenu.Item>
                <DropdownMenu.Separator className="h-px bg-border my-1" />
                <DropdownMenu.Item 
                  className="px-2 py-2 text-sm cursor-pointer hover:bg-accent rounded text-red-600"
                  onClick={handleLogout}
                >
                  Logout
                </DropdownMenu.Item>
              </DropdownMenu.Content>
            </DropdownMenu.Portal>
          </DropdownMenu.Root>
        </div>
      </div>
    </header>
  );
}