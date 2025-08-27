import type { Metadata } from 'next';
import { Inter } from 'next/font/google';
import './globals.css';
import { Providers } from '@/components/providers';
import { cn } from '@/lib/utils';

const inter = Inter({ subsets: ['latin'] });

export const metadata: Metadata = {
  title: 'Uprez Valuation Platform',
  description: 'AI-powered IPO valuation analysis platform for institutional investors',
  keywords: ['IPO', 'valuation', 'financial analysis', 'investment', 'prospectus'],
  authors: [{ name: 'Uprez' }],
  viewport: 'width=device-width, initial-scale=1',
  themeColor: '#0ea5e9',
};

export default function RootLayout({
  children,
}: {
  children: React.ReactNode;
}) {
  return (
    <html lang="en" suppressHydrationWarning>
      <body className={cn(inter.className, 'antialiased min-h-screen bg-background')}>
        <Providers>
          {children}
        </Providers>
      </body>
    </html>
  );
}