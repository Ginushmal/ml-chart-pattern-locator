'use client';

import dynamic from 'next/dynamic';
import { Suspense } from 'react';

const ChartContainer = dynamic(() => import('@/components/ChartContainer'), {
  ssr: false,
  loading: () => (
    <div className="flex items-center justify-center h-[500px] bg-gray-800 rounded-lg">
      <div className="animate-spin rounded-full h-12 w-12 border-b-2 border-blue-500"></div>
    </div>
  ),
});

export default function Home() {
  return (
    <main className="container mx-auto px-4 py-8">
      <h1 className="text-3xl font-bold mb-8">Pattern Detection System</h1>
      <Suspense fallback={
        <div className="flex items-center justify-center h-[500px] bg-gray-800 rounded-lg">
          <div className="animate-spin rounded-full h-12 w-12 border-b-2 border-blue-500"></div>
        </div>
      }>
        <ChartContainer />
      </Suspense>
    </main>
  );
}
