"use client";

import { useEffect, useState } from "react";

export default function Home() {
  const [data, setData] = useState<{ message: string; status?: string } | null>(null);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    // 백엔드 API 호출 테스트
    fetch("http://localhost:8000/api/health")
      .then((res) => res.json())
      .then((data) => {
        setData(data);
        setLoading(false);
      })
      .catch((err) => {
        console.error("Backend fetch error:", err);
        setLoading(false);
      });
  }, []);

  return (
    <main className="flex min-h-screen flex-col items-center justify-center p-24 bg-gray-50">
      <div className="z-10 max-w-5xl w-full items-center justify-between font-mono text-sm">
        <h1 className="text-4xl font-bold mb-8 text-center text-blue-600">
          Cloud Automation Chatbot
        </h1>
        
        <div className="bg-white p-8 rounded-xl shadow-md border border-gray-200">
          <h2 className="text-xl font-semibold mb-4 border-b pb-2">Backend Connection Test</h2>
          {loading ? (
            <p className="text-gray-500">Connecting to FastAPI...</p>
          ) : data ? (
            <div className="space-y-2">
              <p className="text-green-600 font-medium flex items-center">
                <span className="mr-2">●</span> Status: {data.status}
              </p>
              <p className="text-gray-700">Message: {data.message}</p>
            </div>
          ) : (
            <p className="text-red-500">Failed to connect to backend (Check if port 8000 is running)</p>
          )}
        </div>

        <div className="mt-8 grid grid-cols-1 md:grid-cols-2 gap-4">
          <div className="p-4 border rounded bg-white">
            <h3 className="font-bold mb-2">Frontend</h3>
            <p className="text-xs text-gray-600">Next.js (App Router), Tailwind CSS</p>
          </div>
          <div className="p-4 border rounded bg-white">
            <h3 className="font-bold mb-2">Backend</h3>
            <p className="text-xs text-gray-600">FastAPI, Uvicorn, Python venv</p>
          </div>
        </div>
      </div>
    </main>
  );
}
