import type { Metadata } from "next";
import { Outfit, JetBrains_Mono } from "next/font/google";
import "./globals.css";
import { cn } from "@/lib/utils";

const outfit = Outfit({ 
  subsets: ["latin"], 
  variable: "--font-outfit",
  display: "swap"
});

const mono = JetBrains_Mono({ 
  subsets: ["latin"], 
  variable: "--font-mono",
  display: "swap"
});

export const metadata: Metadata = {
  title: "Lingual | Multilingual Document Intelligence",
  description: "Explainable Hybrid Summarization Engine",
};

export default function RootLayout({
  children,
}: Readonly<{
  children: React.ReactNode;
}>) {
  return (
    <html lang="en">
      <body className={cn(
        "min-h-screen bg-slate-950 font-sans antialiased text-slate-100 selection:bg-cyan-500/30 selection:text-cyan-200",
        outfit.variable,
        mono.variable
      )}>
        {children}
      </body>
    </html>
  );
}
