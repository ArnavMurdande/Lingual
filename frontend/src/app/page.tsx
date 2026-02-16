"use client";

import { useState, useRef } from "react";
import {
  Upload,
  FileText,
  Zap,
  BarChart3,
  Clock,
  AlignLeft,
  CheckCircle2,
  Download,
  Settings2,
  Languages,
  FileOutput,
} from "lucide-react";
import { motion } from "framer-motion";
import { MetricCard } from "@/components/MetricCard";
import { HeatMapHighlighter } from "@/components/HeatMapHighlighter";
import { cn } from "@/lib/utils";

export default function Home() {
  const [file, setFile] = useState<File | null>(null);
  const [isProcessing, setIsProcessing] = useState(false);
  const [result, setResult] = useState<any>(null);
  const [textInput, setTextInput] = useState("");

  // Configuration State
  const [summarySize, setSummarySize] = useState<"small" | "medium" | "large">("medium");
  const [outputLanguage, setOutputLanguage] = useState("en");
  const [summaryMode, setSummaryMode] = useState("extractive");
  const [isDragging, setIsDragging] = useState(false);

  const fileInputRef = useRef<HTMLInputElement>(null);

  const handleFileUpload = (e: React.ChangeEvent<HTMLInputElement>) => {
    if (e.target.files?.[0]) {
      setFile(e.target.files[0]);
      setTextInput(""); // Clear text input if file selected
    }
  };

  const handleDragOver = (e: React.DragEvent) => {
    e.preventDefault();
    setIsDragging(true);
  };

  const handleDragLeave = (e: React.DragEvent) => {
    e.preventDefault();
    setIsDragging(false);
  };

  const handleDrop = (e: React.DragEvent) => {
    e.preventDefault();
    setIsDragging(false);
    
    if (e.dataTransfer.files && e.dataTransfer.files[0]) {
      const droppedFile = e.dataTransfer.files[0];
      const validTypes = ['text/plain', 'application/pdf', 'application/vnd.openxmlformats-officedocument.wordprocessingml.document'];
      
      // Basic validation (or just trust the backend/extension check)
      if (droppedFile.name.endsWith('.txt') || droppedFile.name.endsWith('.pdf') || droppedFile.name.endsWith('.docx')) {
        setFile(droppedFile);
        setTextInput("");
      } else {
        alert("Please upload PDF, DOCX, or TXT files only.");
      }
    }
  };

  const processFile = async () => {
    setIsProcessing(true);
    setResult(null);

    try {
      let data;
      const endpoint = file
        ? "http://localhost:8000/summarize/file"
        : "http://localhost:8000/summarize/text";

      const ratioMap = { small: 0.3, medium: 0.5, large: 0.7 };
      const ratio = ratioMap[summarySize];

      if (file) {
        const formData = new FormData();
        formData.append("file", file);
        // formData.append("sentences_count", "5"); // Default fallback
        formData.append("summary_ratio", ratio.toString());
        formData.append("language", outputLanguage);
        formData.append("mode", summaryMode);

        const res = await fetch(endpoint, {
          method: "POST",
          body: formData,
        });
        if (!res.ok) throw new Error(await res.text());
        data = await res.json();
      } else if (textInput.trim()) {
        const res = await fetch(endpoint, {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({
            text: textInput,
            // sentences_count: 5,
            summary_ratio: ratio,
            language: outputLanguage,
            mode: summaryMode,
          }),
        });
        if (!res.ok) throw new Error(await res.text());
        data = await res.json();
      }

      setResult(data);
    } catch (err: any) {
      console.error(err);
      alert("Error processing document: " + err.message);
    } finally {
      setIsProcessing(false);
    }
  };

  const downloadReport = async () => {
    if (!result) return;
    try {
      const res = await fetch("http://localhost:8000/export/word", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(result),
      });

      if (!res.ok) throw new Error("Export failed");

      const blob = await res.blob();
      const url = window.URL.createObjectURL(blob);
      const a = document.createElement("a");
      a.href = url;
      a.download = `Summary_Report_${new Date().toISOString().slice(0, 10)}.docx`;
      document.body.appendChild(a);
      a.click();
      window.URL.revokeObjectURL(url);
      document.body.removeChild(a);
    } catch (err) {
      console.error(err);
      alert("Failed to download report");
    }
  };

  return (
    <main className="min-h-screen bg-[radial-gradient(ellipse_at_top,_var(--tw-gradient-stops))] from-slate-900 via-slate-950 to-black text-slate-100 p-6 md:p-12 font-sans selection:bg-cyan-500/30">
      {/* Header */}
      <header className="max-w-7xl mx-auto mb-12 flex justify-between items-center">
        <div>
          <h1 className="text-4xl md:text-5xl font-bold tracking-tighter bg-gradient-to-r from-cyan-400 to-indigo-400 bg-clip-text text-transparent pb-2 pr-1">
            Lingual
          </h1>
          <p className="text-slate-400 mt-2 text-lg">
            Explainable Hybrid Summarization Platform
          </p>
        </div>
        <div className="flex items-center gap-2 px-3 py-1 bg-slate-900/50 rounded-full border border-slate-800 text-xs font-mono text-cyan-400">
          <div className="w-2 h-2 rounded-full bg-emerald-500 animate-pulse" />
          SYSTEM ONLINE (CPU-ONLY)
        </div>
      </header>

      {/* Main Content */}
      <div className="max-w-7xl mx-auto grid grid-cols-1 lg:grid-cols-12 gap-8">
        {/* Left Column: Input & Controls */}
        <div className="lg:col-span-4 space-y-6">
          <div className="bg-slate-900/40 border border-slate-800 rounded-2xl p-6 backdrop-blur-sm shadow-xl">
            <h2 className="text-xl font-semibold mb-6 flex items-center gap-2 text-slate-200">
              <Settings2 size={20} className="text-cyan-400" />
              Configuration
            </h2>

            {/* Controls Grid */}
            <div className="space-y-6">
              {/* Mode Selection */}
              <div className="space-y-3">
                <label className="text-sm font-medium text-slate-400">
                  Summarization Mode
                </label>
                <div className="grid grid-cols-3 gap-2 bg-slate-950/50 p-1 rounded-lg border border-slate-800">
                  {["extractive", "abstractive", "hybrid"].map((mode) => (
                    <button
                      key={mode}
                      onClick={() => setSummaryMode(mode)}
                      className={cn(
                        "text-xs font-semibold py-2 px-3 rounded-md capitalize transition-all",
                        summaryMode === mode
                          ? "bg-gradient-to-r from-cyan-600 to-indigo-600 text-white shadow-lg"
                          : "text-slate-500 hover:text-slate-300 hover:bg-slate-800",
                      )}
                    >
                      {mode}
                    </button>
                  ))}
                </div>
              </div>

              {/* Language Selector */}
              <div className="space-y-3">
                <label className="text-sm font-medium text-slate-400 flex items-center gap-2">
                  <Languages size={14} /> Output Language
                </label>
                <select
                  value={outputLanguage}
                  onChange={(e) => setOutputLanguage(e.target.value)}
                  className="w-full bg-slate-950/50 border border-slate-800 rounded-lg p-2.5 text-sm text-slate-200 focus:ring-1 focus:ring-cyan-500 outline-none"
                >
                  <option value="en">English (Default)</option>
                  <option value="es">Spanish (Español)</option>
                  <option value="fr">French (Français)</option>
                  <option value="de">German (Deutsch)</option>
                  <option value="hi">Hindi (हिन्दी)</option>
                  <option value="ja">Japanese (日本語)</option>
                  <option value="ko">Korean (한국어)</option>
                  <option value="mr">Marathi (मराठी)</option>
                  <option value="gu">Gujarati (ગુજરાતી)</option>
                  <option value="kn">Kannada (ಕನ್ನಡ)</option>
                  <option value="te">Telugu (తెలుగు)</option>
                </select>
              </div>

              {/* Summary Length Controls */}
              <div className="space-y-3">
                <div className="flex justify-between">
                  <label className="text-sm font-medium text-slate-400">
                    Summary Size
                  </label>
                  <span className="text-xs text-cyan-400 font-mono capitalize">
                    {summarySize} ({summarySize === 'small' ? '30%' : summarySize === 'medium' ? '50%' : '70%'})
                  </span>
                </div>
                <div className="grid grid-cols-3 gap-2 bg-slate-950/50 p-1 rounded-lg border border-slate-800">
                  {(['small', 'medium', 'large'] as const).map((size) => (
                    <button
                      key={size}
                      onClick={() => setSummarySize(size)}
                      className={cn(
                        "text-xs font-semibold py-2 px-3 rounded-md capitalize transition-all",
                        summarySize === size
                          ? "bg-gradient-to-r from-emerald-600 to-teal-600 text-white shadow-lg"
                          : "text-slate-500 hover:text-slate-300 hover:bg-slate-800",
                      )}
                    >
                      {size}
                    </button>
                  ))}
                </div>
              </div>
            </div>

            <div className="my-6 border-t border-slate-800" />

            <h2 className="text-xl font-semibold mb-4 flex items-center gap-2 text-slate-200">
              <Upload size={20} className="text-cyan-400" />
              Input Source
            </h2>

            {/* Drag & Drop Area */}
            <div
              onClick={() => fileInputRef.current?.click()}
              onDragOver={handleDragOver}
              onDragLeave={handleDragLeave}
              onDrop={handleDrop}
              className={cn(
                "border-2 border-dashed border-slate-700 rounded-xl p-6 text-center cursor-pointer transition-all hover:border-cyan-500/50 hover:bg-slate-800/50 group relative overflow-hidden",
                file ? "border-cyan-500 bg-cyan-950/20" : "",
                isDragging ? "border-emerald-500 bg-emerald-950/20 scale-[1.02]" : ""
              )}
            >
              <input
                ref={fileInputRef}
                type="file"
                className="hidden"
                accept=".txt,.pdf,.docx"
                onChange={handleFileUpload}
              />
              <div className="flex flex-col items-center gap-3 relative z-10">
                <div className="p-3 bg-slate-800 rounded-full group-hover:scale-110 transition-transform shadow-lg shadow-black/50">
                  {file ? (
                    <FileText className="text-cyan-400" size={24} />
                  ) : (
                    <Upload className="text-slate-400" size={24} />
                  )}
                </div>
                {file ? (
                  <span className="text-cyan-400 font-medium text-sm truncate max-w-full">
                    {file.name}
                  </span>
                ) : (
                  <span className="text-slate-400 text-sm">
                    Upload PDF, DOCX, TXT
                  </span>
                )}
              </div>
            </div>

            <div className="relative my-6">
              <div className="absolute inset-0 flex items-center">
                <div className="w-full border-t border-slate-800"></div>
              </div>
              <div className="relative flex justify-center text-xs uppercase">
                <span className="bg-slate-900 px-2 text-slate-500">
                  Or paste text
                </span>
              </div>
            </div>

            <textarea
              className="w-full h-32 bg-slate-950/50 border border-slate-800 rounded-lg p-3 text-sm focus:ring-1 focus:ring-cyan-500 outline-none resize-none font-mono placeholder:text-slate-600 text-slate-300"
              placeholder="Paste content here..."
              value={textInput}
              onChange={(e) => {
                setTextInput(e.target.value);
                setFile(null);
              }}
            />

            <button
              onClick={processFile}
              disabled={isProcessing || (!file && !textInput)}
              className="w-full mt-6 bg-gradient-to-r from-cyan-600 to-indigo-600 hover:from-cyan-500 hover:to-indigo-500 text-white font-bold py-3.5 px-6 rounded-xl transition-all flex items-center justify-center gap-2 disabled:opacity-50 disabled:cursor-not-allowed shadow-lg shadow-cyan-900/20 active:scale-[0.98]"
            >
              {isProcessing ? (
                <>
                  <Clock className="animate-spin" size={18} /> Processing...
                </>
              ) : (
                <>
                  <Zap size={18} /> Analyze & Summarize
                </>
              )}
            </button>
          </div>

          {/* Metrics Panel (Bottom Left) */}
          {result && (
            <div className="grid grid-cols-1 gap-3">
              <MetricCard
                label="Compression"
                value={`${((result.metrics?.compression_ratio ?? 0) * 100).toFixed(0)}%`}
                icon={BarChart3}
                delay={0.1}
              />
              <MetricCard
                label="Sentences"
                value={result.metrics?.sentence_count ?? 0}
                icon={AlignLeft}
                delay={0.2}
              />
              <MetricCard
                label="Kept"
                value={result.metrics?.kept_sentences ?? 0}
                icon={CheckCircle2}
                delay={0.3}
              />
            </div>
          )}
        </div>

        {/* Right Column: Visualization */}
        <div className="lg:col-span-8 space-y-6">
          {result ? (
            <motion.div
              initial={{ opacity: 0, y: 20 }}
              animate={{ opacity: 1, y: 0 }}
              transition={{ duration: 0.5 }}
              className="space-y-6"
            >
              {/* Summary Block */}
              <div className="bg-slate-900/40 border border-slate-800 rounded-2xl p-8 backdrop-blur-sm relative overflow-hidden shadow-2xl">
                <div className="absolute top-0 left-0 w-1 h-full bg-gradient-to-b from-cyan-500 to-indigo-500" />
                <div className="flex justify-between items-start mb-6">
                  <div>
                    <h3 className="text-2xl font-bold text-cyan-50">
                      Executive Summary
                    </h3>
                    <p className="text-slate-400 text-sm mt-1">
                      Generated via {summaryMode} mode •{" "}
                      {outputLanguage.toUpperCase()} output
                    </p>
                  </div>
                  <button
                    onClick={downloadReport}
                    className="flex items-center gap-2 px-4 py-2 bg-slate-800 hover:bg-slate-700 text-slate-200 rounded-lg text-sm font-medium transition-colors border border-slate-700"
                  >
                    <Download size={16} /> Export Report
                  </button>
                </div>

                <div className="prose prose-invert prose-lg max-w-none">
                  <p className="leading-relaxed text-slate-300 whitespace-pre-line">
                    {result.summary_text}
                  </p>
                </div>
              </div>

              {/* Explainable View */}
              <div className="bg-slate-900/30 border border-slate-800/50 rounded-2xl p-8 shadow-xl">
                <div className="flex justify-between items-center mb-6">
                  <div className="flex items-center gap-3">
                    <div className="p-2 bg-indigo-500/10 rounded-lg">
                      <FileOutput size={20} className="text-indigo-400" />
                    </div>
                    <h3 className="text-xl font-bold text-slate-200">
                      Explainable Analysis
                    </h3>
                  </div>
                  <div className="flex gap-4 text-xs text-slate-400 bg-slate-950/50 px-3 py-1.5 rounded-full border border-slate-800">
                    <div className="flex items-center gap-2">
                      <div className="w-2 h-2 rounded-full bg-cyan-500/50" />{" "}
                      KEY INSIGHT
                    </div>
                    <div className="flex items-center gap-2">
                      <div className="w-2 h-2 rounded-full bg-slate-700" />{" "}
                      CONTEXT
                    </div>
                  </div>
                </div>

                {/* Heatmap Component */}
                <div className="max-h-[600px] overflow-y-auto pr-2 custom-scrollbar p-1">
                  <HeatMapHighlighter
                    sentences={
                      result.ranking_data
                        ? [...result.ranking_data].sort((a: any, b: any) => a.index - b.index)
                        : []
                    }
                  />
                </div>
              </div>
            </motion.div>
          ) : (
            // Empty State
            <div className="h-full flex flex-col items-center justify-center text-slate-500 border-2 border-dashed border-slate-800 rounded-2xl bg-slate-900/10 min-h-[500px]">
              <div className="w-24 h-24 bg-slate-900 rounded-full flex items-center justify-center mb-6 shadow-inner ring-1 ring-slate-800">
                <Settings2 className="text-slate-700" size={40} />
              </div>
              <h3 className="text-xl font-semibold text-slate-400 mb-2">
                Ready for Analysis
              </h3>
              <p className="max-w-md text-center text-slate-500">
                Configure your summarization preferences on the left and upload
                a document to generate an explainable report.
              </p>
            </div>
          )}
        </div>
      </div>
    </main>
  );
}
