"use client";

import { useState } from "react";
import { motion, AnimatePresence } from "framer-motion";
import { cn } from "@/lib/utils";
import { Info } from "lucide-react";

interface SentenceScore {
  text: string;
  index: number;
  score: number;
  rank: number;
  reasons: { [key: string]: number };
}

interface HeatMapHighlighterProps {
  sentences: SentenceScore[];
}

export function HeatMapHighlighter({ sentences }: HeatMapHighlighterProps) {
  const [hoveredIdx, setHoveredIdx] = useState<number | null>(null);
  const [hoveredPos, setHoveredPos] = useState<{ x: number, y: number } | null>(null);

  // Normalize scores for visualization (0.0 - 1.0)
  const maxScore = Math.max(...sentences.map(s => s.score), 0.001);

  const handleMouseEnter = (e: React.MouseEvent, idx: number) => {
    const rect = e.currentTarget.getBoundingClientRect();
    setHoveredIdx(idx);
    setHoveredPos({
      x: rect.left + rect.width / 2,
      y: rect.top
    });
  };

  return (
    <div className="space-y-1 font-serif text-lg leading-relaxed text-slate-300">
      {sentences.map((sent, idx) => {
        const intensity = sent.score / maxScore;
        const isHovered = hoveredIdx === idx;
        const isTop = sent.rank <= 5; // Default top 5 highlight

        return (
          <span key={idx} className="relative inline">
            <motion.span
              onMouseEnter={(e) => handleMouseEnter(e, idx)}
              onMouseLeave={() => {
                setHoveredIdx(null);
                setHoveredPos(null);
              }}
              className={cn(
                "cursor-pointer transition-colors duration-300 rounded px-0.5 mx-0.5 box-decoration-clone",
                isTop ? "bg-cyan-900/40 text-cyan-50 border-b border-cyan-500/30" : "hover:bg-slate-800",
                isHovered && "bg-cyan-800/60 ring-1 ring-cyan-500/50"
              )}
              style={{
                // Subtle highlight proportional to score
                backgroundColor: isTop ? undefined : `rgba(6, 182, 212, ${intensity * 0.15})`
              }}
            >
              {sent.text}{" "}
            </motion.span>
          </span>
        );
      })}

      {/* Render tooltip outside the loop/span to avoid stacking context issues if possible, 
          but simpler to just use fixed positioning portal-style. 
          Actually, we can just render ONE tooltip at the end based on state. */}
      
      <AnimatePresence>
        {hoveredIdx !== null && hoveredPos && sentences[hoveredIdx] && (
          <motion.div
            initial={{ opacity: 0, y: 10, scale: 0.95 }}
            animate={{ opacity: 1, y: -10, scale: 1 }}
            exit={{ opacity: 0, scale: 0.95 }}
            style={{
              position: 'fixed',
              left: hoveredPos.x,
              top: hoveredPos.y,
              transform: 'translateX(-50%)' // We'll handle this in the className or style
            }}
            className="z-[100] -translate-x-1/2 -translate-y-full mb-2 w-64 bg-slate-900 border border-slate-700 shadow-xl rounded-lg p-3 text-xs pointer-events-none"
          >
            <div className="flex justify-between items-center mb-2 border-b border-slate-800 pb-1">
              <span className="font-bold text-cyan-400">Idx {sentences[hoveredIdx].index} (Rank #{sentences[hoveredIdx].rank})</span>
              <span className="font-mono text-slate-300">{sentences[hoveredIdx].score.toFixed(3)}</span>
            </div>
            
            <div className="space-y-1">
              <div className="flex justify-between">
                <span className="text-slate-400">Centrality</span>
                <div className="w-24 h-1.5 bg-slate-800 rounded-full mt-1">
                  <div className="h-full bg-indigo-500 rounded-full" style={{ width: `${sentences[hoveredIdx].reasons.centrality * 100}%` }} />
                </div>
              </div>
              <div className="flex justify-between">
                <span className="text-slate-400">Keywords</span>
                <div className="w-24 h-1.5 bg-slate-800 rounded-full mt-1">
                  <div className="h-full bg-emerald-500 rounded-full" style={{ width: `${sentences[hoveredIdx].reasons.tfidf * 100}%` }} />
                </div>
              </div>
              <div className="flex justify-between">
                <span className="text-slate-400">Graph</span>
                <div className="w-24 h-1.5 bg-slate-800 rounded-full mt-1">
                  <div className="h-full bg-amber-500 rounded-full" style={{ width: `${sentences[hoveredIdx].reasons.textrank * 100}%` }} />
                </div>
              </div>
            </div>
          </motion.div>
        )}
      </AnimatePresence>
    </div>
  );
}
