import { motion } from "framer-motion";
import { cn } from "@/lib/utils";
import { LucideIcon } from "lucide-react";

interface MetricCardProps {
  label: string;
  value: string | number;
  icon: LucideIcon;
  trend?: string;
  delay?: number;
}

export function MetricCard({
  label,
  value,
  icon: Icon,
  trend,
  delay = 0,
}: MetricCardProps) {
  return (
    <motion.div
      initial={{ opacity: 0, y: 20 }}
      animate={{ opacity: 1, y: 0 }}
      transition={{ duration: 0.5, delay }}
      className="bg-slate-900/50 border border-slate-800 backdrop-blur-sm p-4 rounded-xl flex items-center gap-4 hover:border-slate-700 transition-colors"
    >
      <div className="p-3 bg-cyan-950/30 rounded-lg text-cyan-400">
        <Icon size={20} />
      </div>
      <div className="min-w-0 flex-1">
        <div className="text-slate-400 text-xs font-medium uppercase tracking-wider truncate">
          {label}
        </div>
        <div
          className="text-xl font-bold font-mono text-slate-100 mt-1 truncate"
          title={String(value)}
        >
          {value}
        </div>
        {trend && (
          <div className="text-emerald-400 text-xs mt-1 truncate">{trend}</div>
        )}
      </div>
    </motion.div>
  );
}
