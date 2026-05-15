import { useState } from "react";
import { useStore, AwardCategory } from "../store";
import { motion, AnimatePresence } from "motion/react";
import { Trash2, Award, Calendar } from "lucide-react";

const CATEGORIES: AwardCategory[] = ["科创竞赛", "文化艺术", "体育竞技", "社会实践"];

export const Home = () => {
  const { records, deleteRecord } = useStore();
  const [activeTab, setActiveTab] = useState<AwardCategory | "全部">("全部");

  const filteredRecords = activeTab === "全部" 
    ? records 
    : records.filter(r => r.category === activeTab);

  const tabs = ["全部", ...CATEGORIES];

  return (
    <div className="min-h-full flex flex-col">
      <header className="pt-14 pb-4 px-6 z-10 sticky top-0 bg-[var(--surface)] backdrop-blur-xl border-b border-[var(--border)] transition-colors duration-500">
        <h1 className="text-2xl font-semibold tracking-tight text-[var(--text)] pr-24">Records</h1>
        <p className="text-[var(--text-muted)] text-sm mt-1 font-medium">Your achievements & growth</p>
      </header>

      <div className="px-6 mt-4 z-10">
        <div className="flex space-x-2 overflow-x-auto pb-4 scrollbar-hide" style={{ scrollbarWidth: 'none' }}>
          {tabs.map(tab => (
            <button
              key={tab}
              onClick={() => setActiveTab(tab as any)}
              className={`whitespace-nowrap px-4 py-1.5 rounded-full text-[13px] font-medium transition-all duration-300 border ${
                activeTab === tab 
                  ? "bg-[var(--primary)] text-[var(--primary-text)] border-[var(--primary)] shadow-[var(--shadow)]" 
                  : "bg-[var(--surface-elevated)] text-[var(--text-muted)] hover:text-[var(--text)] border-[var(--border)] hover:border-[var(--border-hover)]"
              }`}
            >
              {tab}
            </button>
          ))}
        </div>
      </div>

      <div className="flex-1 p-6 overflow-y-auto pt-2">
        <AnimatePresence>
          {filteredRecords.length === 0 ? (
            <motion.div 
              initial={{ opacity: 0 }}
              animate={{ opacity: 1 }}
              className="flex flex-col items-center justify-center h-48 text-[var(--text-muted)] space-y-4"
            >
              <Award size={48} strokeWidth={1} />
              <p className="text-sm">No records found.</p>
            </motion.div>
          ) : (
            <div className="space-y-4">
              {filteredRecords.map(record => (
                <motion.div
                  key={record.id}
                  initial={{ opacity: 0, y: 10 }}
                  animate={{ opacity: 1, y: 0 }}
                  exit={{ opacity: 0, scale: 0.95 }}
                  transition={{ duration: 0.2 }}
                  className="bg-[var(--surface-container-high)] rounded-[var(--radius)] border border-[var(--border)] overflow-hidden group relative hover:border-[var(--border-hover)] transition-all duration-200 shadow-[var(--shadow)]"
                >
                  <div className="flex h-32">
                    <div className="w-1/3 relative bg-[var(--surface)]">
                      <img 
                        src={record.image} 
                        alt="Certificate" 
                        className="w-full h-full object-cover opacity-90 group-hover:opacity-100 transition-opacity"
                      />
                      <div className="absolute top-2 left-2 px-1.5 py-0.5 bg-[var(--surface)]/80 backdrop-blur-md rounded-[calc(var(--radius)*0.5)] border border-[var(--border)] text-[9px] text-[var(--text)] font-medium tracking-wide shadow-sm">
                        {record.category}
                      </div>
                    </div>
                    <div className="w-2/3 p-4 flex flex-col justify-between">
                      <div>
                        <h3 className="font-semibold text-[var(--text)] line-clamp-1 text-sm tracking-tight">{record.competitionName}</h3>
                        <p className="text-[var(--text-muted)] text-xs font-medium mt-1">{record.awardLevel}</p>
                      </div>
                      
                      <div className="flex items-center justify-between mt-2 text-[11px] text-[var(--text-muted)] font-mono">
                        <div className="flex items-center">
                          <Calendar size={12} className="mr-1.5 opacity-70" />
                          {record.date}
                        </div>
                      </div>
                    </div>
                  </div>

                  <div className="absolute top-3 right-3 flex space-x-1 opacity-0 group-hover:opacity-100 transition-opacity">
                    <button 
                      onClick={(e) => { e.stopPropagation(); deleteRecord(record.id); }}
                      className="p-1.5 bg-[var(--surface)]/90 backdrop-blur-md rounded-[calc(var(--radius)*0.5)] text-[var(--text-muted)] hover:text-red-500 border border-[var(--border)] shadow-sm transition-all"
                    >
                      <Trash2 size={14} />
                    </button>
                  </div>
                </motion.div>
              ))}
            </div>
          )}
        </AnimatePresence>
      </div>
    </div>
  );
};
