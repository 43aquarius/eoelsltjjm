import { useRef } from "react";
import { useStore } from "../store";
import { motion } from "motion/react";
import { Download, Share2, Award, Star } from "lucide-react";
import { toast } from "sonner";

export const MemorialCard = () => {
  const { records } = useStore();
  const cardRef = useRef<HTMLDivElement>(null);
  
  const studentInfo = {
    name: "张三",
    major: "计算机科学与技术",
    className: "计科2101",
    avatar: "https://images.unsplash.com/photo-1596247290824-e9f12b8c574f?crop=entropy&cs=tinysrgb&fit=max&fm=jpg&ixid=M3w3Nzg4Nzd8MHwxfHNlYXJjaHwxfHxjb2xsZWdlJTIwc3R1ZGVudCUyMHN0dWR5aW5nfGVufDF8fHx8MTc3ODY3NTA2Mnww&ixlib=rb-4.1.0&q=80&w=1080"
  };

  const handleDownload = () => {
    toast.success("Saved to gallery.");
  };

  return (
    <div className="min-h-full flex flex-col items-center selection:bg-[var(--primary)] selection:text-[var(--primary-text)]">
      <header className="w-full text-center py-8">
        <h1 className="text-xl font-semibold tracking-tight text-[var(--text)]">Memorial Card</h1>
        <p className="text-[var(--text-muted)] text-[13px] mt-1 font-medium">Your digital achievement showcase</p>
      </header>

      <div className="flex-1 w-full px-6 flex flex-col items-center overflow-y-auto pb-12">
        <motion.div 
          initial={{ y: 20, opacity: 0 }}
          animate={{ y: 0, opacity: 1 }}
          ref={cardRef}
          className="w-full max-w-sm bg-[var(--surface)] border border-[var(--border)] rounded-[var(--radius)] shadow-[var(--shadow)] relative overflow-hidden group"
        >
          {/* Subtle Glow Background */}
          <div className="absolute -top-24 -left-24 w-48 h-48 bg-[var(--primary)] opacity-[0.05] blur-3xl rounded-full pointer-events-none group-hover:opacity-[0.08] transition-opacity" />
          
          <div className="p-6 relative z-10">
            {/* Header */}
            <div className="flex items-center space-x-4 mb-8">
              <div className="w-12 h-12 rounded-[50%] border border-[var(--border)] overflow-hidden bg-[var(--secondary-container)]">
                <img src={studentInfo.avatar} alt="Avatar" className="w-full h-full object-cover grayscale opacity-90" />
              </div>
              <div>
                <h2 className="text-lg font-semibold text-[var(--text)] tracking-tight">{studentInfo.name}</h2>
                <p className="text-[11px] text-[var(--text-muted)] mt-0.5 font-mono">{studentInfo.major} · {studentInfo.className}</p>
              </div>
            </div>

            {/* Stats */}
            <div className="grid grid-cols-2 gap-3 mb-8">
              <div className="bg-[var(--surface-elevated)] border border-[var(--border)] rounded-[calc(var(--radius)*0.8)] p-4 text-center shadow-sm">
                <div className="text-[var(--text)] mb-1.5 flex justify-center"><Award size={18} strokeWidth={1.5} /></div>
                <div className="text-xl font-semibold text-[var(--text)]">{records.length}</div>
                <div className="text-[10px] text-[var(--text-muted)] font-medium mt-1 uppercase tracking-wider">Total Awards</div>
              </div>
              <div className="bg-[var(--surface-elevated)] border border-[var(--border)] rounded-[calc(var(--radius)*0.8)] p-4 text-center shadow-sm">
                <div className="text-[var(--text)] mb-1.5 flex justify-center"><Star size={18} strokeWidth={1.5} /></div>
                <div className="text-xl font-semibold text-[var(--text)]">
                  {records.filter(r => r.category === "科创竞赛").length}
                </div>
                <div className="text-[10px] text-[var(--text-muted)] font-medium mt-1 uppercase tracking-wider">Tech & Sci</div>
              </div>
            </div>

            {/* Recent Awards List */}
            <div className="space-y-4">
              <h3 className="text-[11px] font-medium text-[var(--text-muted)] uppercase tracking-widest mb-4">Milestones</h3>
              <div className="space-y-4 relative before:absolute before:inset-0 before:ml-[5px] before:h-full before:w-[1px] before:bg-[var(--border)]">
                {records.slice(0, 3).map((record) => (
                  <div key={record.id} className="relative flex items-start pl-6">
                    <div className="absolute left-[3px] top-1.5 w-1.5 h-1.5 bg-[var(--primary)] rounded-full ring-4 ring-[var(--surface)]" />
                    <div className="flex-1">
                      <h4 className="text-[13px] font-medium text-[var(--text)] line-clamp-1">{record.competitionName}</h4>
                      <div className="flex items-center space-x-2 text-[10px] mt-1 font-mono">
                        <span className="text-[var(--text-muted)]">{record.awardLevel}</span>
                        <span className="text-[var(--border-hover)]">·</span>
                        <span className="text-[var(--text-muted)]">{record.date.substring(0,4)}</span>
                      </div>
                    </div>
                  </div>
                ))}
                {records.length > 3 && (
                  <div className="text-[11px] text-[var(--text-muted)] font-mono mt-4 pl-6">
                    + {records.length - 3} more records
                  </div>
                )}
                {records.length === 0 && (
                  <div className="text-center text-[12px] text-[var(--text-muted)] py-4">
                    No records found.
                  </div>
                )}
              </div>
            </div>
          </div>

          {/* Footer stamp */}
          <div className="bg-[var(--surface-elevated)] px-6 py-4 flex justify-between items-center border-t border-[var(--border)] z-10 relative">
            <div className="text-[9px] text-[var(--text-muted)] font-mono uppercase tracking-wider">
              <p>Generated {new Date().toLocaleDateString()}</p>
              <p className="mt-0.5">Awards Tracker System</p>
            </div>
            <div className="w-8 h-8 bg-[var(--surface)] border border-[var(--border)] rounded-[calc(var(--radius)*0.3)] flex items-center justify-center text-[8px] text-[var(--text-muted)]">
              QR
            </div>
          </div>
        </motion.div>

        <div className="flex space-x-4 mt-8">
          <button onClick={handleDownload} className="flex items-center space-x-2 bg-[var(--primary)] text-[var(--primary-text)] px-6 py-2.5 rounded-full text-[14px] font-medium tracking-wide hover:shadow-[var(--shadow-elevated)] transition-all shadow-[var(--shadow)]">
            <Download size={16} />
            <span>Download</span>
          </button>
          <button className="flex items-center space-x-2 bg-[var(--secondary-container)] text-[var(--secondary)] px-6 py-2.5 rounded-full text-[14px] font-medium tracking-wide hover:shadow-[var(--shadow)] transition-all">
            <Share2 size={16} />
            <span>Share</span>
          </button>
        </div>
      </div>
    </div>
  );
};
