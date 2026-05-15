import { useState } from "react";
import { useNavigate } from "react-router";
import { useStore, AwardCategory } from "../store";
import { motion, AnimatePresence } from "motion/react";
import { Camera, Loader2, Sparkles, AlertCircle } from "lucide-react";
import { toast } from "sonner";

export const AddRecord = () => {
  const { addRecord } = useStore();
  const navigate = useNavigate();
  
  const [step, setStep] = useState<"upload" | "scanning" | "form">("upload");
  const [preview, setPreview] = useState<string | null>(null);
  
  const [category, setCategory] = useState<AwardCategory>("科创竞赛");
  const [formData, setFormData] = useState({
    competitionName: "",
    awardLevel: "",
    date: "",
    studentId: "20210001",
    studentName: "张三",
    major: "计算机科学与技术",
    className: "计科2101",
    workName: "",
    teamName: "",
    teamSize: "1",
    advisor: "",
    rank: "",
    level: "",
    remarks: ""
  });

  const handleSimulateUpload = () => {
    setPreview("https://images.unsplash.com/photo-1766722906733-609eebf3b63a?crop=entropy&cs=tinysrgb&fit=max&fm=jpg&ixid=M3w3Nzg4Nzd8MHwxfHNlYXJjaHwxfHxjZXJ0aWZpY2F0ZSUyMGF3YXJkfGVufDF8fHx8MTc3ODgzNjYzOXww&ixlib=rb-4.1.0&q=80&w=1080");
    setStep("scanning");
    
    setTimeout(() => {
      setFormData(prev => ({
        ...prev,
        competitionName: "全国大学生互联网+创新创业大赛",
        awardLevel: "省级金奖",
        date: "2023-08-20",
        level: "省级"
      }));
      setCategory("科创竞赛");
      setStep("form");
      toast("Data extracted successfully.", {
        icon: <Sparkles size={14} />
      });
    }, 2000);
  };

  const handleChange = (e: React.ChangeEvent<HTMLInputElement | HTMLSelectElement | HTMLTextAreaElement>) => {
    const { name, value } = e.target;
    setFormData(prev => ({ ...prev, [name]: value }));
  };

  const handleSubmit = (e: React.FormEvent) => {
    e.preventDefault();
    if (!formData.competitionName || !formData.awardLevel) {
      toast.error("Required fields missing");
      return;
    }
    
    addRecord({
      category,
      image: preview || "",
      ...formData,
      teamSize: parseInt(formData.teamSize) || 1,
    });
    
    toast.success("Record saved.");
    navigate("/");
  };

  const InputLabel = ({ children }: { children: React.ReactNode }) => (
    <label className="block text-[11px] font-medium text-[var(--text-muted)] mb-1.5 uppercase tracking-wider">{children}</label>
  );

  const inputClass = "w-full p-2.5 bg-[var(--surface-elevated)] border border-[var(--border)] rounded-[calc(var(--radius)*0.5)] text-[13px] text-[var(--text)] focus:border-[var(--primary)] outline-none transition-colors placeholder:text-[var(--text-muted)]";

  return (
    <div className="min-h-full flex flex-col">
      <header className="px-6 py-5 z-10 sticky top-0 bg-[var(--surface)] backdrop-blur-xl border-b border-[var(--border)] transition-colors duration-500">
        <h1 className="text-lg font-semibold tracking-tight text-[var(--text)] pr-24">New Entry</h1>
      </header>

      <div className="flex-1 p-6 overflow-y-auto">
        <AnimatePresence mode="wait">
          {step === "upload" && (
            <motion.div 
              key="upload"
              initial={{ opacity: 0 }}
              animate={{ opacity: 1 }}
              exit={{ opacity: 0 }}
              className="h-[60vh] flex flex-col items-center justify-center space-y-8"
            >
              <div className="text-center space-y-2">
                <h2 className="text-lg font-medium text-[var(--text)]">Upload Certificate</h2>
                <p className="text-[13px] text-[var(--text-muted)]">AI will automatically extract information.</p>
              </div>
              
              <button 
                onClick={handleSimulateUpload}
                className="w-full max-w-xs aspect-[4/3] bg-[var(--surface-container)] border border-dashed border-[var(--border)] rounded-[var(--radius)] flex flex-col items-center justify-center space-y-4 hover:border-[var(--primary)] hover:bg-[var(--primary-container)] transition-all duration-300 group shadow-[var(--shadow)]"
              >
                <div className="p-3 bg-[var(--surface-elevated)] rounded-full group-hover:bg-[var(--border)] transition-colors shadow-sm">
                  <Camera size={24} className="text-[var(--text-muted)] group-hover:text-[var(--text)] transition-colors" />
                </div>
                <span className="text-[13px] text-[var(--text-muted)] font-medium group-hover:text-[var(--text)]">Tap to capture or upload</span>
              </button>
              
              <div className="flex items-center space-x-2 text-[11px] text-[var(--text-muted)] font-medium">
                <AlertCircle size={12} />
                <span>Supports JPG, PNG</span>
              </div>
            </motion.div>
          )}

          {step === "scanning" && (
            <motion.div 
              key="scanning"
              initial={{ opacity: 0, scale: 0.98 }}
              animate={{ opacity: 1, scale: 1 }}
              exit={{ opacity: 0 }}
              className="h-[60vh] flex flex-col items-center justify-center space-y-8"
            >
              <div className="relative w-64 h-48 rounded-[var(--radius)] overflow-hidden border border-[var(--border)] shadow-[var(--shadow)]">
                <img src={preview!} alt="Preview" className="w-full h-full object-cover opacity-50 grayscale" />
                <div className="absolute inset-0 bg-black/10" />
                <motion.div 
                  initial={{ top: 0 }}
                  animate={{ top: "100%" }}
                  transition={{ repeat: Infinity, duration: 1.5, ease: "linear" }}
                  className="absolute left-0 right-0 h-px bg-[var(--primary)] shadow-[0_0_15px_2px_var(--primary)]"
                />
              </div>
              
              <div className="flex items-center space-x-3 text-[var(--text-muted)]">
                <Loader2 className="animate-spin" size={16} />
                <span className="text-[13px] font-medium">Analyzing document...</span>
              </div>
            </motion.div>
          )}

          {step === "form" && (
            <motion.div 
              key="form"
              initial={{ opacity: 0, y: 10 }}
              animate={{ opacity: 1, y: 0 }}
              className="pb-8"
            >
              <form onSubmit={handleSubmit} className="space-y-6">
                
                <div className="bg-[var(--primary-container)] p-4 rounded-[var(--radius)] flex items-start space-x-3 text-[var(--on-primary-container)] shadow-sm">
                  <Sparkles className="shrink-0 mt-0.5" size={16} />
                  <div>
                    <h3 className="font-medium text-[13px]">Auto-filled via AI</h3>
                    <p className="text-[11px] mt-1 opacity-80">Please verify and complete the missing details.</p>
                  </div>
                </div>

                <div className="space-y-5 bg-[var(--surface)] rounded-[var(--radius)] p-5 border border-[var(--border)] shadow-[var(--shadow)]">
                  <div>
                    <InputLabel>Category</InputLabel>
                    <div className="relative">
                      <select 
                        value={category}
                        onChange={(e) => setCategory(e.target.value as AwardCategory)}
                        className={`${inputClass} appearance-none`}
                      >
                        <option value="科创竞赛">科创竞赛</option>
                        <option value="文化艺术">文化艺术</option>
                        <option value="体育竞技">体育竞技</option>
                        <option value="社会实践">社会实践</option>
                      </select>
                      <div className="absolute right-3 top-1/2 -translate-y-1/2 pointer-events-none text-[var(--text-muted)] text-[10px]">▼</div>
                    </div>
                  </div>
                  
                  <div>
                    <InputLabel>Competition Name *</InputLabel>
                    <input 
                      required
                      name="competitionName"
                      value={formData.competitionName}
                      onChange={handleChange}
                      className={inputClass}
                    />
                  </div>

                  <div className="grid grid-cols-2 gap-4">
                    <div>
                      <InputLabel>Award Level *</InputLabel>
                      <input 
                        required
                        name="awardLevel"
                        value={formData.awardLevel}
                        onChange={handleChange}
                        className={inputClass}
                      />
                    </div>
                    <div>
                      <InputLabel>Date</InputLabel>
                      <input 
                        type="date"
                        name="date"
                        value={formData.date}
                        onChange={handleChange}
                        className={`${inputClass} [color-scheme:dark]`}
                      />
                    </div>
                  </div>
                </div>

                <div className="space-y-5 bg-[var(--surface)] rounded-[var(--radius)] p-5 border border-[var(--border)] shadow-[var(--shadow)]">
                  <h3 className="text-[13px] font-semibold text-[var(--text)] mb-2">Additional Details</h3>
                  
                  {category === "科创竞赛" && (
                    <>
                      <div>
                        <InputLabel>Work Name</InputLabel>
                        <input name="workName" value={formData.workName} onChange={handleChange} className={inputClass} placeholder="Optional" />
                      </div>
                      <div className="grid grid-cols-2 gap-4">
                        <div>
                          <InputLabel>Team Name</InputLabel>
                          <input name="teamName" value={formData.teamName} onChange={handleChange} className={inputClass} placeholder="Optional" />
                        </div>
                        <div>
                          <InputLabel>Team Size</InputLabel>
                          <input type="number" name="teamSize" value={formData.teamSize} onChange={handleChange} className={inputClass} min="1" />
                        </div>
                      </div>
                      <div>
                        <InputLabel>Advisor</InputLabel>
                        <input name="advisor" value={formData.advisor} onChange={handleChange} className={inputClass} placeholder="Optional" />
                      </div>
                    </>
                  )}

                  {category === "文化艺术" && (
                    <>
                      <div>
                        <InputLabel>Rank</InputLabel>
                        <input name="rank" value={formData.rank} onChange={handleChange} className={inputClass} placeholder="e.g. 1/5" />
                      </div>
                      <div>
                        <InputLabel>Advisor</InputLabel>
                        <input name="advisor" value={formData.advisor} onChange={handleChange} className={inputClass} placeholder="Optional" />
                      </div>
                    </>
                  )}
                  
                  <div>
                    <InputLabel>Remarks</InputLabel>
                    <textarea 
                      name="remarks"
                      value={formData.remarks}
                      onChange={handleChange}
                      rows={2}
                      className={inputClass} 
                      placeholder="Optional notes"
                    />
                  </div>
                </div>

                <button
                  type="submit"
                  className="w-full py-3 bg-[var(--primary)] text-[var(--primary-text)] rounded-full font-medium text-[14px] tracking-wide hover:shadow-[var(--shadow-elevated)] active:scale-[0.98] transition-all shadow-[var(--shadow)]"
                >
                  Save Record
                </button>
              </form>
            </motion.div>
          )}
        </AnimatePresence>
      </div>
    </div>
  );
};
