import React, { createContext, useContext, useState, ReactNode } from "react";

export type AwardCategory = "科创竞赛" | "文化艺术" | "体育竞技" | "社会实践";
export type ThemeType = "linear" | "apple" | "md3" | "notion" | "framer";

export interface AwardRecord {
  id: string;
  category: AwardCategory;
  competitionName: string;
  awardLevel: string;
  date: string;
  image: string;
  
  studentId: string;
  studentName: string;
  major: string;
  className: string;
  
  workName?: string;
  teamName?: string;
  teamSize?: number;
  advisor?: string;
  rank?: string;
  level?: string; 
  remarks?: string;
  teamMembers?: string;
}

interface StoreContextType {
  records: AwardRecord[];
  addRecord: (record: Omit<AwardRecord, "id">) => void;
  updateRecord: (id: string, record: Partial<AwardRecord>) => void;
  deleteRecord: (id: string) => void;
  theme: ThemeType;
  setTheme: (theme: ThemeType) => void;
}

const mockRecords: AwardRecord[] = [
  {
    id: "1",
    category: "科创竞赛",
    competitionName: "全国大学生挑战杯",
    awardLevel: "全国一等奖",
    date: "2023-11-15",
    image: "https://images.unsplash.com/photo-1766722906733-609eebf3b63a?crop=entropy&cs=tinysrgb&fit=max&fm=jpg&ixid=M3w3Nzg4Nzd8MHwxfHNlYXJjaHwxfHxjZXJ0aWZpY2F0ZSUyMGF3YXJkfGVufDF8fHx8MTc3ODgzNjYzOXww&ixlib=rb-4.1.0&q=80&w=1080",
    studentId: "20210001",
    studentName: "张三",
    major: "计算机科学与技术",
    className: "计科2101",
    workName: "智能农业监测系统",
    teamName: "星火燎原队",
    teamSize: 3,
    advisor: "李四教授",
    level: "国家级",
  },
  {
    id: "2",
    category: "体育竞技",
    competitionName: "校秋季运动会",
    awardLevel: "男子100米冠军",
    date: "2023-10-20",
    image: "https://images.unsplash.com/photo-1578269174936-2709b6aeb913?crop=entropy&cs=tinysrgb&fit=max&fm=jpg&ixid=M3w3Nzg4Nzd8MHwxfHNlYXJjaHwxfHx0cm9waHklMjBhY2hpZXZlbWVudHxlbnwxfHx8fDE3Nzg4MzY2Mzl8MA&ixlib=rb-4.1.0&q=80&w=1080",
    studentId: "20210001",
    studentName: "张三",
    major: "计算机科学与技术",
    className: "计科2101",
    level: "校级",
  }
];

const StoreContext = createContext<StoreContextType | undefined>(undefined);

export const StoreProvider = ({ children }: { children: ReactNode }) => {
  const [records, setRecords] = useState<AwardRecord[]>(mockRecords);
  const [theme, setTheme] = useState<ThemeType>("linear");

  const addRecord = (record: Omit<AwardRecord, "id">) => {
    const newRecord = { ...record, id: Math.random().toString(36).substr(2, 9) };
    setRecords([newRecord, ...records]);
  };

  const updateRecord = (id: string, partial: Partial<AwardRecord>) => {
    setRecords(records.map(r => r.id === id ? { ...r, ...partial } : r));
  };

  const deleteRecord = (id: string) => {
    setRecords(records.filter(r => r.id !== id));
  };

  return (
    <StoreContext.Provider value={{ records, addRecord, updateRecord, deleteRecord, theme, setTheme }}>
      {children}
    </StoreContext.Provider>
  );
};

export const useStore = () => {
  const context = useContext(StoreContext);
  if (context === undefined) {
    throw new Error("useStore must be used within a StoreProvider");
  }
  return context;
};
