import { useState, useMemo } from "react";
import { useStore } from "../store";
import { 
  BarChart, Bar, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer,
  PieChart, Pie, Cell
} from "recharts";
import { Download, Search, Filter } from "lucide-react";
import { toast } from "sonner";

export const Admin = () => {
  const { records, theme } = useStore();
  const [searchTerm, setSearchTerm] = useState("");

  const categoryStats = [
    { name: '科创竞赛', value: records.filter(r => r.category === '科创竞赛').length },
    { name: '文化艺术', value: records.filter(r => r.category === '文化艺术').length },
    { name: '体育竞技', value: records.filter(r => r.category === '体育竞技').length },
    { name: '社会实践', value: records.filter(r => r.category === '社会实践').length },
  ];

  const getChartColors = () => {
    switch (theme) {
      case 'apple': return ['#007aff', '#34c759', '#5856d6', '#ff9500'];
      case 'md3': return ['#6750a4', '#625b71', '#7d5260', '#49454f'];
      case 'notion': return ['#37352f', '#787774', '#9a9a97', '#d3d1cb'];
      case 'framer': return ['#0099ff', '#ff0055', '#00cc66', '#ffaa00'];
      case 'linear':
      default: return ['#ffffff', '#a3a3a3', '#525252', '#262626'];
    }
  };

  const chartColors = useMemo(getChartColors, [theme]);

  const deptStats = [
    { name: 'CS', count: 120 },
    { name: 'Business', count: 86 },
    { name: 'Arts', count: 65 },
    { name: 'Sports', count: 42 },
  ];

  const handleExport = () => {
    toast("Generating CSV export...");
    setTimeout(() => {
      toast.success("Export completed.");
    }, 1500);
  };

  const filteredRecords = records.filter(r => 
    r.competitionName.includes(searchTerm) || 
    r.studentName.includes(searchTerm)
  );

  return (
    <div className="min-h-full flex flex-col">
      <header className="pt-10 pb-4 px-6 z-10 sticky top-0 bg-[var(--surface)] backdrop-blur-xl border-b border-[var(--border)] transition-colors duration-500">
        <div className="flex justify-between items-center mb-6 pr-24">
          <h1 className="text-xl font-semibold tracking-tight text-[var(--text)]">Dashboard</h1>
          <button onClick={handleExport} className="flex items-center space-x-1.5 bg-[var(--secondary-container)] hover:shadow-[var(--shadow)] px-4 py-2 rounded-full text-[12px] text-[var(--secondary)] font-medium transition-all shadow-sm">
            <Download size={14} />
            <span>Export</span>
          </button>
        </div>
        
        <div className="relative pr-24">
          <Search size={16} className="absolute left-3 top-1/2 -translate-y-1/2 text-[var(--text-muted)]" />
          <input 
            type="text" 
            placeholder="Search events or students..." 
            value={searchTerm}
            onChange={e => setSearchTerm(e.target.value)}
            className="w-full bg-[var(--surface-container-highest)] text-[var(--text)] placeholder:text-[var(--text-muted)] border border-[var(--border)] rounded-full py-2.5 pl-9 pr-4 text-[13px] focus:border-[var(--primary)] focus:border-2 focus:outline-none transition-colors shadow-sm"
          />
        </div>
      </header>

      <div className="flex-1 overflow-y-auto p-6 space-y-4">
        
        {/* Dashboard Cards */}
        <div className="grid grid-cols-2 gap-4">
          <div className="bg-[var(--surface-container)] p-4 rounded-[var(--radius)] border border-[var(--border)] flex flex-col justify-between h-24 shadow-[var(--shadow)]">
            <div className="text-[var(--text-muted)] text-[11px] font-medium uppercase tracking-wider">Total Awards</div>
            <div className="text-2xl font-semibold text-[var(--text)] tracking-tight">313</div>
          </div>
          <div className="bg-[var(--surface-container)] p-4 rounded-[var(--radius)] border border-[var(--border)] flex flex-col justify-between h-24 shadow-[var(--shadow)]">
            <div className="text-[var(--text-muted)] text-[11px] font-medium uppercase tracking-wider">National Level</div>
            <div className="text-2xl font-semibold text-[var(--text)] tracking-tight">45</div>
          </div>
        </div>

        {/* Chart 1: Categories */}
        <div className="bg-[var(--surface-container)] p-5 rounded-[var(--radius)] border border-[var(--border)] shadow-[var(--shadow)]">
          <h2 className="text-[13px] font-semibold text-[var(--text)] mb-4">Distribution by Category</h2>
          <div className="h-40 w-full flex items-center">
            <div className="w-1/2 h-full">
              <ResponsiveContainer width="100%" height="100%">
                <PieChart>
                  <Pie
                    data={categoryStats}
                    cx="50%"
                    cy="50%"
                    innerRadius={30}
                    outerRadius={50}
                    paddingAngle={2}
                    dataKey="value"
                    stroke="none"
                  >
                    {categoryStats.map((entry, index) => (
                      <Cell key={`cell-${index}`} fill={chartColors[index % chartColors.length]} />
                    ))}
                  </Pie>
                  <Tooltip 
                    contentStyle={{ backgroundColor: 'var(--surface-elevated)', borderColor: 'var(--border)', borderRadius: '8px', fontSize: '12px', color: 'var(--text)' }}
                    itemStyle={{ color: 'var(--text)' }}
                  />
                </PieChart>
              </ResponsiveContainer>
            </div>
            <div className="w-1/2 flex flex-col justify-center space-y-2 pl-4 border-l border-[var(--border)]">
              {categoryStats.map((stat, i) => (
                <div key={stat.name} className="flex items-center text-[10px]">
                  <div className="w-2 h-2 rounded-full mr-2" style={{ backgroundColor: chartColors[i] }} />
                  <span className="text-[var(--text-muted)] flex-1">{stat.name}</span>
                  <span className="text-[var(--text)] font-mono">{stat.value}</span>
                </div>
              ))}
            </div>
          </div>
        </div>

        {/* Chart 2: Departments */}
        <div className="bg-[var(--surface-container)] p-5 rounded-[var(--radius)] border border-[var(--border)] shadow-[var(--shadow)]">
          <h2 className="text-[13px] font-semibold text-[var(--text)] mb-4">Top Departments</h2>
          <div className="h-40 w-full">
            <ResponsiveContainer width="100%" height="100%">
              <BarChart data={deptStats} margin={{ top: 0, right: 0, left: -25, bottom: 0 }}>
                <CartesianGrid strokeDasharray="3 3" vertical={false} stroke="var(--border)" />
                <XAxis dataKey="name" tick={{ fontSize: 10, fill: 'var(--text-muted)' }} axisLine={false} tickLine={false} />
                <YAxis tick={{ fontSize: 10, fill: 'var(--text-muted)' }} axisLine={false} tickLine={false} />
                <Tooltip 
                  cursor={{ fill: 'var(--surface-elevated)' }} 
                  contentStyle={{ backgroundColor: 'var(--surface-elevated)', borderColor: 'var(--border)', borderRadius: '8px', fontSize: '12px', color: 'var(--text)' }}
                />
                <Bar dataKey="count" fill={chartColors[0]} radius={[4, 4, 0, 0]} barSize={20} />
              </BarChart>
            </ResponsiveContainer>
          </div>
        </div>

        {/* Records List */}
        <div className="bg-[var(--surface-container)] p-5 rounded-[var(--radius)] border border-[var(--border)] shadow-[var(--shadow)]">
          <div className="flex justify-between items-center mb-4">
            <h2 className="text-[13px] font-semibold text-[var(--text)]">Recent Entries</h2>
            <button className="text-[var(--text-muted)] hover:text-[var(--text)] transition-colors"><Filter size={14}/></button>
          </div>
          
          <div className="space-y-4">
            {filteredRecords.length === 0 ? (
              <div className="text-center text-[var(--text-muted)] text-[12px] py-4">No matching records.</div>
            ) : (
              filteredRecords.slice(0, 5).map(record => (
                <div key={record.id} className="flex justify-between items-center pb-3 border-b border-[var(--border)] last:border-0 last:pb-0">
                  <div className="flex-1 pr-4">
                    <h3 className="text-[13px] font-medium text-[var(--text)] line-clamp-1">{record.competitionName}</h3>
                    <div className="flex items-center text-[10px] text-[var(--text-muted)] mt-1 space-x-2 font-mono">
                      <span>{record.studentName}</span>
                      <span className="w-1 h-1 bg-[var(--border-hover)] rounded-full" />
                      <span>{record.category}</span>
                    </div>
                  </div>
                  <div className="text-right">
                    <div className="text-[11px] text-[var(--text)] font-medium opacity-80">{record.awardLevel}</div>
                    <div className="text-[9px] text-[var(--text-muted)] font-mono mt-0.5">{record.date}</div>
                  </div>
                </div>
              ))
            )}
          </div>
        </div>

      </div>
    </div>
  );
};
