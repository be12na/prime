
import React, { useState, useEffect, useMemo } from 'react';
import { getHistory, deletePrompt } from '../../services/promptHistoryService.ts';
import { SavedPrompt, Studio } from '../../types.ts';

const HistoryStudio: React.FC = () => {
    const [history, setHistory] = useState<SavedPrompt[]>([]);
    const [copiedId, setCopiedId] = useState<number | null>(null);

    useEffect(() => {
        setHistory(getHistory());
    }, []);

    const handleDelete = (id: number) => {
        setHistory(deletePrompt(id));
    };

    const handleCopy = (id: number, content: string) => {
        navigator.clipboard.writeText(content).catch(() => {});
        setCopiedId(id);
        setTimeout(() => setCopiedId(null), 2000);
    };

    const studioIcons: { [key in Studio]?: React.ReactNode } = {
        'photo-video': <svg xmlns="http://www.w3.org/2000/svg" className="h-5 w-5" fill="none" viewBox="0 0 24 24" stroke="currentColor" strokeWidth="2"><path strokeLinecap="round" strokeLinejoin="round" d="M3 9a2 2 0 012-2h.93a2 2 0 001.664-.89l.812-1.22A2 2 0 0110.07 4h3.86a2 2 0 011.664.89l.812 1.22A2 2 0 0018.07 7H19a2 2 0 012 2v9a2 2 0 01-2 2H5a2 2 0 01-2-2V9z" /><path strokeLinecap="round" strokeLinejoin="round" d="M15 13a3 3 0 11-6 0 3 3 0 016 0z" /></svg>,
        'product-launch': <svg xmlns="http://www.w3.org/2000/svg" className="h-5 w-5" fill="none" viewBox="0 0 24 24" stroke="currentColor" strokeWidth="2"><path strokeLinecap="round" strokeLinejoin="round" d="M13 10V3L4 14h7v7l9-11h-7z" /></svg>,
        'live': <svg xmlns="http://www.w3.org/2000/svg" className="h-5 w-5" fill="none" viewBox="0 0 24 24" stroke="currentColor" strokeWidth="2"><path strokeLinecap="round" strokeLinejoin="round" d="M15.536 8.464a5 5 0 010 7.072m2.828-9.9a9 9 0 010 12.728M5.636 5.636a9 9 0 0112.728 0m-12.728 0l12.728 12.728" /></svg>,
        'post': <svg xmlns="http://www.w3.org/2000/svg" className="h-5 w-5" fill="none" viewBox="0 0 24 24" stroke="currentColor" strokeWidth="2"><path strokeLinecap="round" strokeLinejoin="round" d="M11 5H6a2 2 0 00-2 2v11a2 2 0 002 2h11a2 2 0 002-2v-5m-1.414-9.414a2 2 0 112.828 2.828L11.828 15H9v-2.828l8.586-8.586z" /></svg>,
        'analytic': <svg xmlns="http://www.w3.org/2000/svg" className="h-5 w-5" fill="none" viewBox="0 0 24 24" stroke="currentColor" strokeWidth="2"><path strokeLinecap="round" strokeLinejoin="round" d="M9 19v-6a2 2 0 00-2-2H5a2 2 0 00-2 2v6a2 2 0 002 2h2a2 2 0 002-2zm0 0V9a2 2 0 012-2h2a2 2 0 012 2v10m-6 0a2 2 0 002 2h2a2 2 0 002-2m0 0V5a2 2 0 012-2h2a2 2 0 012 2v14a2 2 0 01-2 2h-2a2 2 0 01-2-2z" /></svg>,
        'guide': <svg xmlns="http://www.w3.org/2000/svg" className="h-5 w-5" fill="none" viewBox="0 0 24 24" stroke="currentColor" strokeWidth="2"><path strokeLinecap="round" strokeLinejoin="round" d="M12 6.253v13m0-13C10.832 5.477 9.246 5 7.5 5S4.168 5.477 3 6.253v13C4.168 18.477 5.754 18 7.5 18s3.332.477 4.5 1.253m0-13C13.168 5.477 14.754 5 16.5 5c1.747 0 3.332.477 4.5 1.253v13C19.832 18.477 18.247 18 16.5 18c-1.746 0-3.332.477-4.5 1.253" /></svg>,
        'history': <svg xmlns="http://www.w3.org/2000/svg" className="h-5 w-5" fill="none" viewBox="0 0 24 24" stroke="currentColor" strokeWidth="2"><path strokeLinecap="round" strokeLinejoin="round" d="M12 8v4l3 3m6-3a9 9 0 11-18 0 9 9 0 0118 0z" /></svg>,
    };

    const studioNames: { [key in Studio]?: string } = {
        'photo-video': 'Studio Foto & Video',
        'product-launch': 'Studio Launching',
        'live': 'Studio Live',
        'post': 'Studio Post',
        'analytic': 'Studio Analitik',
        'guide': 'Panduan & Pengaturan',
        'history': 'Riwayat',
    };

    const groupedHistory = useMemo(() => {
        return history.reduce((acc, prompt) => {
            const studioKey = prompt.studio;
            if (!acc[studioKey]) {
                acc[studioKey] = [];
            }
            acc[studioKey].push(prompt);
            return acc;
        }, {} as { [key in Studio]?: SavedPrompt[] });
    }, [history]);

    const orderedStudioKeys = Object.keys(groupedHistory).sort() as Studio[];


    return (
        <div id="history-studio" className="space-y-8">
            <div className="text-center px-4">
                <h2 className="text-3xl sm:text-4xl md:text-5xl font-display font-bold tracking-tight text-transparent bg-clip-text bg-gradient-to-r from-orange-400 to-amber-400">Riwayat Prompt</h2>
                <p className="mt-4 text-base sm:text-lg text-slate-400 max-w-2xl mx-auto">
                    Akses, salin, dan kelola semua prompt dan konten yang pernah Anda simpan.
                </p>
            </div>

            {history.length === 0 ? (
                <div className="text-center text-slate-400 p-8 bg-slate-800/50 rounded-xl border border-slate-700 max-w-2xl mx-auto">
                    <p>Anda belum menyimpan prompt apapun. Prompt yang Anda simpan dari studio lain akan muncul di sini.</p>
                </div>
            ) : (
                <div className="space-y-10">
                    {orderedStudioKeys.map(studio => (
                        groupedHistory[studio] && (
                        <div key={studio}>
                            {/* FIX: The h3 element was corrupted and incomplete. It is now fixed to display the studio icon and name. */}
                            <h3 className="flex items-center gap-3 text-xl font-semibold text-orange-400 mb-4 px-4 sm:px-0">
                                {studioIcons[studio]}
                                <span>{studioNames[studio]}</span>
                            </h3>
                            <div className="space-y-4">
                                {groupedHistory[studio]?.map(prompt => (
                                    <div key={prompt.id} className="bg-slate-800/50 p-4 sm:p-5 rounded-xl border border-slate-700">
                                        <div className="flex justify-between items-start gap-4">
                                            <div className="flex-1">
                                                <p className="text-xs text-slate-500 mb-2">{new Date(prompt.timestamp).toLocaleString()}</p>
                                                <h4 className="font-semibold text-slate-200 mb-2">{prompt.type}</h4>
                                                <p className="text-sm text-slate-300 whitespace-pre-wrap">{prompt.content}</p>
                                            </div>
                                            <div className="flex-shrink-0 flex items-center gap-2">
                                                <button onClick={() => handleCopy(prompt.id, prompt.content)} title="Salin" className="p-2 text-slate-400 hover:text-orange-400 bg-slate-700/50 hover:bg-slate-700 rounded-md transition-colors">
                                                    {copiedId === prompt.id ? (
                                                        <svg className="w-5 h-5 text-green-400" xmlns="http://www.w3.org/2000/svg" fill="none" viewBox="0 0 24 24" stroke="currentColor"><path strokeLinecap="round" strokeLinejoin="round" strokeWidth="2" d="M5 13l4 4L19 7" /></svg>
                                                    ) : (
                                                        <svg className="w-5 h-5" xmlns="http://www.w3.org/2000/svg" fill="none" viewBox="0 0 24 24" stroke="currentColor"><path strokeLinecap="round" strokeLinejoin="round" strokeWidth="2" d="M8 16H6a2 2 0 01-2-2V6a2 2 0 012-2h8a2 2 0 012 2v2m-6 12h8a2 2 0 002-2v-8a2 2 0 00-2-2h-8a2 2 0 00-2 2v8a2 2 0 002 2z" /></svg>
                                                    )}
                                                </button>
                                                <button onClick={() => handleDelete(prompt.id)} title="Hapus" className="p-2 text-slate-400 hover:text-red-400 bg-slate-700/50 hover:bg-slate-700 rounded-md transition-colors">
                                                    <svg xmlns="http://www.w3.org/2000/svg" className="h-5 w-5" fill="none" viewBox="0 0 24 24" stroke="currentColor" strokeWidth="2"><path strokeLinecap="round" strokeLinejoin="round" d="M19 7l-.867 12.142A2 2 0 0116.138 21H7.862a2 2 0 01-1.995-1.858L5 7m5 4v6m4-6v6m1-10V4a1 1 0 00-1-1h-4a1 1 0 00-1 1v3M4 7h16" /></svg>
                                                </button>
                                            </div>
                                        </div>
                                    </div>
                                ))}
                            </div>
                        </div>
                        )
                    ))}
                </div>
            )}
        </div>
    );
};

// FIX: Added the missing default export to resolve the module import error.
export default HistoryStudio;
