import React, { useState } from 'react';
import { analyzePerformance } from '../../services/geminiService.ts';
import Markdown from 'react-markdown';

interface ParsedOutput {
    title: string;
    content: string;
}

const AnalyticStudio: React.FC = () => {
    const [isLoading, setIsLoading] = useState(false);
    const [parsedOutput, setParsedOutput] = useState<ParsedOutput[]>([]);
    const [errorMessage, setErrorMessage] = useState('');
    const [formState, setFormState] = useState({
        transcript: '',
        views: '',
        likes: '',
        comments: '',
        shares: '',
        sales: '',
    });

    const handleChange = (e: React.ChangeEvent<HTMLInputElement | HTMLTextAreaElement>) => {
        const { id, value } = e.target;
         const keyMap: { [key: string]: string } = {
            'analytic-transcript': 'transcript',
            'analytic-views': 'views',
            'analytic-likes': 'likes',
            'analytic-comments': 'comments',
            'analytic-shares': 'shares',
            'analytic-sales': 'sales'
        };
        const stateKey = keyMap[id] as keyof typeof formState;
        if (stateKey) {
            setFormState(prev => ({ ...prev, [stateKey]: value }));
        }
    };

    const parseMarkdownOutput = (markdown: string): ParsedOutput[] => {
        if (!markdown || typeof markdown !== 'string') return [];
        // Split by markdown headings (###)
        const sections = markdown.split(/\n### /).filter(s => s.trim() !== '');
        
        return sections.map(section => {
            const parts = section.split('\n');
            const title = parts[0].replace(/[*_]/g, '').trim();
            const content = parts.slice(1).join('\n').trim();
            return { title, content };
        }).filter(s => s.title && s.content);
    };

    const handleGenerate = async () => {
        setIsLoading(true);
        setParsedOutput([]);
        setErrorMessage('');
        try {
            const result = await analyzePerformance(formState);
            if (result.startsWith('Error:')) {
                setErrorMessage(result);
            } else {
                const parsed = parseMarkdownOutput(result);
                if (parsed.length === 0) {
                    setErrorMessage('AI tidak menghasilkan output yang valid. Silakan coba lagi.');
                } else {
                    setParsedOutput(parsed);
                }
            }
        } catch (error) {
            const msg = error instanceof Error ? error.message : 'Terjadi kesalahan.';
            setErrorMessage(`Error: ${msg}`);
        } finally {
            setIsLoading(false);
        }
    };
    
    // Fix: Defined props interface and used React.FC to correctly type the component for use with a 'key' prop in a list.
    interface OutputCardProps {
        title: string;
        content: string;
    }
     const OutputCard: React.FC<OutputCardProps> = ({ title, content }) => (
        <div className="relative bg-slate-900/50 p-6 rounded-xl border border-slate-700">
            <h4 className="font-semibold text-lg text-orange-400 mb-3">{title}</h4>
            <div className="absolute top-4 right-4">
                <button onClick={() => navigator.clipboard.writeText(content).catch(() => {})} title="Salin" className="p-2 text-slate-400 hover:text-white bg-slate-700/50 rounded-md transition-colors">
                    <svg className="h-5 w-5" fill="none" viewBox="0 0 24 24" stroke="currentColor"><path strokeLinecap="round" strokeLinejoin="round" strokeWidth="2" d="M8 16H6a2 2 0 01-2-2V6a2 2 0 012-2h8a2 2 0 012 2v2m-6 12h8a2 2 0 002-2v-8a2 2 0 00-2-2h-8a2 2 0 00-2 2v8a2 2 0 002 2z" /></svg>
                </button>
            </div>
            <div className="prose prose-sm prose-invert text-slate-300 max-w-none prose-headings:text-slate-200">
                <Markdown>{content}</Markdown>
            </div>
        </div>
    );
    
    const SkeletonLoader = () => (
        <div className="space-y-6">
            <div className="bg-slate-800/50 p-6 rounded-xl space-y-3 animate-pulse"><div className="h-5 w-1/3 bg-slate-700 rounded"></div><div className="h-4 w-full bg-slate-700 rounded"></div><div className="h-4 w-5/6 bg-slate-700 rounded"></div></div>
            <div className="bg-slate-800/50 p-6 rounded-xl space-y-3 animate-pulse"><div className="h-5 w-1/4 bg-slate-700 rounded"></div><div className="h-4 w-full bg-slate-700 rounded"></div><div className="h-4 w-full bg-slate-700 rounded"></div></div>
            <div className="bg-slate-800/50 p-6 rounded-xl space-y-3 animate-pulse"><div className="h-5 w-1/3 bg-slate-700 rounded"></div><div className="h-4 w-2/3 bg-slate-700 rounded"></div></div>
        </div>
    );


    return (
       <div id="analytic-studio" className="space-y-8">
        <div className="text-center px-4">
          <h2 className="text-3xl sm:text-4xl md:text-5xl font-display font-bold tracking-tight text-transparent bg-clip-text bg-gradient-to-r from-orange-400 to-amber-400">Studio Analitik</h2>
          <p className="mt-4 text-base sm:text-lg text-slate-400 max-w-3xl mx-auto">
            Ubah data performa konten Anda menjadi wawasan strategis dan rekomendasi aksi yang cerdas.
          </p>
        </div>
        <div className="grid grid-cols-1 lg:grid-cols-2 gap-8 items-start">
            <div className="bg-slate-800/50 backdrop-blur-lg border border-slate-700 rounded-xl p-4 sm:p-6 md:p-8 space-y-8 shadow-2xl shadow-black/30 lg:sticky top-24">
                <div className="space-y-6">
                    <h3 className="flex items-center text-lg font-semibold gap-3 text-transparent bg-clip-text bg-gradient-to-r from-orange-400 to-amber-400">
                        <svg xmlns="http://www.w3.org/2000/svg" className="h-6 w-6 text-orange-500" fill="none" viewBox="0 0 24 24" stroke="currentColor" strokeWidth="2"><path strokeLinecap="round" strokeLinejoin="round" d="M9 17v-2a4 4 0 00-4-4H3V9a4 4 0 004-4v-2m11 14v-2a4 4 0 00-4-4h-2.59a1 1 0 00-.7.29l-2.42 2.42a1 1 0 01-1.42 0l-2.42-2.42a1 1 0 00-.7-.29H3a4 4 0 00-4 4v2" /></svg>
                        <span>1. Data Konten & Performa</span>
                    </h3>
                    <div className="space-y-4">
                        <div>
                            <label htmlFor="analytic-transcript" className="block text-sm font-medium text-slate-400 mb-2">Transkrip Video</label>
                            {/* Fix: The `rows` attribute for a textarea expects a number, not a string. */}
                            <textarea id="analytic-transcript" rows={8} value={formState.transcript} onChange={handleChange} className="block w-full bg-slate-700/50 border-slate-600 rounded-md py-2.5 px-4 text-slate-100 sm:text-sm transition focus:ring-orange-500 focus:border-orange-500" placeholder="Tempel transkrip atau skrip lengkap dari video yang ingin Anda analisis..."></textarea>
                        </div>
                        <div>
                            <p className="block text-sm font-medium text-slate-400 mb-2">Metrik Performa Utama</p>
                            <div className="grid grid-cols-2 sm:grid-cols-3 gap-4">
                                {/* Fix: Changed input type from "number" to "text" with inputMode="numeric" to resolve a TypeScript error. */}
                                <div><label htmlFor="analytic-views" className="sr-only">Views</label><input type="text" inputMode="numeric" id="analytic-views" value={formState.views} onChange={handleChange} className="block w-full bg-slate-700/50 border-slate-600 rounded-md py-2.5 px-4 text-slate-100 sm:text-sm transition focus:ring-orange-500 focus:border-orange-500" placeholder="Views"/></div>
                                <div><label htmlFor="analytic-likes" className="sr-only">Likes</label><input type="text" inputMode="numeric" id="analytic-likes" value={formState.likes} onChange={handleChange} className="block w-full bg-slate-700/50 border-slate-600 rounded-md py-2.5 px-4 text-slate-100 sm:text-sm transition focus:ring-orange-500 focus:border-orange-500" placeholder="Likes"/></div>
                                <div><label htmlFor="analytic-comments" className="sr-only">Comments</label><input type="text" inputMode="numeric" id="analytic-comments" value={formState.comments} onChange={handleChange} className="block w-full bg-slate-700/50 border-slate-600 rounded-md py-2.5 px-4 text-slate-100 sm:text-sm transition focus:ring-orange-500 focus:border-orange-500" placeholder="Comments"/></div>
                                <div><label htmlFor="analytic-shares" className="sr-only">Shares</label><input type="text" inputMode="numeric" id="analytic-shares" value={formState.shares} onChange={handleChange} className="block w-full bg-slate-700/50 border-slate-600 rounded-md py-2.5 px-4 text-slate-100 sm:text-sm transition focus:ring-orange-500 focus:border-orange-500" placeholder="Shares"/></div>
                                <div className="col-span-2"><label htmlFor="analytic-sales" className="sr-only">Sales (Opsional)</label><input type="text" inputMode="numeric" id="analytic-sales" value={formState.sales} onChange={handleChange} className="block w-full bg-slate-700/50 border-slate-600 rounded-md py-2.5 px-4 text-slate-100 sm:text-sm transition focus:ring-orange-500 focus:border-orange-500" placeholder="Penjualan dari video ini (opsional)"/></div>
                            </div>
                        </div>
                    </div>
                </div>
                <div className="pt-4">
                    <button id="generate-analytic-button" onClick={handleGenerate} disabled={isLoading} className="w-full inline-flex items-center justify-center px-12 py-4 text-lg font-semibold text-white bg-gradient-to-r from-orange-500 to-amber-500 rounded-lg shadow-lg shadow-orange-500/40 hover:shadow-xl hover:shadow-orange-500/40 focus:outline-none focus:ring-2 focus:ring-offset-2 focus:ring-offset-slate-900 focus:ring-orange-500 transition-all duration-300 ease-in-out transform hover:-translate-y-0.5 disabled:from-slate-600 disabled:to-slate-600 disabled:shadow-none disabled:translate-y-0 disabled:cursor-not-allowed">
                        {isLoading && <svg id="spinner-icon-analytic" className="animate-spin -ml-1 mr-3 h-5 w-5 text-white" xmlns="http://www.w3.org/2000/svg" fill="none" viewBox="0 0 24 24"><circle className="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" strokeWidth="4"></circle><path className="opacity-75" fill="currentColor" d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4zm2 5.291A7.962 7.962 0 014 12H0c0 3.042 1.135 5.824 3 7.938l3-2.647z"></path></svg>}
                        <span id="button-text-analytic">{isLoading ? 'Menganalisis...' : 'Analisis Performa'}</span>
                    </button>
                </div>
            </div>
            
            <div className="space-y-6">
                {isLoading && <SkeletonLoader />}
                {!isLoading && parsedOutput.length > 0 && parsedOutput.map((item, index) => (
                    <OutputCard key={index} title={item.title} content={item.content} />
                ))}
                {!isLoading && parsedOutput.length === 0 && (
                    <div className="text-center text-slate-400 p-8 bg-slate-800/50 rounded-xl border border-slate-700">
                        {errorMessage ? <p className="text-red-400">{errorMessage}</p> : <p>Hasil analisis dan rekomendasi akan muncul di sini.</p>}
                    </div>
                )}
            </div>
        </div>
      </div>
    );
};

export default AnalyticStudio;