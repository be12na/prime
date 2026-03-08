import React, { useState, useMemo } from 'react';
import { LAUNCH_OPTIONS as OPTIONS } from '../../constants.ts';
import { generateLaunchContent } from '../../services/geminiService.ts';
import { savePrompt } from '../../services/promptHistoryService.ts';
import Markdown from 'react-markdown';

interface ParsedOutput {
    title: string;
    content: string;
}

const LaunchStudio: React.FC = () => {
    const [activeTab, setActiveTab] = useState<'product' | 'ad'>('product');
    const [isLoading, setIsLoading] = useState(false);
    const [parsedOutput, setParsedOutput] = useState<ParsedOutput[]>([]);
    const [savedStatuses, setSavedStatuses] = useState<{ [key: string]: boolean }>({});
    const [errorMessage, setErrorMessage] = useState('');


    const [formState, setFormState] = useState({
        productName: '',
        brandName: '',
        productType: OPTIONS.productType[0],
        productDesc: '',
        productPrice: '',
        audienceAge: '',
        audienceGender: 'Semua Jenis Kelamin',
        audienceLocation: '',
        audienceDesc: '',
        adObjective: OPTIONS.adObjective[0],
        adFeeling: OPTIONS.adFeeling[0],
        adCTA: OPTIONS.adCTA[0],
        adPromo: '',
    });

    const handleChange = (e: React.ChangeEvent<HTMLInputElement | HTMLTextAreaElement | HTMLSelectElement>) => {
        const { id, value } = e.target;
        // Map HTML IDs to state keys
        const keyMap: { [key: string]: string } = {
            'launch-product-name': 'productName',
            'launch-brand-name': 'brandName',
            'launch-product-type': 'productType',
            'launch-product-desc': 'productDesc',
            'launch-product-price': 'productPrice',
            'launch-audience-age': 'audienceAge',
            'launch-audience-gender': 'audienceGender',
            'launch-audience-location': 'audienceLocation',
            'launch-audience-desc': 'audienceDesc',
            'ad-launch-objective': 'adObjective',
            'ad-launch-feeling': 'adFeeling',
            'ad-launch-cta': 'adCTA',
            'ad-launch-promo': 'adPromo',
        };
        const stateKey = keyMap[id] as keyof typeof formState;
        if (stateKey) {
            setFormState(prev => ({ ...prev, [stateKey]: value }));
        }
    };
    
    const parseMarkdownOutput = (markdown: string): ParsedOutput[] => {
        if (!markdown || typeof markdown !== 'string') return [];
        // Split by markdown headings (## or ###)
        const sections = markdown.split(/\n###? /).filter(s => s.trim() !== '');
        
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
            const result = await generateLaunchContent(formState, activeTab);
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

    const copyToClipboard = (text: string) => {
        navigator.clipboard.writeText(text).catch(() => {});
    };
    
    const handleSave = (title: string, content: string) => {
        if (!content) return;
        savePrompt('product-launch', title, content);
        setSavedStatuses(prev => ({...prev, [title]: true}));
        setTimeout(() => {
            setSavedStatuses(prev => ({...prev, [title]: false}));
        }, 2000);
    };

    const buttonText = useMemo(() => {
        if (isLoading) return 'Menghasilkan...';
        return activeTab === 'product' ? 'Hasilkan Konten Produk' : 'Hasilkan Konten Iklan';
    }, [isLoading, activeTab]);

    // Fix: Defined props interface and used React.FC to correctly type the component for use with a 'key' prop in a list.
    interface OutputCardProps {
        title: string;
        content: string;
    }
    const OutputCard: React.FC<OutputCardProps> = ({ title, content }) => (
        <div className="relative bg-slate-900/50 p-6 rounded-xl border border-slate-700">
            <h4 className="font-semibold text-lg text-orange-400 mb-3">{title}</h4>
            <div className="absolute top-4 right-4 flex items-center gap-2">
                <button onClick={() => handleSave(title, content)} title="Simpan" className="p-2 text-slate-400 hover:text-orange-400 bg-slate-700/50 hover:bg-slate-700 rounded-md transition-colors" disabled={savedStatuses[title]}>
                    {savedStatuses[title] ? <svg className="w-5 h-5 text-green-400" xmlns="http://www.w3.org/2000/svg" fill="none" viewBox="0 0 24 24" stroke="currentColor"><path strokeLinecap="round" strokeLinejoin="round" strokeWidth="2" d="M5 13l4 4L19 7" /></svg> : <svg className="w-5 h-5" xmlns="http://www.w3.org/2000/svg" fill="none" viewBox="0 0 24 24" stroke="currentColor" strokeWidth="2"><path strokeLinecap="round" strokeLinejoin="round" d="M5 5a2 2 0 012-2h8l2 2v13.5a1.5 1.5 0 01-3 0V6H5a1 1 0 011-1z" /><path strokeLinecap="round" strokeLinejoin="round" d="M15 12a3 3 0 11-6 0 3 3 0 016 0z" /></svg>}
                </button>
                <button onClick={() => copyToClipboard(content)} title="Salin" className="p-2 text-slate-400 hover:text-white bg-slate-700/50 rounded-md transition-colors">
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
            <div className="bg-slate-800/50 p-6 rounded-xl space-y-3 animate-pulse"><div className="h-5 w-1/3 bg-slate-700 rounded"></div><div className="h-4 w-2/3 bg-slate-700 rounded"></div><div className="h-4 w-1/2 bg-slate-700 rounded"></div></div>
            <div className="bg-slate-800/50 p-6 rounded-xl space-y-3 animate-pulse"><div className="h-5 w-1/3 bg-slate-700 rounded"></div><div className="h-4 w-2/3 bg-slate-700 rounded"></div><div className="h-4 w-1/2 bg-slate-700 rounded"></div></div>
            <div className="bg-slate-800/50 p-6 rounded-xl space-y-3 animate-pulse"><div className="h-5 w-1/4 bg-slate-700 rounded"></div><div className="h-4 w-full bg-slate-700 rounded"></div><div className="h-4 w-full bg-slate-700 rounded"></div><div className="h-4 w-5/6 bg-slate-700 rounded"></div></div>
        </div>
    );
    
    return (
        <div id="product-launch-studio" className="space-y-8">
            <div className="text-center px-4">
                <h2 className="text-3xl sm:text-4xl md:text-5xl font-display font-bold tracking-tight text-transparent bg-clip-text bg-gradient-to-r from-orange-400 to-amber-400">Studio Launching</h2>
                <p className="mt-4 text-base sm:text-lg text-slate-400 max-w-2xl mx-auto">
                    Ciptakan aset copywriting dan strategi iklan untuk produk baru Anda.
                </p>
            </div>

            <div className="grid grid-cols-1 lg:grid-cols-2 gap-8 items-start">
                <div className="bg-slate-800/50 backdrop-blur-lg border border-slate-700 rounded-xl p-4 sm:p-6 md:p-8 space-y-8 shadow-2xl shadow-black/30 lg:sticky top-24">
                    <div className="flex border-b border-slate-700">
                        <button onClick={() => setActiveTab('product')} className={`flex-1 py-3 text-sm font-semibold border-b-2 transition-colors duration-200 ${activeTab === 'product' ? 'text-orange-400 border-orange-400' : 'text-slate-400 border-transparent hover:border-slate-500'}`}>
                            <span className="flex items-center justify-center gap-2">
                                <svg xmlns="http://www.w3.org/2000/svg" className="h-5 w-5" viewBox="0 0 20 20" fill="currentColor"><path d="M5 4a2 2 0 012-2h6a2 2 0 012 2v1H5V4zM5 8h10a1 1 0 011 1v6a1 1 0 01-1 1H5a1 1 0 01-1-1V9a1 1 0 011-1z" /></svg>
                                Luncurkan Produk
                            </span>
                        </button>
                        <button onClick={() => setActiveTab('ad')} className={`flex-1 py-3 text-sm font-semibold border-b-2 transition-colors duration-200 ${activeTab === 'ad' ? 'text-orange-400 border-orange-400' : 'text-slate-400 border-transparent hover:border-slate-500'}`}>
                            <span className="flex items-center justify-center gap-2">
                                <svg xmlns="http://www.w3.org/2000/svg" className="h-5 w-5" viewBox="0 0 20 20" fill="currentColor"><path d="M15 5.5a3 3 0 11-6 0 3 3 0 016 0zM8.5 12.5a4.5 4.5 0 109 0 4.5 4.5 0 00-9 0zM2 10a5 5 0 1110 0 5 5 0 01-10 0zM15 10a5 5 0 11-10 0 5 5 0 0110 0z" /></svg>
                                Luncurkan Iklan
                            </span>
                        </button>
                    </div>

                    <div className="space-y-6">
                        <h3 className="flex items-center text-lg font-semibold gap-3 text-transparent bg-clip-text bg-gradient-to-r from-orange-400 to-amber-400">
                            <svg xmlns="http://www.w3.org/2000/svg" className="h-6 w-6 text-orange-500" fill="none" viewBox="0 0 24 24" stroke="currentColor" strokeWidth="2"><path strokeLinecap="round" strokeLinejoin="round" d="M9 5H7a2 2 0 00-2 2v12a2 2 0 002 2h10a2 2 0 002-2V7a2 2 0 00-2-2h-2M9 5a2 2 0 002 2h2a2 2 0 002-2M9 5a2 2 0 012-2h2a2 2 0 012 2m-3 7h3m-3 4h3m-6-4h.01M9 16h.01" /></svg>
                            <span>1. Informasi Produk & Audiens</span>
                        </h3>
                        <div className="space-y-4">
                            <div className="grid grid-cols-1 sm:grid-cols-2 gap-4">
                                <div>
                                    <label htmlFor="launch-product-name" className="block text-sm font-medium text-slate-400 mb-2">Nama Produk</label>
                                    <input type="text" id="launch-product-name" value={formState.productName} onChange={handleChange} className="block w-full bg-slate-700/50 border-slate-600 rounded-md shadow-sm py-2.5 px-4 text-slate-100 focus:outline-none focus:ring-2 focus:ring-orange-500 focus:border-orange-500 sm:text-sm transition" placeholder="Contoh: Kemeja Flanel 'Nusantara'" />
                                </div>
                                <div>
                                    <label htmlFor="launch-brand-name" className="block text-sm font-medium text-slate-400 mb-2">Nama Brand / Merk</label>
                                    <input type="text" id="launch-brand-name" value={formState.brandName} onChange={handleChange} className="block w-full bg-slate-700/50 border-slate-600 rounded-md shadow-sm py-2.5 px-4 text-slate-100 focus:outline-none focus:ring-2 focus:ring-orange-500 focus:border-orange-500 sm:text-sm transition" placeholder="Contoh: Gaya Lokal" />
                                </div>
                            </div>
                            <div>
                                <label htmlFor="launch-product-type" className="block text-sm font-medium text-slate-400 mb-2">Jenis Produk</label>
                                <select id="launch-product-type" value={formState.productType} onChange={handleChange} className="block w-full bg-slate-700/50 border-slate-600 rounded-md shadow-sm py-2.5 px-4 text-slate-100 focus:outline-none focus:ring-2 focus:ring-orange-500 focus:border-orange-500 sm:text-sm transition">
                                    {OPTIONS.productType.map(opt => <option key={opt} value={opt}>{opt}</option>)}
                                </select>
                            </div>
                            <div>
                                <label htmlFor="launch-product-desc" className="block text-sm font-medium text-slate-400 mb-2">Deskripsi & Fitur Kunci</label>
                                <textarea id="launch-product-desc" rows={4} value={formState.productDesc} onChange={handleChange} className="block w-full bg-slate-700/50 border-slate-600 rounded-md shadow-sm py-2.5 px-4 text-slate-100 focus:outline-none focus:ring-2 focus:ring-orange-500 focus:border-orange-500 sm:text-sm transition" placeholder="- Bahan katun premium, adem dan lembut&#10;- Jahitan kuat dengan benang ganda&#10;- Motif eksklusif terinspirasi budaya lokal"></textarea>
                            </div>
                            <div>
                                <label htmlFor="launch-product-price" className="block text-sm font-medium text-slate-400 mb-2">Harga Produk</label>
                                <input type="text" id="launch-product-price" value={formState.productPrice} onChange={handleChange} className="block w-full bg-slate-700/50 border-slate-600 rounded-md shadow-sm py-2.5 px-4 text-slate-100 focus:outline-none focus:ring-2 focus:ring-orange-500 focus:border-orange-500 sm:text-sm transition" placeholder="Rp 199.000" />
                            </div>
                             <div className="pt-2">
                                <p className="block text-sm font-medium text-slate-400 mb-2">Detail Target Audiens</p>
                                <div className="grid grid-cols-1 sm:grid-cols-2 gap-4">
                                   <div>
                                      <label htmlFor="launch-audience-age" className="sr-only">Usia</label>
                                      <input type="text" id="launch-audience-age" value={formState.audienceAge} onChange={handleChange} className="block w-full bg-slate-700/50 border-slate-600 rounded-md shadow-sm py-2.5 px-4 text-slate-100 focus:outline-none focus:ring-2 focus:ring-orange-500 focus:border-orange-500 sm:text-sm transition" placeholder="Usia, cth: 25-40" />
                                   </div>
                                   <div>
                                      <label htmlFor="launch-audience-gender" className="sr-only">Jenis Kelamin</label>
                                      <select id="launch-audience-gender" value={formState.audienceGender} onChange={handleChange} className="block w-full bg-slate-700/50 border-slate-600 rounded-md shadow-sm py-2.5 px-4 text-slate-100 focus:outline-none focus:ring-2 focus:ring-orange-500 focus:border-orange-500 sm:text-sm transition">
                                        <option>Semua Jenis Kelamin</option>
                                        <option>Pria</option>
                                        <option>Wanita</option>
                                      </select>
                                   </div>
                                   <div className="sm:col-span-2">
                                      <label htmlFor="launch-audience-location" className="sr-only">Lokasi</label>
                                      <input type="text" id="launch-audience-location" value={formState.audienceLocation} onChange={handleChange} className="block w-full bg-slate-700/50 border-slate-600 rounded-md shadow-sm py-2.5 px-4 text-slate-100 focus:outline-none focus:ring-2 focus:ring-orange-500 focus:border-orange-500 sm:text-sm transition" placeholder="Lokasi, cth: Kota besar di Indonesia" />
                                   </div>
                                   <div className="sm:col-span-2">
                                        <label htmlFor="launch-audience-desc" className="sr-only">Deskripsi Target Audience</label>
                                        <textarea id="launch-audience-desc" rows={3} value={formState.audienceDesc} onChange={handleChange} className="block w-full bg-slate-700/50 border-slate-600 rounded-md shadow-sm py-2.5 px-4 text-slate-100 focus:outline-none focus:ring-2 focus:ring-orange-500 focus:border-orange-500 sm:text-sm transition" placeholder="Jelaskan target audiens Anda. Cth: Ibu rumah tangga yang mencari baju anak berkualitas, atau pekerja kantoran yang ingin hemat."></textarea>
                                   </div>
                                </div>
                            </div>
                        </div>
                    </div>
                    
                    <div className="pt-6 border-t border-slate-700">
                        <div id="prod-launch-specifics" className={`${activeTab === 'product' ? '' : 'hidden'}`}>
                             <h3 className="flex items-center text-lg font-semibold gap-3 text-transparent bg-clip-text bg-gradient-to-r from-orange-400 to-amber-400">
                                <svg xmlns="http://www.w3.org/2000/svg" className="h-6 w-6 text-orange-500" fill="none" viewBox="0 0 24 24" stroke="currentColor" strokeWidth="2"><path strokeLinecap="round" strokeLinejoin="round" d="M13 10V3L4 14h7v7l9-11h-7z" /></svg>
                                <span>2. Hasilkan Aset Copywriting</span>
                              </h3>
                            <p className="text-sm text-slate-400 mt-2">AI akan menganalisis detail di atas untuk menghasilkan nama, deskripsi, dan highlight produk untuk halaman e-commerce Anda.</p>
                        </div>

                        <div id="ad-launch-specifics" className={`${activeTab === 'ad' ? '' : 'hidden'} space-y-6`}>
                          <h3 className="flex items-center text-lg font-semibold gap-3 text-transparent bg-clip-text bg-gradient-to-r from-orange-400 to-amber-400">
                            <svg xmlns="http://www.w3.org/2000/svg" className="h-6 w-6 text-orange-500" fill="none" viewBox="0 0 24 24" stroke="currentColor" strokeWidth="2"><path strokeLinecap="round" strokeLinejoin="round" d="M15 15l-2 5L9 9l11 4-5 2zm0 0l5 5M7.188 2.239l.777 2.897M5.136 7.965l-2.898-.777M13.95 4.05l-2.122 2.122m-5.657 5.656l-2.12 2.122" /></svg>
                            <span>2. Rancang Iklan Konversi Tinggi</span>
                          </h3>
                          <p className="text-sm text-slate-400 -mt-4">Lengkapi detail strategis di bawah untuk merancang kampanye TikTok Ads yang viral.</p>
                          <div className="space-y-4">
                              <div className="grid grid-cols-1 sm:grid-cols-2 gap-4">
                                  <div>
                                    <label htmlFor="ad-launch-objective" className="block text-sm font-medium text-slate-400 mb-2">Tujuan Iklan</label>
                                    <select id="ad-launch-objective" value={formState.adObjective} onChange={handleChange} className="block w-full bg-slate-700/50 border-slate-600 rounded-md shadow-sm py-2.5 px-4 text-slate-100 focus:outline-none focus:ring-2 focus:ring-orange-500 focus:border-orange-500 sm:text-sm transition">
                                        {OPTIONS.adObjective.map(opt => <option key={opt} value={opt}>{opt}</option>)}
                                    </select>
                                  </div>
                                  <div>
                                      <label htmlFor="ad-launch-feeling" className="block text-sm font-medium text-slate-400 mb-2">Emosi yang Diinginkan</label>
                                      <select id="ad-launch-feeling" value={formState.adFeeling} onChange={handleChange} className="block w-full bg-slate-700/50 border-slate-600 rounded-md shadow-sm py-2.5 px-4 text-slate-100 focus:outline-none focus:ring-2 focus:ring-orange-500 focus:border-orange-500 sm:text-sm transition">
                                        {OPTIONS.adFeeling.map(opt => <option key={opt} value={opt}>{opt}</option>)}
                                      </select>
                                  </div>
                              </div>
                              <div className="grid grid-cols-1 sm:grid-cols-2 gap-4">
                                  <div>
                                      <label htmlFor="ad-launch-cta" className="block text-sm font-medium text-slate-400 mb-2">Call to Action (CTA)</label>
                                      <select id="ad-launch-cta" value={formState.adCTA} onChange={handleChange} className="block w-full bg-slate-700/50 border-slate-600 rounded-md shadow-sm py-2.5 px-4 text-slate-100 focus:outline-none focus:ring-2 focus:ring-orange-500 focus:border-orange-500 sm:text-sm transition">
                                        {OPTIONS.adCTA.map(opt => <option key={opt} value={opt}>{opt}</option>)}
                                      </select>
                                  </div>
                                  <div>
                                      <label htmlFor="ad-launch-promo" className="block text-sm font-medium text-slate-400 mb-2">Promo / Penawaran (Opsional)</label>
                                      <input type="text" id="ad-launch-promo" value={formState.adPromo} onChange={handleChange} className="block w-full bg-slate-700/50 border-slate-600 rounded-md shadow-sm py-2.5 px-4 text-slate-100 focus:outline-none focus:ring-2 focus:ring-orange-500 focus:border-orange-500 sm:text-sm transition" placeholder="Cth: Diskon 50%, Beli 1 Gratis 1" />
                                  </div>
                              </div>
                          </div>
                        </div>
                    </div>

                    <div className="pt-4">
                        <button onClick={handleGenerate} disabled={isLoading} className="w-full inline-flex items-center justify-center px-12 py-4 text-lg font-semibold text-white bg-gradient-to-r from-orange-500 to-amber-500 rounded-lg shadow-lg shadow-orange-500/40 hover:shadow-xl hover:shadow-orange-500/40 focus:outline-none focus:ring-2 focus:ring-offset-2 focus:ring-offset-slate-900 focus:ring-orange-500 transition-all duration-300 ease-in-out transform hover:-translate-y-0.5 disabled:from-slate-600 disabled:to-slate-600 disabled:shadow-none disabled:translate-y-0 disabled:cursor-not-allowed">
                            {isLoading && <svg className="animate-spin -ml-1 mr-3 h-5 w-5 text-white" xmlns="http://www.w3.org/2000/svg" fill="none" viewBox="0 0 24 24"><circle className="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" strokeWidth="4"></circle><path className="opacity-75" fill="currentColor" d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4zm2 5.291A7.962 7.962 0 014 12H0c0 3.042 1.135 5.824 3 7.938l3-2.647z"></path></svg>}
                            <span>{buttonText}</span>
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
                            {errorMessage ? <p className="text-red-400">{errorMessage}</p> : <p>Hasil copywriting akan muncul di sini setelah Anda menekan tombol generate.</p>}
                        </div>
                   )}
                </div>
            </div>
        </div>
    );
};

export default LaunchStudio;