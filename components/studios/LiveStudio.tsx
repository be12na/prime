import React, { useState } from 'react';
import { LIVE_OPTIONS } from '../../constants.ts';
import { generateLiveScript, generatePepTalk } from '../../services/geminiService.ts';
import { savePrompt } from '../../services/promptHistoryService.ts';
import { LiveProduct, TimelineSegment } from '../../types.ts';

type LiveMode = 'shopping' | 'gift' | 'qa';

const initialProduct: LiveProduct = {
    id: 1, name: '', price: '', info: '', features: '', benefits: ''
};

const LiveStudio: React.FC = () => {
    const [view, setView] = useState<'setup' | 'output'>('setup');
    const [activeTab, setActiveTab] = useState<LiveMode>('shopping');
    const [isLoading, setIsLoading] = useState(false);
    const [isPepTalkLoading, setIsPepTalkLoading] = useState(false);
    const [pepTalk, setPepTalk] = useState('');
    const [isScriptSaved, setIsScriptSaved] = useState(false);
    const [errorMessage, setErrorMessage] = useState('');
    
    const [shoppingForm, setShoppingForm] = useState({
        goal: LIVE_OPTIONS.goal[0],
        style: LIVE_OPTIONS.style[0],
        audienceTone: '',
        duration: '60',
        strategy: '',
        products: [initialProduct],
    });
    
    const [giftForm, setGiftForm] = useState({
        goal: LIVE_OPTIONS.giftGoal[0],
        style: LIVE_OPTIONS.giftStyle[0],
        audienceTone: '',
        duration: '60',
        giftList: '',
        strategy: '',
    });

    const [qaForm, setQaForm] = useState({
        topic: '',
        style: LIVE_OPTIONS.qaStyle[0],
        audienceTone: '',
        duration: '45',
        expertise: '',
        strategy: '',
    });

    const [mentalForm, setMentalForm] = useState({ mood: 'nervous', reason: '' });
    const [scriptMode, setScriptMode] = useState(LIVE_OPTIONS.scriptMode[0].id);
    
    const [timeline, setTimeline] = useState<TimelineSegment[]>([]);

    const handleShoppingChange = (e: React.ChangeEvent<HTMLInputElement | HTMLTextAreaElement | HTMLSelectElement>) => {
        const { id, value } = e.target;
        const keyMap: {[key: string]: string} = { 'live-goal': 'goal', 'live-style': 'style', 'live-audience-tone': 'audienceTone', 'live-duration': 'duration', 'live-shopping-strategy': 'strategy' };
        const stateKey = keyMap[id];
        if (stateKey) setShoppingForm(prev => ({ ...prev, [stateKey]: value }));
    };

    const handleGiftChange = (e: React.ChangeEvent<HTMLInputElement | HTMLTextAreaElement | HTMLSelectElement>) => {
        const { id, value } = e.target;
        const keyMap: {[key: string]: string} = { 'gift-live-goal': 'goal', 'gift-live-style': 'style', 'gift-live-audience-tone': 'audienceTone', 'gift-live-duration': 'duration', 'gift-live-gift-list': 'giftList', 'live-gift-strategy': 'strategy' };
        const stateKey = keyMap[id];
        if (stateKey) setGiftForm(prev => ({ ...prev, [stateKey]: value }));
    };

    const handleQaChange = (e: React.ChangeEvent<HTMLInputElement | HTMLTextAreaElement | HTMLSelectElement>) => {
        const { id, value } = e.target;
        const keyMap: {[key: string]: string} = { 'qa-live-topic': 'topic', 'qa-live-style': 'style', 'qa-live-audience-tone': 'audienceTone', 'qa-live-duration': 'duration', 'qa-live-expertise': 'expertise', 'live-qa-strategy': 'strategy' };
        const stateKey = keyMap[id];
        if (stateKey) setQaForm(prev => ({ ...prev, [stateKey]: value }));
    };
    
    const handleMentalChange = (e: React.ChangeEvent<HTMLInputElement | HTMLTextAreaElement | HTMLSelectElement>) => {
        const { id, value } = e.target;
        const keyMap: {[key: string]: string} = { 'live-mental-mood': 'mood', 'live-mental-reason': 'reason' };
        const stateKey = keyMap[id];
        if (stateKey) setMentalForm(prev => ({ ...prev, [stateKey]: value }));
    };

    const handleAddProduct = () => {
        setShoppingForm(prev => ({
            ...prev,
            products: [...prev.products, { ...initialProduct, id: Date.now() }]
        }));
    };
    
    const handleProductChange = (id: number, field: keyof Omit<LiveProduct, 'id'>, value: string) => {
        setShoppingForm(prev => ({
            ...prev,
            products: prev.products.map(p => p.id === id ? { ...p, [field]: value } : p)
        }));
    };
    
    const handleRemoveProduct = (id: number) => {
        setShoppingForm(prev => ({
            ...prev,
            products: prev.products.length > 1 ? prev.products.filter(p => p.id !== id) : prev.products
        }));
    };

    const handleGenerateScript = async () => {
        setIsLoading(true);
        setTimeline([]);
        setErrorMessage('');
        try {
            let specificFormData = {};
            switch (activeTab) {
                case 'shopping': specificFormData = shoppingForm; break;
                case 'gift': specificFormData = giftForm; break;
                case 'qa': specificFormData = qaForm; break;
            }
            
            const formData = {
                mode: activeTab,
                scriptMode,
                mentalNote: `Host feels ${mentalForm.mood} because: ${mentalForm.reason}`,
                ...specificFormData,
            };
            const resultJson = await generateLiveScript(formData);
            const result = JSON.parse(resultJson);
            if (result.timeline) {
                setTimeline(result.timeline);
                setView('output');
            } else {
                setErrorMessage(`Gagal membuat skrip: ${result.error || "Format respons tidak valid."}`);
            }
        } catch(e) {
            console.error("Failed to parse timeline JSON:", e);
            setErrorMessage("Gagal memproses respons dari AI. Silakan coba lagi.");
        } finally {
            setIsLoading(false);
        }
    };

    const handlePepTalk = async () => {
        if (!mentalForm.reason.trim()) return;
        setIsPepTalkLoading(true);
        setPepTalk('');
        try {
            const talk = await generatePepTalk(mentalForm.mood, mentalForm.reason);
            setPepTalk(talk);
        } catch (error) {
            const msg = error instanceof Error ? error.message : 'Terjadi kesalahan.';
            setPepTalk(`Error: ${msg}`);
        } finally {
            setIsPepTalkLoading(false);
        }
    };
    
    const handleSaveScript = () => {
        if (timeline.length === 0) return;
        
        let title = 'Skrip Live';
        if (activeTab === 'shopping') title = `Skrip Live Shopping: ${shoppingForm.goal}`;
        else if (activeTab === 'gift') title = `Skrip Live Gift: ${giftForm.goal}`;
        else if (activeTab === 'qa') title = `Skrip Live Q&A: ${qaForm.topic}`;

        const content = timeline.map(t => `[${t.time}] ${t.title}\n${t.script}`).join('\n\n');
        savePrompt('live', title, content);
        setIsScriptSaved(true);
        setTimeout(() => setIsScriptSaved(false), 2000);
    };
    
    // Fix: Defined props interface and used React.FC to correctly type the component for use with a 'key' prop in a list.
    interface ProductCardProps {
        product: LiveProduct;
        index: number;
    }

    const ProductCard: React.FC<ProductCardProps> = ({ product, index }) => (
        <div className="bg-slate-900/50 p-4 rounded-lg space-y-3 border border-slate-700 relative">
            <div className="flex justify-between items-center">
                <span className="font-bold text-slate-400">Produk #{index + 1}</span>
                <button onClick={() => handleRemoveProduct(product.id)} disabled={shoppingForm.products.length <= 1} className="text-slate-400 hover:text-red-400 font-bold text-2xl leading-none disabled:opacity-30 disabled:cursor-not-allowed">&times;</button>
            </div>
             <div className="grid grid-cols-1 sm:grid-cols-2 gap-3">
                <input type="text" placeholder="Nama Produk" value={product.name} onChange={(e) => handleProductChange(product.id, 'name', e.target.value)} className="w-full bg-slate-700/50 p-2 rounded-md placeholder-slate-500 text-sm"/>
                <input type="text" placeholder="Harga (cth: Rp 99.000)" value={product.price} onChange={(e) => handleProductChange(product.id, 'price', e.target.value)} className="w-full bg-slate-700/50 p-2 rounded-md placeholder-slate-500 text-sm"/>
            </div>
            <textarea placeholder="Info singkat (cth: Bahan, ukuran, warna)" value={product.info} onChange={(e) => handleProductChange(product.id, 'info', e.target.value)} rows={2} className="w-full bg-slate-700/50 p-2 rounded-md placeholder-slate-500 text-sm" />
            <textarea placeholder="Fitur Kunci (1 per baris)" value={product.features} onChange={(e) => handleProductChange(product.id, 'features', e.target.value)} rows={3} className="w-full bg-slate-700/50 p-2 rounded-md placeholder-slate-500 text-sm" />
            <textarea placeholder="Manfaat untuk Customer (1 per baris)" value={product.benefits} onChange={(e) => handleProductChange(product.id, 'benefits', e.target.value)} rows={3} className="w-full bg-slate-700/50 p-2 rounded-md placeholder-slate-500 text-sm" />
        </div>
    );

    const ScriptSkeleton = () => (
        <div className="p-4">
            <div className="space-y-4 animate-pulse">
                {[...Array(3)].map((_, i) => (
                    <div key={i} className="flex gap-4 items-start p-4 border-b border-slate-700 last:border-0"><div className="h-5 w-16 bg-slate-700 rounded mt-1"></div><div className="h-8 w-8 bg-slate-700 rounded-full"></div><div className="flex-1 space-y-2"><div className="h-5 w-1/3 bg-slate-700 rounded"></div><div className="h-4 w-full bg-slate-700 rounded"></div><div className="h-4 w-3/4 bg-slate-700 rounded"></div></div></div>
                ))}
            </div>
        </div>
    );

    return (
        <div id="live-studio" className="space-y-8">
            {view === 'setup' ? (
                <div id="live-setup-view">
                    <div className="text-center px-4">
                        <h2 className="text-3xl sm:text-4xl md:text-5xl font-display font-bold tracking-tight text-transparent bg-clip-text bg-gradient-to-r from-orange-400 to-amber-400">Studio Live</h2>
                        <p className="mt-4 text-base sm:text-lg text-slate-400 max-w-2xl mx-auto">Asisten cerdas untuk siaran langsung Anda, baik untuk jualan, hiburan, maupun edukasi.</p>
                    </div>

                    <div id="live-setup-controls" className="mt-8 max-w-4xl mx-auto bg-slate-800/50 backdrop-blur-lg border border-slate-700 rounded-xl p-4 sm:p-6 md:p-8 space-y-8 shadow-2xl shadow-black/30">
                        <div className="flex border-b border-slate-700">
                           <button onClick={() => setActiveTab('shopping')} className={`flex-1 py-3 text-sm font-semibold border-b-2 transition-colors duration-200 ${activeTab === 'shopping' ? 'text-orange-400 border-orange-400' : 'text-slate-400 border-transparent hover:border-slate-500'}`}><span className="flex items-center justify-center gap-2"><svg xmlns="http://www.w3.org/2000/svg" className="h-5 w-5" viewBox="0 0 20 20" fill="currentColor"><path d="M3 1a1 1 0 000 2h1.22l.305 1.222a.997.997 0 00.01.042l1.358 5.43-.893.892C3.74 11.846 4.632 14 6.414 14H15a1 1 0 000-2H6.414l1-1H14a1 1 0 00.894-.553l3-6A1 1 0 0017 3H6.28l-.31-1.243A1 1 0 005 1H3zM16 16.5a1.5 1.5 0 11-3 0 1.5 1.5 0 013 0zM6.5 18a1.5 1.5 0 100-3 1.5 1.5 0 000 3z" /></svg>Live Shopping</span></button>
                           <button onClick={() => setActiveTab('gift')} className={`flex-1 py-3 text-sm font-semibold border-b-2 transition-colors duration-200 ${activeTab === 'gift' ? 'text-orange-400 border-orange-400' : 'text-slate-400 border-transparent hover:border-slate-500'}`}><span className="flex items-center justify-center gap-2"><svg xmlns="http://www.w3.org/2000/svg" className="h-5 w-5" viewBox="0 0 20 20" fill="currentColor"><path fillRule="evenodd" d="M5 5a3 3 0 015-2.236A3 3 0 0114.83 6H16a2 2 0 110 4h-5v3a1 1 0 01-2 0v-3H4a2 2 0 110-4h1.17C5.06 5.687 5 5.35 5 5zm4 1V5a1 1 0 10-2 0v1h2zm3 0V5a1 1 0 10-2 0v1h2z" clipRule="evenodd" /></svg>Live Gift</span></button>
                           <button onClick={() => setActiveTab('qa')} className={`flex-1 py-3 text-sm font-semibold border-b-2 transition-colors duration-200 ${activeTab === 'qa' ? 'text-orange-400 border-orange-400' : 'text-slate-400 border-transparent hover:border-slate-500'}`}><span className="flex items-center justify-center gap-2"><svg xmlns="http://www.w3.org/2000/svg" className="h-5 w-5" viewBox="0 0 20 20" fill="currentColor"><path fillRule="evenodd" d="M18 10a8 8 0 11-16 0 8 8 0 0116 0zm-8-3a1 1 0 00-.867.5 1 1 0 11-1.731-1A3 3 0 0113 8a3.001 3.001 0 01-2 2.83V11a1 1 0 11-2 0v-1a1 1 0 011-1 1 1 0 100-2zm0 8a1 1 0 100-2 1 1 0 000 2z" clipRule="evenodd" /></svg>Live Q&A</span></button>
                        </div>
                        
                        {/* Tab Contents */}
                        <div id="live-shopping-tab" className={activeTab !== 'shopping' ? 'hidden' : ''}>
                            {/* Shopping form content here */}
                            <div className="space-y-4 pt-6">
                               <h3 className="flex items-center text-xl font-semibold gap-3 text-transparent bg-clip-text bg-gradient-to-r from-orange-400 to-amber-400"><svg xmlns="http://www.w3.org/2000/svg" className="h-6 w-6 text-orange-500" fill="none" viewBox="0 0 24 24" stroke="currentColor" strokeWidth="2"><path strokeLinecap="round" strokeLinejoin="round" d="M9 5H7a2 2 0 00-2 2v12a2 2 0 002 2h10a2 2 0 002-2V7a2 2 0 00-2-2h-2M9 5a2 2 0 002 2h2a2 2 0 002-2M9 5a2 2 0 012-2h2a2 2 0 012 2m-3 7h3m-3 4h3m-6-4h.01M9 16h.01" /></svg><span>Perencanaan Sesi Live Shopping</span></h3>

                               <div className="grid grid-cols-1 sm:grid-cols-2 gap-6">
                                    <div><label htmlFor="live-goal" className="block text-sm font-medium text-slate-400 mb-2">Tujuan Utama Live</label><select id="live-goal" value={shoppingForm.goal} onChange={handleShoppingChange} className="block w-full bg-slate-700/50 border-slate-600 rounded-md shadow-sm py-2.5 px-4 text-slate-100 focus:outline-none focus:ring-2 focus:ring-orange-500 focus:border-orange-500 sm:text-sm transition">{LIVE_OPTIONS.goal.map(o=><option key={o}>{o}</option>)}</select></div>
                                    <div><label htmlFor="live-style" className="block text-sm font-medium text-slate-400 mb-2">Gaya Host</label><select id="live-style" value={shoppingForm.style} onChange={handleShoppingChange} className="block w-full bg-slate-700/50 border-slate-600 rounded-md shadow-sm py-2.5 px-4 text-slate-100 focus:outline-none focus:ring-2 focus:ring-orange-500 focus:border-orange-500 sm:text-sm transition">{LIVE_OPTIONS.style.map(o=><option key={o}>{o}</option>)}</select></div>
                                    <div><label htmlFor="live-audience-tone" className="block text-sm font-medium text-slate-400 mb-2">Sapaan Audiens</label><input id="live-audience-tone" type="text" value={shoppingForm.audienceTone} onChange={handleShoppingChange} className="block w-full bg-slate-700/50 border-slate-600 rounded-md shadow-sm py-2.5 px-4 text-slate-100 focus:outline-none focus:ring-2 focus:ring-orange-500 focus:border-orange-500 sm:text-sm transition" placeholder="Contoh: Juragan, Kak, Guys, Bund..."/></div>
                                    <div><label htmlFor="live-duration" className="block text-sm font-medium text-slate-400 mb-2">Durasi Live (Menit)</label><input id="live-duration" type="number" value={shoppingForm.duration} onChange={handleShoppingChange} className="block w-full bg-slate-700/50 border-slate-600 rounded-md shadow-sm py-2.5 px-4 text-slate-100 focus:outline-none focus:ring-2 focus:ring-orange-500 focus:border-orange-500 sm:text-sm transition"/></div>
                                    <div className="sm:col-span-2"><label htmlFor="live-shopping-strategy" className="block text-sm font-medium text-slate-400 mb-2">Strategi Live Anda (Opsional)</label><textarea id="live-shopping-strategy" value={shoppingForm.strategy} onChange={handleShoppingChange} rows={3} className="block w-full bg-slate-700/50 border-slate-600 rounded-md py-2.5 px-4 text-slate-100 sm:text-sm transition focus:ring-orange-500 focus:border-orange-500" placeholder="Jelaskan strategi Anda (cth: Fokus pada diskon, bangun interaksi dulu baru jualan, dll). Kosongkan jika ingin AI menentukan strategi terbaik."></textarea></div>
                                </div>
                            </div>
                            <div className="space-y-4 pt-8 border-t border-slate-700">
                                <h3 className="flex items-center text-xl font-semibold gap-3 text-transparent bg-clip-text bg-gradient-to-r from-orange-400 to-amber-400"><svg xmlns="http://www.w3.org/2000/svg" className="h-6 w-6 text-orange-500" fill="none" viewBox="0 0 24 24" stroke="currentColor" strokeWidth="2"><path strokeLinecap="round" strokeLinejoin="round" d="M16 11V7a4 4 0 00-8 0v4M5 9h14l1 12H4L5 9z" /></svg><span>Daftar Produk (Seret untuk Mengurutkan)</span></h3>
                                <div id="live-product-list" className="space-y-4">{shoppingForm.products.map((p, i) => <ProductCard key={p.id} product={p} index={i} />)}</div>
                                <button id="live-add-product-btn" onClick={handleAddProduct} className="w-full text-sm font-semibold text-orange-400 border-2 border-dashed border-slate-600 hover:border-orange-500 hover:text-orange-500 rounded-lg py-3 transition-colors">+ Tambah Produk Lain</button>
                                <div className="text-center mt-2"><span className="text-xs text-slate-400">atau </span><button id="live-bulk-upload-btn" type="button" onClick={() => { /* Fitur upload massal akan ditambahkan di versi mendatang */ }} className="text-xs font-semibold text-orange-400 hover:text-orange-300 hover:underline transition-colors" title="Fitur ini akan segera hadir">Unggah Massal (Segera Hadir)</button></div>
                            </div>
                        </div>

                        <div id="live-gift-tab" className={activeTab !== 'gift' ? 'hidden' : ''}>
                            {/* Gift form content here */}
                             <div className="space-y-4 pt-6">
                                <h3 className="flex items-center text-xl font-semibold gap-3 text-transparent bg-clip-text bg-gradient-to-r from-orange-400 to-amber-400"><svg xmlns="http://www.w3.org/2000/svg" className="h-6 w-6 text-orange-500" fill="none" viewBox="0 0 24 24" stroke="currentColor" strokeWidth="2"><path strokeLinecap="round" strokeLinejoin="round" d="M14.828 14.828a4 4 0 01-5.656 0M9 10h.01M15 10h.01M21 12a9 9 0 11-18 0 9 9 0 0118 0z" /></svg><span>Perencanaan Sesi Live Gift</span></h3>
                                <div className="grid grid-cols-1 sm:grid-cols-2 gap-6">
                                    <div><label htmlFor="gift-live-goal" className="block text-sm font-medium text-slate-400 mb-2">Tujuan Utama Live</label><select id="gift-live-goal" value={giftForm.goal} onChange={handleGiftChange} className="block w-full bg-slate-700/50 border-slate-600 rounded-md shadow-sm py-2.5 px-4 text-slate-100 focus:outline-none focus:ring-2 focus:ring-orange-500 focus:border-orange-500 sm:text-sm transition">{LIVE_OPTIONS.giftGoal.map(o=><option key={o}>{o}</option>)}</select></div>
                                    <div><label htmlFor="gift-live-style" className="block text-sm font-medium text-slate-400 mb-2">Gaya Host</label><select id="gift-live-style" value={giftForm.style} onChange={handleGiftChange} className="block w-full bg-slate-700/50 border-slate-600 rounded-md shadow-sm py-2.5 px-4 text-slate-100 focus:outline-none focus:ring-2 focus:ring-orange-500 focus:border-orange-500 sm:text-sm transition">{LIVE_OPTIONS.giftStyle.map(o=><option key={o}>{o}</option>)}</select></div>
                                    <div><label htmlFor="gift-live-audience-tone" className="block text-sm font-medium text-slate-400 mb-2">Sapaan Audiens</label><input id="gift-live-audience-tone" type="text" value={giftForm.audienceTone} onChange={handleGiftChange} className="block w-full bg-slate-700/50 border-slate-600 rounded-md shadow-sm py-2.5 px-4 text-slate-100 focus:outline-none focus:ring-2 focus:ring-orange-500 focus:border-orange-500 sm:text-sm transition" placeholder="Contoh: Warga, Sobat, Gengs..."/></div>
                                    <div><label htmlFor="gift-live-duration" className="block text-sm font-medium text-slate-400 mb-2">Durasi Live (Menit)</label><input id="gift-live-duration" type="number" value={giftForm.duration} onChange={handleGiftChange} className="block w-full bg-slate-700/50 border-slate-600 rounded-md shadow-sm py-2.5 px-4 text-slate-100 focus:outline-none focus:ring-2 focus:ring-orange-500 focus:border-orange-500 sm:text-sm transition"/></div>
                                    <div className="sm:col-span-2"><label htmlFor="gift-live-gift-list" className="block text-sm font-medium text-slate-400 mb-2">Sebutkan Gift atau Target yang Ingin Dicapai (Opsional)</label><textarea id="gift-live-gift-list" value={giftForm.giftList} onChange={handleGiftChange} rows={3} className="block w-full bg-slate-700/50 border-slate-600 rounded-md shadow-sm py-2.5 px-4 text-slate-100 focus:outline-none focus:ring-2 focus:ring-orange-500 focus:border-orange-500 sm:text-sm transition" placeholder="Contoh:&#10;Target 10 gift Paus&#10;Masuk top 100 rising star hari ini"></textarea></div>
                                    <div className="sm:col-span-2"><label htmlFor="live-gift-strategy" className="block text-sm font-medium text-slate-400 mb-2">Strategi Live Anda (Opsional)</label><textarea id="live-gift-strategy" value={giftForm.strategy} onChange={handleGiftChange} rows={3} className="block w-full bg-slate-700/50 border-slate-600 rounded-md py-2.5 px-4 text-slate-100 sm:text-sm transition focus:ring-orange-500 focus:border-orange-500" placeholder="Jelaskan strategi Anda (cth: Adakan challenge, balas semua gift dengan pantun, dll). Kosongkan jika ingin AI menentukan strategi terbaik."></textarea></div>
                                </div>
                            </div>
                        </div>

                        <div id="live-qa-tab" className={activeTab !== 'qa' ? 'hidden' : ''}>
                            {/* Q&A form content here */}
                             <div className="space-y-4 pt-6">
                                <h3 className="flex items-center text-xl font-semibold gap-3 text-transparent bg-clip-text bg-gradient-to-r from-orange-400 to-amber-400"><svg xmlns="http://www.w3.org/2000/svg" className="h-6 w-6 text-orange-500" viewBox="0 0 20 20" fill="currentColor"><path d="M11 17a1 1 0 001.447.894l4-2A1 1 0 0017 15V5a1 1 0 00-1.447-.894l-4 2A1 1 0 0011 7v10zM6 17a1 1 0 001.447.894l4-2A1 1 0 0012 15V5a1 1 0 00-1.447-.894l-4 2A1 1 0 006 7v10z" /></svg><span>Perencanaan Sesi Live Q&A</span></h3>
                                <div className="grid grid-cols-1 sm:grid-cols-2 gap-6">
                                    <div><label htmlFor="qa-live-topic" className="block text-sm font-medium text-slate-400 mb-2">Topik Utama Sesi</label><input id="qa-live-topic" type="text" value={qaForm.topic} onChange={handleQaChange} className="block w-full bg-slate-700/50 border-slate-600 rounded-md shadow-sm py-2.5 px-4 text-slate-100 focus:outline-none focus:ring-2 focus:ring-orange-500 focus:border-orange-500 sm:text-sm transition" placeholder="Contoh: Tips Investasi Saham Pemula"/></div>
                                    <div><label htmlFor="qa-live-style" className="block text-sm font-medium text-slate-400 mb-2">Gaya Host</label><select id="qa-live-style" value={qaForm.style} onChange={handleQaChange} className="block w-full bg-slate-700/50 border-slate-600 rounded-md shadow-sm py-2.5 px-4 text-slate-100 focus:outline-none focus:ring-2 focus:ring-orange-500 focus:border-orange-500 sm:text-sm transition">{LIVE_OPTIONS.qaStyle.map(o=><option key={o}>{o}</option>)}</select></div>
                                    <div><label htmlFor="qa-live-audience-tone" className="block text-sm font-medium text-slate-400 mb-2">Sapaan Audiens</label><input id="qa-live-audience-tone" type="text" value={qaForm.audienceTone} onChange={handleQaChange} className="block w-full bg-slate-700/50 border-slate-600 rounded-md shadow-sm py-2.5 px-4 text-slate-100 focus:outline-none focus:ring-2 focus:ring-orange-500 focus:border-orange-500 sm:text-sm transition" placeholder="Contoh: Teman Cerdas, Investor Muda..."/></div>
                                    <div><label htmlFor="qa-live-duration" className="block text-sm font-medium text-slate-400 mb-2">Durasi Live (Menit)</label><input id="qa-live-duration" type="number" value={qaForm.duration} onChange={handleQaChange} className="block w-full bg-slate-700/50 border-slate-600 rounded-md shadow-sm py-2.5 px-4 text-slate-100 focus:outline-none focus:ring-2 focus:ring-orange-500 focus:border-orange-500 sm:text-sm transition"/></div>
                                    <div className="sm:col-span-2"><label htmlFor="qa-live-expertise" className="block text-sm font-medium text-slate-400 mb-2">Keahlian Anda</label><input id="qa-live-expertise" type="text" value={qaForm.expertise} onChange={handleQaChange} className="block w-full bg-slate-700/50 border-slate-600 rounded-md shadow-sm py-2.5 px-4 text-slate-100 focus:outline-none focus:ring-2 focus:ring-orange-500 focus:border-orange-500 sm:text-sm transition" placeholder="Contoh: Saya adalah seorang Perencana Keuangan"/></div>
                                    <div className="sm:col-span-2"><label htmlFor="live-qa-strategy" className="block text-sm font-medium text-slate-400 mb-2">Strategi Live Anda (Opsional)</label><textarea id="live-qa-strategy" value={qaForm.strategy} onChange={handleQaChange} rows={3} className="block w-full bg-slate-700/50 border-slate-600 rounded-md py-2.5 px-4 text-slate-100 sm:text-sm transition focus:ring-orange-500 focus:border-orange-500" placeholder="Jelaskan strategi Anda (cth: Siapkan 3 pertanyaan pancingan, adakan polling di tengah sesi, dll). Kosongkan jika ingin AI menentukan strategi terbaik."></textarea></div>
                                </div>
                            </div>
                        </div>

                        {/* Shared Sections */}
                        <div className="space-y-4 pt-8 border-t border-slate-700">
                             <h3 className="flex items-center text-xl font-semibold gap-3 text-transparent bg-clip-text bg-gradient-to-r from-orange-400 to-amber-400"><svg xmlns="http://www.w3.org/2000/svg" className="h-6 w-6 text-orange-500" fill="none" viewBox="0 0 24 24" stroke="currentColor" strokeWidth="2"><path strokeLinecap="round" strokeLinejoin="round" d="M12 9v3.75m-9.303 3.376c-.866 1.5.217 3.374 1.948 3.374h14.71c1.73 0 2.813-1.874 1.948-3.374L13.949 3.378c-.866-1.5-3.032-1.5-3.898 0L2.697 16.126zM12 15.75h.007v.008H12v-.008z" /></svg><span>Pemanasan Mental (Opsional)</span></h3>
                             <div className="space-y-4">
                                <div><label htmlFor="live-mental-mood" className="block text-sm font-medium text-slate-400 mb-2">Bagaimana perasaan Anda?</label><select id="live-mental-mood" value={mentalForm.mood} onChange={handleMentalChange} className="block w-full bg-slate-700/50 border-slate-600 rounded-md shadow-sm py-2.5 px-4 text-slate-100 focus:outline-none focus:ring-2 focus:ring-orange-500 focus:border-orange-500 sm:text-sm transition"><option value="nervous">Sangat Gugup / Cemas</option><option value="calm">Biasa Saja / Sedikit Cemas</option><option value="ready">Siap Tempur! 🔥</option></select></div>
                                <div><label htmlFor="live-mental-reason" className="block text-sm font-medium text-slate-400 mb-2">Alasan nya apa?</label><textarea id="live-mental-reason" value={mentalForm.reason} onChange={handleMentalChange} rows={2} className="block w-full bg-slate-700/50 border-slate-600 rounded-md shadow-sm py-2.5 px-4 text-slate-100 focus:outline-none focus:ring-2 focus:ring-orange-500 focus:border-orange-500 sm:text-sm transition" placeholder="Contoh: Takut salah ngomong, cemas tidak ada yang nonton..."></textarea></div>
                                <div><button id="live-pep-talk-btn" onClick={handlePepTalk} disabled={isPepTalkLoading} className="w-full inline-flex items-center justify-center px-4 py-2.5 text-sm font-semibold text-orange-400 bg-slate-700/50 hover:bg-slate-700 rounded-md transition-colors">{isPepTalkLoading ? 'Memuat...' : 'Beri Saya Semangat!'}</button></div>
                            </div>
                            {pepTalk && <div id="live-pep-talk-output" className="mt-4 bg-slate-900/50 p-4 rounded-md text-slate-300 border border-slate-700 text-sm">{pepTalk}</div>}
                        </div>
                        <div className="space-y-3 pt-8 border-t border-slate-700">
                            <h3 className="flex items-center text-xl font-semibold gap-3 text-transparent bg-clip-text bg-gradient-to-r from-orange-400 to-amber-400"><svg xmlns="http://www.w3.org/2000/svg" className="h-6 w-6 text-orange-500" fill="none" viewBox="0 0 24 24" stroke="currentColor" strokeWidth="2"><path strokeLinecap="round" strokeLinejoin="round" d="M9 12h6m-6 4h6m2 5H7a2 2 0 01-2-2V5a2 2 0 012-2h5.586a1 1 0 01.707.293l5.414 5.414a1 1 0 01.293.707V19a2 2 0 01-2 2z" /></svg><span>Pilih Mode Skrip</span></h3>
                            <div id="live-script-mode-group" className="flex items-center space-x-2 bg-slate-700/50 rounded-lg p-1 h-auto">
                                {LIVE_OPTIONS.scriptMode.map(mode => <button key={mode.id} onClick={()=>setScriptMode(mode.id)} className={`flex-1 text-sm rounded-md py-1.5 transition-colors ${scriptMode === mode.id ? 'bg-orange-500 text-white' : 'hover:bg-slate-600 text-slate-300'}`}>{mode.label}</button>)}
                            </div>
                        </div>
                        <div className="pt-6">
                          <button id="generate-live-button" onClick={handleGenerateScript} disabled={isLoading} className="w-full inline-flex items-center justify-center px-12 py-4 text-lg font-semibold text-white bg-gradient-to-r from-orange-500 to-amber-500 rounded-lg shadow-lg shadow-orange-500/40 hover:shadow-xl hover:shadow-orange-500/40 focus:outline-none focus:ring-2 focus:ring-offset-2 focus:ring-offset-slate-900 focus:ring-orange-500 transition-all duration-300 ease-in-out transform hover:-translate-y-0.5 disabled:from-slate-600 disabled:to-slate-600 disabled:shadow-none disabled:translate-y-0 disabled:cursor-not-allowed">
                            {isLoading && <svg className="animate-spin -ml-1 mr-3 h-5 w-5 text-white" xmlns="http://www.w3.org/2000/svg" fill="none" viewBox="0 0 24 24"><circle className="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" strokeWidth="4"></circle><path className="opacity-75" fill="currentColor" d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4zm2 5.291A7.962 7.962 0 014 12H0c0 3.042 1.135 5.824 3 7.938l3-2.647z"></path></svg>}
                            <span>{isLoading ? 'Menghasilkan...' : 'Hasilkan Skrip Live'}</span>
                          </button>
                          {errorMessage && <p className="text-sm text-red-400 text-center mt-3">{errorMessage}</p>}
                        </div>
                    </div>
                </div>
            ) : (
                <div id="live-output-view" className="max-w-4xl mx-auto space-y-6">
                    <div className="flex justify-between items-center flex-wrap gap-4">
                        <div className="flex items-center gap-4">
                            <button id="live-back-to-setup-btn" onClick={() => setView('setup')} className="inline-flex items-center justify-center gap-2 px-4 py-2 text-sm font-semibold text-slate-300 bg-slate-700/50 rounded-lg hover:bg-slate-700 transition-colors"><svg xmlns="http://www.w3.org/2000/svg" className="h-5 w-5" viewBox="0 0 20 20" fill="currentColor"><path fillRule="evenodd" d="M12.707 5.293a1 1 0 010 1.414L9.414 10l3.293 3.293a1 1 0 01-1.414 1.414l-4-4a1 1 0 010-1.414l4-4a1 1 0 011.414 0z" clipRule="evenodd" /></svg><span>Kembali & Edit</span></button>
                             <button onClick={handleSaveScript} disabled={isScriptSaved} className="inline-flex items-center justify-center gap-2 px-4 py-2 text-sm font-semibold text-slate-300 bg-slate-700/50 rounded-lg hover:bg-slate-700 transition-colors disabled:opacity-50">
                                {isScriptSaved ? <><svg className="w-5 h-5 text-green-400" xmlns="http://www.w3.org/2000/svg" fill="none" viewBox="0 0 24 24" stroke="currentColor"><path strokeLinecap="round" strokeLinejoin="round" strokeWidth="2" d="M5 13l4 4L19 7" /></svg>Tersimpan!</> : <><svg className="w-5 h-5" xmlns="http://www.w3.org/2000/svg" fill="none" viewBox="0 0 24 24" stroke="currentColor" strokeWidth="2"><path strokeLinecap="round" strokeLinejoin="round" d="M5 5a2 2 0 012-2h8l2 2v13.5a1.5 1.5 0 01-3 0V6H5a1 1 0 011-1z" /><path strokeLinecap="round" strokeLinejoin="round" d="M15 12a3 3 0 11-6 0 3 3 0 016 0z" /></svg>Simpan Skrip</>}
                            </button>
                        </div>
                        <button id="launch-teleprompter-btn" onClick={() => {
                            const scriptText = timeline.map(t => `[${t.time}] ${t.title}\n${t.script}`).join('\n\n');
                            const teleWindow = window.open('', '_blank');
                            if (teleWindow) {
                                teleWindow.document.write(`<html><head><title>Teleprompter - PRIME</title><style>body{background:#000;color:#fff;font-family:'Inter',sans-serif;font-size:2.5em;line-height:1.8;padding:2em;white-space:pre-wrap;}</style></head><body>${scriptText.replace(/</g,'&lt;').replace(/>/g,'&gt;')}</body></html>`);
                                teleWindow.document.close();
                            }
                        }} className="inline-flex items-center justify-center gap-2 px-5 py-2.5 text-sm font-semibold text-white bg-gradient-to-r from-orange-500 to-amber-500 rounded-lg shadow-lg shadow-orange-500/20 hover:from-orange-600 hover:to-amber-600 transition-all"><svg xmlns="http://www.w3.org/2000/svg" className="h-5 w-5" viewBox="0 0 20 20" fill="currentColor"><path d="M11 17a1 1 0 001.447.894l4-2A1 1 0 0017 15V5a1 1 0 00-1.447-.894l-4 2A1 1 0 0011 7v10zM6 17a1 1 0 001.447.894l4-2A1 1 0 0012 15V5a1 1 0 00-1.447-.894l-4 2A1 1 0 006 7v10z" /></svg>Luncurkan Teleprompter</button>
                    </div>
                     <div id="live-timeline-output" className="bg-slate-800/50 backdrop-blur-lg border border-slate-700 rounded-xl shadow-2xl shadow-black/30">
                        {isLoading && <ScriptSkeleton />}
                        {!isLoading && timeline.length > 0 ? timeline.map((item, index) => (
                             <div key={index} className="flex gap-4 sm:gap-6 border-b border-slate-700 last:border-b-0 p-4 sm:p-6">
                                <div className="flex flex-col items-center flex-shrink-0 w-20">
                                    <div className="text-xs font-semibold text-center font-mono bg-slate-700/50 text-slate-300 rounded-full px-2 py-0.5">{item.time}</div>
                                    <div className="text-3xl sm:text-4xl mt-3">{item.icon}</div>
                                </div>
                                <div className="flex-grow pt-1">
                                    <h4 className="font-bold text-lg text-orange-400">{item.title}</h4>
                                    <p className="text-slate-300 whitespace-pre-wrap mt-2 text-sm sm:text-base leading-relaxed">{item.script}</p>
                                </div>
                            </div>
                        )) : !isLoading && <p className="p-6 text-center text-slate-400">Tidak ada timeline yang dihasilkan.</p>}
                    </div>
                </div>
            )}
        </div>
    );
};

export default LiveStudio;