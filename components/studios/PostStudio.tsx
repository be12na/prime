import React, { useState } from 'react';
import { POST_OPTIONS } from '../../constants.ts';
import { generateSocialPost } from '../../services/geminiService.ts';
import { savePrompt } from '../../services/promptHistoryService.ts';
import Markdown from 'react-markdown';

type PostMode = 'product' | 'creator' | 'affiliate' | 'literacy';

interface ParsedOutput {
    title: string;
    content: string;
}

const PostStudio: React.FC = () => {
    const [activeTab, setActiveTab] = useState<PostMode>('product');
    const [isLoading, setIsLoading] = useState(false);
    const [parsedOutput, setParsedOutput] = useState<ParsedOutput[]>([]);
    const [useTrends, setUseTrends] = useState(false);
    const [savedStatuses, setSavedStatuses] = useState<{ [key: string]: boolean }>({});
    const [errorMessage, setErrorMessage] = useState('');


    const [formState, setFormState] = useState({
        product: { name: '', price: '', detail: '', market: '' },
        creator: { type: POST_OPTIONS.creatorType[0], topic: '', audience: '', duration: POST_OPTIONS.videoDuration[0] },
        affiliate: { product: '', price: '', detail: '', market: '', angle: POST_OPTIONS.affiliateAngle[0], tone: POST_OPTIONS.affiliateTone[0], duration: POST_OPTIONS.videoDuration[0] },
        literacy: { brandVoice: '', source: '', caption: '', hashtags: '', context: '', angle: POST_OPTIONS.literacyAngle[0] }
    });
    
    const handleChange = (tab: PostMode, field: string, value: string) => {
        setFormState(prev => ({
            ...prev,
            [tab]: { ...prev[tab], [field]: value }
        }));
    };
    
    const parseMarkdownOutput = (markdown: string): ParsedOutput[] => {
        if (!markdown || typeof markdown !== 'string') return [];
        const sections = markdown.split(/\n## /).filter(s => s.trim() !== '');
        
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
            const dataToSend = { type: activeTab, ...formState[activeTab] };
            const result = await generateSocialPost(dataToSend, useTrends);
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
        savePrompt('post', title, content);
        setSavedStatuses(prev => ({...prev, [title]: true}));
        setTimeout(() => {
            setSavedStatuses(prev => ({...prev, [title]: false}));
        }, 2000);
    };

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
            <div className="bg-slate-800/50 p-6 rounded-xl space-y-3 animate-pulse"><div className="h-5 w-1/3 bg-slate-700 rounded"></div><div className="h-4 w-full bg-slate-700 rounded"></div><div className="h-4 w-full bg-slate-700 rounded"></div><div className="h-4 w-5/6 bg-slate-700 rounded"></div></div>
            <div className="bg-slate-800/50 p-6 rounded-xl space-y-3 animate-pulse"><div className="h-5 w-1/4 bg-slate-700 rounded"></div><div className="flex flex-wrap gap-2 mt-2"><div className="h-6 w-20 bg-slate-700 rounded-full"></div><div className="h-6 w-24 bg-slate-700 rounded-full"></div><div className="h-6 w-16 bg-slate-700 rounded-full"></div></div></div>
            <div className="bg-slate-800/50 p-6 rounded-xl space-y-3 animate-pulse"><div className="h-5 w-1/3 bg-slate-700 rounded"></div><div className="h-4 w-2/3 bg-slate-700 rounded"></div></div>
        </div>
    );

    return (
        <div id="post-studio" className="space-y-8">
          <div className="text-center px-4">
            <h2 className="text-3xl sm:text-4xl md:text-5xl font-display font-bold tracking-tight text-transparent bg-clip-text bg-gradient-to-r from-orange-400 to-amber-400">Studio Post Media Sosial</h2>
            <p className="mt-4 text-base sm:text-lg text-slate-400 max-w-3xl mx-auto">
              Hasilkan ide konten, caption, dan skrip video pendek yang strategis dan menarik khusus untuk TikTok.
            </p>
          </div>
          <div className="grid grid-cols-1 lg:grid-cols-2 gap-8 items-start">
              <div className="bg-slate-800/50 backdrop-blur-lg border border-slate-700 rounded-xl p-4 sm:p-6 md:p-8 space-y-8 shadow-2xl shadow-black/30 lg:sticky top-24">
                  <div className="flex border-b border-slate-700 text-sm overflow-x-auto pb-1">
                      <button onClick={() => setActiveTab('product')} className={`flex-shrink-0 px-4 py-3 font-semibold border-b-2 transition-colors duration-200 ${activeTab === 'product' ? 'text-orange-400 border-orange-400' : 'text-slate-400 border-transparent hover:border-slate-500'}`}>Post Produk</button>
                      <button onClick={() => setActiveTab('creator')} className={`flex-shrink-0 px-4 py-3 font-semibold border-b-2 transition-colors duration-200 ${activeTab === 'creator' ? 'text-orange-400 border-orange-400' : 'text-slate-400 border-transparent hover:border-slate-500'}`}>Post Kreator</button>
                      <button onClick={() => setActiveTab('affiliate')} className={`flex-shrink-0 px-4 py-3 font-semibold border-b-2 transition-colors duration-200 ${activeTab === 'affiliate' ? 'text-orange-400 border-orange-400' : 'text-slate-400 border-transparent hover:border-slate-500'}`}>Post Afiliasi</button>
                      <button onClick={() => setActiveTab('literacy')} className={`flex-shrink-0 px-4 py-3 font-semibold border-b-2 transition-colors duration-200 ${activeTab === 'literacy' ? 'text-orange-400 border-orange-400' : 'text-slate-400 border-transparent hover:border-slate-500'}`}>Literasi Konten</button>
                  </div>

                  <div>
                    <div id="post-product-tab" className={`${activeTab !== 'product' ? 'hidden' : ''} space-y-6`}>
                        <h3 className="flex items-center text-lg font-semibold gap-3 text-transparent bg-clip-text bg-gradient-to-r from-orange-400 to-amber-400">Postingan Produk</h3>
                        <p className="text-sm -mt-4 text-slate-400">AI akan menganalisis detail untuk menghasilkan ide & skrip konten.</p>
                        <div className="grid grid-cols-1 sm:grid-cols-2 gap-4">
                           <div><label htmlFor="post-product-name" className="block text-sm font-medium text-slate-400 mb-2">Nama Produk</label><input type="text" id="post-product-name" value={formState.product.name} onChange={e => handleChange('product', 'name', e.target.value)} className="block w-full bg-slate-700/50 border-slate-600 rounded-md py-2.5 px-4 text-slate-100 sm:text-sm transition focus:ring-orange-500 focus:border-orange-500" placeholder="Contoh: Serum 'Aurora'" /></div>
                           <div><label htmlFor="post-product-price" className="block text-sm font-medium text-slate-400 mb-2">Harga Produk</label><input type="text" id="post-product-price" value={formState.product.price} onChange={e => handleChange('product', 'price', e.target.value)} className="block w-full bg-slate-700/50 border-slate-600 rounded-md py-2.5 px-4 text-slate-100 sm:text-sm transition focus:ring-orange-500 focus:border-orange-500" placeholder="Contoh: Rp 150.000" /></div>
                        </div>
                        <div><label htmlFor="post-product-detail" className="block text-sm font-medium text-slate-400 mb-2">Detail Produk</label><textarea id="post-product-detail" value={formState.product.detail} onChange={e => handleChange('product', 'detail', e.target.value)} rows={4} className="block w-full bg-slate-700/50 border-slate-600 rounded-md py-2.5 px-4 text-slate-100 sm:text-sm transition focus:ring-orange-500 focus:border-orange-500" placeholder="Jelaskan produk Anda secara detail. AI akan mengidentifikasi fitur & benefitnya."></textarea></div>
                        <div><label htmlFor="post-product-market" className="block text-sm font-medium text-slate-400 mb-2">Target Market</label><textarea id="post-product-market" value={formState.product.market} onChange={e => handleChange('product', 'market', e.target.value)} rows={3} className="block w-full bg-slate-700/50 border-slate-600 rounded-md py-2.5 px-4 text-slate-100 sm:text-sm transition focus:ring-orange-500 focus:border-orange-500" placeholder="Jelaskan target pasar Anda. AI akan menentukan Call to Action yang paling sesuai. Cth: Wanita karir usia 25-35, peduli lingkungan, tinggal di kota besar."></textarea></div>
                    </div>

                    <div id="post-creator-tab" className={`${activeTab !== 'creator' ? 'hidden' : ''} space-y-6`}>
                        <h3 className="flex items-center text-lg font-semibold gap-3 text-transparent bg-clip-text bg-gradient-to-r from-orange-400 to-amber-400">Postingan Kreator</h3>
                        <p className="text-sm -mt-4 text-slate-400">Hasilkan berbagai format konten sesuai pilar Anda.</p>
                        <div><label htmlFor="post-creator-type" className="block text-sm font-medium text-slate-400 mb-2">Jenis Konten</label><select id="post-creator-type" value={formState.creator.type} onChange={e => handleChange('creator', 'type', e.target.value)} className="block w-full bg-slate-700/50 border-slate-600 rounded-md py-2.5 px-4 text-slate-100 sm:text-sm transition focus:ring-orange-500 focus:border-orange-500">{POST_OPTIONS.creatorType.map(o=><option key={o}>{o}</option>)}</select></div>
                        <div><label htmlFor="post-creator-topic" className="block text-sm font-medium text-slate-400 mb-2">Topik Spesifik</label><input type="text" id="post-creator-topic" value={formState.creator.topic} onChange={e => handleChange('creator', 'topic', e.target.value)} className="block w-full bg-slate-700/50 border-slate-600 rounded-md py-2.5 px-4 text-slate-100 sm:text-sm transition focus:ring-orange-500 focus:border-orange-500" placeholder="Contoh: 3 kesalahan public speaking pemula" /></div>
                        <div><label htmlFor="post-creator-audience" className="block text-sm font-medium text-slate-400 mb-2">Target Audiens Anda</label><textarea id="post-creator-audience" value={formState.creator.audience} onChange={e => handleChange('creator', 'audience', e.target.value)} rows={2} className="block w-full bg-slate-700/50 border-slate-600 rounded-md py-2.5 px-4 text-slate-100 sm:text-sm transition focus:ring-orange-500 focus:border-orange-500" placeholder="Contoh: Fresh graduate, freelancer, pemilik bisnis kecil"></textarea></div>
                        <div><label htmlFor="post-creator-duration" className="block text-sm font-medium text-slate-400 mb-2">Durasi Video</label><select id="post-creator-duration" value={formState.creator.duration} onChange={e => handleChange('creator', 'duration', e.target.value)} className="block w-full bg-slate-700/50 border-slate-600 rounded-md py-2.5 px-4 text-slate-100 sm:text-sm transition focus:ring-orange-500 focus:border-orange-500">{POST_OPTIONS.videoDuration.map(o=><option key={o}>{o}</option>)}</select></div>
                    </div>

                    <div id="post-affiliate-tab" className={`${activeTab !== 'affiliate' ? 'hidden' : ''} space-y-6`}>
                        <h3 className="flex items-center text-lg font-semibold gap-3 text-transparent bg-clip-text bg-gradient-to-r from-orange-400 to-amber-400">Postingan Afiliasi</h3>
                        <p className="text-sm -mt-4 text-slate-400">Buat skrip afiliasi TikTok dengan hook tinggi.</p>
                        <div className="grid grid-cols-1 sm:grid-cols-2 gap-4">
                           <div><label htmlFor="post-affiliate-product" className="block text-sm font-medium text-slate-400 mb-2">Nama Produk</label><input type="text" id="post-affiliate-product" value={formState.affiliate.product} onChange={e => handleChange('affiliate', 'product', e.target.value)} className="block w-full bg-slate-700/50 border-slate-600 rounded-md py-2.5 px-4 text-slate-100 sm:text-sm transition focus:ring-orange-500 focus:border-orange-500" placeholder="Contoh: Mic Wireless Taffware" /></div>
                           <div><label htmlFor="post-affiliate-price" className="block text-sm font-medium text-slate-400 mb-2">Harga Produk</label><input type="text" id="post-affiliate-price" value={formState.affiliate.price} onChange={e => handleChange('affiliate', 'price', e.target.value)} className="block w-full bg-slate-700/50 border-slate-600 rounded-md py-2.5 px-4 text-slate-100 sm:text-sm transition focus:ring-orange-500 focus:border-orange-500" placeholder="Contoh: Rp 99.000" /></div>
                        </div>
                        <div><label htmlFor="post-affiliate-detail" className="block text-sm font-medium text-slate-400 mb-2">Detail Produk</label><textarea id="post-affiliate-detail" value={formState.affiliate.detail} onChange={e => handleChange('affiliate', 'detail', e.target.value)} rows={3} className="block w-full bg-slate-700/50 border-slate-600 rounded-md py-2.5 px-4 text-slate-100 sm:text-sm transition focus:ring-orange-500 focus:border-orange-500" placeholder="Jelaskan produk secara singkat."></textarea></div>
                        <div><label htmlFor="post-affiliate-market" className="block text-sm font-medium text-slate-400 mb-2">Untuk Siapa Produk Ini?</label><textarea id="post-affiliate-market" value={formState.affiliate.market} onChange={e => handleChange('affiliate', 'market', e.target.value)} rows={2} className="block w-full bg-slate-700/50 border-slate-600 rounded-md py-2.5 px-4 text-slate-100 sm:text-sm transition focus:ring-orange-500 focus:border-orange-500" placeholder="Contoh: Konten kreator pemula, guru online, podcaster"></textarea></div>
                        <div className="grid grid-cols-1 sm:grid-cols-2 gap-4">
                           <div><label htmlFor="post-affiliate-angle" className="block text-sm font-medium text-slate-400 mb-2">Sudut Pandang</label><select id="post-affiliate-angle" value={formState.affiliate.angle} onChange={e => handleChange('affiliate', 'angle', e.target.value)} className="block w-full bg-slate-700/50 border-slate-600 rounded-md py-2.5 px-4 text-slate-100 sm:text-sm transition focus:ring-orange-500 focus:border-orange-500">{POST_OPTIONS.affiliateAngle.map(o=><option key={o}>{o}</option>)}</select></div>
                           <div><label htmlFor="post-affiliate-tone" className="block text-sm font-medium text-slate-400 mb-2">Gaya Bicara</label><select id="post-affiliate-tone" value={formState.affiliate.tone} onChange={e => handleChange('affiliate', 'tone', e.target.value)} className="block w-full bg-slate-700/50 border-slate-600 rounded-md py-2.5 px-4 text-slate-100 sm:text-sm transition focus:ring-orange-500 focus:border-orange-500">{POST_OPTIONS.affiliateTone.map(o=><option key={o}>{o}</option>)}</select></div>
                        </div>
                         <div><label htmlFor="post-affiliate-duration" className="block text-sm font-medium text-slate-400 mb-2">Durasi Video</label><select id="post-affiliate-duration" value={formState.affiliate.duration} onChange={e => handleChange('affiliate', 'duration', e.target.value)} className="block w-full bg-slate-700/50 border-slate-600 rounded-md py-2.5 px-4 text-slate-100 sm:text-sm transition focus:ring-orange-500 focus:border-orange-500">{POST_OPTIONS.videoDuration.map(o=><option key={o}>{o}</option>)}</select></div>
                    </div>

                    <div id="post-literacy-tab" className={`${activeTab !== 'literacy' ? 'hidden' : ''} space-y-6`}>
                        <h3 className="flex items-center text-lg font-semibold gap-3 text-transparent bg-clip-text bg-gradient-to-r from-orange-400 to-amber-400">Literasi Konten</h3>
                         <p className="text-sm -mt-4 text-slate-400">Adaptasi atau daur ulang konten yang sudah ada.</p>
                         <div><label htmlFor="post-literacy-brand-voice" className="block text-sm font-medium text-slate-400 mb-2">Persona / Brand Voice Anda</label><input type="text" id="post-literacy-brand-voice" value={formState.literacy.brandVoice} onChange={e => handleChange('literacy', 'brandVoice', e.target.value)} className="block w-full bg-slate-700/50 border-slate-600 rounded-md py-2.5 px-4 text-slate-100 sm:text-sm transition focus:ring-orange-500 focus:border-orange-500" placeholder="Contoh: Ceria & jenaka, Informatif & terpercaya" /></div>
                        <div><label htmlFor="post-literacy-source" className="block text-sm font-medium text-slate-400 mb-2">Sumber Konten (Artikel, Berita, dll)</label><textarea id="post-literacy-source" value={formState.literacy.source} onChange={e => handleChange('literacy', 'source', e.target.value)} rows={5} className="block w-full bg-slate-700/50 border-slate-600 rounded-md py-2.5 px-4 text-slate-100 sm:text-sm transition focus:ring-orange-500 focus:border-orange-500" placeholder="Tempel artikel, berita, atau materi lain di sini..."></textarea></div>
                        <div className="grid grid-cols-1 sm:grid-cols-2 gap-4">
                           <div><label htmlFor="post-literacy-caption" className="block text-sm font-medium text-slate-400 mb-2">Caption Asli (Opsional)</label><input type="text" id="post-literacy-caption" value={formState.literacy.caption} onChange={e => handleChange('literacy', 'caption', e.target.value)} className="block w-full bg-slate-700/50 border-slate-600 rounded-md py-2.5 px-4 text-slate-100 sm:text-sm transition focus:ring-orange-500 focus:border-orange-500" /></div>
                           <div><label htmlFor="post-literacy-hashtags" className="block text-sm font-medium text-slate-400 mb-2">Hashtag Asli (Opsional)</label><input type="text" id="post-literacy-hashtags" value={formState.literacy.hashtags} onChange={e => handleChange('literacy', 'hashtags', e.target.value)} className="block w-full bg-slate-700/50 border-slate-600 rounded-md py-2.5 px-4 text-slate-100 sm:text-sm transition focus:ring-orange-500 focus:border-orange-500" /></div>
                        </div>
                        <div><label htmlFor="post-literacy-context" className="block text-sm font-medium text-slate-400 mb-2">Konteks Tambahan (Opsional)</label><textarea id="post-literacy-context" value={formState.literacy.context} onChange={e => handleChange('literacy', 'context', e.target.value)} rows={2} className="block w-full bg-slate-700/50 border-slate-600 rounded-md py-2.5 px-4 text-slate-100 sm:text-sm transition focus:ring-orange-500 focus:border-orange-500" placeholder="Ada hal lain yang perlu AI ketahui?"></textarea></div>
                        <div><label htmlFor="post-literacy-angle" className="block text-sm font-medium text-slate-400 mb-2">Pilih Aksi</label><select id="post-literacy-angle" value={formState.literacy.angle} onChange={e => handleChange('literacy', 'angle', e.target.value)} className="block w-full bg-slate-700/50 border-slate-600 rounded-md py-2.5 px-4 text-slate-100 sm:text-sm transition focus:ring-orange-500 focus:border-orange-500">{POST_OPTIONS.literacyAngle.map(o=><option key={o}>{o}</option>)}</select></div>
                    </div>
                  </div>

                  <div className="pt-6 border-t border-slate-700 space-y-4">
                      <div className="flex items-center justify-between">
                          <label htmlFor="post-use-trends" className="flex items-center gap-2 text-sm font-medium text-slate-400 cursor-pointer">
                              <svg xmlns="http://www.w3.org/2000/svg" className="h-5 w-5 text-sky-400" fill="none" viewBox="0 0 24 24" stroke="currentColor" strokeWidth="2"><path strokeLinecap="round" strokeLinejoin="round" d="M13 7h8m0 0v8m0-8l-8 8-4-4-6 6" /></svg>
                              <span>Gunakan Tren Terkini (Google)</span>
                          </label>
                          <label className="toggle-switch">
                              <input type="checkbox" id="post-use-trends" checked={useTrends} onChange={e => setUseTrends(e.target.checked)} />
                              <span className="toggle-slider"></span>
                          </label>
                      </div>
                      <button onClick={handleGenerate} disabled={isLoading} className="w-full inline-flex items-center justify-center px-6 py-3 text-base sm:px-12 sm:py-4 sm:text-lg font-semibold text-white bg-gradient-to-r from-orange-500 to-amber-500 rounded-lg shadow-lg shadow-orange-500/40 hover:shadow-xl hover:shadow-orange-500/40 focus:outline-none focus:ring-2 focus:ring-offset-2 focus:ring-offset-slate-900 focus:ring-orange-500 transition-all duration-300 ease-in-out transform hover:-translate-y-0.5 disabled:from-slate-600 disabled:to-slate-600 disabled:shadow-none disabled:translate-y-0 disabled:cursor-not-allowed">
                          {isLoading && <svg className="animate-spin -ml-1 mr-3 h-5 w-5 text-white" xmlns="http://www.w3.org/2000/svg" fill="none" viewBox="0 0 24 24"><circle className="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" strokeWidth="4"></circle><path className="opacity-75" fill="currentColor" d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4zm2 5.291A7.962 7.962 0 014 12H0c0 3.042 1.135 5.824 3 7.938l3-2.647z"></path></svg>}
                          <span>{isLoading ? 'Menghasilkan...' : 'Hasilkan Paket Konten'}</span>
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
                          {errorMessage ? <p className="text-red-400">{errorMessage}</p> : <p>Paket konten Anda (ide, skrip, caption) akan muncul di sini.</p>}
                      </div>
                  )}
              </div>
          </div>
        </div>
    );
};

export default PostStudio;