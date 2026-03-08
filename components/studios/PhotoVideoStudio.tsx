
import React, { useState, useCallback, useMemo } from 'react';
// Fix: Corrected typo from PHOTO_VIDEO_MANual_OPTIONS to PHOTO_VIDEO_MANUAL_OPTIONS
import { PHOTO_VIDEO_MANUAL_OPTIONS as OPTIONS } from '../../constants.ts';
import { generatePhotoVideoPrompt } from '../../services/geminiService.ts';
import { savePrompt } from '../../services/promptHistoryService.ts';

type FormState = {
    concept: string;
    model: {
        gender: string;
        ethnicity: string;
        age: string;
        bodyShape: string;
        hair: string;
        makeup: string;
        shotType: string;
        expression: string;
        pose: string;
        interaction: string[];
    };
    art: {
        productType: string;
        style: string;
        mood: string;
        location: string;
        background: string;
        composition: string;
        productAngle: string;
        lighting: string;
        props: string;
        productPosition: string;
        vfx: string;
        sound: string;
    };
    output: {
        aspectRatio: string;
    };
};

type AiFormState = {
    productName: string;
    productDetail: string;
    imageBase64: string;
    mimeType: string;
    concept: string;
    model: FormState['model'];
    art: FormState['art'] & { aiImageStyle: string, aiArtistStyle: string };
    output: FormState['output'];
}

const initialFormState: FormState = {
    concept: OPTIONS.concept[1].value,
    model: {
        gender: OPTIONS.gender[1],
        ethnicity: OPTIONS.ethnicity[1],
        age: OPTIONS.age[1],
        bodyShape: OPTIONS.bodyShape[1],
        hair: OPTIONS.hair[1],
        makeup: OPTIONS.makeup[1],
        shotType: OPTIONS.shotType[1],
        expression: OPTIONS.expression[1],
        pose: OPTIONS.pose[1],
        interaction: [OPTIONS.interaction[0].value],
    },
    art: {
        productType: OPTIONS.productType[1],
        style: OPTIONS.style[1],
        mood: OPTIONS.mood[1],
        location: OPTIONS.location[1],
        background: OPTIONS.background[1],
        composition: OPTIONS.composition[1],
        productAngle: OPTIONS.productAngle[1],
        lighting: OPTIONS.lighting[1],
        props: OPTIONS.props[1],
        productPosition: OPTIONS.productPosition[1],
        vfx: OPTIONS.vfx[1],
        sound: OPTIONS.sound[1],
    },
    output: {
        aspectRatio: OPTIONS.aspectRatio[2],
    }
};

const initialAiFormState: AiFormState = {
    productName: '',
    productDetail: '',
    imageBase64: '',
    mimeType: '',
    concept: OPTIONS.concept[0].value,
    model: {
        gender: OPTIONS.gender[0],
        ethnicity: OPTIONS.ethnicity[0],
        age: OPTIONS.age[0],
        bodyShape: OPTIONS.bodyShape[0],
        hair: OPTIONS.hair[0],
        makeup: OPTIONS.makeup[0],
        shotType: OPTIONS.shotType[0],
        expression: OPTIONS.expression[0],
        pose: OPTIONS.pose[0],
        interaction: [OPTIONS.interaction[0].value],
    },
    art: {
        productType: OPTIONS.productType[0],
        style: OPTIONS.style[0],
        mood: OPTIONS.mood[0],
        location: OPTIONS.location[0],
        background: OPTIONS.background[0],
        composition: OPTIONS.composition[0],
        productAngle: OPTIONS.productAngle[0],
        lighting: OPTIONS.lighting[0],
        props: OPTIONS.props[0],
        productPosition: OPTIONS.productPosition[0],
        vfx: OPTIONS.vfx[0],
        sound: OPTIONS.sound[0],
        aiImageStyle: OPTIONS.aiImageStyle[0],
        aiArtistStyle: OPTIONS.aiArtistStyle[0],
    },
    output: {
        aspectRatio: OPTIONS.aspectRatio[2],
    }
}


const PhotoVideoStudio: React.FC = () => {
    const [activeTab, setActiveTab] = useState<'manual' | 'ai'>('manual');
    const [isLoading, setIsLoading] = useState(false);
    const [isImageLoading, setIsImageLoading] = useState(false);
    
    const [formState, setFormState] = useState<FormState>(initialFormState);
    const [aiFormState, setAiFormState] = useState<AiFormState>(initialAiFormState);
    
    const [photoPrompt, setPhotoPrompt] = useState('');
    const [videoPrompt, setVideoPrompt] = useState('');
    const [savedPrompts, setSavedPrompts] = useState<{ [key: string]: boolean }>({});
    const [copiedType, setCopiedType] = useState<string | null>(null);

    // Manual mode image uploads
    const [manualModelImage, setManualModelImage] = useState<{ base64: string; mimeType: string; name: string } | null>(null);
    const [manualProductImage, setManualProductImage] = useState<{ base64: string; mimeType: string; name: string } | null>(null);


    const isModelVisible = useMemo(() => activeTab === 'manual' ? formState.concept === 'produk-dengan-model' : aiFormState.concept === 'produk-dengan-model', [activeTab, formState.concept, aiFormState.concept]);
    const isStillLifeControlsVisible = useMemo(() => activeTab === 'manual' ? formState.concept !== 'produk-dengan-model' : aiFormState.concept !== 'produk-dengan-model', [activeTab, formState.concept, aiFormState.concept]);

    // Manual Form Handlers
    const handleFormChange = useCallback((section: keyof Omit<FormState, 'concept'>, field: string, value: string) => {
        setFormState(prev => ({ ...prev, [section]: { ...prev[section], [field]: value } }));
    }, []);
    const handleModelChange = useCallback((field: keyof FormState['model'], value: string) => handleFormChange('model', field, value), [handleFormChange]);
    const handleArtChange = useCallback((field: keyof FormState['art'], value: string) => handleFormChange('art', field, value), [handleFormChange]);
    const handleInteractionChange = useCallback((value: string) => {
        setFormState(prev => {
            const current = prev.model.interaction.filter(i => i !== 'Tidak ada');
            let updated;
            if (value === 'Tidak ada') {
                updated = ['Tidak ada'];
            } else if (current.includes(value)) {
                updated = current.filter(i => i !== value);
            } else {
                updated = [...current, value];
            }
            if (updated.length === 0) updated = ['Tidak ada'];
            return { ...prev, model: { ...prev.model, interaction: updated } };
        });
    }, []);

    // AI Form Handlers
    const handleAiFormChange = useCallback((section: keyof Omit<AiFormState, 'concept' | 'productName' | 'productDetail' | 'imageBase64' | 'mimeType'>, field: string, value: string) => {
        setAiFormState(prev => ({ ...prev, [section]: { ...prev[section], [field]: value } }));
    }, []);
    const handleAiModelChange = useCallback((field: keyof AiFormState['model'], value: string) => handleAiFormChange('model', field, value), [handleAiFormChange]);
    const handleAiArtChange = useCallback((field: keyof AiFormState['art'], value: string) => handleAiFormChange('art', field, value), [handleAiFormChange]);
    const handleAiInteractionChange = useCallback((value: string) => {
        setAiFormState(prev => {
            const current = prev.model.interaction.filter(i => i !== 'Tidak ada');
            let updated;
            if (value === 'Tidak ada') {
                updated = ['Tidak ada'];
            } else if (current.includes(value)) {
                updated = current.filter(i => i !== value);
            } else {
                updated = [...current, value];
            }
            if (updated.length === 0) updated = ['Tidak ada'];
            return { ...prev, model: { ...prev.model, interaction: updated } };
        });
    }, []);


    const handleGenerate = async () => {
        setIsLoading(true);
        setPhotoPrompt('');
        setVideoPrompt('');
        setGeneratedImage('');
        setImageGenStatus('');
        try {
            const dataToSend = activeTab === 'manual' 
                ? { ...formState, modelImageBase64: manualModelImage?.base64 || '', modelImageMimeType: manualModelImage?.mimeType || '', productImageBase64: manualProductImage?.base64 || '', productImageMimeType: manualProductImage?.mimeType || '' }
                : aiFormState;
            const result = await generatePhotoVideoPrompt(dataToSend, activeTab);
            setPhotoPrompt(result.photo);
            setVideoPrompt(result.video);
        } catch (error) {
            const msg = error instanceof Error ? error.message : 'Terjadi kesalahan.';
            setPhotoPrompt(`Error: ${msg}`);
            setVideoPrompt(`Error: ${msg}`);
        } finally {
            setIsLoading(false);
        }
    };


    
    const handleAiImageUpload = (e: React.ChangeEvent<HTMLInputElement>) => {
        const file = e.target.files?.[0];
        if (file) {
            const reader = new FileReader();
            reader.onloadend = () => {
                const base64String = reader.result?.toString().split(',')[1] || '';
                setAiFormState(prev => ({ ...prev, imageBase64: base64String, mimeType: file.type }));
            };
            reader.readAsDataURL(file);
        }
    };

    const handleManualImageUpload = (type: 'model' | 'product') => (e: React.ChangeEvent<HTMLInputElement>) => {
        const file = e.target.files?.[0];
        if (file) {
            const reader = new FileReader();
            reader.onloadend = () => {
                const base64String = reader.result?.toString().split(',')[1] || '';
                const imageData = { base64: base64String, mimeType: file.type, name: file.name };
                if (type === 'model') setManualModelImage(imageData);
                else setManualProductImage(imageData);
            };
            reader.readAsDataURL(file);
        }
    };
    
    const copyToClipboard = (text: string, type: 'image' | 'video') => {
        navigator.clipboard.writeText(text).catch(() => {});
        setCopiedType(type);
        setTimeout(() => setCopiedType(null), 2000);
    };
    
    const handleSavePrompt = (type: 'Foto' | 'Video', content: string) => {
        if (!content || content.startsWith('Gagal')) return;
        savePrompt('photo-video', type, content);
        setSavedPrompts(prev => ({ ...prev, [type]: true }));
        setTimeout(() => {
            setSavedPrompts(prev => ({ ...prev, [type]: false }));
        }, 2000);
    };
    
    const renderSelect = (id: string, label: string, options: any[], value: string, onChange: (e: React.ChangeEvent<HTMLSelectElement>) => void, containerClassName = "") => (
        <div className={containerClassName}>
            <label htmlFor={id} className="block text-sm font-medium text-slate-400 mb-2">{label}</label>
            <select id={id} name={id} value={value} onChange={onChange} className="block w-full bg-slate-700/50 border-slate-600 rounded-md shadow-sm py-2.5 px-4 text-slate-100 focus:outline-none focus:ring-2 focus:ring-orange-500 focus:border-orange-500 sm:text-sm transition appearance-none">
                {options.map((opt: any) => (
                    typeof opt === 'string' ? <option key={opt} value={opt}>{opt}</option> : <option key={opt.value} value={opt.value}>{opt.label}</option>
                ))}
            </select>
        </div>
    );
    
    return (
      <div id="photo-video-studio" className="space-y-8">
        <div className="text-center px-4">
          <h2 className="text-3xl sm:text-4xl md:text-5xl font-display font-bold tracking-tight text-transparent bg-clip-text bg-gradient-to-r from-orange-400 to-amber-400">Studio Foto & Video</h2>
          <p className="mt-4 text-base sm:text-lg text-slate-400 max-w-2xl mx-auto">
            Ciptakan PROMPT FOTO dan PROMPT VIDEO produk yang ultra-realistis dan profesional.
          </p>
        </div>

        <div className="grid grid-cols-1 lg:grid-cols-2 gap-8 items-start">
          <div className="bg-slate-800/50 backdrop-blur-lg border border-slate-700 rounded-xl p-4 sm:p-6 md:p-8 space-y-8 shadow-2xl shadow-black/30 lg:sticky top-24">
            
            <div className="flex border-b border-slate-700">
              <button onClick={() => setActiveTab('manual')} className={`flex-1 py-3 text-sm font-semibold border-b-2 transition-colors duration-200 ${activeTab === 'manual' ? 'text-orange-400 border-orange-400' : 'text-slate-400 border-transparent hover:border-slate-500'}`}>
                Manual
              </button>
              <button onClick={() => setActiveTab('ai')} className={`flex-1 py-3 text-sm font-semibold border-b-2 transition-colors duration-200 ${activeTab === 'ai' ? 'text-orange-400 border-orange-400' : 'text-slate-400 border-transparent hover:border-slate-500'}`}>
                AI Cerdas
              </button>
            </div>

            <div id="ai-controls" className={`space-y-8 ${activeTab === 'ai' ? '' : 'hidden'}`}>
                <div className="space-y-4">
                    <h3 className="flex items-center text-lg font-semibold gap-3 text-transparent bg-clip-text bg-gradient-to-r from-orange-400 to-amber-400">
                        <svg xmlns="http://www.w3.org/2000/svg" className="h-6 w-6 text-orange-500" fill="none" viewBox="0 0 24 24" stroke="currentColor" strokeWidth="2"><path strokeLinecap="round" strokeLinejoin="round" d="M11 5H6a2 2 0 00-2 2v11a2 2 0 002 2h11a2 2 0 002-2v-5m-1.414-9.414a2 2 0 112.828 2.828L11.828 15H9v-2.828l8.586-8.586z" /></svg>
                        <span>1. Detail Produk Kunci</span>
                    </h3>
                    <div className="space-y-4">
                        <div>
                            <label htmlFor="ai-product-name" className="block text-sm font-medium text-slate-400 mb-2">Nama Produk</label>
                            <input type="text" id="ai-product-name" value={aiFormState.productName} onChange={e => setAiFormState(s => ({...s, productName: e.target.value}))} className="block w-full bg-slate-700/50 border-slate-600 rounded-md shadow-sm py-2.5 px-4 text-slate-100 focus:outline-none focus:ring-2 focus:ring-orange-500 focus:border-orange-500 sm:text-sm transition" placeholder="Contoh: Sepatu Lari Trail 'Kilat V2'" />
                        </div>
                        <div>
                            <label htmlFor="ai-product-detail" className="block text-sm font-medium text-slate-400 mb-2">Deskripsi & Fitur Produk</label>
                            <textarea id="ai-product-detail" rows={3} value={aiFormState.productDetail} onChange={e => setAiFormState(s => ({...s, productDetail: e.target.value}))} className="block w-full bg-slate-700/50 border-slate-600 rounded-md shadow-sm py-2.5 px-4 text-slate-100 focus:outline-none focus:ring-2 focus:ring-orange-500 focus:border-orange-500 sm:text-sm transition" placeholder="Contoh: Sepatu lari trail pria, tahan air, sol Vibram, warna biru navy dengan aksen oranye..."></textarea>
                        </div>
                    </div>
                </div>
                
                <div className="space-y-4 pt-8 border-t border-slate-700">
                    <h3 className="flex items-center text-lg font-semibold gap-3 text-transparent bg-clip-text bg-gradient-to-r from-orange-400 to-amber-400">
                        <svg xmlns="http://www.w3.org/2000/svg" className="h-6 w-6 text-orange-500" fill="none" viewBox="0 0 24 24" stroke="currentColor" strokeWidth="2">
                            <path strokeLinecap="round" strokeLinejoin="round" d="M4 16v1a3 3 0 003 3h10a3 3 0 003-3v-1m-4-8l-4-4m0 0L8 8m4-4v12" />
                        </svg>
                        <span>2. Unggah Gambar Produk</span>
                    </h3>
                    {aiFormState.imageBase64 && (
                        <div id="ai-image-preview-container" className="relative w-full h-48 bg-slate-700/50 rounded-lg flex items-center justify-center border-2 border-dashed border-slate-600">
                            <img src={`data:${aiFormState.mimeType};base64,${aiFormState.imageBase64}`} alt="preview" className="max-h-full max-w-full object-contain rounded"/>
                        </div>
                    )}
                    <label htmlFor="ai-image-upload" className="cursor-pointer w-full inline-flex items-center justify-center px-4 py-2.5 text-sm font-semibold text-orange-400 bg-slate-700/50 hover:bg-slate-700 rounded-md transition-colors">
                        {aiFormState.imageBase64 ? 'Ganti Gambar Produk...' : 'Pilih Gambar Produk...'}
                    </label>
                    <input type="file" id="ai-image-upload" className="hidden" accept="image/png, image/jpeg, image/webp" onChange={handleAiImageUpload} />
                    <p className="text-xs text-slate-500 text-center">Gambar ini akan menjadi dasar untuk diedit oleh AI.</p>
                </div>

                <div className="space-y-4 pt-8 border-t border-slate-700">
                  <h3 className="flex items-center text-lg font-semibold gap-3 text-transparent bg-clip-text bg-gradient-to-r from-orange-400 to-amber-400">
                    <svg xmlns="http://www.w3.org/2000/svg" className="h-6 w-6 text-orange-500" fill="none" viewBox="0 0 24 24" stroke="currentColor" strokeWidth="2"><path strokeLinecap="round" strokeLinejoin="round" d="M19.428 15.428a2 2 0 00-1.022-.547l-2.387-.477a6 6 0 00-3.86.517l-.318.158a6 6 0 01-3.86.517L6.05 15.21a2 2 0 00-1.806.547M8 4h8l-1 1v5.172a2 2 0 00.586 1.414l5 5c1.26 1.26.367 3.414-1.415 3.414H4.828c-1.782 0-2.674-2.154-1.414-3.414l5-5A2 2 0 009 10.172V5L8 4z" /></svg>
                    <span>3. Konsep Utama (AI)</span>
                  </h3>
                  <select id="ai-concept-select" value={aiFormState.concept} onChange={e => setAiFormState(s => ({ ...s, concept: e.target.value }))} className="block w-full bg-slate-700/50 border-slate-600 rounded-md shadow-sm py-2.5 px-4 text-slate-100 focus:outline-none focus:ring-2 focus:ring-orange-500 focus:border-orange-500 sm:text-sm transition">
                      {OPTIONS.concept.map(opt => <option key={opt.value} value={opt.value}>{opt.label}</option>)}
                  </select>
                </div>

                {isModelVisible && (
                    <div id="ai-model-controls" className="collapsible">
                        <div className="space-y-8 pt-8 border-t border-slate-700">
                            <h3 className="flex items-center text-lg font-semibold gap-3 text-transparent bg-clip-text bg-gradient-to-r from-orange-400 to-amber-400">
                                <svg xmlns="http://www.w3.org/2000/svg" className="h-6 w-6 text-orange-500" fill="none" viewBox="0 0 24 24" stroke="currentColor" strokeWidth="2"><path strokeLinecap="round" strokeLinejoin="round" d="M16 7a4 4 0 11-8 0 4 4 0 018 0zM12 14a7 7 0 00-7 7h14a7 7 0 00-7-7z" /></svg>
                                <span>4. Detail Model (AI)</span>
                            </h3>
                            <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-3 gap-6">
                                {renderSelect('ai-model-gender-select', 'Jenis Kelamin', OPTIONS.gender, aiFormState.model.gender, e => handleAiModelChange('gender', e.target.value))}
                                {renderSelect('ai-model-ethnicity-select', 'Etnisitas', OPTIONS.ethnicity, aiFormState.model.ethnicity, e => handleAiModelChange('ethnicity', e.target.value))}
                                {renderSelect('ai-model-age-select', 'Perkiraan Usia', OPTIONS.age, aiFormState.model.age, e => handleAiModelChange('age', e.target.value))}
                                {renderSelect('ai-model-body-shape-select', 'Bentuk Badan', OPTIONS.bodyShape, aiFormState.model.bodyShape, e => handleAiModelChange('bodyShape', e.target.value))}
                                {renderSelect('ai-model-hair-select', 'Gaya Rambut', OPTIONS.hair, aiFormState.model.hair, e => handleAiModelChange('hair', e.target.value))}
                                {renderSelect('ai-model-makeup-select', 'Gaya Riasan', OPTIONS.makeup, aiFormState.model.makeup, e => handleAiModelChange('makeup', e.target.value))}
                                {renderSelect('ai-shot-type-select', 'Tipe Bidikan', OPTIONS.shotType, aiFormState.model.shotType, e => handleAiModelChange('shotType', e.target.value), 'sm:col-span-2 lg:col-span-3')}
                                {renderSelect('ai-model-expression-select', 'Ekspresi Model', OPTIONS.expression, aiFormState.model.expression, e => handleAiModelChange('expression', e.target.value), 'sm:col-span-2 lg:col-span-3')}
                                {renderSelect('ai-model-pose-select', 'Atur Gaya (Pose)', OPTIONS.pose, aiFormState.model.pose, e => handleAiModelChange('pose', e.target.value), 'sm:col-span-2 lg:col-span-3')}
                                <div className="sm:col-span-2 lg:col-span-3">
                                    <label className="block text-sm font-medium text-slate-400 mb-2">Interaksi Model</label>
                                    <div id="ai-model-interaction-group" className="grid grid-cols-2 sm:grid-cols-4 gap-1 bg-slate-700/50 rounded-lg p-1 h-auto">
                                        {OPTIONS.interaction.map(opt => (
                                            <button
                                                key={opt.value}
                                                onClick={() => handleAiInteractionChange(opt.value)}
                                                className={`text-sm rounded-md py-1.5 transition-colors ${aiFormState.model.interaction.includes(opt.value) ? 'bg-orange-500 text-white' : 'hover:bg-slate-600 text-slate-300'}`}
                                            >
                                                {opt.label}
                                            </button>
                                        ))}
                                    </div>
                                </div>
                            </div>
                        </div>
                    </div>
                )}
                
                <div className="space-y-8 pt-8 border-t border-slate-700">
                  <h3 className="flex items-center text-lg font-semibold gap-3 text-transparent bg-clip-text bg-gradient-to-r from-orange-400 to-amber-400">
                    <svg xmlns="http://www.w3.org/2000/svg" className="h-6 w-6 text-orange-500" fill="none" viewBox="0 0 24 24" stroke="currentColor" strokeWidth="2"><path strokeLinecap="round" strokeLinejoin="round" d="M4 5a1 1 0 011-1h14a1 1 0 011 1v2a1 1 0 01-1 1H5a1 1 0 01-1-1V5zM4 13a1 1 0 011-1h6a1 1 0 011 1v6a1 1 0 01-1 1H5a1 1 0 01-1-1v-6zM16 13a1 1 0 011-1h2a1 1 0 011 1v6a1 1 0 01-1 1h-2a1 1 0 01-1-1v-6z" /></svg>
                    <span id="ai-art-direction-title">{isModelVisible ? '5.' : '4.'} Arahan Gaya & Teknis (AI)</span>
                  </h3>
                  <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-3 gap-6">
                      {renderSelect('ai-product-type-select', 'Jenis Produk', OPTIONS.productType, aiFormState.art.productType, e => handleAiArtChange('productType', e.target.value))}
                      {renderSelect('ai-style-select', 'Gaya Fotografi', OPTIONS.style, aiFormState.art.style, e => handleAiArtChange('style', e.target.value))}
                      {renderSelect('ai-mood-select', 'Mood', OPTIONS.mood, aiFormState.art.mood, e => handleAiArtChange('mood', e.target.value))}
                      {renderSelect('ai-location-select', 'Lokasi', OPTIONS.location, aiFormState.art.location, e => handleAiArtChange('location', e.target.value), 'sm:col-span-2 lg:col-span-3')}
                      {renderSelect('ai-background-select', 'Latar Belakang & Tekstur', OPTIONS.background, aiFormState.art.background, e => handleAiArtChange('background', e.target.value), 'sm:col-span-2 lg:col-span-3')}
                      {renderSelect('ai-composition-select', 'Komposisi & Framing', OPTIONS.composition, aiFormState.art.composition, e => handleAiArtChange('composition', e.target.value), 'sm:col-span-2 lg:col-span-3')}
                      {isStillLifeControlsVisible && renderSelect('ai-product-angle-select', 'Sudut Pandang Kamera (Produk Saja)', OPTIONS.productAngle, aiFormState.art.productAngle, e => handleAiArtChange('productAngle', e.target.value), 'sm:col-span-2 lg:col-span-3')}
                      {renderSelect('ai-image-style-select', 'Gaya Gambar (AI)', OPTIONS.aiImageStyle, aiFormState.art.aiImageStyle, e => handleAiArtChange('aiImageStyle', e.target.value))}
                      {renderSelect('ai-artist-style-select', 'Gaya Seniman (AI)', OPTIONS.aiArtistStyle, aiFormState.art.aiArtistStyle, e => handleAiArtChange('aiArtistStyle', e.target.value))}
                      {renderSelect('ai-lighting-feel-select', 'Nuansa Cahaya', OPTIONS.lighting, aiFormState.art.lighting, e => handleAiArtChange('lighting', e.target.value))}
                      {renderSelect('ai-prop-select', 'Detail Properti', OPTIONS.props, aiFormState.art.props, e => handleAiArtChange('props', e.target.value))}
                      {isStillLifeControlsVisible && renderSelect('ai-product-position-select', 'Posisi Produk', OPTIONS.productPosition, aiFormState.art.productPosition, e => handleAiArtChange('productPosition', e.target.value))}
                      {renderSelect('ai-vfx-select', 'Efek Visual (VFX)', OPTIONS.vfx, aiFormState.art.vfx, e => handleAiArtChange('vfx', e.target.value))}
                      {renderSelect('ai-sound-design-select', 'Desain Suara', OPTIONS.sound, aiFormState.art.sound, e => handleAiArtChange('sound', e.target.value), 'sm:col-span-2 lg:col-span-3')}
                  </div>
                </div>
            </div>

            <div id="manual-controls" className={`space-y-8 ${activeTab === 'manual' ? '' : 'hidden'}`}>
                <div className="space-y-4">
                    <h3 className="flex items-center text-lg font-semibold gap-3 text-transparent bg-clip-text bg-gradient-to-r from-orange-400 to-amber-400">
                        <svg xmlns="http://www.w3.org/2000/svg" className="h-6 w-6 text-orange-500" fill="none" viewBox="0 0 24 24" stroke="currentColor" strokeWidth="2"><path strokeLinecap="round" strokeLinejoin="round" d="M19.428 15.428a2 2 0 00-1.022-.547l-2.387-.477a6 6 0 00-3.86.517l-.318.158a6 6 0 01-3.86.517L6.05 15.21a2 2 0 00-1.806.547M8 4h8l-1 1v5.172a2 2 0 00.586 1.414l5 5c1.26 1.26.367 3.414-1.415 3.414H4.828c-1.782 0-2.674-2.154-1.414-3.414l5-5A2 2 0 009 10.172V5L8 4z" /></svg>
                        <span>1. Konsep Utama</span>
                    </h3>
                    <select id="concept-select" value={formState.concept} onChange={e => setFormState(s => ({ ...s, concept: e.target.value }))} className="block w-full bg-slate-700/50 border-slate-600 rounded-md shadow-sm py-2.5 px-4 text-slate-100 focus:outline-none focus:ring-2 focus:ring-orange-500 focus:border-orange-500 sm:text-sm transition">
                        {OPTIONS.concept.map(opt => <option key={opt.value} value={opt.value}>{opt.label}</option>)}
                    </select>
                </div>

                {/* Upload Produk */}
                <div className="space-y-4 pt-8 border-t border-slate-700">
                    <h3 className="flex items-center text-lg font-semibold gap-3 text-transparent bg-clip-text bg-gradient-to-r from-orange-400 to-amber-400">
                        <svg xmlns="http://www.w3.org/2000/svg" className="h-6 w-6 text-orange-500" fill="none" viewBox="0 0 24 24" stroke="currentColor" strokeWidth="2"><path strokeLinecap="round" strokeLinejoin="round" d="M4 16v1a3 3 0 003 3h10a3 3 0 003-3v-1m-4-8l-4-4m0 0L8 8m4-4v12" /></svg>
                        <span>Upload Produk</span>
                    </h3>
                    <p className="text-xs text-slate-500">Unggah foto produk sebagai referensi untuk AI. (Opsional)</p>
                    {manualProductImage && (
                        <div className="relative w-full h-40 bg-slate-700/50 rounded-lg flex items-center justify-center border-2 border-dashed border-slate-600">
                            <img src={`data:${manualProductImage.mimeType};base64,${manualProductImage.base64}`} alt="Preview produk" className="max-h-full max-w-full object-contain rounded" />
                            <button onClick={() => setManualProductImage(null)} className="absolute top-2 right-2 p-1 bg-red-500/80 hover:bg-red-500 rounded-full text-white transition-colors" title="Hapus gambar">
                                <svg xmlns="http://www.w3.org/2000/svg" className="h-4 w-4" fill="none" viewBox="0 0 24 24" stroke="currentColor" strokeWidth="2"><path strokeLinecap="round" strokeLinejoin="round" d="M6 18L18 6M6 6l12 12" /></svg>
                            </button>
                        </div>
                    )}
                    <label htmlFor="manual-product-upload" className="cursor-pointer w-full inline-flex items-center justify-center px-4 py-2.5 text-sm font-semibold text-orange-400 bg-slate-700/50 hover:bg-slate-700 rounded-md transition-colors border border-dashed border-slate-600">
                        <svg xmlns="http://www.w3.org/2000/svg" className="h-5 w-5 mr-2" fill="none" viewBox="0 0 24 24" stroke="currentColor" strokeWidth="2"><path strokeLinecap="round" strokeLinejoin="round" d="M4 16v1a3 3 0 003 3h10a3 3 0 003-3v-1m-4-8l-4-4m0 0L8 8m4-4v12" /></svg>
                        {manualProductImage ? `Ganti: ${manualProductImage.name}` : 'Pilih Foto Produk...'}
                    </label>
                    <input type="file" id="manual-product-upload" className="hidden" accept="image/png, image/jpeg, image/webp" onChange={handleManualImageUpload('product')} />
                </div>

                {/* Upload Model - only when concept requires model */}
                {isModelVisible && (
                    <div className="space-y-4 pt-8 border-t border-slate-700">
                        <h3 className="flex items-center text-lg font-semibold gap-3 text-transparent bg-clip-text bg-gradient-to-r from-orange-400 to-amber-400">
                            <svg xmlns="http://www.w3.org/2000/svg" className="h-6 w-6 text-orange-500" fill="none" viewBox="0 0 24 24" stroke="currentColor" strokeWidth="2"><path strokeLinecap="round" strokeLinejoin="round" d="M16 7a4 4 0 11-8 0 4 4 0 018 0zM12 14a7 7 0 00-7 7h14a7 7 0 00-7-7z" /></svg>
                            <span>Upload Model</span>
                        </h3>
                        <p className="text-xs text-slate-500">Unggah foto referensi model/talent. AI akan menyesuaikan prompt dengan referensi ini. (Opsional)</p>
                        {manualModelImage && (
                            <div className="relative w-full h-40 bg-slate-700/50 rounded-lg flex items-center justify-center border-2 border-dashed border-slate-600">
                                <img src={`data:${manualModelImage.mimeType};base64,${manualModelImage.base64}`} alt="Preview model" className="max-h-full max-w-full object-contain rounded" />
                                <button onClick={() => setManualModelImage(null)} className="absolute top-2 right-2 p-1 bg-red-500/80 hover:bg-red-500 rounded-full text-white transition-colors" title="Hapus gambar">
                                    <svg xmlns="http://www.w3.org/2000/svg" className="h-4 w-4" fill="none" viewBox="0 0 24 24" stroke="currentColor" strokeWidth="2"><path strokeLinecap="round" strokeLinejoin="round" d="M6 18L18 6M6 6l12 12" /></svg>
                                </button>
                            </div>
                        )}
                        <label htmlFor="manual-model-upload" className="cursor-pointer w-full inline-flex items-center justify-center px-4 py-2.5 text-sm font-semibold text-orange-400 bg-slate-700/50 hover:bg-slate-700 rounded-md transition-colors border border-dashed border-slate-600">
                            <svg xmlns="http://www.w3.org/2000/svg" className="h-5 w-5 mr-2" fill="none" viewBox="0 0 24 24" stroke="currentColor" strokeWidth="2"><path strokeLinecap="round" strokeLinejoin="round" d="M16 7a4 4 0 11-8 0 4 4 0 018 0zM12 14a7 7 0 00-7 7h14a7 7 0 00-7-7z" /></svg>
                            {manualModelImage ? `Ganti: ${manualModelImage.name}` : 'Pilih Foto Referensi Model...'}
                        </label>
                        <input type="file" id="manual-model-upload" className="hidden" accept="image/png, image/jpeg, image/webp" onChange={handleManualImageUpload('model')} />
                    </div>
                )}

                {isModelVisible && (
                    <div id="model-controls" className="collapsible">
                        <div className="space-y-8 pt-8 border-t border-slate-700">
                            <h3 className="flex items-center text-lg font-semibold gap-3 text-transparent bg-clip-text bg-gradient-to-r from-orange-400 to-amber-400">
                                <svg xmlns="http://www.w3.org/2000/svg" className="h-6 w-6 text-orange-500" fill="none" viewBox="0 0 24 24" stroke="currentColor" strokeWidth="2"><path strokeLinecap="round" strokeLinejoin="round" d="M16 7a4 4 0 11-8 0 4 4 0 018 0zM12 14a7 7 0 00-7 7h14a7 7 0 00-7-7z" /></svg>
                                <span>2. Detail Model</span>
                            </h3>
                            <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-3 gap-6">
                                {renderSelect('model-gender-select', 'Jenis Kelamin', OPTIONS.gender, formState.model.gender, e => handleModelChange('gender', e.target.value))}
                                {renderSelect('model-ethnicity-select', 'Etnisitas', OPTIONS.ethnicity, formState.model.ethnicity, e => handleModelChange('ethnicity', e.target.value))}
                                {renderSelect('model-age-select', 'Perkiraan Usia', OPTIONS.age, formState.model.age, e => handleModelChange('age', e.target.value))}
                                {renderSelect('model-body-shape-select', 'Bentuk Badan', OPTIONS.bodyShape, formState.model.bodyShape, e => handleModelChange('bodyShape', e.target.value))}
                                {renderSelect('model-hair-select', 'Gaya Rambut', OPTIONS.hair, formState.model.hair, e => handleModelChange('hair', e.target.value))}
                                {renderSelect('model-makeup-select', 'Gaya Riasan', OPTIONS.makeup, formState.model.makeup, e => handleModelChange('makeup', e.target.value))}
                                {renderSelect('shot-type-select', 'Tipe Bidikan', OPTIONS.shotType, formState.model.shotType, e => handleModelChange('shotType', e.target.value), 'sm:col-span-2 lg:col-span-3')}
                                {renderSelect('model-expression-select', 'Ekspresi Model', OPTIONS.expression, formState.model.expression, e => handleModelChange('expression', e.target.value), 'sm:col-span-2 lg:col-span-3')}
                                {renderSelect('model-pose-select', 'Atur Gaya (Pose)', OPTIONS.pose, formState.model.pose, e => handleModelChange('pose', e.target.value), 'sm:col-span-2 lg:col-span-3')}
                                <div className="sm:col-span-2 lg:col-span-3">
                                    <label className="block text-sm font-medium text-slate-400 mb-2">Interaksi Model</label>
                                    <div id="model-interaction-group" className="grid grid-cols-2 sm:grid-cols-4 gap-1 bg-slate-700/50 rounded-lg p-1 h-auto">
                                        {OPTIONS.interaction.map(opt => (
                                            <button
                                                key={opt.value}
                                                onClick={() => handleInteractionChange(opt.value)}
                                                className={`text-sm rounded-md py-1.5 transition-colors ${formState.model.interaction.includes(opt.value) ? 'bg-orange-500 text-white' : 'hover:bg-slate-600 text-slate-300'}`}
                                            >
                                                {opt.label}
                                            </button>
                                        ))}
                                    </div>
                                </div>
                            </div>
                        </div>
                    </div>
                )}
                
                <div className="space-y-8 pt-8 border-t border-slate-700">
                  <h3 className="flex items-center text-lg font-semibold gap-3 text-transparent bg-clip-text bg-gradient-to-r from-orange-400 to-amber-400">
                    <svg xmlns="http://www.w3.org/2000/svg" className="h-6 w-6 text-orange-500" fill="none" viewBox="0 0 24 24" stroke="currentColor" strokeWidth="2"><path strokeLinecap="round" strokeLinejoin="round" d="M4 5a1 1 0 011-1h14a1 1 0 011 1v2a1 1 0 01-1 1H5a1 1 0 01-1-1V5zM4 13a1 1 0 011-1h6a1 1 0 011 1v6a1 1 0 01-1 1H5a1 1 0 01-1-1v-6zM16 13a1 1 0 011-1h2a1 1 0 011 1v6a1 1 0 01-1 1h-2a1 1 0 01-1-1v-6z" /></svg>
                    <span id="art-direction-title">{isModelVisible ? '3.' : '2.'} Arahan Gaya & Teknis</span>
                  </h3>
                  <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-3 gap-6">
                      {renderSelect('product-type-select', 'Jenis Produk', OPTIONS.productType, formState.art.productType, e => handleArtChange('productType', e.target.value))}
                      {renderSelect('style-select', 'Gaya Fotografi', OPTIONS.style, formState.art.style, e => handleArtChange('style', e.target.value))}
                      {renderSelect('mood-select', 'Mood', OPTIONS.mood, formState.art.mood, e => handleArtChange('mood', e.target.value))}
                      {renderSelect('location-select', 'Lokasi', OPTIONS.location, formState.art.location, e => handleArtChange('location', e.target.value), 'sm:col-span-2 lg:col-span-3')}
                      {renderSelect('background-select', 'Latar Belakang & Tekstur', OPTIONS.background, formState.art.background, e => handleArtChange('background', e.target.value), 'sm:col-span-2 lg:col-span-3')}
                      {renderSelect('composition-select', 'Komposisi & Framing', OPTIONS.composition, formState.art.composition, e => handleArtChange('composition', e.target.value), 'sm:col-span-2 lg:col-span-3')}
                      {isStillLifeControlsVisible && renderSelect('product-angle-select', 'Sudut Pandang Kamera (Produk Saja)', OPTIONS.productAngle, formState.art.productAngle, e => handleArtChange('productAngle', e.target.value), 'sm:col-span-2 lg:col-span-3')}
                      {renderSelect('lighting-feel-select', 'Nuansa Cahaya', OPTIONS.lighting, formState.art.lighting, e => handleArtChange('lighting', e.target.value))}
                      {renderSelect('prop-select', 'Detail Properti', OPTIONS.props, formState.art.props, e => handleArtChange('props', e.target.value))}
                      {isStillLifeControlsVisible && renderSelect('product-position-select', 'Posisi Produk', OPTIONS.productPosition, formState.art.productPosition, e => handleArtChange('productPosition', e.target.value))}
                      {renderSelect('vfx-select', 'Efek Visual (VFX)', OPTIONS.vfx, formState.art.vfx, e => handleArtChange('vfx', e.target.value))}
                      {renderSelect('sound-design-select', 'Desain Suara', OPTIONS.sound, formState.art.sound, e => handleArtChange('sound', e.target.value), 'sm:col-span-2 lg:col-span-3')}
                  </div>
                </div>
            </div>

            <div className="space-y-4 pt-8 border-t border-slate-700">
              <h3 className="flex items-center text-lg font-semibold gap-3 text-transparent bg-clip-text bg-gradient-to-r from-orange-400 to-amber-400">Pengaturan Output</h3>
              <div id="aspect-ratio-group" className="flex items-center space-x-2 bg-slate-700/50 rounded-lg p-1 h-[42px]">
                  {OPTIONS.aspectRatio.map(ratio => (
                      <button 
                        key={ratio} 
                        onClick={() => {
                            if (activeTab === 'manual') setFormState(s => ({...s, output: {...s.output, aspectRatio: ratio}}));
                            else setAiFormState(s => ({...s, output: {...s.output, aspectRatio: ratio}}));
                        }}
                        className={`flex-1 text-sm rounded-md py-1.5 transition-colors ${(activeTab === 'manual' ? formState.output.aspectRatio : aiFormState.output.aspectRatio) === ratio ? 'bg-orange-500 text-white' : 'hover:bg-slate-600'}`}>
                          {ratio}
                      </button>
                  ))}
              </div>
            </div>

            <div className="pt-6">
              <button id="generate-button" onClick={handleGenerate} disabled={isLoading} className="w-full inline-flex items-center justify-center px-12 py-4 text-lg font-semibold text-white bg-gradient-to-r from-orange-500 to-amber-500 rounded-lg shadow-lg shadow-orange-500/40 hover:shadow-xl hover:shadow-orange-500/40 focus:outline-none focus:ring-2 focus:ring-offset-2 focus:ring-offset-slate-900 focus:ring-orange-500 transition-all duration-300 ease-in-out transform hover:-translate-y-0.5 disabled:from-slate-600 disabled:to-slate-600 disabled:shadow-none disabled:translate-y-0 disabled:cursor-not-allowed">
                {isLoading && <svg className="animate-spin -ml-1 mr-3 h-5 w-5 text-white" xmlns="http://www.w3.org/2000/svg" fill="none" viewBox="0 0 24 24"><circle className="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" strokeWidth="4"></circle><path className="opacity-75" fill="currentColor" d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4zm2 5.291A7.962 7.962 0 014 12H0c0 3.042 1.135 5.824 3 7.938l3-2.647z"></path></svg>}
                <span>{isLoading ? 'Generating...' : 'Generate New Prompt'}</span>
              </button>
            </div>
          </div>
          
          <div className="space-y-8">
            {/* New Instructions Block */}
            <div className="bg-slate-800/50 backdrop-blur-lg border border-slate-700 rounded-xl p-6 shadow-2xl shadow-black/30">
                <h3 className="text-xl font-bold text-transparent bg-clip-text bg-gradient-to-r from-orange-400 to-amber-400 mb-6 flex items-center gap-3">
                    <svg xmlns="http://www.w3.org/2000/svg" className="h-6 w-6 text-orange-500" fill="none" viewBox="0 0 24 24" stroke="currentColor" strokeWidth="2">
                        <path strokeLinecap="round" strokeLinejoin="round" d="M13 16h-1v-4h-1m1-4h.01M21 12a9 9 0 11-18 0 9 9 0 0118 0z" />
                    </svg>
                    Instruksi Pembuatan Gambar
                </h3>
                <div className="space-y-4">
                    <div className="flex items-start gap-4 p-4 bg-slate-700/30 rounded-lg border border-slate-600/50 hover:border-orange-500/30 transition-colors">
                        <div className="flex-shrink-0 w-10 h-10 rounded-full bg-gradient-to-br from-orange-500 to-amber-600 flex items-center justify-center text-white font-bold text-lg shadow-lg shadow-orange-500/20">
                            1
                        </div>
                        <div>
                            <h4 className="text-lg font-semibold text-slate-200 mb-1">Gunakan Akun Gemini Pro</h4>
                            <p className="text-slate-400 text-sm leading-relaxed">Pastikan Anda telah login menggunakan akun Gemini Pro untuk mengakses fitur pembuatan gambar tingkat lanjut.</p>
                        </div>
                    </div>
                    
                    <div className="flex items-start gap-4 p-4 bg-slate-700/30 rounded-lg border border-slate-600/50 hover:border-orange-500/30 transition-colors">
                        <div className="flex-shrink-0 w-10 h-10 rounded-full bg-slate-600 flex items-center justify-center text-white font-bold text-lg border border-slate-500">
                           <svg xmlns="http://www.w3.org/2000/svg" className="h-6 w-6" fill="none" viewBox="0 0 24 24" stroke="currentColor" strokeWidth="2">
                                <path strokeLinecap="round" strokeLinejoin="round" d="M4 16l4.586-4.586a2 2 0 012.828 0L16 16m-2-2l1.586-1.586a2 2 0 012.828 0L20 14m-6-6h.01M6 20h12a2 2 0 002-2V6a2 2 0 00-2-2H6a2 2 0 00-2 2v12a2 2 0 002 2z" />
                            </svg>
                        </div>
                        <div>
                            <h4 className="text-lg font-semibold text-slate-200 mb-1">Pilih Menu "Create Image"</h4>
                            <p className="text-slate-400 text-sm leading-relaxed">Di dalam platform Gemini Pro, navigasikan ke menu atau opsi "Create Image" untuk memulai sesi.</p>
                        </div>
                    </div>

                    <div className="flex items-start gap-4 p-4 bg-slate-700/30 rounded-lg border border-slate-600/50 hover:border-orange-500/30 transition-colors">
                        <div className="flex-shrink-0 w-10 h-10 rounded-full bg-slate-600 flex items-center justify-center text-white font-bold text-lg border border-slate-500">
                            <svg xmlns="http://www.w3.org/2000/svg" className="h-6 w-6" fill="none" viewBox="0 0 24 24" stroke="currentColor" strokeWidth="2">
                                <path strokeLinecap="round" strokeLinejoin="round" d="M19.428 15.428a2 2 0 00-1.022-.547l-2.387-.477a6 6 0 00-3.86.517l-.318.158a6 6 0 01-3.86.517L6.05 15.21a2 2 0 00-1.806.547M8 4h8l-1 1v5.172a2 2 0 00.586 1.414l5 5c1.26 1.26.367 3.414-1.415 3.414H4.828c-1.782 0-2.674-2.154-1.414-3.414l5-5A2 2 0 009 10.172V5L8 4z" />
                            </svg>
                        </div>
                        <div>
                            <h4 className="text-lg font-semibold text-slate-200 mb-1">Aktifkan Model "Nano Banana Pro"</h4>
                            <p className="text-slate-400 text-sm leading-relaxed">Pilih model "Nano Banana Pro" sebagai engine generasi gambar untuk hasil yang optimal.</p>
                        </div>
                    </div>
                </div>
            </div>

             <div id="prompt-output-image" className="relative bg-slate-800/50 backdrop-blur-lg border border-slate-700 rounded-xl p-4 sm:p-6 shadow-2xl shadow-black/30 min-h-[300px] flex flex-col">
                <h3 className="text-lg font-semibold text-slate-100 mb-4">Prompt Foto</h3>
                <div className="absolute top-4 right-4 flex items-center gap-2">
                    <button onClick={() => handleSavePrompt('Foto', photoPrompt)} title="Simpan Prompt" className="p-2 text-slate-400 hover:text-orange-400 bg-slate-700/50 hover:bg-slate-700 rounded-lg transition-all duration-200" disabled={!photoPrompt || savedPrompts['Foto']}>
                        {savedPrompts['Foto'] ? <svg className="w-5 h-5 text-green-400" xmlns="http://www.w3.org/2000/svg" fill="none" viewBox="0 0 24 24" stroke="currentColor"><path strokeLinecap="round" strokeLinejoin="round" strokeWidth="2" d="M5 13l4 4L19 7" /></svg> : <svg className="w-5 h-5" xmlns="http://www.w3.org/2000/svg" fill="none" viewBox="0 0 24 24" stroke="currentColor" strokeWidth="2"><path strokeLinecap="round" strokeLinejoin="round" d="M5 5a2 2 0 012-2h8l2 2v13.5a1.5 1.5 0 01-3 0V6H5a1 1 0 011-1z" /><path strokeLinecap="round" strokeLinejoin="round" d="M15 12a3 3 0 11-6 0 3 3 0 016 0z" /></svg>}
                    </button>
                    <button onClick={() => copyToClipboard(photoPrompt, 'image')} title="Salin Prompt" className="p-2 text-slate-400 hover:text-orange-400 bg-slate-700/50 hover:bg-slate-700 rounded-lg transition-all duration-200" disabled={!photoPrompt}>
                        {copiedType === 'image' ? <svg className="w-5 h-5 text-green-400" xmlns="http://www.w3.org/2000/svg" fill="none" viewBox="0 0 24 24" stroke="currentColor"><path strokeLinecap="round" strokeLinejoin="round" strokeWidth="2" d="M5 13l4 4L19 7" /></svg> : <svg className="w-5 h-5" xmlns="http://www.w3.org/2000/svg" fill="none" viewBox="0 0 24 24" stroke="currentColor"><path strokeLinecap="round" strokeLinejoin="round" strokeWidth="2" d="M8 16H6a2 2 0 01-2-2V6a2 2 0 012-2h8a2 2 0 012 2v2m-6 12h8a2 2 0 002-2v-8a2 2 0 00-2-2h-8a2 2 0 00-2 2v8a2 2 0 002 2z" /></svg>}
                    </button>
                </div>
                <div className="flex-grow flex items-center justify-center">
                    {isLoading && <p>Generating...</p>}
                    <p className="text-slate-300 text-base leading-relaxed whitespace-pre-wrap">{photoPrompt || 'Hasil prompt foto akan muncul di sini...'}</p>
                </div>
            </div>
            
            <div id="prompt-output-video" className="relative bg-slate-800/50 backdrop-blur-lg border border-slate-700 rounded-xl p-4 sm:p-6 shadow-2xl shadow-black/30 min-h-[300px] flex flex-col">
                <h3 className="text-lg font-semibold text-slate-100 mb-4">Prompt Video</h3>
                <div className="absolute top-4 right-4 flex items-center gap-2">
                    <button onClick={() => handleSavePrompt('Video', videoPrompt)} title="Simpan Prompt" className="p-2 text-slate-400 hover:text-orange-400 bg-slate-700/50 hover:bg-slate-700 rounded-lg transition-all duration-200" disabled={!videoPrompt || savedPrompts['Video']}>
                        {savedPrompts['Video'] ? <svg className="w-5 h-5 text-green-400" xmlns="http://www.w3.org/2000/svg" fill="none" viewBox="0 0 24 24" stroke="currentColor"><path strokeLinecap="round" strokeLinejoin="round" strokeWidth="2" d="M5 13l4 4L19 7" /></svg> : <svg className="w-5 h-5" xmlns="http://www.w3.org/2000/svg" fill="none" viewBox="0 0 24 24" stroke="currentColor" strokeWidth="2"><path strokeLinecap="round" strokeLinejoin="round" d="M5 5a2 2 0 012-2h8l2 2v13.5a1.5 1.5 0 01-3 0V6H5a1 1 0 011-1z" /><path strokeLinecap="round" strokeLinejoin="round" d="M15 12a3 3 0 11-6 0 3 3 0 016 0z" /></svg>}
                    </button>
                    <button onClick={() => copyToClipboard(videoPrompt, 'video')} title="Salin Prompt" className="p-2 text-slate-400 hover:text-orange-400 bg-slate-700/50 hover:bg-slate-700 rounded-lg transition-all duration-200" disabled={!videoPrompt}>
                       {copiedType === 'video' ? <svg className="w-5 h-5 text-green-400" xmlns="http://www.w3.org/2000/svg" fill="none" viewBox="0 0 24 24" stroke="currentColor"><path strokeLinecap="round" strokeLinejoin="round" strokeWidth="2" d="M5 13l4 4L19 7" /></svg> : <svg className="w-5 h-5" xmlns="http://www.w3.org/2000/svg" fill="none" viewBox="0 0 24 24" stroke="currentColor"><path strokeLinecap="round" strokeLinejoin="round" strokeWidth="2" d="M8 16H6a2 2 0 01-2-2V6a2 2 0 012-2h8a2 2 0 012 2v2m-6 12h8a2 2 0 002-2v-8a2 2 0 00-2-2h-8a2 2 0 00-2 2v8a2 2 0 002 2z" /></svg>}
                    </button>
                </div>
                <div className="flex-grow flex items-center justify-center">
                    {isLoading && <p>Generating...</p>}
                    <p className="text-slate-300 text-base leading-relaxed whitespace-pre-wrap">{videoPrompt || 'Hasil prompt video akan muncul di sini...'}</p>
                </div>
            </div>
          </div>
        </div>
      </div>
    );
};

export default PhotoVideoStudio;
