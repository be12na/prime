
import { GoogleGenAI } from "@google/genai";

// Mengambil kunci API dari sesi browser (prioritas) atau lingkungan
const getAI = () => {
    const userApiKey = sessionStorage.getItem('geminiApiKey');
    let apiKey = userApiKey;

    // Cek `process.env` dengan aman sebagai fallback.
    // Ini mencegah error "process is not defined" di lingkungan browser murni.
    if (!apiKey && typeof process !== 'undefined' && process.env && process.env.API_KEY) {
        apiKey = process.env.API_KEY;
    }

    if (!apiKey) {
        throw new Error("Kunci API Gemini tidak diatur. Harap masukkan kunci Anda di menu 'Panduan & Pengaturan'.");
    }
    return { ai: new GoogleGenAI({ apiKey }), apiKey };
};

/**
 * Extracts a JSON string from a larger string, tolerating surrounding text.
 * @param text The text containing the JSON block.
 * @returns The extracted JSON string.
 * @throws An error if no valid JSON is found.
 */
const extractJson = (text: string): string => {
    // First, try to find the JSON within Markdown code fences
    const fencedMatch = text.match(/```json\n([\s\S]*?)\n```/);
    if (fencedMatch && fencedMatch[1]) {
        return fencedMatch[1];
    }

    // If not found, find the first and last brace to extract the JSON object
    const firstBrace = text.indexOf('{');
    const lastBrace = text.lastIndexOf('}');
    if (firstBrace !== -1 && lastBrace > firstBrace) {
        return text.substring(firstBrace, lastBrace + 1);
    }
    
    // Fallback for arrays
    const firstBracket = text.indexOf('[');
    const lastBracket = text.lastIndexOf(']');
    if (firstBracket !== -1 && lastBracket > firstBracket) {
        return text.substring(firstBracket, lastBracket + 1);
    }

    throw new Error("Tidak dapat menemukan blok JSON yang valid dalam respons AI.");
};


const generateContent = async (model: string, prompt: string, useSearch: boolean = false): Promise<string> => {
    try {
        const { ai } = getAI();
        const config = useSearch ? { tools: [{ googleSearch: {} }] } : {};
        const response = await ai.models.generateContent({
            model,
            contents: prompt,
            config
        });
        
        let text = response.text;

        if (!text && response.candidates?.[0]?.content?.parts) {
            text = response.candidates[0].content.parts.map(p => p.text).join('');
        }

        if (useSearch && response.candidates?.[0]?.groundingMetadata?.groundingChunks) {
            const citations = response.candidates[0].groundingMetadata.groundingChunks
                .map((chunk: any) => chunk.web?.uri)
                .filter(Boolean);
            if (citations.length > 0) {
                text = (text || '') + `\n\n**Sumber:**\n${citations.map((c: string) => `- ${c}`).join('\n')}`;
            }
        }

        if (text) {
            return text;
        }

        throw new Error("No text response from Gemini API");

    } catch (error) {
        console.error("Gemini API call failed:", error);
        if (error instanceof Error) {
            return `Error: Terjadi masalah saat menghubungi AI. Detail: ${error.message}`;
        }
        return "Error: Terjadi masalah saat menghubungi AI. Silakan coba lagi.";
    }
};

export const generatePhotoVideoPrompt = async (formData: any, activeMode: 'manual' | 'ai'): Promise<{ photo: string, video: string }> => {
    const promptData = { ...formData };

    if (activeMode === 'ai' && promptData.imageBase64) {
        // Ganti data base64 yang besar dengan placeholder untuk menghindari error token limit.
        promptData.imageBase64 = `[Gambar produk telah diunggah oleh pengguna, ukuran: ${Math.round(promptData.imageBase64.length * 3/4 / 1024)} KB]`;
        delete promptData.mimeType;
    }

    if (activeMode === 'manual') {
        if (promptData.modelImageBase64) {
            promptData.modelImageBase64 = `[Foto referensi MODEL telah diunggah oleh pengguna, ukuran: ${Math.round(promptData.modelImageBase64.length * 3/4 / 1024)} KB. Sesuaikan deskripsi model dalam prompt dengan referensi gambar ini.]`;
            delete promptData.modelImageMimeType;
        } else {
            delete promptData.modelImageBase64;
            delete promptData.modelImageMimeType;
        }
        if (promptData.productImageBase64) {
            promptData.productImageBase64 = `[Foto PRODUK telah diunggah oleh pengguna, ukuran: ${Math.round(promptData.productImageBase64.length * 3/4 / 1024)} KB. Gunakan foto ini sebagai referensi utama tampilan produk.]`;
            delete promptData.productImageMimeType;
        } else {
            delete promptData.productImageBase64;
            delete promptData.productImageMimeType;
        }
    }
    
    const prompt = `
Anda adalah seorang Creative Director dan Prompt Engineer ahli untuk AI Image Generator (seperti Midjourney, DALL-E, Leonardo AI).
Tugas Anda adalah membuat 2 prompt terpisah (satu untuk FOTO, satu untuk VIDEO) berdasarkan detail berikut.
Pastikan prompt sangat detail, dalam bahasa Inggris, dan mengikuti format terbaik untuk hasil fotorealistis dan sinematik.

Mode yang digunakan: ${activeMode}

**Detail Input dari User:**
${JSON.stringify(promptData, null, 2)}

**Tugas Anda:**
1.  **Analisis Input:** Pahami semua detail yang diberikan. Jika ada input yang saling bertentangan secara logika (misal: "close-up shot" dengan "full body pose"), gunakan penilaian profesional Anda untuk memilih yang paling masuk akal atau menggabungkannya secara kreatif.
2.  **Buat Prompt FOTO:**
    *   Mulai dengan deskripsi subjek (model/produk).
    *   Jelaskan secara detail gaya fotografi, komposisi, angle, pencahayaan, dan mood.
    *   Tambahkan detail teknis seperti jenis kamera, lensa, dan pengaturan (misal: "photorealistic, shot on Sony A7III with 85mm f1.4 lens, sharp focus, beautiful bokeh").
    *   Akhiri dengan parameter teknis seperti "--ar {aspectRatio} --v 6.0 --style raw".
3.  **Buat Prompt VIDEO:**
    *   Mulai dengan deskripsi adegan sinematik.
    *   Fokus pada pergerakan (model, kamera, atau elemen).
    *   Jelaskan gaya sinematik, color grading, mood, dan pencahayaan.
    *   Tambahkan detail teknis video (misal: "cinematic shot, 4K, shallow depth of field, anamorphic lens flare, graded with moody teal and orange tones").
    *   Jika ada Sound Design, jelaskan jenis suara atau musik yang cocok.
    *   Akhiri dengan parameter teknis seperti "--ar {aspectRatio} --v 6.0".

**Format Output (WAJIB):**
Berikan jawaban dalam format JSON Markdown yang valid seperti ini, tanpa teks pembuka atau penutup lainnya:
\`\`\`json
{
  "photo_prompt": "prompt foto bahasa inggris di sini",
  "video_prompt": "prompt video bahasa inggris di sini"
}
\`\`\`
`;
    // Gunakan gemini-3-flash-preview untuk respons teks yang cepat
    const responseText = await generateContent('gemini-3-flash-preview', prompt);
    try {
        const jsonString = extractJson(responseText);
        const result = JSON.parse(jsonString);
        return { photo: result.photo_prompt, video: result.video_prompt };
    } catch (e) {
        console.error("Failed to parse JSON from Gemini response", e, "Raw response:", responseText);
        const errorMessage = e instanceof Error ? e.message : "Unknown parsing error";
        return { 
            photo: `Gagal mem-parsing hasil dari AI: ${errorMessage}`, 
            video: `Gagal mem-parsing hasil dari AI: ${errorMessage}`
        };
    }
};


export const generateLaunchContent = async (formData: any, mode: 'product' | 'ad'): Promise<string> => {
    const prompt = `
Anda adalah seorang Senior Copywriter dan Brand Strategist.
Tugas Anda adalah membuat aset copywriting berdasarkan data yang diberikan.

**Mode:** ${mode === 'product' ? 'Luncurkan Produk' : 'Luncurkan Iklan'}
**Data dari User:**
${JSON.stringify(formData, null, 2)}

**Tugas Spesifik:**

${mode === 'product' ? `
**Mode: Luncurkan Produk**
Buat 4 bagian copywriting untuk halaman produk e-commerce. Gunakan bahasa yang persuasif dan sesuai dengan target audiens.
1.  **Nama Produk (3 Pilihan):** Berikan 3 alternatif nama produk yang menarik dan mudah diingat.
2.  **Tagline (3 Pilihan):** Buat 3 tagline yang menangkap esensi produk.
3.  **Deskripsi Produk:** Tulis deskripsi produk yang menarik (sekitar 3-5 paragraf), fokus pada storytelling dan emosi.
4.  **Highlight / Poin Utama (5 Poin):** Tulis 5 poin utama dalam format bullet points, ubah fitur menjadi manfaat langsung bagi pelanggan.
` : `
**Mode: Luncurkan Iklan**
Rancang sebuah "Paket Iklan TikTok" yang lengkap.
1.  **Angle / Sudut Pandang Iklan:** Tentukan 1 angle paling kuat untuk iklan ini (Contoh: "Solusi Masalah", "FOMO/Fear of Missing Out", "Social Proof").
2.  **Hook (3 Detik Pertama):** Tulis 3 alternatif kalimat pembuka yang sangat kuat untuk menghentikan scroll.
3.  **Skrip Video Iklan (30-45 detik):** Tulis skrip lengkap dengan format: [VISUAL] dan [AUDIO/VOICEOVER]. Skrip harus mencakup hook, problem/solution, product showcase, dan Call to Action (CTA).
4.  **Teks Caption Iklan:** Tulis caption yang singkat, menarik, dan diakhiri dengan CTA.
5.  **Rekomendasi Hashtag:** Berikan 5-7 hashtag yang relevan.
`}

Gunakan format Markdown dengan heading (##) untuk setiap bagian. Jaga agar bahasa tetap natural dan sesuai dengan platform (e-commerce atau TikTok).
`;
    // Gunakan gemini-3-flash-preview
    return await generateContent('gemini-3-flash-preview', prompt);
};


export const generateLiveScript = async (formData: any): Promise<string> => {
    const prompt = `
Anda adalah seorang Showrunner dan Scriptwriter profesional untuk acara siaran langsung (live streaming).
Tugas Anda adalah membuat timeline dan skrip/poin-poin untuk sesi live berdasarkan detail berikut.

**Detail Sesi Live:**
${JSON.stringify(formData, null, 2)}

**Tugas Anda:**
1.  **Analisis & Strategi:** Pahami tujuan live, gaya host, durasi, dan detail lainnya. Jika ada produk, susun urutannya secara strategis (misal: produk hero di tengah, produk flash sale di akhir).
2.  **Buat Timeline Menit-per-Menit:** Bagi durasi total menjadi segmen-segmen logis. Setiap segmen harus memiliki estimasi waktu, judul, dan isi.
3.  **Tulis Konten Segmen:** Berdasarkan "Mode Skrip" yang dipilih:
    *   **Jika 'full' (Skrip Penuh):** Tulis dialog lengkap untuk host, termasuk sapaan, interaksi, penjelasan produk, dan CTA. Buat seolah-olah host sedang berbicara langsung.
    *   **Jika 'flow' (Alur Live):** Tulis poin-poin utama yang harus disampaikan di setiap segmen. Beri kebebasan host untuk berimprovisasi.
4.  **Integrasikan Elemen Interaktif:** Sisipkan ajakan untuk "tap-tap layar", "share", "komen", "follow", dan "cek keranjang kuning" secara natural di sepanjang skrip.
5.  **Strukturkan Output:** Gunakan format JSON Markdown yang ketat. Buat array dari objek segmen. Setiap objek harus memiliki "time", "title", "script", dan "icon". Untuk live shopping, tambahkan "productIndex" jika segmen tersebut membahas produk tertentu.

**Format Output (WAJIB):**
Berikan jawaban dalam format JSON Markdown yang valid. Pastikan JSON bisa di-parse.
\`\`\`json
{
  "timeline": [
    {
      "time": "00:00 - 05:00",
      "title": "Pembukaan & Pemanasan",
      "script": "Script atau poin-poin untuk segmen ini...",
      "icon": "👋"
    },
    {
      "time": "05:00 - 15:00",
      "title": "Produk Pertama: Kemeja Flanel",
      "script": "Script atau poin-poin untuk produk ini...",
      "icon": "👕",
      "productIndex": 0
    }
  ]
}
\`\`\`
`;
    // Gunakan gemini-3-flash-preview
    const responseText = await generateContent('gemini-3-flash-preview', prompt);
    try {
        const jsonString = extractJson(responseText);
        // Test parsing before returning
        JSON.parse(jsonString); 
        return jsonString;
    } catch (e) {
        console.error("Failed to parse JSON from Gemini response", e, "Raw response:", responseText);
        const errorMessage = e instanceof Error ? e.message : "Unknown parsing error";
        return `{"error": "Gagal mem-parsing hasil dari AI: ${errorMessage}"}`;
    }
};


export const generatePepTalk = async (mood: string, reason: string): Promise<string> => {
    const prompt = `
Anda adalah seorang motivator dan mental coach. Seseorang akan melakukan live streaming dan merasa ${mood}.
Alasan mereka: "${reason}".
Berikan mereka sebuah "pep talk" singkat (2-3 paragraf) yang membangkitkan semangat, memberikan perspektif positif, dan meyakinkan mereka bahwa mereka bisa melakukannya. Gunakan bahasa yang hangat dan mendukung.
`;
    // Gunakan gemini-3-flash-preview
    return await generateContent('gemini-3-flash-preview', prompt);
};

export const generateSocialPost = async (formData: any, useSearch: boolean): Promise<string> => {
    const prompt = `
Anda adalah seorang Social Media Strategist dan Content Creator ahli untuk TikTok.
Tugas Anda adalah menghasilkan "Paket Konten" lengkap berdasarkan detail berikut.

**Detail Permintaan:**
${JSON.stringify(formData, null, 2)}
**Gunakan Tren Terkini (Google Search):** ${useSearch ? 'Ya' : 'Tidak'}

**Tugas Anda:**
1.  **Analisis:** Pahami target audiens, produk/topik, dan tujuan konten.
2.  **Ide Konten (3 Pilihan):** Hasilkan 3 ide konten video pendek yang unik dan berpotensi viral. Setiap ide harus memiliki judul yang menarik.
3.  **Skrip Video (Untuk Ide Terbaik):** Pilih ide terbaik dari 3 ide di atas, lalu tulis skrip lengkap untuk video berdurasi yang disarankan. Gunakan format:
    *   **Hook (3 detik pertama):** Kalimat pembuka yang kuat.
    *   **Isi Konten:** Penjelasan/demonstrasi yang jelas dan padat.
    *   **CTA (Call to Action):** Ajakan bertindak yang relevan.
    *   **[Saran Visual]:** Deskripsi adegan atau B-roll.
    *   **[Saran Audio]:** Rekomendasi musik/sound yang sedang tren.
4.  **Caption TikTok:** Tulis 2 alternatif caption yang menarik dan relevan dengan video.
5.  **Hashtag:** Berikan 5-7 hashtag yang merupakan campuran dari hashtag umum dan niche.

Gunakan format Markdown dengan heading (##) untuk setiap bagian.
`;
    // Gunakan gemini-3-flash-preview
    return await generateContent('gemini-3-flash-preview', prompt, useSearch);
};

export const analyzePerformance = async (formData: any): Promise<string> => {
    const prompt = `
Anda adalah seorang Analis Data dan Ahli Strategi Konten.
Tugas Anda adalah menganalisis transkrip dan data performa sebuah video TikTok, lalu memberikan wawasan dan rekomendasi yang bisa dijalankan.

**Data Video:**
${JSON.stringify(formData, null, 2)}

**Format Analisis (WAJIB ikuti format ini):**

### 📊 Ringkasan Performa
*   **Performa Keseluruhan:** Berikan penilaian singkat (misal: "Performa Cukup Baik", "Viral", "Perlu Peningkatan").
*   **Metrik Unggulan:** Sebutkan metrik mana yang paling menonjol (positif atau negatif) dan apa artinya.
*   **Engagement Rate:** (Jika memungkinkan) Hitung dan interpretasikan engagement rate ( (Likes + Comments + Shares) / Views * 100% ).

### ✅ Analisis Kekuatan (Apa yang Berhasil)
*   **Hook (Pembuka):** Analisis 3-5 detik pertama transkrip. Apakah efektif? Mengapa?
*   **Struktur & Alur Cerita:** Bagaimana alur kontennya? Apakah mudah diikuti? Di mana puncaknya?
*   **Pesan Kunci:** Apa pesan utama yang tersampaikan dengan baik?
*   **CTA (Call to Action):** Apakah CTA-nya jelas dan efektif?

### ⚠️ Area Peningkatan (Apa yang Bisa Diperbaiki)
*   **Potensi Drop-off:** Identifikasi bagian mana dari transkrip yang mungkin membuat penonton kehilangan minat.
*   **Kejelasan Pesan:** Apakah ada bagian yang membingungkan atau bisa disederhanakan?
*   **Peluang Engagement:** Di mana Anda bisa menambahkan lebih banyak pemicu interaksi (pertanyaan, polling, dll)?

### 🚀 Rekomendasi Aksi
Berikan 3-5 poin rekomendasi konkret yang bisa diterapkan pada konten berikutnya. Contoh:
1.  **Untuk Hook:** "Coba gunakan pertanyaan retoris di awal seperti..."
2.  **Untuk Struktur:** "Percepat bagian [X] dan tambahkan B-roll di bagian [Y] untuk menjaga dinamika."
3.  **Untuk CTA:** "Ganti CTA menjadi lebih spesifik, misalnya..."
4.  **Ide Konten Turunan:** "Berdasarkan performa ini, buat konten lanjutan tentang [Topik]..."

Gunakan format Markdown dan berikan analisis yang tajam dan membangun.
`;
    // Gunakan gemini-3-flash-preview
    return await generateContent('gemini-3-flash-preview', prompt);
};
