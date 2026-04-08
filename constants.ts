
export const PHOTO_VIDEO_MANUAL_OPTIONS = {
  concept: [
    { value: 'ai-determine', label: 'AI Tentukan' },
    { value: 'produk-dengan-model', label: 'Produk dengan Model' },
    { value: 'produk-saja', label: 'Produk Saja (Still Life)' },
    { value: 'flatlay', label: 'Flatlay / Atas ke Bawah' },
    { value: 'detail-produk', label: 'Fokus Detail Produk (Macro)' },
  ],
  gender: ['AI Tentukan', 'Wanita', 'Pria', 'Tidak Spesifik'],
  ethnicity: ['AI Tentukan', 'Asia (Indonesia)', 'Asia (Timur)', 'Kaukasia', 'Timur Tengah', 'Afrika', 'Latin'],
  age: ['AI Tentukan', '18-25 tahun', '25-35 tahun', '35-45 tahun', '45+ tahun'],
  bodyShape: ['AI Tentukan', 'Ramping', 'Atletis', 'Berisi (Curvy)', 'Plus-size'],
  hair: ['AI Tentukan', 'Rambut Panjang Lurus', 'Rambut Panjang Bergelombang', 'Rambut Pendek', 'Rambut Keriting', 'Rambut Diikat (Updo)'],
  makeup: ['AI Tentukan', 'Natural / Tanpa Riasan', 'Riasan Bold / Glamor', 'Riasan Editorial / Artistik'],
  shotType: ['AI Tentukan', 'Close-Up Shot', 'Medium Shot', 'Full-Body Shot', 'Wide Shot'],
  expression: ['AI Tentukan', 'Senyum Bahagia', 'Tampilan Serius / Fierce', 'Ekspresi Melamun / Tenang', 'Tertawa Lepas'],
  pose: ['AI Tentukan', 'Berdiri Tegak', 'Duduk Santai', 'Berjalan', 'Berbaring', 'Pose Dinamis (Melompat/Menari)'],
  interaction: [
    { value: 'Tidak ada', label: 'Tidak ada' },
    { value: 'Dengan Produk', label: 'Dengan Produk' },
    { value: 'Dengan Properti', label: 'Dengan Properti' },
    { value: 'Dengan Model Lain', label: 'Dengan Model Lain' },
  ],
  productType: ['AI Tentukan', 'Fashion (Baju, Celana)', 'Aksesoris (Tas, Sepatu)', 'Kecantikan (Skincare, Makeup)', 'Makanan & Minuman', 'Elektronik', 'Dekorasi Rumah', 'Produk Digital'],
  style: ['AI Tentukan', 'Gaya Komersial Bersih', 'Gaya Editorial Majalah', 'Gaya Hidup (Lifestyle)', 'Sinematik Dramatis', 'Minimalis Modern'],
  mood: ['AI Tentukan', 'Ceria & Bersemangat', 'Mewah & Elegan', 'Tenang & Damai', 'Misterius & Gelap', 'Nostalgia & Vintage'],
  location: ['AI Tentukan', 'Studio (Latar Polos)', 'Perkotaan Modern', 'Alam Terbuka (Pantai, Hutan)', 'Interior Rumah yang Nyaman', 'Kafe / Restoran Estetik', 'Latar Belakang Abstrak'],
  background: ['AI Tentukan', 'Warna Solid', 'Tekstur Beton', 'Kain Sutra', 'Kayu Alami', 'Gradient Halus', 'Dedaunan Tropis'],
  composition: ['AI Tentukan', 'Aturan Sepertiga (Rule of Thirds)', 'Simetris Tengah', 'Leading Lines', 'Framing', 'Negative Space'],
  productAngle: ['AI Tentukan', 'Eye-level', 'High-Angle (Dari Atas)', 'Low-Angle (Dari Bawah)', 'Dutch Angle (Miring)'],
  lighting: ['AI Tentukan', 'Cahaya Alami (Natural Light)', 'Cahaya Studio Terang', 'Cahaya Dramatis (Low-key)', 'Cahaya Lembut (Soft Light)', 'Neon / Cahaya Berwarna'],
  props: ['AI Tentukan', 'Tanpa Properti', 'Properti Minimalis', 'Properti Sesuai Tema', 'Properti Mewah'],
  productPosition: ['AI Tentukan', 'Di Tengah', 'Di Tangan Model', 'Di Atas Meja', 'Menggantung', 'Melayang'],
  vfx: ['AI Tentukan', 'Tidak Ada', 'Asap / Kabut', 'Percikan Air', 'Partikel Cahaya (Bokeh)', 'Efek Api'],
  sound: ['AI Tentukan', 'Tidak Ada', 'Musik Latar Ceria', 'Musik Latar Epik', 'Suara Alam', 'Efek Suara Produk'],
  aspectRatio: ['1:1', '4:5', '9:16', '16:9', '4:3'],
  aiImageStyle: ['AI Tentukan', 'Fotorealistis', 'Lukisan Digital', 'Gaya Anime', 'Seni Konsep', 'Model 3D'],
  aiArtistStyle: ['AI Tentukan', 'Gaya Annie Leibovitz', 'Gaya Wes Anderson', 'Gaya Tim Walker', 'Gaya Studio Ghibli'],
};

export const LAUNCH_OPTIONS = {
  productType: ['Fashion', 'Kecantikan', 'Makanan & Minuman', 'Elektronik', 'Kesehatan', 'Hobi', 'Produk Digital', 'Lainnya'],
  adObjective: ['Konversi / Penjualan', 'Meningkatkan Brand Awareness', 'Engagement (Suka, Komentar, Bagikan)', 'Mendapatkan Leads'],
  adFeeling: ['Senang / Gembira', 'Penasaran / Misterius', 'Terinspirasi / Termotivasi', 'Rasa Mendesak (Urgency)', 'Terhibur / Lucu'],
  adCTA: ['Beli Sekarang', 'Pelajari Lebih Lanjut', 'Geser ke Atas', 'Kunjungi Toko', 'Daftar Sekarang'],
};

export const LIVE_OPTIONS = {
    goal: ['Meningkatkan penjualan', 'Menghabiskan stok lama', 'Meningkatkan interaksi', 'Membangun komunitas'],
    style: ['Energik & Heboh', 'Tenang & Informatif', 'Lucu & Menghibur', 'Profesional & Terpercaya'],
    giftGoal: ['Meningkatkan engagement', 'Masuk ranking rising star', 'Mendapatkan gift spesifik', 'Menjalin kedekatan dengan penonton'],
    giftStyle: ['Santai & Ngobrol', 'Penuh Challenge & Game', 'PK (Player Kill) Battle', 'Fokus Menjawab Komentar'],
    qaStyle: ['Formal seperti seminar', 'Santai seperti talkshow', 'Interaktif & Penuh Tanya Jawab', 'Kombinasi Edukasi & Hiburan'],
    scriptMode: [
        { id: 'full', label: 'Skrip Penuh' },
        { id: 'flow', label: 'Alur Live (Poin)' },
    ],
};

export const POST_OPTIONS = {
  creatorType: [
    'Tips & Trik',
    'Tutorial Langkah-demi-Langkah',
    'Studi Kasus / Cerita Sukses',
    'POV (Point of View)',
    'Review / Unboxing',
  ],
  videoDuration: ['15-30 detik', '30-60 detik', 'Lebih dari 60 detik'],
  affiliateAngle: ['Solusi Masalah', 'Fitur Unik', 'Perbandingan Produk', 'Demonstrasi Penggunaan', 'Testimoni Pribadi'],
  affiliateTone: ['Jujur & Terbuka', 'Antusias & Bersemangat', 'Informatif & Detail', 'Menghibur & Lucu'],
  literacyAngle: ['Sederhanakan & Persingkat', 'Perdalam & Beri Contoh', 'Ubah menjadi Cerita', 'Bantah & Beri Perspektif Baru', 'Adaptasi untuk Niche Lain'],
};

export const BRAND_ROLES = [
  { id: 'produk', label: 'Brand Produk' },
  { id: 'kreator', label: 'Kreator / Personal Brand' },
  { id: 'afiliasi', label: 'Afiliator' },
];

export const GEMINI_MODEL_OPTIONS = [
  { value: 'api-default', label: 'Ikuti Pengaturan API' },
  { value: 'flash', label: 'Flash (Cepat)' },
  { value: 'pro', label: 'Pro (Kualitas Tinggi)' },
  { value: 'ultra', label: 'Ultra (Kualitas Maksimal)' },
];

export const GEMINI_API_MODEL_PRESETS = [
  { value: 'gemini-2.5-flash', label: 'Text - gemini-2.5-flash (stabil, cepat)' },
  { value: 'gemini-2.5-pro', label: 'Text - gemini-2.5-pro (stabil, kualitas tinggi)' },
  { value: 'gemini-2.5-flash-lite', label: 'Text - gemini-2.5-flash-lite (hemat biaya)' },
  { value: 'gemini-3-flash-preview', label: 'Text - gemini-3-flash-preview (preview)' },
  { value: 'gemini-3.1-pro-preview', label: 'Text - gemini-3.1-pro-preview (preview)' },
  { value: 'gemini-3.1-flash-lite-preview', label: 'Text - gemini-3.1-flash-lite-preview (preview)' },
  { value: 'gemini-3.1-flash-live-preview', label: 'Audio/Live - gemini-3.1-flash-live-preview' },
  { value: 'gemini-2.5-flash-native-audio-preview-12-2025', label: 'Audio/Live - gemini-2.5-flash-native-audio-preview-12-2025' },
  { value: 'gemini-2.5-flash-preview-tts', label: 'Audio/TTS - gemini-2.5-flash-preview-tts' },
  { value: 'gemini-2.5-pro-preview-tts', label: 'Audio/TTS - gemini-2.5-pro-preview-tts' },
  { value: 'gemini-3.1-flash-image-preview', label: 'Image - gemini-3.1-flash-image-preview (Nano Banana 2)' },
  { value: 'gemini-3-pro-image-preview', label: 'Image - gemini-3-pro-image-preview (Nano Banana Pro)' },
  { value: 'gemini-2.5-flash-image', label: 'Image - gemini-2.5-flash-image (Nano Banana)' },
  { value: 'imagen', label: 'Image - imagen (Imagen 4 alias)' },
  { value: 'veo-3.1-generate-preview', label: 'Video - veo-3.1-generate-preview' },
  { value: 'veo-3.1-lite-generate-preview', label: 'Video - veo-3.1-lite-generate-preview' },
  { value: 'lyria-3-pro-preview', label: 'Music - lyria-3-pro-preview' },
  { value: 'lyria-3-clip-preview', label: 'Music - lyria-3-clip-preview' },
  { value: 'lyria-realtime-exp', label: 'Music - lyria-realtime-exp' },
  { value: 'gemini-2.5-computer-use-preview-10-2025', label: 'Agent - gemini-2.5-computer-use-preview-10-2025' },
  { value: 'deep-research-pro-preview-12-2025', label: 'Agent - deep-research-pro-preview-12-2025' },
  { value: 'gemini-embedding-2-preview', label: 'Embedding - gemini-embedding-2-preview' },
  { value: 'gemini-embedding-001', label: 'Embedding - gemini-embedding-001' },
  { value: 'gemini-robotics-er-1.5-preview', label: 'Robotics - gemini-robotics-er-1.5-preview' },
  { value: 'gemini-2.0-flash', label: 'Deprecated - gemini-2.0-flash' },
  { value: 'gemini-2.0-flash-lite', label: 'Deprecated - gemini-2.0-flash-lite' },
  { value: 'gemini-3-pro-preview', label: 'Shut down - gemini-3-pro-preview' },
];
