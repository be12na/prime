
import React from 'react';

interface FullScreenModalProps {
    onEnter: () => void;
    onSkip: () => void;
}

const FullScreenModal: React.FC<FullScreenModalProps> = ({ onEnter, onSkip }) => {
    return (
        <div className="fixed inset-0 bg-slate-900/80 backdrop-blur-md z-50 flex items-center justify-center p-4">
            <div className="w-full max-w-md bg-slate-800 border border-slate-700 rounded-xl p-8 text-center shadow-2xl shadow-black/30">
                <div className="flex justify-center items-center space-x-3 mb-4">
                    <svg className="w-10 h-10 text-orange-500" xmlns="http://www.w3.org/2000/svg" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                        <path strokeLinecap="round" strokeLinejoin="round" strokeWidth="2" d="M4 8V4m0 0h4M4 4l5 5m11-1v4m0 0h-4m4 0l-5-5M4 16v4m0 0h4m-4 0l5-5m11 5v-4m0 0h-4m4 0l-5 5" />
                    </svg>
                </div>
                <h2 className="text-xl font-bold text-slate-100 mb-2">Pengalaman Terbaik Menanti Anda</h2>
                <p className="text-slate-400 mb-6">Untuk pengalaman yang lebih fokus dan imersif, kami merekomendasikan penggunaan mode layar penuh.</p>
                <div className="flex flex-col sm:flex-row gap-3">
                    <button 
                        onClick={onEnter} 
                        className="w-full inline-flex items-center justify-center px-4 py-3 font-semibold text-white bg-gradient-to-r from-orange-500 to-amber-500 rounded-lg shadow-lg hover:shadow-xl focus:outline-none focus:ring-2 focus:ring-offset-2 focus:ring-offset-slate-800 focus:ring-orange-500 transition-all duration-300"
                    >
                        Aktifkan Layar Penuh
                    </button>
                    <button 
                        onClick={onSkip}
                        className="w-full sm:w-auto inline-flex items-center justify-center px-6 py-3 font-semibold text-slate-300 bg-slate-700/50 hover:bg-slate-700 rounded-lg transition-colors"
                    >
                        Lain Kali
                    </button>
                </div>
            </div>
        </div>
    );
};

export default FullScreenModal;