
import React, { useState } from 'react';

interface PasswordModalProps {
    correctPassword: string;
    onLoginSuccess: () => void;
}

const PasswordModal: React.FC<PasswordModalProps> = ({ correctPassword, onLoginSuccess }) => {
    const [password, setPassword] = useState('');
    const [error, setError] = useState(false);

    const handleSubmit = (e: React.FormEvent<HTMLFormElement>) => {
        e.preventDefault();
        if (password.trim() === correctPassword) {
            setError(false);
            onLoginSuccess();
        } else {
            setError(true);
        }
    };

    return (
        <div id="password-modal" className="fixed inset-0 bg-slate-900/80 backdrop-blur-md z-50 flex items-center justify-center p-4">
            <div className="w-full max-w-sm bg-slate-800 border border-slate-700 rounded-xl p-8 shadow-2xl shadow-black/30">
                <div className="text-center mb-6">
                    <div className="flex justify-center items-center space-x-3 mb-4">
                        <svg className="w-8 h-8 text-orange-500" xmlns="http://www.w3.org/2000/svg" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                            <path strokeLinecap="round" strokeLinejoin="round" strokeWidth="2" d="M9.663 17h4.673M12 3v1m6.364 1.636l-.707.707M21 12h-1M4 12H3m3.343-5.657l-.707-.707m2.828 9.9a5 5 0 117.072 0l-.548.547A3.374 3.374 0 0014 18.469V19a2 2 0 11-4 0v-.531c0-.895-.356-1.754-.988-2.386l-.548-.547z" />
                        </svg>
                        <span className="text-xl font-semibold text-slate-100">PRIME</span>
                    </div>
                    <h2 className="text-lg font-semibold text-slate-200">Akses Terbatas</h2>
                    <p className="text-sm text-slate-400 mt-1">Silakan masukkan kata sandi untuk melanjutkan.</p>
                </div>
                <form id="password-form" className="space-y-4" onSubmit={handleSubmit}>
                    <div>
                        <label htmlFor="password-input" className="sr-only">Password</label>
                        <input 
                            type="password" 
                            id="password-input" 
                            value={password}
                            onChange={(e) => setPassword(e.target.value)}
                            className="block w-full bg-slate-700/50 border-slate-600 rounded-md shadow-sm py-2.5 px-4 text-slate-100 focus:outline-none focus:ring-2 focus:ring-orange-500 focus:border-orange-500 sm:text-sm transition" 
                            placeholder="Kata Sandi" 
                        />
                    </div>
                    {error && <p id="password-error" className="text-sm text-red-400 text-center">Kata sandi salah. Silakan coba lagi.</p>}
                    <button type="submit" className="w-full inline-flex items-center justify-center px-4 py-2.5 font-semibold text-white bg-gradient-to-r from-orange-500 to-amber-500 rounded-lg shadow-lg hover:shadow-xl focus:outline-none focus:ring-2 focus:ring-offset-2 focus:ring-offset-slate-800 focus:ring-orange-500 transition-all duration-300">
                        Masuk
                    </button>
                </form>
                <div className="text-center mt-6">
                    <a href="#" className="text-xs text-slate-500 hover:text-orange-400 transition-colors">Lupa Kata Sandi?</a>
                </div>
            </div>
        </div>
    );
};

export default PasswordModal;
