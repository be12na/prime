
import React, { useState } from 'react';
import { Studio } from '../types.ts';

interface HeaderProps {
    activeStudio: Studio;
    setActiveStudio: (studio: Studio) => void;
}

// Fix: Changed JSX.Element to React.ReactNode to fix "Cannot find namespace 'JSX'" error.
const NAV_ITEMS: { id: Studio; label: string; icon: React.ReactNode }[] = [
    { id: 'photo-video', label: 'Studio Foto & Video', icon: <svg xmlns="http://www.w3.org/2000/svg" className="h-5 w-5" fill="none" viewBox="0 0 24 24" stroke="currentColor" strokeWidth="2"><path strokeLinecap="round" strokeLinejoin="round" d="M3 9a2 2 0 012-2h.93a2 2 0 001.664-.89l.812-1.22A2 2 0 0110.07 4h3.86a2 2 0 011.664.89l.812 1.22A2 2 0 0018.07 7H19a2 2 0 012 2v9a2 2 0 01-2 2H5a2 2 0 01-2-2V9z" /><path strokeLinecap="round" strokeLinejoin="round" d="M15 13a3 3 0 11-6 0 3 3 0 016 0z" /></svg> },
    { id: 'product-launch', label: 'Studio Launching', icon: <svg xmlns="http://www.w3.org/2000/svg" className="h-5 w-5" fill="none" viewBox="0 0 24 24" stroke="currentColor" strokeWidth="2"><path strokeLinecap="round" strokeLinejoin="round" d="M13 10V3L4 14h7v7l9-11h-7z" /></svg> },
    { id: 'live', label: 'Studio Live', icon: <svg xmlns="http://www.w3.org/2000/svg" className="h-5 w-5" fill="none" viewBox="0 0 24 24" stroke="currentColor" strokeWidth="2"><path strokeLinecap="round" strokeLinejoin="round" d="M15.536 8.464a5 5 0 010 7.072m2.828-9.9a9 9 0 010 12.728M5.636 5.636a9 9 0 0112.728 0m-12.728 0l12.728 12.728" /></svg> },
    { id: 'post', label: 'Studio Post', icon: <svg xmlns="http://www.w3.org/2000/svg" className="h-5 w-5" fill="none" viewBox="0 0 24 24" stroke="currentColor" strokeWidth="2"><path strokeLinecap="round" strokeLinejoin="round" d="M11 5H6a2 2 0 00-2 2v11a2 2 0 002 2h11a2 2 0 002-2v-5m-1.414-9.414a2 2 0 112.828 2.828L11.828 15H9v-2.828l8.586-8.586z" /></svg> },
    { id: 'analytic', label: 'Studio Analitik', icon: <svg xmlns="http://www.w3.org/2000/svg" className="h-5 w-5" fill="none" viewBox="0 0 24 24" stroke="currentColor" strokeWidth="2"><path strokeLinecap="round" strokeLinejoin="round" d="M7 12l3-3 3 3 4-4M8 21l4-4 4 4M3 4h18M4 4h16v12a2 2 0 01-2 2H6a2 2 0 01-2-2V4z" /></svg> },
    { id: 'history', label: 'Riwayat', icon: <svg xmlns="http://www.w3.org/2000/svg" className="h-5 w-5" fill="none" viewBox="0 0 24 24" stroke="currentColor" strokeWidth="2"><path strokeLinecap="round" strokeLinejoin="round" d="M12 8v4l3 3m6-3a9 9 0 11-18 0 9 9 0 0118 0z" /></svg> },
    { id: 'guide', label: 'Panduan & Pengaturan', icon: <svg xmlns="http://www.w3.org/2000/svg" className="h-5 w-5" fill="none" viewBox="0 0 24 24" stroke="currentColor" strokeWidth="2"><path strokeLinecap="round" strokeLinejoin="round" d="M12 6.253v13m0-13C10.832 5.477 9.246 5 7.5 5S4.168 5.477 3 6.253v13C4.168 18.477 5.754 18 7.5 18s3.332.477 4.5 1.253m0-13C13.168 5.477 14.754 5 16.5 5c1.747 0 3.332.477 4.5 1.253v13C19.832 18.477 18.246 18 16.5 18c-1.746 0-3.332.477-4.5 1.253" /></svg> },
];

const Header: React.FC<HeaderProps> = ({ activeStudio, setActiveStudio }) => {
    const [isMobileMenuOpen, setIsMobileMenuOpen] = useState(false);

    const handleNavClick = (studio: Studio) => {
        setActiveStudio(studio);
        setIsMobileMenuOpen(false);
    };

    return (
        <>
            <header className="bg-slate-900/50 backdrop-blur-sm border-b border-slate-700 sticky top-0 z-20">
              <div className="max-w-screen-xl mx-auto px-4 sm:px-6 lg:px-8">
                <div className="flex items-center justify-between h-16">
                  <div className="flex items-center space-x-3">
                    <svg className="w-8 h-8 text-orange-500" xmlns="http://www.w3.org/2000/svg" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                      <path strokeLinecap="round" strokeLinejoin="round" strokeWidth="2" d="M9.663 17h4.673M12 3v1m6.364 1.636l-.707.707M21 12h-1M4 12H3m3.343-5.657l-.707-.707m2.828 9.9a5 5 0 117.072 0l-.548.547A3.374 3.374 0 0014 18.469V19a2 2 0 11-4 0v-.531c0-.895-.356-1.754-.988-2.386l-.548-.547z" />
                    </svg>
                    <span className="text-xl font-semibold text-slate-100">PRIME</span>
                  </div>
                  <nav className="hidden md:flex items-center space-x-1">
                    {NAV_ITEMS.map((item) => (
                         <button key={item.id} onClick={() => handleNavClick(item.id)} className={`flex items-center gap-2 px-3 py-2 text-sm font-medium rounded-md transition-colors ${activeStudio === item.id ? 'bg-orange-500/10 text-orange-400' : 'text-slate-300 hover:bg-slate-700/50'}`}>
                            {item.icon}
                            <span>{item.label}</span>
                         </button>
                    ))}
                  </nav>
                  <div className="md:hidden">
                      <button onClick={() => setIsMobileMenuOpen(true)} type="button" className="inline-flex items-center justify-center p-2 rounded-md text-slate-400 hover:text-white hover:bg-slate-700 focus:outline-none focus:ring-2 focus:ring-inset focus:ring-orange-500" aria-controls="mobile-menu" aria-expanded="false">
                          <span className="sr-only">Buka menu</span>
                          <svg className="block h-6 w-6" xmlns="http://www.w3.org/2000/svg" fill="none" viewBox="0 0 24 24" stroke="currentColor" aria-hidden="true">
                              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth="2" d="M4 6h16M4 12h16M4 18h16" />
                          </svg>
                      </button>
                  </div>
                </div>
              </div>
            </header>

            {/* Mobile Menu */}
            <div className={`md:hidden ${isMobileMenuOpen ? '' : 'pointer-events-none'}`}>
                <div className={`fixed inset-0 bg-black/60 z-30 transition-opacity duration-300 ease-in-out ${isMobileMenuOpen ? 'opacity-100' : 'opacity-0'}`} onClick={() => setIsMobileMenuOpen(false)}></div>
                <div className={`fixed top-0 right-0 h-full w-4/5 max-w-xs bg-slate-800/80 backdrop-blur-lg shadow-2xl z-40 transform transition-transform duration-300 ease-in-out ${isMobileMenuOpen ? 'translate-x-0' : 'translate-x-full'}`}>
                    <div className="p-4 flex justify-between items-center border-b border-slate-700">
                         <div className="flex items-center space-x-3">
                            <svg className="w-7 h-7 text-orange-500" xmlns="http://www.w3.org/2000/svg" fill="none" viewBox="0 0 24 24" stroke="currentColor"><path strokeLinecap="round" strokeLinejoin="round" strokeWidth="2" d="M9.663 17h4.673M12 3v1m6.364 1.636l-.707.707M21 12h-1M4 12H3m3.343-5.657l-.707-.707m2.828 9.9a5 5 0 117.072 0l-.548.547A3.374 3.374 0 0014 18.469V19a2 2 0 11-4 0v-.531c0-.895-.356-1.754-.988-2.386l-.548-.547z" /></svg>
                            <span className="text-lg font-semibold text-slate-100">PRIME</span>
                         </div>
                         <button onClick={() => setIsMobileMenuOpen(false)} type="button" className="p-2 rounded-md text-slate-400 hover:bg-slate-700 hover:text-white transition-colors">
                            <span className="sr-only">Tutup menu</span>
                            <svg className="h-6 w-6" xmlns="http://www.w3.org/2000/svg" fill="none" viewBox="0 0 24 24" stroke="currentColor" aria-hidden="true"><path strokeLinecap="round" strokeLinejoin="round" strokeWidth="2" d="M6 18L18 6M6 6l12 12" /></svg>
                         </button>
                    </div>
                    <nav className="p-4 flex flex-col space-y-2">
                        {NAV_ITEMS.map((item) => (
                            <button key={item.id} onClick={() => handleNavClick(item.id)} className={`w-full flex items-center gap-4 px-4 py-3 rounded-md text-base font-medium transition-colors ${activeStudio === item.id ? 'bg-orange-500/10 text-orange-400' : 'text-slate-300 hover:bg-slate-700 hover:text-white'}`}>
                                {item.icon}
                                <span>{item.label}</span>
                            </button>
                        ))}
                    </nav>
                </div>
            </div>
        </>
    );
};

export default Header;
