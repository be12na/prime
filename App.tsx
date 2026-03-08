
import React, { useState, useCallback, useEffect } from 'react';
import { Studio } from './types.ts';
import PasswordModal from './components/PasswordModal.tsx';
import Header from './components/Header.tsx';
import FullScreenModal from './components/FullScreenModal.tsx';
import PhotoVideoStudio from './components/studios/PhotoVideoStudio.tsx';
import LaunchStudio from './components/studios/LaunchStudio.tsx';
import LiveStudio from './components/studios/LiveStudio.tsx';
import PostStudio from './components/studios/PostStudio.tsx';
import AnalyticStudio from './components/studios/AnalyticStudio.tsx';
import GuideStudio from './components/studios/GuideStudio.tsx';
import HistoryStudio from './components/studios/HistoryStudio.tsx';


const CORRECT_PASSWORD = 'Prime@2026!';

const App: React.FC = () => {
    const [isAuthenticated, setIsAuthenticated] = useState<boolean>(() => sessionStorage.getItem('isVerified') === 'true');
    const [activeStudio, setActiveStudio] = useState<Studio>(() => {
        // First visit: direct to Guide studio if no API key set
        const hasApiKey = sessionStorage.getItem('geminiApiKey');
        return hasApiKey ? 'photo-video' : 'guide';
    });
    const [showFullScreenModal, setShowFullScreenModal] = useState(false);

    useEffect(() => {
        // Show fullscreen prompt only after authentication and only once per session
        if (isAuthenticated && !sessionStorage.getItem('fullscreenPrompted')) {
            // Check if fullscreen is supported by the browser
            if (document.documentElement.requestFullscreen) {
                setShowFullScreenModal(true);
            } else {
                // If not supported, just mark as prompted and don't show the modal
                sessionStorage.setItem('fullscreenPrompted', 'true');
            }
        }
    }, [isAuthenticated]);

    const handleLoginSuccess = useCallback(() => {
        sessionStorage.setItem('isVerified', 'true');
        setIsAuthenticated(true);
    }, []);

    const handleEnterFullScreen = () => {
      if (document.documentElement.requestFullscreen) {
        document.documentElement.requestFullscreen().catch(err => {
          console.error(`Gagal masuk mode layar penuh: ${err.message} (${err.name})`);
        });
      }
      sessionStorage.setItem('fullscreenPrompted', 'true');
      setShowFullScreenModal(false);
    };

    const handleSkipFullScreen = () => {
      sessionStorage.setItem('fullscreenPrompted', 'true');
      setShowFullScreenModal(false);
    };


    const renderStudio = () => {
        switch (activeStudio) {
            case 'photo-video':
                return <PhotoVideoStudio />;
            case 'product-launch':
                return <LaunchStudio />;
            case 'live':
                return <LiveStudio />;
            case 'post':
                return <PostStudio />;
            case 'analytic':
                return <AnalyticStudio />;
            case 'history':
                return <HistoryStudio />;
            case 'guide':
                return <GuideStudio />;
            default:
                return <PhotoVideoStudio />;
        }
    };
    
    if (!isAuthenticated) {
        return <PasswordModal correctPassword={CORRECT_PASSWORD} onLoginSuccess={handleLoginSuccess} />;
    }

    return (
        <>
            {showFullScreenModal && <FullScreenModal onEnter={handleEnterFullScreen} onSkip={handleSkipFullScreen} />}
            <div id="app-container" className="flex-grow flex flex-col">
                <Header activeStudio={activeStudio} setActiveStudio={setActiveStudio} />
                <main className="flex-grow w-full max-w-screen-xl mx-auto p-4 sm:p-6 md:p-8">
                    {renderStudio()}
                </main>
            </div>
        </>
    );
};

export default App;
