
import { SavedPrompt, Studio } from '../types';

const HISTORY_KEY = 'provaPromptHistory';

export const getHistory = (): SavedPrompt[] => {
    try {
        const historyJson = localStorage.getItem(HISTORY_KEY);
        if (historyJson) {
            return JSON.parse(historyJson);
        }
    } catch (e) {
        console.error("Failed to parse prompt history from localStorage", e);
    }
    return [];
};

export const savePrompt = (studio: Studio, type: string, content: string): SavedPrompt[] => {
    if (!content.trim()) return getHistory();

    const newPrompt: SavedPrompt = {
        id: Date.now(),
        studio,
        type,
        content,
        timestamp: Date.now(),
    };

    const currentHistory = getHistory();
    const updatedHistory = [newPrompt, ...currentHistory];

    try {
        localStorage.setItem(HISTORY_KEY, JSON.stringify(updatedHistory));
    } catch (e) {
        console.error("Failed to save prompt history to localStorage", e);
    }
    
    return updatedHistory;
};

export const deletePrompt = (id: number): SavedPrompt[] => {
    const currentHistory = getHistory();
    const updatedHistory = currentHistory.filter(prompt => prompt.id !== id);

    try {
        localStorage.setItem(HISTORY_KEY, JSON.stringify(updatedHistory));
    } catch (e) {
        console.error("Failed to delete prompt from localStorage", e);
    }

    return updatedHistory;
};
