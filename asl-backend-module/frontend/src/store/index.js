import { create } from 'zustand';

const getStoredTokens = () => {
    if (typeof window === 'undefined') {
        return { accessToken: null, refreshToken: null };
    }
    return {
        accessToken: localStorage.getItem('accessToken'),
        refreshToken: localStorage.getItem('refreshToken'),
    };
};

export const useAuthStore = create((set) => ({
    user: null,
    ...getStoredTokens(),
    isLoading: false,
    error: null,

    login: async(username, password) => {
        set({ isLoading: true });
        try {
            const response = await fetch('http://localhost:8000/api/v1/auth/login', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ username, password }),
            });

            if (!response.ok) throw new Error('Login failed');

            const data = await response.json();
            localStorage.setItem('accessToken', data.access_token);
            localStorage.setItem('refreshToken', data.refresh_token);

            set({
                accessToken: data.access_token,
                refreshToken: data.refresh_token,
                error: null,
            });
        } catch (error) {
            set({ error: error.message });
        } finally {
            set({ isLoading: false });
        }
    },

    register: async(username, email, password) => {
        set({ isLoading: true });
        try {
            const response = await fetch('http://localhost:8000/api/v1/auth/register', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ username, email, password }),
            });

            if (!response.ok) throw new Error('Registration failed');

            const data = await response.json();
            localStorage.setItem('accessToken', data.access_token);
            localStorage.setItem('refreshToken', data.refresh_token);

            set({
                accessToken: data.access_token,
                refreshToken: data.refresh_token,
                error: null,
            });
        } catch (error) {
            set({ error: error.message });
        } finally {
            set({ isLoading: false });
        }
    },

    logout: () => {
        localStorage.removeItem('accessToken');
        localStorage.removeItem('refreshToken');
        set({ user: null, accessToken: null, refreshToken: null });
    },

    setUser: (user) => set({ user }),
}));

export const useGameStore = create((set) => ({
    currentSession: null,
    userStats: null,
    streak: null,
    sessions: [],
    isLoading: false,

    startSession: async(lessonId) => {
        set({ isLoading: true });
        try {
            const response = await fetch('http://localhost:8000/api/v1/game/session/start', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                    'Authorization': `Bearer ${useAuthStore.getState().accessToken}`,
                },
                body: JSON.stringify({ lesson_id: lessonId }),
            });

            if (!response.ok) throw new Error('Failed to start session');

            const session = await response.json();
            set({ currentSession: session });
            return session;
        } catch (error) {
            console.error(error);
        } finally {
            set({ isLoading: false });
        }
    },

    endSession: async(sessionId, score, accuracy, duration) => {
        try {
            const response = await fetch(
                `http://localhost:8000/api/v1/game/session/${sessionId}/end?score=${score}&accuracy=${accuracy}&duration_seconds=${duration}`, {
                    method: 'POST',
                    headers: {
                        'Authorization': `Bearer ${useAuthStore.getState().accessToken}`,
                    },
                }
            );

            if (!response.ok) throw new Error('Failed to end session');

            const data = await response.json();
            set({ currentSession: null });
            return data;
        } catch (error) {
            console.error(error);
        }
    },

    fetchUserStats: async() => {
        try {
            const response = await fetch('http://localhost:8000/api/v1/game/stats', {
                headers: {
                    'Authorization': `Bearer ${useAuthStore.getState().accessToken}`,
                },
            });

            if (!response.ok) throw new Error('Failed to fetch stats');

            const stats = await response.json();
            set({ userStats: stats });
            return stats;
        } catch (error) {
            console.error(error);
        }
    },

    fetchStreak: async() => {
        try {
            const response = await fetch('http://localhost:8000/api/v1/game/streak', {
                headers: {
                    'Authorization': `Bearer ${useAuthStore.getState().accessToken}`,
                },
            });

            if (!response.ok) throw new Error('Failed to fetch streak');

            const streak = await response.json();
            set({ streak });
            return streak;
        } catch (error) {
            console.error(error);
        }
    },

    fetchSessions: async(limit = 20) => {
        try {
            const response = await fetch(`http://localhost:8000/api/v1/game/sessions?limit=${limit}`, {
                headers: {
                    'Authorization': `Bearer ${useAuthStore.getState().accessToken}`,
                },
            });

            if (!response.ok) throw new Error('Failed to fetch sessions');

            const sessions = await response.json();
            set({ sessions });
            return sessions;
        } catch (error) {
            console.error(error);
        }
    },
}));