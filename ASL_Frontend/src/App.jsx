import React, { useState, useEffect } from 'react';
import { BrowserRouter, Routes, Route, Navigate } from 'react-router-dom';
import { useAuthStore, useGameStore } from './store';
import './App.css';
import Layout from './components/Layout';
import PathView from './pages/PathView';
import LessonView from './pages/LessonView';
import Login from './pages/Login';

function MainApp() {
  const { userStats, fetchUserStats, fetchStreak, startSession, endSession, currentSession } = useGameStore();
  const [currentView, setCurrentView] = useState('path'); // 'path' or 'lesson'
  const [completedLevels, setCompletedLevels] = useState([]);
  const [activeLesson, setActiveLesson] = useState(null);
  const startTimeRef = React.useRef(null);

  // Fetch stats on mount
  useEffect(() => {
    fetchUserStats();
    fetchStreak();
  }, [fetchUserStats, fetchStreak]);

  const handleStartLesson = async (lessonConfig) => {
    if (userStats.hearts !== undefined && userStats.hearts <= 0) {
        alert("You are out of hearts! Please refill them first.");
        return;
    }
    setActiveLesson(lessonConfig);
    setCurrentView('lesson');
    startTimeRef.current = Date.now();
    // Try to start a backend session, but don't block the lesson if it fails
    try {
      await startSession(lessonIdToNumeric(lessonConfig.id));
    } catch (e) {
      console.warn('Could not start backend session:', e);
    }
  };

  const handleFinishLesson = async (success, lessonId, earnedXP = 100, accuracy = 1.0) => {
    if (success) {
      if (currentSession) {
        try {
          const duration = Math.floor((Date.now() - startTimeRef.current) / 1000);
          await endSession(currentSession.id, earnedXP, accuracy, duration);
        } catch (e) {
          console.warn('Could not end backend session:', e);
        }
      }
      if (!completedLevels.includes(lessonId)) {
        setCompletedLevels(prev => [...prev, lessonId]);
      }
    }
    setActiveLesson(null);
    setCurrentView('path');
  };

  // Helper to map 'A', 'B' etc to numeric IDs if needed by backend
  // In our ALPHABET_PATH, id is 'A', 'B'... let's assume backend handles them or map them
  const lessonIdToNumeric = (id) => {
    if (typeof id === 'number') return id;
    if (id === 'Test-1') return 101;
    if (id === 'Test-2') return 102;
    return id.charCodeAt(0); // Simple mapping: 'A' -> 65
  };

  return (
    <>
      {currentView === 'path' ? (
        <Layout 
          streak={userStats.current_streak} 
          gems={userStats.total_xp} 
          hearts={userStats.hearts !== undefined ? userStats.hearts : 5}
        >
          <PathView 
            completedLevels={completedLevels} 
            onStartLesson={handleStartLesson} 
          />
        </Layout>
      ) : (
        <LessonView 
           lessonConfig={activeLesson}
           onFinish={handleFinishLesson} 
        />
      )}
    </>
  );
}

function ProtectedRoute({ children }) {
  const { accessToken } = useAuthStore();
  if (!accessToken) {
    return <Navigate to="/login" replace />;
  }
  return children;
}

function App() {
  // Try to load user profile on mount if we have a token (optional check)
  return (
    <BrowserRouter>
      <Routes>
        <Route path="/login" element={<Login />} />
        <Route 
          path="/*" 
          element={
            <ProtectedRoute>
              <MainApp />
            </ProtectedRoute>
          } 
        />
      </Routes>
    </BrowserRouter>
  );
}

export default App;
