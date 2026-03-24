import React, { useState, useEffect } from 'react';
import { BrowserRouter, Routes, Route, Navigate } from 'react-router-dom';
import { useAuthStore } from './store';
import './App.css';
import Layout from './components/Layout';
import PathView from './pages/PathView';
import LessonView from './pages/LessonView';
import Login from './pages/Login';

function MainApp() {
  const [currentView, setCurrentView] = useState('path'); // 'path' or 'lesson'
  const [completedLevels, setCompletedLevels] = useState([]);
  const [activeLesson, setActiveLesson] = useState(null);

  const handleStartLesson = (lessonConfig) => {
    setActiveLesson(lessonConfig);
    setCurrentView('lesson');
  };

  const handleFinishLesson = (success, lessonId) => {
    if (success && !completedLevels.includes(lessonId)) {
      setCompletedLevels(prev => [...prev, lessonId]);
    }
    setActiveLesson(null);
    setCurrentView('path');
  };

  return (
    <>
      {currentView === 'path' ? (
        <Layout streak={12} gems={450} hearts={5}>
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
