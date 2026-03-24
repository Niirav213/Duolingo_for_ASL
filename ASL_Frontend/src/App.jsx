import React, { useState } from 'react'
import './App.css'
import Layout from './components/Layout'
import PathView from './pages/PathView'
import LessonView from './pages/LessonView'

function App() {
  const [currentView, setCurrentView] = useState('path'); // 'path' or 'lesson'
  
  // Track which alphabet letters the user has successfully completed
  const [completedLevels, setCompletedLevels] = useState([]);
  
  // The specific lesson node being played (e.g. { id: 'A', mode: 'training' })
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
  )
}

export default App
