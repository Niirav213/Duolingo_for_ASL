import React, { useState } from 'react';
import './LessonView.css';
import Button from '../components/Button';
import Mascot from '../components/Mascot';
import CameraDetector from '../components/CameraDetector';

const LessonView = ({ lessonConfig, onFinish }) => {
  // If it's a test, we sequence through multiple letters.
  // If it's a normal lesson, it's just one letter.
  const targetSequence = lessonConfig.type === 'test' ? lessonConfig.letters : [lessonConfig.id];
  
  const [currentIndex, setCurrentIndex] = useState(0);
  const [isChecking, setIsChecking] = useState(false);
  const [isCorrect, setIsCorrect] = useState(false);

  const currentLetter = targetSequence[currentIndex];
  const progressPercent = (currentIndex / targetSequence.length) * 100;
  const isTestMode = lessonConfig.type === 'test';

  const handleCorrectSign = (score) => {
    if (!isChecking && !isCorrect) {
      setIsChecking(true);
      setIsCorrect(true);
    }
  };

  const handleContinue = () => {
    if (currentIndex < targetSequence.length - 1) {
      setCurrentIndex(curr => curr + 1);
      setIsChecking(false);
      setIsCorrect(false);
    } else {
      // Finished the lesson/test successfully
      onFinish(true, lessonConfig.id);
    }
  };

  const handleQuit = () => {
    onFinish(false, lessonConfig.id);
  };

  return (
    <div className="lesson-container">
      <header className="lesson-header">
        <button className="close-btn" onClick={handleQuit}>✖</button>
        <div className="progress-bar-container">
          <div className="progress-bar-fill" style={{ width: `${progressPercent}%` }}>
            <div className="progress-bar-highlight"></div>
          </div>
        </div>
        <div className="heart-counter">
          ❤️ 5
        </div>
      </header>

      <main className="lesson-content">
        <h2 className="question-title">
          {isTestMode ? (
            <span>Test Mode: Sign the letter <strong>{currentLetter}</strong></span>
          ) : (
            <span>Training: Learn the letter <strong>{currentLetter}</strong></span>
          )}
        </h2>
        
        <div className="question-media" style={{ marginBottom: 32 }}>
          {(!isChecking || !isCorrect) ? (
            <>
              {!isTestMode && (
                <div style={{ marginBottom: '16px', padding: '12px', backgroundColor: '#f0f4f8', borderRadius: '12px', display: 'flex', alignItems: 'center', gap: '16px', justifyContent: 'center' }}>
                  <div style={{
                    width: '80px', height: '80px', 
                    backgroundColor: 'var(--color-primary)', 
                    color: 'white', 
                    fontSize: '3rem', 
                    fontWeight: '900', 
                    display: 'flex', 
                    alignItems: 'center', 
                    justifyContent: 'center', 
                    borderRadius: '12px',
                    boxShadow: '0 4px 0 var(--color-primary-dark)'
                  }}>
                    {currentLetter}
                  </div>
                  <div>
                    <span style={{ fontSize: '1.5rem', display: 'block', marginBottom: '4px' }}>💡</span>
                    <p style={{ margin: 0, color: 'var(--color-primary)', fontSize: '1.1rem' }}><strong>Hint:</strong> Sign the letter <strong>{currentLetter}</strong> securely with your hand facing the camera.</p>
                  </div>
                </div>
              )}
              <CameraDetector 
                 targetSign={currentLetter} 
                 onCorrectSign={handleCorrectSign} 
              />
            </>
          ) : (
            <div className="video-placeholder" style={{ background: '#e0ffe0' }}>
              <Mascot variant="celebrate" size="large" />
              <p style={{ marginTop: 16, color: 'var(--color-success)', fontWeight: 'bold', fontSize: '1.2rem' }}>
                Perfect! That's an {currentLetter}!
              </p>
            </div>
          )}
        </div>

      </main>

      <footer className={`lesson-footer ${isChecking && isCorrect ? 'is-correct' : ''}`}>
        <div className="footer-content">
          {isChecking && isCorrect && (
            <div className="feedback-message">
              <h3>Correct!</h3>
            </div>
          )}
          
          <Button 
             variant={isChecking && isCorrect ? 'success' : 'secondary'}
             size="large"
             disabled={!isChecking || !isCorrect}
             onClick={handleContinue}
             style={{ width: '100%' }}
          >
            {isChecking && isCorrect ? (currentIndex === targetSequence.length - 1 ? 'Finish' : 'Continue') : 'Match the sign to continue'}
          </Button>
        </div>
      </footer>
    </div>
  );
};

export default LessonView;


