import React, { useState } from 'react';
import './LessonView.css';
import Button from '../components/Button';
import Mascot from '../components/Mascot';
import CameraDetector from '../components/CameraDetector';
import { useGameStore } from '../store';

const LessonView = ({ lessonConfig, onFinish }) => {
  const targetSequence = lessonConfig.type === 'test' ? lessonConfig.letters : [lessonConfig.id];
  
  const [currentIndex, setCurrentIndex] = useState(0);
  const [isChecking, setIsChecking] = useState(false);
  const [isCorrect, setIsCorrect] = useState(false);
  const [isFailed, setIsFailed] = useState(false);
  const [scores, setScores] = useState([]); // Score per letter
  const [showReward, setShowReward] = useState(false);

  const loseHeart = useGameStore(state => state.loseHeart);

  const currentLetter = targetSequence[currentIndex];
  const progressPercent = ((currentIndex + (isCorrect ? 1 : 0)) / targetSequence.length) * 100;
  const isTestMode = lessonConfig.type === 'test';
  const totalXP = scores.reduce((sum, s) => sum + Math.round(s / 10), 0);

  const handleCorrectSign = (score) => {
    if (!isChecking && !isCorrect && !isFailed) {
      setIsChecking(true);
      setIsCorrect(true);
      setScores(prev => [...prev, score]);
    }
  };

  const handleSkip = async () => {
    if (!isChecking && !isCorrect && !isFailed) {
      await loseHeart();
      setIsChecking(true);
      setIsFailed(true);
      setScores(prev => [...prev, 0]);
    }
  };

  const handleContinue = () => {
    if (currentIndex < targetSequence.length - 1) {
      setCurrentIndex(curr => curr + 1);
      setIsChecking(false);
      setIsCorrect(false);
      setIsFailed(false);
    } else {
      // Show reward screen before finishing
      setShowReward(true);
    }
  };

  const handleFinishReward = () => {
    onFinish(true, lessonConfig.id, totalXP);
  };

  const handleQuit = () => {
    onFinish(false, lessonConfig.id);
  };

  // Reward screen after completing all letters
  if (showReward) {
    return (
      <div className="lesson-container" style={{ display: 'flex', flexDirection: 'column', alignItems: 'center', justifyContent: 'center' }}>
        <div style={{ textAlign: 'center', padding: '40px 20px', maxWidth: '400px' }}>
          <Mascot variant="celebrate" size="large" />
          <h2 style={{ color: 'var(--color-success)', marginTop: '24px', fontSize: '2rem' }}>
            🎉 Lesson Complete!
          </h2>
          <div style={{
            display: 'flex', gap: '24px', justifyContent: 'center', margin: '24px 0',
            flexWrap: 'wrap'
          }}>
            <div style={{
              background: 'linear-gradient(135deg, #667eea 0%, #764ba2 100%)',
              color: 'white', padding: '16px 24px', borderRadius: '16px',
              textAlign: 'center', minWidth: '100px'
            }}>
              <div style={{ fontSize: '2rem', fontWeight: 'bold' }}>+{totalXP}</div>
              <div style={{ fontSize: '0.9rem', opacity: 0.9 }}>XP Earned</div>
            </div>
            <div style={{
              background: 'linear-gradient(135deg, #f093fb 0%, #f5576c 100%)',
              color: 'white', padding: '16px 24px', borderRadius: '16px',
              textAlign: 'center', minWidth: '100px'
            }}>
              <div style={{ fontSize: '2rem', fontWeight: 'bold' }}>🔥</div>
              <div style={{ fontSize: '0.9rem', opacity: 0.9 }}>Streak Updated</div>
            </div>
          </div>
          <div style={{ 
            background: '#f0f4f8', borderRadius: '12px', padding: '16px', marginBottom: '24px'
          }}>
            <p style={{ margin: 0, fontWeight: 'bold', marginBottom: '8px' }}>Letter Scores:</p>
            <div style={{ display: 'flex', gap: '8px', justifyContent: 'center', flexWrap: 'wrap' }}>
              {targetSequence.map((letter, i) => (
                <div key={letter} style={{
                  background: scores[i] >= 75 ? '#00d250' : '#ffa500',
                  color: 'white', width: '40px', height: '40px', borderRadius: '50%',
                  display: 'flex', alignItems: 'center', justifyContent: 'center',
                  fontWeight: 'bold', fontSize: '1.1rem'
                }}>
                  {letter}
                </div>
              ))}
            </div>
          </div>
          <Button variant="success" size="large" onClick={handleFinishReward} style={{ width: '100%' }}>
            Continue to Dashboard
          </Button>
        </div>
      </div>
    );
  }

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
          💎 {totalXP} XP
        </div>
      </header>

      <main className="lesson-content">
        <h2 className="question-title">
          {isTestMode ? (
            <span>Test Mode: Sign the letter <strong>{currentLetter}</strong></span>
          ) : (
            <span>Training: Sign the letter <strong>{currentLetter}</strong></span>
          )}
        </h2>
        
        <div className="question-media" style={{ marginBottom: 32 }}>
          {(!isChecking) ? (
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
                    <p style={{ margin: 0, color: 'var(--color-primary)', fontSize: '1.1rem' }}><strong>Hint:</strong> Sign the letter <strong>{currentLetter}</strong> with your hand facing the camera.</p>
                  </div>
                </div>
              )}
              <CameraDetector 
                 targetSign={currentLetter} 
                 onCorrectSign={handleCorrectSign} 
              />
              <div style={{ marginTop: '16px', textAlign: 'center' }}>
                <Button variant="danger" size="medium" onClick={handleSkip}>
                   I'm stuck (Skip & Lose 1 Heart)
                </Button>
              </div>
            </>
          ) : isCorrect ? (
            <div className="video-placeholder" style={{ background: '#e0ffe0', textAlign: 'center', padding: '32px', borderRadius: '16px' }}>
              <Mascot variant="celebrate" size="large" />
              <p style={{ marginTop: 16, color: 'var(--color-success)', fontWeight: 'bold', fontSize: '1.4rem' }}>
                ✅ Perfect! That's "{currentLetter}"!
              </p>
              <p style={{ color: '#667eea', fontWeight: 'bold', fontSize: '1.1rem' }}>
                +{Math.round((scores[scores.length - 1] || 0) / 10)} XP
              </p>
            </div>
          ) : isFailed ? (
            <div className="video-placeholder" style={{ background: '#ffe0e0', textAlign: 'center', padding: '32px', borderRadius: '16px' }}>
              <Mascot variant="sad" size="large" />
              <p style={{ marginTop: 16, color: '#ff4b4b', fontWeight: 'bold', fontSize: '1.4rem' }}>
                ❌ Skipped "{currentLetter}"!
              </p>
              <p style={{ color: '#ff4b4b', fontWeight: 'bold', fontSize: '1.1rem' }}>
                -1 Heart
              </p>
            </div>
          ) : null}
        </div>

      </main>

      <footer className={`lesson-footer ${isChecking && isCorrect ? 'is-correct' : isChecking && isFailed ? 'is-failed' : ''}`}>
        <div className="footer-content">
          {isChecking && isCorrect && (
            <div className="feedback-message">
              <h3>🎉 Correct!</h3>
            </div>
          )}
          {isChecking && isFailed && (
            <div className="feedback-message" style={{ color: '#ff4b4b' }}>
              <h3>Aww, next time!</h3>
            </div>
          )}
          
          <Button 
             variant={isChecking && isCorrect ? 'success' : isChecking && isFailed ? 'danger' : 'secondary'}
             size="large"
             disabled={!isChecking}
             onClick={handleContinue}
             style={{ width: '100%' }}
          >
             {isChecking ? (currentIndex === targetSequence.length - 1 ? '🏆 See Results' : 'Continue →') : 'Match the sign to continue'}
          </Button>
        </div>
      </footer>
    </div>
  );
};

export default LessonView;
