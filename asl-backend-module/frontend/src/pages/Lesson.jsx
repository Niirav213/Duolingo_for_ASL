import React, { useEffect, useState } from 'react';
import { useParams, useNavigate } from 'react-router-dom';
import { useGameStore, useAuthStore } from '../store';
import { GestureCamera } from '../components/GestureCamera';

export const Lesson = () => {
  const { lessonId } = useParams();
  const navigate = useNavigate();
  const { accessToken } = useAuthStore();
  const { startSession, endSession } = useGameStore();

  const [currentSession, setCurrentSession] = useState(null);
  const [signs, setSigns] = useState([]);
  const [currentSignIndex, setCurrentSignIndex] = useState(0);
  const [score, setScore] = useState(0);
  const [accuracy, setAccuracy] = useState(0);
  const [startTime, setStartTime] = useState(Date.now());
  const [detectedCorrect, setDetectedCorrect] = useState(false);

  // Mock signs data
  const lessonSigns = {
    'A-F': ['A', 'B', 'C', 'D', 'E', 'F'],
    'G-L': ['G', 'H', 'I', 'J', 'K', 'L'],
    'M-R': ['M', 'N', 'O', 'P', 'Q', 'R'],
    'S-Z': ['S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z'],
  };

  useEffect(() => {
    if (!accessToken) {
      navigate('/login');
      return;
    }

    setSigns(lessonSigns[lessonId] || []);
    initializeSession();
  }, [accessToken, lessonId, navigate]);

  const initializeSession = async () => {
    const session = await startSession(lessonId);
    setCurrentSession(session);
    setStartTime(Date.now());
    setScore(0);
    setCurrentSignIndex(0);
  };

  const handleDetection = ({ sign, confidence, correct }) => {
    if (correct && confidence > 0.7) {
      setDetectedCorrect(true);
      setScore((prev) => prev + Math.round(confidence * 100));
      setAccuracy((prev) => (prev + confidence) / 2);

      setTimeout(() => {
        moveToNextSign();
      }, 1500);
    }
  };

  const moveToNextSign = () => {
    if (currentSignIndex < signs.length - 1) {
      setCurrentSignIndex((prev) => prev + 1);
      setDetectedCorrect(false);
    } else {
      finishLesson();
    }
  };

  const finishLesson = async () => {
    const duration = Math.floor((Date.now() - startTime) / 1000);
    const finalAccuracy = Math.min(accuracy, 1);

    if (currentSession) {
      await endSession(currentSession.id, score, finalAccuracy, duration);
    }

    navigate('/results', {
      state: {
        score,
        accuracy: finalAccuracy,
        duration,
        lessonsCompleted: signs.length,
      },
    });
  };

  if (!signs.length) {
    return (
      <div className="flex items-center justify-center min-h-screen bg-gray-900">
        <div className="text-white text-2xl">Loading lesson...</div>
      </div>
    );
  }

  const currentSign = signs[currentSignIndex];
  const progress = ((currentSignIndex + 1) / signs.length) * 100;

  return (
    <div className="min-h-screen bg-gradient-to-br from-gray-900 via-gray-800 to-gray-900 text-white p-6">
      <div className="max-w-4xl mx-auto">
        {/* Header */}
        <div className="flex justify-between items-center mb-8">
          <div>
            <h1 className="text-4xl font-bold">{lessonId}</h1>
            <p className="text-gray-400">Sign {currentSignIndex + 1} of {signs.length}</p>
          </div>
          <div className="text-right">
            <p className="text-gray-400 text-sm">Score</p>
            <p className="text-3xl font-bold text-yellow-400">{score}</p>
          </div>
        </div>

        {/* Progress Bar */}
        <div className="mb-8">
          <div className="flex justify-between mb-2">
            <span className="text-sm text-gray-400">Lesson Progress</span>
            <span className="text-sm text-gray-400">{progress.toFixed(0)}%</span>
          </div>
          <div className="w-full bg-gray-700 rounded-full h-3 overflow-hidden">
            <div
              className="bg-gradient-to-r from-green-400 to-blue-500 h-full rounded-full transition-all duration-500"
              style={{ width: `${progress}%` }}
            ></div>
          </div>
        </div>

        {/* Main Content */}
        <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
          {/* Camera */}
          <div className="lg:col-span-2">
            <GestureCamera
              expectedSign={currentSign}
              onDetection={handleDetection}
              lessonId={lessonId}
            />

            {detectedCorrect && (
              <div className="mt-4 p-4 bg-green-600 bg-opacity-20 border border-green-500 rounded-lg text-center">
                <p className="text-green-400 text-lg font-bold">✓ Correct! Moving to next sign...</p>
              </div>
            )}
          </div>

          {/* Sidebar - Signs List */}
          <div className="bg-gray-800 bg-opacity-50 border border-gray-700 rounded-lg p-6">
            <h3 className="text-xl font-bold mb-4">Signs in Lesson</h3>
            <div className="space-y-2 max-h-96 overflow-y-auto">
              {signs.map((sign, idx) => (
                <div
                  key={idx}
                  className={`p-3 rounded-lg text-center font-bold transition-all ${
                    idx === currentSignIndex
                      ? 'bg-purple-600 text-white scale-105'
                      : idx < currentSignIndex
                        ? 'bg-green-600 bg-opacity-20 text-green-400'
                        : 'bg-gray-700 text-gray-400'
                  }`}
                >
                  {sign}
                </div>
              ))}
            </div>

            <div className="mt-6 p-4 bg-blue-600 bg-opacity-20 border border-blue-500 rounded-lg">
              <p className="text-sm text-gray-300 mb-2">Tips:</p>
              <ul className="text-xs text-gray-400 space-y-1">
                <li>• Keep your hands within frame</li>
                <li>• Ensure good lighting</li>
                <li>• Make clear sign movements</li>
                <li>• Hold each sign for 1-2 seconds</li>
              </ul>
            </div>

            <button
              onClick={() => {
                finishLesson();
              }}
              className="w-full mt-6 bg-red-600 hover:bg-red-700 text-white font-bold py-2 px-4 rounded-lg transition-colors"
            >
              Exit Lesson
            </button>
          </div>
        </div>
      </div>
    </div>
  );
};

export default Lesson;
