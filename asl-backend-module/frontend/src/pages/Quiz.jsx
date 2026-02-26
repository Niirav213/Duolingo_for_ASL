import React, { useEffect, useState } from 'react';
import { useNavigate } from 'react-router-dom';
import { useAuthStore } from '../store';

export const Quiz = () => {
  const navigate = useNavigate();
  const { accessToken, logout } = useAuthStore();
  const [quizzes, setQuizzes] = useState([]);
  const [selectedQuiz, setSelectedQuiz] = useState(null);
  const [currentQuestion, setCurrentQuestion] = useState(0);
  const [score, setScore] = useState(0);
  const [showResults, setShowResults] = useState(false);

  // Mock quiz data
  const mockQuizzes = [
    {
      id: 1,
      title: 'Alphabet Quiz',
      description: 'Test your knowledge of the ASL alphabet',
      difficulty: 'Beginner',
      questions: [
        {
          id: 1,
          question: 'Show the sign for the letter A',
          correctAnswer: 'A',
          options: ['A', 'B', 'C', 'D'],
        },
        {
          id: 2,
          question: 'Show the sign for the letter B',
          correctAnswer: 'B',
          options: ['A', 'B', 'C', 'D'],
        },
        {
          id: 3,
          question: 'Show the sign for the letter C',
          correctAnswer: 'C',
          options: ['A', 'B', 'C', 'D'],
        },
      ],
    },
    {
      id: 2,
      title: 'Numbers Quiz',
      description: 'Learn numbers in ASL',
      difficulty: 'Beginner',
      questions: [],
    },
    {
      id: 3,
      title: 'Phrases Quiz',
      description: 'Common phrases in ASL',
      difficulty: 'Intermediate',
      questions: [],
    },
  ];

  useEffect(() => {
    if (!accessToken) {
      navigate('/login');
      return;
    }
    setQuizzes(mockQuizzes);
  }, [accessToken, navigate]);

  if (!selectedQuiz) {
    return (
      <div className="min-h-screen bg-gradient-to-br from-gray-900 via-gray-800 to-gray-900 text-white p-6">
        <div className="max-w-4xl mx-auto">
          <div className="flex justify-between items-center mb-8">
            <h1 className="text-4xl font-bold">Quizzes</h1>
            <button
              onClick={logout}
              className="bg-red-600 hover:bg-red-700 text-white font-bold py-2 px-4 rounded-lg"
            >
              Logout
            </button>
          </div>

          <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-6">
            {quizzes.map((quiz) => (
              <div
                key={quiz.id}
                className="bg-gradient-to-br from-blue-600 to-cyan-600 rounded-lg p-6 cursor-pointer hover:shadow-lg transform hover:scale-105 transition-all"
                onClick={() => setSelectedQuiz(quiz)}
              >
                <h3 className="text-2xl font-bold mb-2">{quiz.title}</h3>
                <p className="text-gray-100 mb-4">{quiz.description}</p>
                <div className="flex justify-between items-center">
                  <span className="text-sm bg-black bg-opacity-30 px-3 py-1 rounded-full">
                    {quiz.difficulty}
                  </span>
                  <span className="text-sm font-bold">{quiz.questions.length} Q's</span>
                </div>
                <button className="mt-4 w-full bg-white text-blue-600 font-bold py-2 px-4 rounded-lg hover:bg-gray-100">
                  Start Quiz →
                </button>
              </div>
            ))}
          </div>
        </div>
      </div>
    );
  }

  if (showResults) {
    return (
      <div className="min-h-screen bg-gradient-to-br from-gray-900 via-gray-800 to-gray-900 text-white p-6 flex items-center justify-center">
        <div className="text-center">
          <h1 className="text-5xl font-bold mb-4">Quiz Complete!</h1>
          <div className="text-6xl font-bold text-yellow-400 mb-4">
            {Math.round((score / selectedQuiz.questions.length) * 100)}%
          </div>
          <p className="text-2xl mb-8">
            You got {score} out of {selectedQuiz.questions.length} questions correct!
          </p>
          <button
            onClick={() => {
              setSelectedQuiz(null);
              setCurrentQuestion(0);
              setScore(0);
              setShowResults(false);
            }}
            className="bg-blue-600 hover:bg-blue-700 text-white font-bold py-3 px-6 rounded-lg mr-4"
          >
            Back to Quizzes
          </button>
          <button
            onClick={() => navigate('/')}
            className="bg-green-600 hover:bg-green-700 text-white font-bold py-3 px-6 rounded-lg"
          >
            Go to Home
          </button>
        </div>
      </div>
    );
  }

  const question = selectedQuiz.questions[currentQuestion];

  if (!question) {
    return (
      <div className="min-h-screen bg-gradient-to-br from-gray-900 via-gray-800 to-gray-900 text-white p-6 flex items-center justify-center">
        <div className="text-center">
          <p className="text-2xl">No questions available</p>
          <button
            onClick={() => setSelectedQuiz(null)}
            className="mt-4 bg-blue-600 hover:bg-blue-700 text-white font-bold py-2 px-4 rounded-lg"
          >
            Back
          </button>
        </div>
      </div>
    );
  }

  return (
    <div className="min-h-screen bg-gradient-to-br from-gray-900 via-gray-800 to-gray-900 text-white p-6">
      <div className="max-w-2xl mx-auto">
        <div className="flex justify-between items-center mb-8">
          <h1 className="text-3xl font-bold">{selectedQuiz.title}</h1>
          <button
            onClick={() => {
              setSelectedQuiz(null);
              setCurrentQuestion(0);
              setScore(0);
            }}
            className="text-gray-400 hover:text-white"
          >
            ✕
          </button>
        </div>

        {/* Progress */}
        <div className="mb-8">
          <div className="flex justify-between mb-2">
            <span className="text-sm text-gray-400">
              Question {currentQuestion + 1} of {selectedQuiz.questions.length}
            </span>
            <span className="text-sm text-gray-400">Score: {score}</span>
          </div>
          <div className="w-full bg-gray-700 rounded-full h-3 overflow-hidden">
            <div
              className="bg-gradient-to-r from-green-400 to-blue-500 h-full rounded-full transition-all duration-500"
              style={{
                width: `${((currentQuestion + 1) / selectedQuiz.questions.length) * 100}%`,
              }}
            ></div>
          </div>
        </div>

        {/* Question */}
        <div className="bg-gradient-to-br from-purple-600 to-pink-600 rounded-lg p-8 mb-8">
          <p className="text-2xl font-bold mb-6">{question.question}</p>

          <div className="grid grid-cols-2 gap-4">
            {question.options.map((option, idx) => (
              <button
                key={idx}
                className="bg-white bg-opacity-20 hover:bg-opacity-40 text-white font-bold py-4 px-6 rounded-lg transition-all"
                onClick={() => {
                  if (option === question.correctAnswer) {
                    setScore((prev) => prev + 1);
                  }

                  if (currentQuestion < selectedQuiz.questions.length - 1) {
                    setCurrentQuestion((prev) => prev + 1);
                  } else {
                    setShowResults(true);
                  }
                }}
              >
                {option}
              </button>
            ))}
          </div>
        </div>
      </div>
    </div>
  );
};

export default Quiz;
