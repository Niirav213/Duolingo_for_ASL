import React, { useEffect } from 'react';
import { useNavigate } from 'react-router-dom';
import { useAuthStore, useGameStore } from '../store';
import { XPBar } from '../components/XPBar';

export const Home = () => {
  const navigate = useNavigate();
  const { accessToken } = useAuthStore();
  const { userStats, streak, fetchUserStats, fetchStreak } = useGameStore();

  useEffect(() => {
    if (!accessToken) {
      navigate('/login');
      return;
    }

    fetchUserStats();
    fetchStreak();
  }, [accessToken, navigate, fetchUserStats, fetchStreak]);

  if (!userStats || !streak) {
    return (
      <div className="flex items-center justify-center min-h-screen bg-gray-900">
        <div className="text-white text-2xl">Loading...</div>
      </div>
    );
  }

  const level = Math.floor(userStats.total_xp / 1000) + 1;
  const currentLevelXP = userStats.total_xp % 1000;
  const nextLevelXP = 1000;

  return (
    <div className="min-h-screen bg-gradient-to-br from-gray-900 via-gray-800 to-gray-900 text-white p-6">
      <div className="max-w-7xl mx-auto">
        {/* Header */}
        <div className="text-center mb-8">
          <h1 className="text-5xl font-bold mb-2">ASL Learning Platform</h1>
          <p className="text-gray-400">Master American Sign Language through interactive games</p>
        </div>

        {/* Stats Cards */}
        <div className="grid grid-cols-1 md:grid-cols-3 gap-4 mb-8">
          <div className="bg-blue-600 bg-opacity-20 border border-blue-500 rounded-lg p-6">
            <p className="text-gray-300 text-sm">Total XP</p>
            <p className="text-4xl font-bold">{userStats.total_xp.toLocaleString()}</p>
          </div>

          <div className="bg-green-600 bg-opacity-20 border border-green-500 rounded-lg p-6">
            <p className="text-gray-300 text-sm">Lessons Completed</p>
            <p className="text-4xl font-bold">{userStats.lessons_completed}</p>
          </div>

          <div className="bg-yellow-600 bg-opacity-20 border border-yellow-500 rounded-lg p-6">
            <p className="text-gray-300 text-sm">Current Streak</p>
            <p className="text-4xl font-bold">🔥 {streak.current_streak}</p>
          </div>
        </div>

        {/* XP Progress Bar */}
        <div className="mb-8">
          <XPBar
            currentXP={currentLevelXP}
            nextLevelXP={nextLevelXP}
            level={level}
            streak={streak.current_streak}
          />
        </div>

        {/* Lessons Grid */}
        <div className="mb-8">
          <h2 className="text-3xl font-bold mb-6">Lessons</h2>
          <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-4">
            {['A-F', 'G-L', 'M-R', 'S-Z', 'Numbers', 'Phrases'].map((lesson) => (
              <div
                key={lesson}
                className="bg-gradient-to-br from-purple-600 to-pink-600 rounded-lg p-6 cursor-pointer hover:shadow-lg transform hover:scale-105 transition-all"
                onClick={() => navigate(`/lesson/${lesson}`)}
              >
                <h3 className="text-2xl font-bold mb-2">{lesson}</h3>
                <p className="text-gray-100">Learn {lesson} signs</p>
                <button className="mt-4 w-full bg-white text-purple-600 font-bold py-2 px-4 rounded-lg hover:bg-gray-100">
                  Start Lesson →
                </button>
              </div>
            ))}
          </div>
        </div>

        {/* Leaderboard Preview */}
        <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
          <div className="bg-gray-800 bg-opacity-50 border border-gray-700 rounded-lg p-6">
            <h3 className="text-2xl font-bold mb-4">Top Learners (XP)</h3>
            <div className="space-y-3">
              {[
                { rank: 1, name: 'Alex', xp: 15000 },
                { rank: 2, name: 'Jordan', xp: 12500 },
                { rank: 3, name: 'Casey', xp: 10200 },
              ].map((user) => (
                <div key={user.rank} className="flex justify-between items-center">
                  <span className="font-bold text-lg">#{user.rank} {user.name}</span>
                  <span className="text-yellow-400 font-bold">{user.xp.toLocaleString()} XP</span>
                </div>
              ))}
            </div>
          </div>

          <div className="bg-gray-800 bg-opacity-50 border border-gray-700 rounded-lg p-6">
            <h3 className="text-2xl font-bold mb-4">Hot Streaks 🔥</h3>
            <div className="space-y-3">
              {[
                { rank: 1, name: 'Taylor', streak: 42 },
                { rank: 2, name: 'Morgan', streak: 38 },
                { rank: 3, name: 'Casey', streak: streak.current_streak },
              ].map((user) => (
                <div key={user.rank} className="flex justify-between items-center">
                  <span className="font-bold text-lg">#{user.rank} {user.name}</span>
                  <span className="text-green-400 font-bold">{user.streak} days</span>
                </div>
              ))}
            </div>
          </div>
        </div>
      </div>
    </div>
  );
};

export default Home;
