import React from 'react';

export const XPBar = ({ currentXP, nextLevelXP, level, streak }) => {
  const progressPercentage = (currentXP / nextLevelXP) * 100;

  return (
    <div className="xp-bar bg-gradient-to-r from-purple-600 to-pink-600 rounded-lg p-4 shadow-lg">
      <div className="flex justify-between items-center mb-3">
        <div>
          <p className="text-gray-200 text-sm">Level</p>
          <p className="text-white text-3xl font-bold">{level}</p>
        </div>
        <div>
          <p className="text-gray-200 text-sm">Streak</p>
          <p className="text-yellow-300 text-2xl font-bold flex items-center gap-1">
            🔥 {streak}
          </p>
        </div>
        <div>
          <p className="text-gray-200 text-sm">XP</p>
          <p className="text-white text-lg font-bold">
            {currentXP} / {nextLevelXP}
          </p>
        </div>
      </div>

      <div className="w-full bg-gray-700 rounded-full h-4 overflow-hidden">
        <div
          className="bg-gradient-to-r from-green-400 to-blue-500 h-full rounded-full transition-all duration-500"
          style={{ width: `${Math.min(progressPercentage, 100)}%` }}
        ></div>
      </div>

      <div className="mt-2 text-right">
        <span className="text-xs text-gray-300">
          {progressPercentage.toFixed(0)}% to next level
        </span>
      </div>
    </div>
  );
};

export default XPBar;
