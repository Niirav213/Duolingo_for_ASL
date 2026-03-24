import React from 'react';
import './TopBar.css';

const TopBar = ({ streak = 0, gems = 0, hearts = 5 }) => {
  return (
    <div className="topbar">
      <div className="topbar-items">
        <div className="stat-item streak">
          <span className="icon">🔥</span>
          <span className="value">{streak}</span>
        </div>
        <div className="stat-item gems">
          <span className="icon">💎</span>
          <span className="value">{gems}</span>
        </div>
        <div className="stat-item hearts">
          <span className="icon">❤️</span>
          <span className="value">{hearts}</span>
        </div>
      </div>
    </div>
  );
};

export default TopBar;
