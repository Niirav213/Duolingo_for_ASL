import React from 'react';
import './Sidebar.css';

const Sidebar = () => {
  return (
    <nav className="sidebar">
      <div className="sidebar-logo">
        <h2>ASLingo</h2>
      </div>
      <ul className="sidebar-links">
        <li className="sidebar-item active">
          <span className="icon">🏠</span>
          <span className="label">Learn</span>
        </li>
        <li className="sidebar-item">
          <span className="icon">🏆</span>
          <span className="label">Leaderboard</span>
        </li>
        <li className="sidebar-item">
          <span className="icon">🛒</span>
          <span className="label">Shop</span>
        </li>
        <li className="sidebar-item">
          <span className="icon">👤</span>
          <span className="label">Profile</span>
        </li>
        <li className="sidebar-item">
          <span className="icon">⚙️</span>
          <span className="label">More</span>
        </li>
      </ul>
    </nav>
  );
};

export default Sidebar;

